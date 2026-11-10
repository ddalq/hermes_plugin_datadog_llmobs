"""datadog-llmobs — Datadog Agent Observability (LLM Observability) for Hermes.

Every hook registered here is a read-only observer: nothing changes what the agent does, and every
callback is wrapped so a Datadog problem can never break a turn. Spans are opened in a ``pre_*``
callback and closed in the matching ``post_*`` callback, keyed by the correlation ids the observer
payloads already carry (``turn_id``, ``api_request_id``, ``tool_call_id``, subagent ids).

Span model — one trace per turn:

* ``pre_llm_call`` / ``post_llm_call`` — the turn's root ``agent`` span.
* ``pre_api_request`` / ``post_api_request`` / ``api_request_error`` — one ``llm`` span per provider
  request, carrying the model, provider, token metrics and (unless content capture is off) the
  messages.
* ``pre_auxiliary_call`` / ``post_auxiliary_call`` — ``llm`` spans for titling, compression, MoA and
  the other auxiliary calls, kept off the turn-scoped path.
* ``pre_tool_call`` / ``post_tool_call`` — one ``tool`` span per tool call.
* ``subagent_start`` / ``subagent_stop`` — one ``agent`` span per delegated child.
* ``on_session_end`` — closes anything the interrupted path left open.
"""

from __future__ import annotations

import functools
import itertools
import json
import logging
from typing import Any, Callable, Optional

from . import spans, tracer

logger = logging.getLogger(__name__)

# Provider payloads are already sanitized and size-capped by the host; this is a second ceiling so a
# single span never carries an unbounded blob.
_MAX_CHARS = 10_000

_LLM_PARAM_KEYS = (
    "model", "max_tokens", "max_completion_tokens", "temperature", "top_p", "stop", "stream",
    "tool_choice", "reasoning_effort",
)
_USAGE_KEYS = {
    "prompt_tokens": "input_tokens",
    "completion_tokens": "output_tokens",
    "input_tokens": "input_tokens",
    "output_tokens": "output_tokens",
    "total_tokens": "total_tokens",
    "cache_read_input_tokens": "cache_read_input_tokens",
    "cache_creation_input_tokens": "cache_creation_input_tokens",
    "reasoning_tokens": "reasoning_tokens",
}
_FAILED_TOOL_STATUSES = frozenset({"error", "blocked", "cancelled"})

_subagent_counter = itertools.count(1)


# --------------------------------------------------------------------------------------------
# payload helpers
# --------------------------------------------------------------------------------------------

def _truncate(value: Any, limit: int = _MAX_CHARS) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            logger.debug("datadog-llmobs: serializing a payload failed", exc_info=True)
            text = str(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… [truncated {len(text) - limit} chars]"


def _bounded(value: Any) -> Any:
    """A value small enough to send as-is, else its truncated JSON form."""
    if value is None:
        return None
    try:
        encoded = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        logger.debug("datadog-llmobs: measuring a payload failed", exc_info=True)
        return _truncate(value)
    return value if len(encoded) <= _MAX_CHARS else _truncate(value)


def _messages(role: str, content: Any) -> Optional[list]:
    text = _truncate(content)
    return [{"role": role, "content": text}] if text else None


def _tags(**values: Any) -> dict[str, str]:
    return {key: str(value) for key, value in values.items() if value is not None and value != ""}


def _capture() -> bool:
    return tracer.capture_content()


def _owner(task_id: str, session_id: str) -> str:
    return task_id or session_id or "default"


def _turn_key(turn_id: str, session_id: str) -> str:
    return turn_id or session_id or "default"


def _tool_key(task_id: str, session_id: str, tool_call_id: str) -> str:
    return f"{_owner(task_id, session_id)}:{tool_call_id}"


def _normalize_messages(messages: Any, limit: int = 50) -> Optional[list]:
    if not isinstance(messages, list) or not messages:
        return None
    out = []
    for message in messages[:limit]:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            parts = [part.get("text") for part in content
                     if isinstance(part, dict) and isinstance(part.get("text"), str)]
            content = "\n".join(parts) or None
        if content is None and message.get("tool_calls"):
            content = json.dumps(message["tool_calls"], ensure_ascii=False, default=str)
        if content is None:
            continue
        out.append({"role": str(message.get("role") or "user"), "content": _truncate(content)})
    return out or None


def _llm_input(request: Any) -> Optional[list]:
    if not isinstance(request, dict):
        return None
    return _normalize_messages(request.get("messages"))


def _llm_output(response: Any) -> Optional[list]:
    if not isinstance(response, dict):
        return None
    content: Any = None
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if content is None and message.get("tool_calls"):
                content = json.dumps(message["tool_calls"], ensure_ascii=False, default=str)
        if content is None:
            content = choices[0].get("text")
    if content is None:
        raw = response.get("content")
        if isinstance(raw, list):
            parts = [part.get("text") for part in raw
                     if isinstance(part, dict) and isinstance(part.get("text"), str)]
            content = "\n".join(parts) or None
        else:
            content = raw
    if content is None:
        return None
    return [{"role": "assistant", "content": _truncate(content)}]


def _llm_parameters(request: Any) -> Optional[dict[str, Any]]:
    if not isinstance(request, dict):
        return None
    parameters = {key: request[key] for key in _LLM_PARAM_KEYS
                  if isinstance(request.get(key), (str, int, float, bool))}
    tools = request.get("tools")
    if isinstance(tools, list):
        parameters["tool_count"] = len(tools)
    return parameters or None


def _usage_metrics(usage: Any) -> dict[str, Any]:
    if not isinstance(usage, dict):
        return {}
    metrics: dict[str, Any] = {}
    for source, target in _USAGE_KEYS.items():
        value = usage.get(source)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics.setdefault(target, value)
    return metrics


def _error_message(error: Any, fallback: str) -> str:
    if isinstance(error, dict):
        parts = [str(error.get(key)) for key in ("type", "message") if error.get(key)]
        if parts:
            return " — ".join(parts)
    return _truncate(error) or fallback


# --------------------------------------------------------------------------------------------
# hooks
# --------------------------------------------------------------------------------------------

def _on_pre_llm_call(session_id: str = "", turn_id: str = "", user_message: str = "",
                     model: str = "", platform: str = "", **_: Any) -> None:
    """Open the turn's root agent span."""
    spans.open_span(
        "agent", _turn_key(turn_id, session_id),
        name=tracer.resolve_ml_app(),
        session_id=session_id or None,
        input_data=_messages("user", user_message) if _capture() else None,
        tags=_tags(platform=platform, model=model),
    )


def _on_post_llm_call(session_id: str = "", turn_id: str = "", assistant_response: str = "",
                      model: str = "", platform: str = "", **_: Any) -> None:
    spans.close_span(
        "agent", _turn_key(turn_id, session_id),
        output_data=_messages("assistant", assistant_response) if _capture() else None,
        tags=_tags(platform=platform, model=model),
    )


def _on_pre_api_request(api_request_id: str = "", session_id: str = "", model: str = "",
                        provider: str = "", request: Any = None, api_call_count: Any = None,
                        message_count: Any = None, tool_count: Any = None, **_: Any) -> None:
    if not api_request_id:
        return
    spans.open_span(
        "llm", api_request_id,
        name=model or "llm",
        session_id=session_id or None,
        model_name=model or None,
        model_provider=provider or None,
        input_data=_llm_input(request) if _capture() else None,
        parameters=_llm_parameters(request),
        tags=_tags(provider=provider, api_call_count=api_call_count,
                   message_count=message_count, tool_count=tool_count),
    )


def _on_post_api_request(api_request_id: str = "", response: Any = None, usage: Any = None,
                         finish_reason: str = "", **_: Any) -> None:
    if not api_request_id:
        return
    spans.close_span(
        "llm", api_request_id,
        output_data=_llm_output(response) if _capture() else None,
        metrics=_usage_metrics(usage),
        tags=_tags(finish_reason=finish_reason),
    )


def _on_api_request_error(api_request_id: str = "", error: Any = None, status_code: Any = None,
                          reason: str = "", **_: Any) -> None:
    if not api_request_id:
        return
    spans.close_span(
        "llm", api_request_id,
        tags=_tags(status_code=status_code, reason=reason),
        error=_error_message(error, reason or "provider request failed"),
    )


def _on_pre_auxiliary_call(api_request_id: str = "", aux_task: str = "", session_id: str = "",
                           model: str = "", provider: str = "", request: Any = None,
                           **_: Any) -> None:
    if not api_request_id:
        return
    spans.open_span(
        "llm", f"aux:{api_request_id}",
        name=f"aux:{aux_task or 'auxiliary'}",
        session_id=session_id or None,
        model_name=model or None,
        model_provider=provider or None,
        input_data=_llm_input(request) if _capture() else None,
        parameters=_llm_parameters(request),
        tags=_tags(aux_task=aux_task, provider=provider),
    )


def _on_post_auxiliary_call(api_request_id: str = "", aux_task: str = "", response: Any = None,
                            usage: Any = None, error: Any = None, error_type: str = "",
                            **_: Any) -> None:
    if not api_request_id:
        return
    key = f"aux:{api_request_id}"
    if error or error_type:
        spans.close_span("llm", key, tags=_tags(aux_task=aux_task, error_type=error_type),
                         error=_error_message(error, error_type or "auxiliary call failed"))
        return
    spans.close_span(
        "llm", key,
        output_data=_llm_output(response) if _capture() else None,
        metrics=_usage_metrics(usage),
        tags=_tags(aux_task=aux_task),
    )


def _on_pre_tool_call(tool_name: str = "", args: Any = None, task_id: str = "", session_id: str = "",
                      tool_call_id: str = "", **_: Any) -> None:
    """Open a tool span. Never raises: a raise here would block the tool."""
    if not tool_call_id or not tool_name:
        return
    spans.open_span(
        "tool", _tool_key(task_id, session_id, tool_call_id),
        name=tool_name,
        session_id=session_id or None,
        input_data=_bounded(args) if _capture() else None,
        tags=_tags(tool_name=tool_name),
    )


def _on_post_tool_call(tool_name: str = "", result: Any = None, task_id: str = "",
                       session_id: str = "", tool_call_id: str = "", duration_ms: Any = None,
                       status: str = "", error_type: str = "", error_message: str = "",
                       **_: Any) -> None:
    key = _tool_key(task_id, session_id, tool_call_id)
    tags = _tags(tool_name=tool_name, status=status, duration_ms=duration_ms, error_type=error_type)
    if error_message or status in _FAILED_TOOL_STATUSES:
        spans.close_span("tool", key, tags=tags,
                         error=_truncate(error_message) or status or "tool call failed")
        return
    spans.close_span("tool", key, output_data=_bounded(result) if _capture() else None, tags=tags)


def _on_subagent_start(parent_session_id: str = "", child_session_id: str = "",
                       child_subagent_id: str = "", child_role: str = "", child_goal: str = "",
                       **_: Any) -> None:
    child_id = child_subagent_id or child_session_id or f"child-{next(_subagent_counter)}"
    spans.open_span(
        "agent", spans.subagent_key(parent_session_id, child_id),
        name=f"subagent:{child_role or 'child'}",
        session_id=child_session_id or parent_session_id or None,
        input_data=_messages("user", child_goal) if _capture() else None,
        tags=_tags(child_role=child_role, parent_session_id=parent_session_id),
    )


def _on_subagent_stop(parent_session_id: str = "", child_role: str = "", child_summary: str = "",
                      child_status: str = "", duration_ms: Any = None, **kwargs: Any) -> None:
    # `child_subagent_id` is not in the documented stop payload, so it is read leniently and the
    # oldest still-open child of this parent stands in when it is absent.
    key = spans.take_subagent(parent_session_id, kwargs.get("child_subagent_id"))
    if key is None:
        return
    tags = _tags(child_role=child_role, status=child_status, duration_ms=duration_ms)
    if child_status and child_status != "completed":
        spans.close_span("agent", key, tags=tags, error=child_status)
        return
    spans.close_span("agent", key,
                     output_data=_messages("assistant", child_summary) if _capture() else None,
                     tags=tags)


def _on_session_end(session_id: str = "", **_: Any) -> None:
    """Close spans an interrupted or blocked turn left open."""
    spans.close_session(session_id)


def _observe(callback: Callable[..., Any]) -> Callable[..., Any]:
    """Hook callbacks are fail-open: Hermes logs and skips a raise, and this keeps that cheap."""
    @functools.wraps(callback)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return callback(*args, **kwargs)
        except Exception:
            logger.debug("datadog-llmobs: %s failed", callback.__name__, exc_info=True)
            return None
    return wrapper


_HOOKS: dict[str, Callable[..., Any]] = {
    "pre_llm_call": _on_pre_llm_call,
    "post_llm_call": _on_post_llm_call,
    "pre_api_request": _on_pre_api_request,
    "post_api_request": _on_post_api_request,
    "api_request_error": _on_api_request_error,
    "pre_auxiliary_call": _on_pre_auxiliary_call,
    "post_auxiliary_call": _on_post_auxiliary_call,
    "pre_tool_call": _on_pre_tool_call,
    "post_tool_call": _on_post_tool_call,
    "subagent_start": _on_subagent_start,
    "subagent_stop": _on_subagent_stop,
    "on_session_end": _on_session_end,
}


def register(ctx: Any) -> None:
    """Wire the observer hooks and read the plugin's settings."""
    tracer.set_settings({
        "capture_content": bool(ctx.get_config("capture_content", default=True)),
        "ml_app": str(ctx.get_config("ml_app", default=tracer.DEFAULT_ML_APP)
                      or tracer.DEFAULT_ML_APP),
        "site": str(ctx.get_config("site", default=tracer.DEFAULT_SITE) or tracer.DEFAULT_SITE),
    })
    for name, callback in _HOOKS.items():
        ctx.register_hook(name, _observe(callback))
