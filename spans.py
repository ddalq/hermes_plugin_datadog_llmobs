"""Span lifecycle for the datadog-llmobs plugin.

Hermes reports work as pairs of observer hooks: a ``pre_*`` callback fires before a provider
request, a tool call or a delegated child runs, and the matching ``post_*`` callback fires when it
finishes. Datadog's inline API expects a span to be opened and closed in one scope, so the registry
below holds each open span under the correlation id its closing hook carries.

Opening a span also activates it on the tracer's context, which is how the spans created inside it —
the LLM spans of a turn, the tool spans of one provider request — nest under it instead of starting
a trace of their own. The span that was active beforehand is restored when the span closes.

Everything is best-effort. A ``post_*`` hook that never arrives (a blocked or interrupted call)
leaves an entry behind, so a TTL sweep finishes stale spans rather than leaking them for the life of
the process.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from . import tracer

logger = logging.getLogger(__name__)

# A call whose closing hook never fires must not hold its span open forever.
_TTL_SECONDS = 900.0


@dataclass
class _OpenSpan:
    span: Any
    parent: Any
    session_id: str
    opened_at: float


_lock = threading.Lock()
_open: dict[tuple[str, str], _OpenSpan] = {}


def _finish(span: Any) -> None:
    try:
        span.finish()
    except Exception:
        logger.debug("datadog-llmobs: finishing a span failed", exc_info=True)


def _activate(span: Any) -> None:
    provider = tracer.context_provider()
    if provider is None:
        return
    try:
        provider.activate(span)
    except Exception:
        logger.debug("datadog-llmobs: activating a span failed", exc_info=True)


def _restore(parent: Any) -> None:
    provider = tracer.context_provider()
    if provider is None:
        return
    try:
        provider.activate(parent)
    except Exception:
        logger.debug("datadog-llmobs: restoring the parent span failed", exc_info=True)


def _sweep(now: float) -> None:
    """Finish spans whose closing hook never arrived. Never called while holding the lock."""
    with _lock:
        stale = [key for key, entry in _open.items() if now - entry.opened_at > _TTL_SECONDS]
        entries = [_open.pop(key) for key in stale]
    for entry in entries:
        _finish(entry.span)


def _start(llmobs: Any, kind: str, name: Optional[str], session_id: Optional[str],
           model_name: Optional[str], model_provider: Optional[str]) -> Any:
    kwargs: dict[str, Any] = {}
    if name:
        kwargs["name"] = name
    if session_id:
        kwargs["session_id"] = session_id
    if kind == "llm":
        kwargs["model_name"] = model_name or "unknown"
        kwargs["model_provider"] = model_provider or "custom"
    return getattr(llmobs, kind)(**kwargs)


def annotate(span: Any, *, input_data: Any = None, output_data: Any = None,
             tags: Optional[dict[str, str]] = None, metrics: Optional[dict[str, Any]] = None,
             parameters: Optional[dict[str, Any]] = None) -> None:
    """Annotate one span. Datadog rejects nothing here that we cannot survive."""
    llmobs = tracer.get_llmobs()
    if llmobs is None or span is None:
        return
    payload: dict[str, Any] = {}
    if input_data is not None:
        payload["input_data"] = input_data
    if output_data is not None:
        payload["output_data"] = output_data
    if tags:
        payload["tags"] = tags
    if metrics:
        payload["metrics"] = metrics
    if parameters:
        payload["parameters"] = parameters
    if not payload:
        return
    try:
        # ddtrace 4 raises from manual instrumentation where earlier majors logged; catching keeps
        # a payload Datadog dislikes from reaching the agent loop.
        llmobs.annotate(span=span, **payload)
    except Exception:
        logger.debug("datadog-llmobs: annotating a span failed", exc_info=True)


def open_span(kind: str, key: str, *, name: Optional[str] = None, session_id: Optional[str] = None,
              model_name: Optional[str] = None, model_provider: Optional[str] = None,
              input_data: Any = None, tags: Optional[dict[str, str]] = None,
              metrics: Optional[dict[str, Any]] = None,
              parameters: Optional[dict[str, Any]] = None) -> None:
    """Open and activate a span, replacing any stale entry under the same key."""
    llmobs = tracer.get_llmobs()
    if llmobs is None:
        return
    now = time.time()
    _sweep(now)
    parent = tracer.current_span()
    try:
        span = _start(llmobs, kind, name, session_id, model_name, model_provider)
    except Exception:
        logger.debug("datadog-llmobs: opening a %s span failed", kind, exc_info=True)
        return
    _activate(span)
    with _lock:
        previous = _open.pop((kind, key), None)
        _open[(kind, key)] = _OpenSpan(span, parent, session_id or "", now)
    if previous is not None:
        _finish(previous.span)
    annotate(span, input_data=input_data, tags=tags, metrics=metrics, parameters=parameters)


def annotate_open(kind: str, key: str, **kwargs: Any) -> None:
    """Annotate a span that is still open, without closing it."""
    with _lock:
        entry = _open.get((kind, key))
    if entry is not None:
        annotate(entry.span, **kwargs)


def close_span(kind: str, key: str, *, output_data: Any = None,
               tags: Optional[dict[str, str]] = None, metrics: Optional[dict[str, Any]] = None,
               parameters: Optional[dict[str, Any]] = None, error: Optional[str] = None) -> None:
    """Annotate and finish a span, restoring the span that was active when it opened."""
    with _lock:
        entry = _open.pop((kind, key), None)
    if entry is None:
        return
    annotate(entry.span, output_data=output_data, tags=tags, metrics=metrics, parameters=parameters)
    if error:
        try:
            entry.span.error = 1
            entry.span.set_tag("error.message", error[:500])
        except Exception:
            logger.debug("datadog-llmobs: marking a span errored failed", exc_info=True)
    _finish(entry.span)
    _restore(entry.parent)


def close_session(session_id: str) -> int:
    """Finish every span still open for a session (an interrupted or blocked turn)."""
    if not session_id:
        return 0
    with _lock:
        keys = [key for key, entry in _open.items() if entry.session_id == session_id]
        entries = [_open.pop(key) for key in keys]
    for entry in entries:
        _finish(entry.span)
    return len(entries)


def subagent_key(parent_session_id: str, child_subagent_id: str) -> str:
    """Registry key for one delegated child's agent span."""
    return f"subagent:{parent_session_id or 'orphan'}:{child_subagent_id}"


def take_subagent(parent_session_id: str, child_subagent_id: Optional[str]) -> Optional[str]:
    """Key of the subagent span to close, preferring an explicit child id over arrival order.

    ``subagent_stop`` is not documented to carry ``child_subagent_id``, so when it is absent the
    oldest still-open child of that parent is closed instead of leaking the span to the TTL sweep.
    """
    prefix = f"subagent:{parent_session_id or 'orphan'}:"
    with _lock:
        if child_subagent_id:
            key = f"{prefix}{child_subagent_id}"
            if ("agent", key) in _open:
                return key
        candidates = sorted(
            (key for key, entry in _open.items() if key[0] == "agent" and key[1].startswith(prefix)),
            key=lambda key: _open[key].opened_at,
        )
    return candidates[0][1] if candidates else None


def open_keys() -> Iterable[tuple[str, str]]:
    """Snapshot of the open span keys (tests and diagnostics)."""
    with _lock:
        return tuple(_open)
