"""Runtime initialization of the Datadog Agent Observability SDK.

Datadog's agentic guide brings the SDK up with ``import ddtrace.auto`` as the first import of the
application entrypoint, which also auto-patches the provider SDKs. A Hermes plugin is imported long
after the process has started, so this module takes the in-code ``LLMObs.enable()`` path instead and
builds every span itself from the observer hooks.

Initialization is lazy: the SDK comes up on the first hook that needs a span, so the spans emitted
before that point are not reported. Everything here is exception-safe and degrades to a no-op, so a
missing or misconfigured SDK can never break a turn.

Configuration precedence is environment variable, then plugin setting, then default:

* ``DD_API_KEY`` — read through the profile secret scope, never straight from ``os.environ``.
* ``DD_SITE`` — default ``datadoghq.com``.
* ``DD_LLMOBS_ML_APP`` — default ``hermes-agent``.
* ``DD_LLMOBS_AGENTLESS_ENABLED`` — default ``true`` (Hermes talks to Datadog intake directly).
* ``DD_LLMOBS_ENABLED`` — when set to a falsey value, the SDK is left off entirely.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from plugins.plugin_utils import SingletonSlot

logger = logging.getLogger(__name__)

DEFAULT_ML_APP = "hermes-agent"
DEFAULT_SITE = "datadoghq.com"

_FALSEY = frozenset({"0", "false", "no", "off", ""})

# Plugin settings read from ctx.get_config() at register() time; environment variables win.
_settings: dict[str, Any] = {
    "capture_content": True,
    "ml_app": DEFAULT_ML_APP,
    "site": DEFAULT_SITE,
}

# The factory below runs at most once per slot lifetime, so its warnings never repeat per hook.
_slot: SingletonSlot = SingletonSlot()


def set_settings(settings: dict[str, Any]) -> None:
    """Record the plugin settings read from the plugin context."""
    _settings.update(settings)


def capture_content() -> bool:
    """Whether prompt/completion text and tool arguments/results may be sent to Datadog."""
    return bool(_settings.get("capture_content", True))


def _env(name: str) -> Optional[str]:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else None


def resolve_ml_app() -> str:
    return _env("DD_LLMOBS_ML_APP") or str(_settings.get("ml_app") or DEFAULT_ML_APP)


def resolve_site() -> str:
    return _env("DD_SITE") or str(_settings.get("site") or DEFAULT_SITE)


def resolve_agentless() -> bool:
    raw = _env("DD_LLMOBS_AGENTLESS_ENABLED")
    return True if raw is None else raw.lower() not in _FALSEY


def _llmobs_disabled_by_env() -> bool:
    raw = _env("DD_LLMOBS_ENABLED")
    return raw is not None and raw.lower() in _FALSEY


def _resolve_api_key() -> Optional[str]:
    """Resolve ``DD_API_KEY`` through the active profile's secret scope."""
    try:
        from agent.secret_scope import get_secret
    except Exception:
        logger.debug("datadog-llmobs: the profile secret scope is unavailable", exc_info=True)
        return None
    try:
        return get_secret("DD_API_KEY") or None
    except Exception:
        # UnscopedSecretError under multiplexing, or an unreadable .env.
        logger.warning("datadog-llmobs: could not read DD_API_KEY", exc_info=True)
        return None


def _build_llmobs() -> Optional[Any]:
    """Import and enable the SDK once. Returns the ``LLMObs`` service, or None when unavailable."""
    if _llmobs_disabled_by_env():
        logger.warning("datadog-llmobs: DD_LLMOBS_ENABLED is falsey; Agent Observability stays off")
        return None
    try:
        from ddtrace.llmobs import LLMObs
    except Exception:
        logger.warning("datadog-llmobs: ddtrace is unavailable; no spans will be sent", exc_info=True)
        return None
    api_key = _resolve_api_key()
    if not api_key:
        logger.warning("datadog-llmobs: DD_API_KEY is not set; no spans will be sent")
        return None
    try:
        LLMObs.enable(
            ml_app=resolve_ml_app(),
            site=resolve_site(),
            api_key=api_key,
            agentless_enabled=resolve_agentless(),
            # Spans come from Hermes' observer hooks, one per provider request. The provider-SDK
            # integrations would emit a second span for the same call, so they stay off.
            integrations_enabled=False,
        )
    except Exception:
        logger.warning("datadog-llmobs: could not enable Agent Observability; no spans will be sent",
                       exc_info=True)
        return None
    logger.info(
        "datadog-llmobs: Agent Observability enabled (ml_app=%s site=%s agentless=%s)",
        resolve_ml_app(), resolve_site(), resolve_agentless(),
    )
    return LLMObs


def get_llmobs() -> Optional[Any]:
    """The enabled ``LLMObs`` service, or None when the plugin cannot emit spans."""
    return _slot.get(_build_llmobs)


def reset() -> None:
    """Drop the cached SDK service (tests, and after a settings change)."""
    _slot.reset()


def current_span() -> Optional[Any]:
    """The span currently active on the tracer, used to parent the span about to be opened."""
    try:
        from ddtrace import tracer as dd_tracer
        return dd_tracer.current_span()
    except Exception:
        logger.debug("datadog-llmobs: reading the active span failed", exc_info=True)
        return None


def context_provider() -> Optional[Any]:
    """The tracer's context provider, used to activate/restore the span in scope."""
    try:
        from ddtrace import tracer as dd_tracer
        return getattr(dd_tracer, "context_provider", None)
    except Exception:
        logger.debug("datadog-llmobs: reading the context provider failed", exc_info=True)
        return None
