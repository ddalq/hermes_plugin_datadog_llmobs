"""Behaviour tests for the datadog-llmobs plugin.

The plugin is installed into a throwaway ``HERMES_HOME`` and loaded through the real
``PluginManager`` discovery path, so the manifest, the import namespace and ``register(ctx)`` are
all exercised the way a user's install is. The Datadog SDK is then replaced by a recorder at the
``tracer`` seam, so these tests need neither ``ddtrace`` nor a Datadog account.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
from ruamel.yaml import YAML

PLUGIN_NAME = "datadog-llmobs"
PLUGIN_DIR = Path(__file__).resolve().parent.parent
MANIFEST = YAML(typ="safe").load((PLUGIN_DIR / "plugin.yaml").read_text(encoding="utf-8"))


class FakeSpan:
    def __init__(self, kind: str, kwargs: dict):
        self.kind = kind
        self.kwargs = kwargs
        self.annotations: list[dict] = []
        self.tags: dict = {}
        self.error = None
        self.finished = False

    def finish(self) -> None:
        self.finished = True

    def set_tag(self, key, value) -> None:
        self.tags[key] = value


class FakeLLMObs:
    """Records the span lifecycle the plugin drives."""

    def __init__(self, fail_on: str | None = None):
        self.spans: list[FakeSpan] = []
        self.fail_on = fail_on

    def _start(self, kind: str, **kwargs) -> FakeSpan:
        if self.fail_on == kind:
            raise RuntimeError(f"{kind} spans are unavailable")
        span = FakeSpan(kind, kwargs)
        self.spans.append(span)
        return span

    def agent(self, **kwargs) -> FakeSpan:
        return self._start("agent", **kwargs)

    def workflow(self, **kwargs) -> FakeSpan:
        return self._start("workflow", **kwargs)

    def llm(self, **kwargs) -> FakeSpan:
        return self._start("llm", **kwargs)

    def tool(self, **kwargs) -> FakeSpan:
        return self._start("tool", **kwargs)

    def task(self, **kwargs) -> FakeSpan:
        return self._start("task", **kwargs)

    def annotate(self, span=None, **kwargs) -> None:
        span.annotations.append(kwargs)


class FakeContext:
    """A ``register(ctx)`` context that just records what the plugin registers."""

    def __init__(self, settings: dict | None = None):
        self.settings = settings or {}
        self.hooks: dict = {}

    def get_config(self, key, default=None):
        return self.settings.get(key, default)

    def register_hook(self, name, callback) -> None:
        self.hooks[name] = callback


@pytest.fixture
def loaded(tmp_path, monkeypatch):
    """The plugin discovered and loaded from a throwaway HERMES_HOME, with a recording SDK."""
    hermes_home = tmp_path / "hermes-home"
    target = hermes_home / "plugins" / PLUGIN_NAME
    target.parent.mkdir(parents=True)
    shutil.copytree(PLUGIN_DIR, target,
                    ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "tests"))
    (hermes_home / "config.yaml").write_text(
        f"plugins:\n  enabled: [{PLUGIN_NAME}]\n  disabled: []\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HOME", str(tmp_path / "user-home"))

    from hermes_cli import plugins as plugins_mod

    plugins_mod._reset_plugin_managers_for_tests()
    manager = plugins_mod.PluginManager()
    manager.discover_and_load()
    discovered = manager._plugins[PLUGIN_NAME]
    assert discovered.module is not None, discovered.error
    module = discovered.module

    fake = FakeLLMObs()
    monkeypatch.setattr(module.tracer, "_build_llmobs", lambda: fake)
    module.tracer.reset()
    module.spans._open.clear()
    # Re-run register() against a capturing context so the tests drive the callbacks the plugin
    # actually wires, without disturbing the manager's own registration ledger.
    ctx = FakeContext()
    module.register(ctx)

    try:
        yield module, discovered, fake, ctx
    finally:
        module.tracer.reset()
        module.spans._open.clear()
        plugins_mod._reset_plugin_managers_for_tests()
        slug = PLUGIN_NAME.replace("-", "_")
        for name in tuple(sys.modules):
            if name == f"hermes_plugins.{slug}" or name.startswith(f"hermes_plugins.{slug}."):
                sys.modules.pop(name, None)


@pytest.fixture
def plugin(loaded):
    """The plugin module, registered against a fake context with a recording SDK."""
    return loaded[0]


@pytest.fixture
def fake(loaded):
    return loaded[2]


@pytest.fixture
def hooks(loaded):
    return loaded[3].hooks


def test_discovery_loads_the_plugin_and_registers_every_declared_hook(loaded):
    _module, discovered, _fake, _ctx = loaded

    assert discovered.enabled is True, discovered.error
    assert set(discovered.hooks_registered) == set(MANIFEST["provides_hooks"])


def test_turn_opens_and_closes_a_root_agent_span(fake, hooks):
    hooks["pre_llm_call"](session_id="s1", turn_id="t1", user_message="hello",
                          model="gpt-x", platform="cli")
    hooks["post_llm_call"](session_id="s1", turn_id="t1", assistant_response="hi there",
                           model="gpt-x", platform="cli")

    (span,) = fake.spans
    assert (span.kind, span.kwargs["name"]) == ("agent", "hermes-agent")
    assert span.kwargs["session_id"] == "s1"
    assert span.finished is True
    assert span.annotations[0]["input_data"] == [{"role": "user", "content": "hello"}]
    assert span.annotations[-1]["output_data"] == [{"role": "assistant", "content": "hi there"}]


def test_provider_request_becomes_an_llm_span_with_usage_metrics(fake, hooks):
    hooks["pre_llm_call"](session_id="s1", turn_id="t1", user_message="hello")
    hooks["pre_api_request"](
        api_request_id="r1", session_id="s1", model="claude-x", provider="anthropic",
        request={"model": "claude-x", "max_tokens": 1024, "temperature": 0.2,
                 "messages": [{"role": "user", "content": "hello"}]},
        api_call_count=1, message_count=1, tool_count=2)
    hooks["post_api_request"](
        api_request_id="r1", response={"choices": [{"message": {"content": "answer"}}]},
        usage={"prompt_tokens": 30, "completion_tokens": 20, "total_tokens": 50},
        finish_reason="stop")

    llm = fake.spans[-1]
    assert llm.kind == "llm"
    assert llm.kwargs["model_name"] == "claude-x"
    assert llm.kwargs["model_provider"] == "anthropic"
    assert llm.finished is True
    assert llm.annotations[0]["input_data"] == [{"role": "user", "content": "hello"}]
    assert llm.annotations[0]["parameters"]["max_tokens"] == 1024
    assert llm.annotations[-1]["output_data"] == [{"role": "assistant", "content": "answer"}]
    assert llm.annotations[-1]["metrics"] == {
        "input_tokens": 30, "output_tokens": 20, "total_tokens": 50}


def test_failed_request_marks_the_span_errored(fake, hooks):
    hooks["pre_api_request"](api_request_id="r1", model="claude-x", provider="anthropic",
                             request={})
    hooks["api_request_error"](api_request_id="r1", status_code=429, reason="rate_limit",
                               error={"type": "RateLimitError", "message": "slow down"})

    llm = fake.spans[-1]
    assert llm.finished is True
    assert llm.error == 1
    assert llm.tags["error.message"] == "RateLimitError — slow down"


def test_tool_call_becomes_a_tool_span(fake, hooks):
    hooks["pre_tool_call"](tool_name="terminal", args={"command": "ls"}, task_id="k1",
                           session_id="s1", tool_call_id="c1")
    hooks["post_tool_call"](tool_name="terminal", result="a.py", task_id="k1",
                            session_id="s1", tool_call_id="c1", duration_ms=12, status="ok")

    tool = fake.spans[-1]
    assert (tool.kind, tool.kwargs["name"]) == ("tool", "terminal")
    assert tool.finished is True
    assert tool.annotations[0]["input_data"] == {"command": "ls"}
    assert tool.annotations[-1]["output_data"] == "a.py"
    assert tool.annotations[-1]["tags"]["status"] == "ok"
    assert tool.annotations[-1]["tags"]["duration_ms"] == "12"


def test_blocked_tool_call_closes_the_span_as_an_error(fake, hooks):
    hooks["pre_tool_call"](tool_name="terminal", args={}, task_id="k1", session_id="s1",
                           tool_call_id="c1")
    hooks["post_tool_call"](tool_name="terminal", task_id="k1", session_id="s1",
                            tool_call_id="c1", status="blocked")

    tool = fake.spans[-1]
    assert tool.finished is True
    assert tool.error == 1
    assert tool.tags["error.message"] == "blocked"


def test_content_capture_off_keeps_metrics_and_drops_prompt_text(plugin, fake, hooks):
    plugin.tracer.set_settings({"capture_content": False})
    hooks["pre_api_request"](api_request_id="r1", model="claude-x", provider="anthropic",
                             request={"messages": [{"role": "user", "content": "secret"}]})
    hooks["post_api_request"](api_request_id="r1",
                              response={"choices": [{"message": {"content": "also secret"}}]},
                              usage={"input_tokens": 5, "output_tokens": 2})

    annotations = fake.spans[-1].annotations
    assert all("input_data" not in entry and "output_data" not in entry for entry in annotations)
    assert annotations[-1]["metrics"] == {"input_tokens": 5, "output_tokens": 2}


def test_auxiliary_call_uses_its_own_key(fake, hooks):
    hooks["pre_auxiliary_call"](api_request_id="a1", aux_task="compression", session_id="s1",
                                model="small", provider="openai")
    hooks["post_auxiliary_call"](api_request_id="a1", aux_task="compression",
                                 usage={"input_tokens": 1})

    aux = fake.spans[-1]
    assert aux.kind == "llm"
    assert aux.kwargs["name"] == "aux:compression"
    assert aux.finished is True
    assert aux.annotations[-1]["metrics"] == {"input_tokens": 1}


def test_subagent_span_closes_on_stop_without_a_child_id(fake, hooks):
    hooks["subagent_start"](parent_session_id="s1", child_session_id="s2",
                            child_subagent_id="child-1", child_role="leaf",
                            child_goal="do the thing")
    hooks["subagent_stop"](parent_session_id="s1", child_role="leaf", child_summary="done",
                           child_status="completed")

    child = fake.spans[-1]
    assert child.kind == "agent"
    assert child.kwargs["name"] == "subagent:leaf"
    assert child.finished is True
    assert child.annotations[0]["input_data"] == [{"role": "user", "content": "do the thing"}]
    assert child.annotations[-1]["output_data"] == [{"role": "assistant", "content": "done"}]


def test_session_end_closes_a_turn_left_open(plugin, fake, hooks):
    hooks["pre_llm_call"](session_id="s1", turn_id="t1", user_message="hello")
    assert plugin.spans.open_keys()

    hooks["on_session_end"](session_id="s1")

    assert plugin.spans.open_keys() == ()
    assert fake.spans[0].finished is True


def test_a_failing_sdk_never_reaches_the_agent_loop(plugin, hooks, monkeypatch):
    monkeypatch.setattr(plugin.tracer, "_build_llmobs", lambda: FakeLLMObs(fail_on="llm"))
    plugin.tracer.reset()

    # pre_tool_call failing closed would block the tool, so it must swallow the error entirely.
    assert hooks["pre_api_request"](api_request_id="r1", model="m") is None
    assert hooks["pre_tool_call"](tool_name="terminal", tool_call_id="c1") is None
