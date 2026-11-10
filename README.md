# datadog-llmobs — Datadog Agent Observability for Hermes

Sends Hermes agent activity to [Datadog Agent Observability](https://docs.datadoghq.com/llm_observability/)
(LLM Observability): one agent trace per turn, one LLM span per provider request, one tool span per
tool call, and one agent span per delegated subagent.

It is a plain Hermes plugin — read-only observer hooks, no core changes. Nothing it does can change
what the agent decides, and every callback swallows its own errors, so a Datadog outage or a missing
SDK never breaks a turn.

## Requirements

- A Datadog API key (`DD_API_KEY`).
- `ddtrace`, declared in this plugin's `pyproject.toml`. Enabling the plugin asks for dependency
  consent and installs it through Hermes' package manager.

## Install

Hermes loads plugins from `<HERMES_HOME>/plugins/<name>/` (usually `~/.hermes/plugins/`):

```bash
cp -R hermes_plugin_datadog_llmobs ~/.hermes/plugins/datadog-llmobs
hermes plugins doctor ~/.hermes/plugins/datadog-llmobs --ci   # optional pre-flight check
hermes plugins enable datadog-llmobs
```

`hermes plugins enable` prompts for `DD_API_KEY` and saves it to the profile `.env`. The plugin is
disabled while that variable is unset.

## Environment variables

Set before Hermes starts, or in the profile `.env`:

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `DD_API_KEY` | yes | — | Datadog API key used to submit Agent Observability data. |
| `DD_SITE` | no | `datadoghq.com` | Datadog site the key belongs to. |
| `DD_LLMOBS_ML_APP` | no | `hermes-agent` | Application all spans are grouped under. |
| `DD_LLMOBS_AGENTLESS_ENABLED` | no | `true` | Submit straight to Datadog intake instead of through a local Agent. Set `false` when a Datadog Agent is running. |
| `DD_LLMOBS_ENABLED` | no | unset | Set to a falsey value to leave the SDK off entirely. |

Each variable wins over the matching plugin setting.

## Settings

Set in `config.yaml` under `plugins.entries.datadog-llmobs.settings`, or from the Desktop app's
plugin page (Settings → Plugins → the gear on this plugin's row):

| Key | Type | Default | Meaning |
|---|---|---|---|
| `capture_content` | bool | `true` | Send prompt/completion text and tool arguments/results. Off = metadata and token metrics only. |
| `ml_app` | str | `hermes-agent` | Application name; `DD_LLMOBS_ML_APP` overrides it. |
| `site` | str | `datadoghq.com` | Datadog site; `DD_SITE` overrides it. |

Content is off-limits to Datadog when `capture_content: false`: no prompt, completion, tool argument
or tool result text is annotated, only span names, tags and token counts.

## Span model

| Span | Opened by | Closed by | Carries |
|---|---|---|---|
| `agent` (root, one per turn) | `pre_llm_call` | `post_llm_call` | user message, assistant response, model, platform, session id |
| `llm` (one per provider request) | `pre_api_request` | `post_api_request` / `api_request_error` | model, provider, messages, sampling parameters, token metrics, finish reason, error |
| `llm` (one per auxiliary call) | `pre_auxiliary_call` | `post_auxiliary_call` | `aux_task` (titling, compression, …), token metrics |
| `tool` (one per tool call) | `pre_tool_call` | `post_tool_call` | tool name, arguments, result, status, duration |
| `agent` (one per delegated child) | `subagent_start` | `subagent_stop` | delegated goal, child role, summary, status |
| — | `on_session_end` | | closes anything an interrupted or blocked turn left open |

Spans opened for a turn nest under its root `agent` span, so a turn is one trace in the Datadog
waterfall. Parentage comes from the tracer's active span, so each opened span is activated on the
context provider and the previously active span is restored when it closes.

## How the tracer is loaded

The Datadog agentic guide starts the SDK with `import ddtrace.auto` as the first import of the
application entrypoint. A Hermes plugin is imported long after the process has started, so this
plugin instead calls `LLMObs.enable(...)` from the first hook that needs a span, and builds every
span itself from Hermes' observer hooks. Provider-SDK auto-instrumentation is deliberately left off
(`integrations_enabled=False`) — it would emit a second span for each call the hooks already cover.

## Known limitations

- The SDK initializes on first use, so the earliest spans of a process are not reported.
- Tool calls that Hermes dispatches on a worker thread can end up as flat traces rather than nested
  under the turn, because the tracer's active span is per-context.
- `subagent_stop` does not document a child id, so a delegated child's span is closed by the oldest
  still-open child of that parent when the id is absent.
- A `post_*` hook that never arrives (blocked, cancelled, interrupted) leaves its span open until a
  sweep finishes it, or until `on_session_end` closes everything for the session.

## Tests

The plugin's own tests load it through the real `PluginManager` against a throwaway `HERMES_HOME`
and drive the hooks with a recording SDK, so they need neither `ddtrace` nor Datadog credentials:

```bash
scripts/run_tests.sh hermes_plugin_datadog_llmobs/tests
```
