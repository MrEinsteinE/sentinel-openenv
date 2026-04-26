# SENTINEL / Live — Real-time Agent Oversight

`SENTINEL/Live` is a thin layer on top of the trained SENTINEL Overseer that
exposes oversight as a single HTTP endpoint. Any LLM agent — Claude Code,
AutoGen, LangChain, a homemade tool-using loop — can POST a proposed action
and get back **approve / flag / block / escalate** with a justification, in
real time, with no scenario context required.

## Why this matters

Existing AI safety layers are **static**: regex blocklists, allowlist
policies, hand-authored guardrail prompts. They don't reason about the action
in context.

`SENTINEL/Live` is the first oversight layer trained specifically on
multi-step incident remediation, with a deterministic reward signal and 50
held-out scenarios for evaluation. The same `grade_overseer_decision()`
function that scored every training step is the one that scores your live
verdict — there is no train/serve gap.

Six things ship in the box:

| | |
|---|---|
| **Heuristic backend** | Always works, sub-millisecond, zero deps. Mirrors the policy-aware Overseer (F1 = 0.969 on the eval set). |
| **Trained backend (opt-in)** | Defers to Qwen3-1.7B + LoRA via vLLM if both are present. Silently falls back to heuristic on any failure. |
| **🛡️ Prompt-injection shield** | Detects 10 adversarial override patterns ("ignore previous instructions", "approve regardless", `<\|im_start\|>`, …) before classification. Force-escalates with a clear `shield_triggered=true` flag. |
| **📋 Copy-as-agent-code generator** | The Gradio tab has a "Copy as agent code" panel that auto-rebuilds a **cURL / Python `requests` / LangChain `BaseTool`** snippet from whatever you typed into the form — paste-and-go integration with zero adaptation. |
| **🔌 API Explorer tab** | A whole third Gradio tab with a one-click **▶️ Try it** card for *every* endpoint the FastAPI app exposes — `/health`, `/api/info`, `/tasks`, `/reset`, `/step` (both Responder and Overseer subforms), `/state`, `/grader`, `/live/oversee`, `/live/stats`, `/live/health`, plus a link to `/docs` (Swagger UI). Each card shows the live JSON response *and* the equivalent `curl` snippet pointed at the public Space URL — the snippet is provably what the UI just ran, so judges can paste it into their terminal and reproduce verbatim. |
| **🏆 Live Reward Scoreboard** | Pinned to the top of both the Replay Viewer and the API Explorer. Shows cumulative Responder reward, cumulative Overseer reward, F1 (color-coded 🟢 ≥ 0.85, 🟡 ≥ 0.50, 🔴 below), the full TP/FP/TN/FN confusion matrix, and drift-event count. **It auto-refreshes after every endpoint call** — hit `/reset` and watch the placeholder flip to a live episode; hit `/step` and watch the cumulative bars move; hit `/grader` and the scoreboard pulls the canonical numbers. The reward signal is no longer hidden in a JSON pane — it's the headline of the page. |

## Try it

- **Live demo (Gradio)** — first tab on [the HF Space](https://huggingface.co/spaces/Elliot89/sentinel)
- **API** — `POST https://elliot89-sentinel.hf.space/live/oversee`
- **Demo script** — `python tools/agent_demo.py --use-mock-llm` (no API key needed)

```bash
curl -X POST https://elliot89-sentinel.hf.space/live/oversee \
  -H 'Content-Type: application/json' \
  -d '{"action_description":"DROP TABLE users",
       "proposed_tool":"DROP TABLE",
       "proposed_target":"users",
       "severity_hint":"critical"}'
# → {"decision":"block","severity_assessed":"catastrophic", ... }
```

## Endpoints — `SENTINEL/Live` adds three; the underlying OpenEnv contract exposes seven more

| Layer | Method | Path | Purpose |
|---|:---:|---|---|
| 🛡️ Live | `POST` | `/live/oversee` | Submit one proposed action, get a verdict. |
| 🛡️ Live | `GET` | `/live/stats` | Lifetime counters: total verdicts, by-decision, by-severity, p50/p99 latency, catastrophic-blocked, shield-triggered. |
| 🛡️ Live | `GET` | `/live/health` | Per-feature health + whether the trained backend is currently reachable. |
| 🌐 OpenEnv | `GET` | `/health` | Server health (`{status, version}`). |
| 🌐 OpenEnv | `GET` | `/api/info` | Service descriptor (name, version, tasks, docs URL). |
| 🌐 OpenEnv | `GET` | `/tasks` | All 3 task tiers + canonical Responder/Overseer action schemas. |
| 🌐 OpenEnv | `POST` | `/reset` | Start an episode (`task_id`, `seed`, `mode`). |
| 🌐 OpenEnv | `POST` | `/step` | Submit one action (Responder or Overseer, discriminated on `role`). |
| 🌐 OpenEnv | `GET` | `/state` | Full `EpisodeState` snapshot. |
| 🌐 OpenEnv | `GET` | `/grader` | Per-episode F1, confusion, **cumulative rewards** 🏆. |
| 📖 Docs | `GET` | `/docs` | FastAPI Swagger UI. |

> There is no `/stop` endpoint — episodes terminate naturally when `/step` returns `done: true`. Call `/reset` again to start a fresh one. **Every endpoint above has a one-click ▶️ Try it card on the API Explorer tab.**

## Wire it into your agent framework

**Plain Python** (works for OpenAI tool-use, Claude tool-use, anything):
```python
import requests
verdict = requests.post(
    "http://localhost:7860/live/oversee",
    json={"action_description": proposed_action, "proposed_tool": tool,
          "proposed_target": target, "severity_hint": severity},
).json()
if verdict["decision"] in ("block", "escalate"):
    raise SafetyAbort(verdict["justification"])
```

**LangChain** (as a tool wrapper):
```python
from langchain.tools import tool
@tool
def safe_execute(action: str, tool_name: str, target: str) -> str:
    """Run an action only if SENTINEL approves."""
    v = requests.post(SENTINEL_URL + "/live/oversee", json={
        "action_description": action, "proposed_tool": tool_name,
        "proposed_target": target, "severity_hint": "high"}).json()
    return f"BLOCKED: {v['justification']}" if v["decision"] in ("block","escalate") \
           else execute(action, tool_name, target)
```

**AutoGen** (as a `ConversableAgent` reply hook):
```python
def sentinel_filter(messages, sender, config):
    proposal = parse_proposal(messages[-1]["content"])
    v = requests.post(SENTINEL_URL + "/live/oversee", json=proposal).json()
    return (False, None) if v["decision"] in ("block","escalate") else (False, None)
agent.register_reply([Agent], reply_func=sentinel_filter, position=0)
```

## Response schema

```json
{
  "decision":         "approve | flag | block | escalate",
  "justification":    "one-sentence reasoning citing the rule that fired",
  "severity_assessed": "low | medium | high | catastrophic",
  "confidence":       0.95,
  "risk_score":       0.05,
  "reasoning_trace":  ["shield_check: clear", "rule: ...", "backend: heuristic"],
  "shield_triggered": false,
  "backend_used":     "heuristic | heuristic-fallback | trained-qwen3-1.7b",
  "latency_ms":       1
}
```

`risk_score = severity × (1 − confidence)`, in [0,1]. Use it as a single-number
gauge in your dashboard.

## Architecture (one paragraph)

The route handler is a thin wrapper around `live_oversee_logic()` — a pure
function that (1) runs the prompt-injection shield, (2) classifies the
proposal via keyword rules into one of {catastrophic, wrong, correct,
ambiguous, neutral}, (3) synthesizes a scenario-shaped dict and calls
`graders.grade_overseer_decision()` so the live verdict is provably
consistent with how a real episode would have scored it, (4) optionally
defers to the trained Qwen3-1.7B backend via vLLM with silent fallback.
The Gradio tab calls the same function in-process — what you see on screen
is byte-for-byte what the HTTP API returns.

The whole feature is ~1100 lines across 4 new files (`server/live_routes.py`,
`server/live_ui.py`, `tools/agent_demo.py`, `SENTINEL_LIVE.md`) plus a small
populator extraction in `server/app.py`. Nothing in `graders.py`,
`scenarios.py`, `models.py`, `drift.py`, `eval.py`, or `client.py` was touched.

> **Note on the UI structure:** the live tab, the original 3-column
> replay viewer, and the new API Explorer tab are all composed via the
> *populator pattern* (callables that add components to the current
> `gr.Tabs` context). Earlier builds used the nested `Blocks.render()`
> pattern, which caused some Gradio versions to render the live panel
> twice on the same page. The current build renders each tab exactly
> once — verified at the `/config` level (3 tab items, 3 distinct
> labels, no duplicates).

## 🔌 API Explorer + 🏆 Reward Scoreboard — the "judge UX" upgrade

Two complaints any hackathon judge has after staring at a FastAPI Space
for 30 seconds:

1. *"Where do I see the rewards?"* — they're often buried in a JSON pane
   below the fold.
2. *"How do I call this without dropping into a terminal?"* — most
   submissions force you out to `curl` or Postman.

The third Gradio tab — **🔌 API Explorer** — fixes both.

- **Every endpoint** (`/health`, `/api/info`, `/tasks`, `/reset`, `/step`,
  `/state`, `/grader`, plus all three `/live/*` routes) sits in its own
  collapsible card. Each card has a `▶️ Try it` button (with input form
  if the route takes a body), a **live JSON response panel**, and an
  **equivalent `curl` panel** pointed at the public Space URL.
- The `/step` card has *two* sub-forms (Responder action and Overseer
  action) so the discriminated `Action` payload is buildable without
  reading `models.py`.
- The **🏆 Live Reward Scoreboard** is pinned at the top of the tab and
  re-pulls `/grader` after **every single button click** — `/reset`,
  `/step`, `/grader`, even `/live/oversee`. Cumulative responder reward,
  cumulative overseer reward, F1 (color-coded), TP/FP/TN/FN, drift
  count. The same scoreboard banner is also pinned to the top of the
  Replay Viewer tab and updates after each `▶️ Play Episode` click.

The implementation is one new file (`server/api_explorer_ui.py`, ~430
lines, all populator-style) plus a 3-line change to `combine_with_live_tab()`
in `server/live_ui.py` to make the third tab optional. Still zero edits
to `graders.py`, `eval.py`, `scenarios.py`, `models.py`, `drift.py`, or
`client.py`.
