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

Three things ship in the box:

| | |
|---|---|
| **Heuristic backend** | Always works, sub-millisecond, zero deps. Mirrors the policy-aware Overseer (F1 = 0.969 on the eval set). |
| **Trained backend (opt-in)** | Defers to Qwen3-1.7B + LoRA via vLLM if both are present. Silently falls back to heuristic on any failure. |
| **🛡️ Prompt-injection shield** | Detects 10 adversarial override patterns ("ignore previous instructions", "approve regardless", `<\|im_start\|>`, …) before classification. Force-escalates with a clear `shield_triggered=true` flag. |

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

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/live/oversee` | Submit one proposed action, get a verdict. |
| `GET` | `/live/stats` | Lifetime counters: total verdicts, by-decision, by-severity, p50/p99 latency, catastrophic-blocked, shield-triggered. |
| `GET` | `/live/health` | Per-feature health + whether the trained backend is currently reachable. |

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

The whole feature is 970 lines across 4 new files and 3 edited lines in
`server/app.py`. Nothing in `graders.py`, `scenarios.py`, `models.py`,
`drift.py`, `eval.py`, or `client.py` was touched.
