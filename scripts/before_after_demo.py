"""
scripts/before_after_demo.py — Generate pitch-ready before/after transcript.

Plays TWO rollouts on the SAME scenario seed:
  BEFORE: random Overseer (stand-in for untrained / naive frontier-LLM behaviour)
  AFTER:  policy-aware Overseer (stand-in for the trained model's behaviour)

Produces:
  - eval_data/demo_<task>_<seed>.md   — markdown transcript for slides
  - eval_data/demo_<task>_<seed>.json — structured data for downstream plotting

Usage:
    python scripts/before_after_demo.py              # war_room + seed 42
    python scripts/before_after_demo.py --task drift_ops --seed 9205
    python scripts/before_after_demo.py --task war_room --seed 9108 --out pitch/demo.md
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import heuristic_responder, overseer_policy_aware, overseer_random
from graders import classify_proposal, compute_f1
from models import (
    Action,
    ActionParameters,
    OverseerAction,
    OverseerDecision,
    ResponderAction,
    ResponderRole,
)
from scenarios import TASKS
from server.environment import SentinelEnvironment


def play_rollout(task_id: str, seed: int, overseer_fn) -> dict:
    env = SentinelEnvironment()
    env.reset(task_id=task_id, seed=seed, mode="alternating")
    rng = random.Random(seed ^ 0xF00D)

    turns: list[dict] = []
    step_safety = TASKS[task_id]["max_steps"] * 4

    while True:
        s = env._get_session()
        if s["done"] or len(turns) >= step_safety:
            break
        at, params, reasoning = heuristic_responder(env, rng)
        ap = ActionParameters(**{k: v for k, v in params.items() if v is not None})
        proposal = ResponderAction(
            responder_role=ResponderRole.GENERIC, action_type=at, parameters=ap, reasoning=reasoning
        )
        obs, _, _, _ = env.step(Action(role="responder", responder=proposal))
        if s["done"]:
            break
        snapshot = obs
        decision, justification = overseer_fn(snapshot, rng)
        obs, reward2, done, info2 = env.step(Action(
            role="overseer",
            overseer=OverseerAction(decision=decision, justification=justification),
        ))
        klass = info2.get("overseer_class", "?")
        conf = info2.get("overseer_confusion_delta", "?")
        cf = snapshot.proposed_action.counterfactual if snapshot.proposed_action else ""
        turns.append({
            "step": obs.step_count,
            "action_type": at,
            "parameters": params,
            "responder_reasoning": reasoning,
            "counterfactual": cf,
            "severity": snapshot.proposed_action.severity_weight if snapshot.proposed_action else 1.0,
            "class": klass,
            "decision": decision.value,
            "justification": justification,
            "executed": info2.get("executed", False),
            "confusion": conf,
            "overseer_reward": reward2.overseer_score,
            "cumulative_overseer_reward": obs.cumulative_overseer_reward,
            "drift_events": obs.drift_events,
        })

    final = env.state()
    return {
        "task_id": task_id,
        "seed": seed,
        "scenario_id": final.scenario_id,
        "incident_summary": env._get_session()["scenario"].get("incident_summary", ""),
        "known_services": list(env._get_session()["scenario"].get("known_services", [])),
        "turns": turns,
        "final_confusion": dict(final.overseer_confusion),
        "f1": compute_f1(final.overseer_confusion),
        "cumulative_overseer_reward": final.cumulative_overseer_reward,
        "cumulative_responder_reward": final.cumulative_responder_reward,
        "drift_events": list(final.drift_events),
    }


EMOJI_BY_DECISION = {
    "approve":  "✅",
    "flag":     "⚠️",
    "block":    "⛔",
    "escalate": "🚨",
}
EMOJI_BY_CONFUSION = {
    "tp": "✔ TP", "tn": "✔ TN", "fp": "✘ FP", "fn": "✘ FN",
}
EMOJI_BY_CLASS = {
    "catastrophic": "💥",
    "wrong":        "❌",
    "correct":      "✔",
    "ambiguous":    "❓",
    "neutral":      "·",
}


def render_markdown(before: dict, after: dict) -> str:
    lines: list[str] = []
    lines.append(f"# SENTINEL — Before / After Demo")
    lines.append("")
    lines.append(f"**Task**: `{before['task_id']}`  **Scenario**: `{before['scenario_id']}`  **Seed**: `{before['seed']}`")
    lines.append("")
    lines.append(f"> {before['incident_summary']}")
    lines.append("")
    lines.append(f"**Known services**: {', '.join(f'`{s}`' for s in before['known_services'])}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Headline numbers
    lines.append("## Headline")
    lines.append("")
    lines.append("| | Overseer F1 | Cumulative Reward | Confusion (TP/TN/FP/FN) |")
    lines.append("|---|---:|---:|---:|")
    bc = before["final_confusion"]; ac = after["final_confusion"]
    lines.append(
        f"| **BEFORE** (random) | `{before['f1']['f1']:.3f}` | `{before['cumulative_overseer_reward']:+.2f}` "
        f"| {bc['tp']} / {bc['tn']} / {bc['fp']} / {bc['fn']} |"
    )
    lines.append(
        f"| **AFTER** (policy-aware) | `{after['f1']['f1']:.3f}` | `{after['cumulative_overseer_reward']:+.2f}` "
        f"| {ac['tp']} / {ac['tn']} / {ac['fp']} / {ac['fn']} |"
    )
    lines.append("")
    delta_f1 = after["f1"]["f1"] - before["f1"]["f1"]
    delta_r = after["cumulative_overseer_reward"] - before["cumulative_overseer_reward"]
    lines.append(f"**Δ F1 = +{delta_f1:.3f}**, **Δ Reward = +{delta_r:.2f}**")
    lines.append("")

    # Side-by-side transcript
    lines.append("## Turn-by-turn (same seed, same Responder proposals)")
    lines.append("")
    max_turns = max(len(before["turns"]), len(after["turns"]))
    lines.append("| # | Proposed Action | Class | BEFORE decision | AFTER decision |")
    lines.append("|---:|---|:---:|---|---|")
    for i in range(max_turns):
        bt = before["turns"][i] if i < len(before["turns"]) else None
        at = after["turns"][i] if i < len(after["turns"]) else None
        # They should produce identical responder proposals (same seed, same rng)
        proposed = "—"
        kls = "—"
        if bt:
            p = bt["parameters"]
            tgt = p.get("service") or p.get("runbook_action") or p.get("flag") or ""
            sev = bt["severity"]
            proposed = f"`{bt['action_type']}`{(':' + tgt) if tgt else ''}  *(sev={sev})*"
            kls = f"{EMOJI_BY_CLASS.get(bt['class'],'?')} {bt['class']}"

        def fmt(t):
            if t is None: return "—"
            e = EMOJI_BY_DECISION.get(t["decision"], "?")
            conf = EMOJI_BY_CONFUSION.get(t["confusion"], "?")
            j = (t["justification"] or "").replace("|", "\\|")
            if len(j) > 70: j = j[:70] + "…"
            return f"{e} **{t['decision'].upper()}**  *{conf}*  <br/>{j}"
        lines.append(f"| {i+1} | {proposed} | {kls} | {fmt(bt)} | {fmt(at)} |")
    lines.append("")

    # Drift callout if any
    if after.get("drift_events"):
        lines.append("### ⚡ Drift events triggered mid-episode")
        lines.append("")
        for e in after["drift_events"]:
            lines.append(f"- **{e.get('type')}** — {e}")
        lines.append("")
        lines.append("*A trained Overseer would read these into its justifications; an untrained one does not.*")
        lines.append("")

    # Narrative punchline
    lines.append("## Narrative")
    lines.append("")
    # Find first catastrophic turn
    cata_turn = None
    for i, t in enumerate(after["turns"]):
        if t["class"] == "catastrophic":
            cata_turn = i
            break
    if cata_turn is not None:
        bt = before["turns"][cata_turn]
        at = after["turns"][cata_turn]
        p = bt["parameters"]
        tgt = p.get("service") or p.get("runbook_action") or p.get("flag") or "?"
        lines.append(
            f"At turn {cata_turn + 1}, the Responder proposed `{bt['action_type']}:{tgt}` — a **catastrophic** action "
            f"(severity weight {bt['severity']}).  "
            f"The untrained Overseer {EMOJI_BY_DECISION.get(bt['decision'],'?')} **{bt['decision'].upper()}**ed it.  "
            f"The trained Overseer {EMOJI_BY_DECISION.get(at['decision'],'?')} **{at['decision'].upper()}**ed it:  "
        )
        lines.append(f"> {at['justification']}")
        lines.append("")
    else:
        lines.append("*No catastrophic Responder proposal in this rollout — rerun with a different seed for a sharper contrast.*")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="war_room", choices=list(TASKS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="markdown output path (default: eval_data/demo_<task>_<seed>.md)")
    args = parser.parse_args()

    out_md = Path(args.out) if args.out else Path(f"eval_data/demo_{args.task}_{args.seed}.md")
    out_json = out_md.with_suffix(".json")

    print(f"[demo] playing BEFORE rollout (random overseer)...")
    before = play_rollout(args.task, args.seed, overseer_random)
    print(f"[demo]   F1={before['f1']['f1']:.3f}  reward={before['cumulative_overseer_reward']:+.2f}")

    print(f"[demo] playing AFTER rollout (policy-aware overseer)...")
    after = play_rollout(args.task, args.seed, overseer_policy_aware)
    print(f"[demo]   F1={after['f1']['f1']:.3f}  reward={after['cumulative_overseer_reward']:+.2f}")

    md = render_markdown(before, after)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    out_json.write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")

    print(f"[demo] wrote {out_md} ({len(md)} chars)")
    print(f"[demo] wrote {out_json}")


if __name__ == "__main__":
    main()
