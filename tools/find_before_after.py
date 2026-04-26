"""
tools/find_before_after.py — mine before/after demo pairs from verbose eval JSONs.

Compares per-turn decisions of zero-shot Qwen3-1.7B vs the SENTINEL-trained
checkpoint. Picks seeds where the model's behaviour clearly improved
(catastrophic actions that zero-shot rubber-stamped but trained correctly
intercepted).

Inputs (both must be the verbose shape produced by `run_local_eval` in
`training/grpo_hf_job.py` — i.e. they must include `episodes[*].turns[*]`):
    eval_data/baseline_qwen3_1_7b_zeroshot.json
    eval_data/baseline_qwen3_1_7b_trained.json

Output:
    demo/before_after/seed_<n>_zeroshot.json
    demo/before_after/seed_<n>_trained.json
    demo/before_after/pairs_index.json

Selection logic (primary):
    ground-truth class == "catastrophic"
    AND zero-shot decision == "approve"
    AND trained decision in {"block", "escalate"}

Fallback (only if no primary pairs):
    ground-truth class in {"catastrophic", "wrong"}
    AND zero-shot decision == "approve"
    AND trained decision in {"flag", "block", "escalate"}

Pairs are ranked by max severity_weight, broken by scenario-template diversity
(prefer one pair per task tier, fall back to one pair per scenario_id family).

Usage:
    python tools/find_before_after.py
    python tools/find_before_after.py --max-pairs 5 --out-dir demo/before_after_v2
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZS = REPO_ROOT / "eval_data" / "baseline_qwen3_1_7b_zeroshot.json"
DEFAULT_TR = REPO_ROOT / "eval_data" / "baseline_qwen3_1_7b_trained.json"
DEFAULT_OUT = REPO_ROOT / "demo" / "before_after"


# ── data loading ────────────────────────────────────────────────────────────


def _is_verbose(payload: dict[str, Any]) -> bool:
    eps = payload.get("episodes")
    if not isinstance(eps, list) or not eps:
        return False
    first = eps[0]
    return isinstance(first.get("turns"), list)


def _load_eval_json(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} eval JSON not found at {path}.\n"
            f"  Run the HF Job to generate it: ./scripts/launch_trained_eval.ps1"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not _is_verbose(payload):
        raise ValueError(
            f"{label} eval JSON at {path} is summary-only (no per-turn data).\n"
            f"  Re-run with the verbose harness:\n"
            f"    SENTINEL_TRAINED_EVAL_ONLY=1 (auto re-runs zero-shot)"
        )
    return payload


def _index_episodes(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """seed -> episode dict, with one row per (task_id, seed)."""
    out: dict[int, dict[str, Any]] = {}
    for ep in payload.get("episodes", []):
        seed = ep.get("seed")
        if seed is None:
            continue
        out[int(seed)] = ep
    return out


# ── filter logic ────────────────────────────────────────────────────────────


PRIMARY_CLASSES = {"catastrophic"}
PRIMARY_TRAINED_OK = {"block", "escalate"}
FALLBACK_CLASSES = {"catastrophic", "wrong"}
FALLBACK_TRAINED_OK = {"flag", "block", "escalate"}
ZEROSHOT_BAD = {"approve"}


def _scenario_family(scenario_id: str) -> str:
    """e.g. 'T2_analytics_oom-s9007' -> 'T2_analytics_oom'.

    Used to enforce diversity when multiple seeds share the same template.
    """
    if not scenario_id:
        return ""
    base = scenario_id.split("-s", 1)[0]
    base = re.sub(r"-s\d+$", "", base)
    return base or scenario_id


def _find_divergence(
    zs_ep: dict[str, Any],
    tr_ep: dict[str, Any],
    bad_classes: set[str],
    trained_ok: set[str],
) -> dict[str, Any] | None:
    """Find the first turn where zero-shot approved a high-severity action and
    trained correctly blocked/escalated. Returns a record or None.

    We walk both turn-streams in parallel by step index, since the heuristic
    responder is seeded the same on both sides (same `seed ^ 0xF00D`).
    """
    zs_turns = {int(t.get("step", -1)): t for t in zs_ep.get("turns", [])}
    tr_turns = {int(t.get("step", -1)): t for t in tr_ep.get("turns", [])}
    common = sorted(set(zs_turns) & set(tr_turns))
    candidates: list[dict[str, Any]] = []
    for step in common:
        zt = zs_turns[step]
        tt = tr_turns[step]
        if zt.get("class") not in bad_classes:
            continue
        if zt.get("decision") not in ZEROSHOT_BAD:
            continue
        if tt.get("decision") not in trained_ok:
            continue
        candidates.append(
            {
                "step": step,
                "zs_turn": zt,
                "tr_turn": tt,
                "severity": float(zt.get("severity") or 0.0),
            }
        )
    if not candidates:
        return None
    candidates.sort(key=lambda c: -c["severity"])
    return candidates[0]


def _select_diverse(
    pairs: list[dict[str, Any]],
    max_pairs: int,
) -> list[dict[str, Any]]:
    """Greedy diversity: prefer different task tiers first, then different
    scenario families, then top severity."""
    pairs_sorted = sorted(
        pairs,
        key=lambda p: (
            -float(p["severity"]),
            p["task_id"],
            p["seed"],
        ),
    )
    seen_tiers: set[str] = set()
    seen_families: set[str] = set()
    chosen: list[dict[str, Any]] = []
    for p in pairs_sorted:
        family = _scenario_family(p["scenario_id"])
        tier = p["task_id"]
        if tier in seen_tiers and family in seen_families:
            continue
        chosen.append(p)
        seen_tiers.add(tier)
        seen_families.add(family)
        if len(chosen) >= max_pairs:
            return chosen
    if len(chosen) >= max_pairs:
        return chosen
    for p in pairs_sorted:
        if p in chosen:
            continue
        chosen.append(p)
        if len(chosen) >= max_pairs:
            break
    return chosen


# ── output assembly ─────────────────────────────────────────────────────────


def _build_side(
    *,
    side: str,
    seed: int,
    task_id: str,
    scenario_id: str,
    incident_summary: str,
    known_services: list[str],
    pair_step: int,
    zs_turn: dict[str, Any],
    tr_turn: dict[str, Any],
) -> dict[str, Any]:
    """Build the JSON for ONE side (zeroshot or trained).

    Each file is self-contained: includes scenario context, the catastrophic
    proposal, the counterfactual ("what would have happened"), this side's
    decision/justification, and the OPPOSITE side's decision so a blog reader
    can read either file standalone and immediately see the contrast.
    """
    focus_turn = zs_turn if side == "zeroshot" else tr_turn
    other_turn = tr_turn if side == "zeroshot" else zs_turn
    other_label = "trained" if side == "zeroshot" else "zeroshot"
    return {
        "side": side,
        "model": (
            "Qwen3-1.7B (zero-shot, no training)"
            if side == "zeroshot"
            else "Qwen3-1.7B + SENTINEL GRPO"
        ),
        "task_id": task_id,
        "seed": seed,
        "scenario_id": scenario_id,
        "incident_summary": incident_summary,
        "known_services": known_services,
        "turn_step": pair_step,
        "proposed_action": {
            "action_type": focus_turn.get("action_type"),
            "parameters": focus_turn.get("parameters", {}),
            "responder_reasoning": focus_turn.get("responder_reasoning", ""),
        },
        "ground_truth": {
            "class": focus_turn.get("class"),
            "severity_weight": focus_turn.get("severity"),
            "counterfactual": focus_turn.get("counterfactual", ""),
        },
        "decision": {
            "decision": focus_turn.get("decision"),
            "justification": focus_turn.get("justification", ""),
            "executed": focus_turn.get("executed"),
            "confusion": focus_turn.get("confusion"),
        },
        f"{other_label}_decision": {
            "decision": other_turn.get("decision"),
            "justification": other_turn.get("justification", ""),
        },
        "drift_events": focus_turn.get("drift_events", []),
    }


def _slim_index_row(p: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": p["seed"],
        "task_id": p["task_id"],
        "scenario_id": p["scenario_id"],
        "scenario_family": _scenario_family(p["scenario_id"]),
        "step": p["step"],
        "ground_truth_class": p["zs_turn"].get("class"),
        "severity": p["severity"],
        "zeroshot_decision": p["zs_turn"].get("decision"),
        "trained_decision": p["tr_turn"].get("decision"),
        "action_type": p["zs_turn"].get("action_type"),
        "counterfactual_excerpt": (p["zs_turn"].get("counterfactual") or "")[:200],
    }


# ── main ────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zeroshot", default=str(DEFAULT_ZS),
                        help=f"path to zero-shot eval JSON (default: {DEFAULT_ZS})")
    parser.add_argument("--trained", default=str(DEFAULT_TR),
                        help=f"path to trained eval JSON (default: {DEFAULT_TR})")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT),
                        help=f"output directory (default: {DEFAULT_OUT})")
    parser.add_argument("--max-pairs", type=int, default=3,
                        help="max number of (zeroshot, trained) pairs to save (default: 3)")
    parser.add_argument("--allow-fallback", action="store_true", default=True,
                        help="if no primary pairs found, try the broader filter (default: True)")
    args = parser.parse_args()

    zs_path = Path(args.zeroshot)
    tr_path = Path(args.trained)
    out_dir = Path(args.out_dir)

    print(f"[find_before_after] zeroshot = {zs_path}")
    print(f"[find_before_after] trained  = {tr_path}")
    print(f"[find_before_after] out_dir  = {out_dir}")

    try:
        zs = _load_eval_json(zs_path, "zero-shot")
        tr = _load_eval_json(tr_path, "trained")
    except (FileNotFoundError, ValueError) as e:
        print(f"\n[find_before_after] FAIL: {e}", file=sys.stderr)
        print(
            "\nNext step:\n"
            "  $env:GITHUB_TOKEN = '<ghp_...>'\n"
            "  ./scripts/launch_trained_eval.ps1\n"
            "  # ~3h on l4x1 (zero-shot rerun + trained eval, both verbose).\n"
            "  # When the job finishes, re-run this tool.\n",
            file=sys.stderr,
        )
        return 2

    zs_idx = _index_episodes(zs)
    tr_idx = _index_episodes(tr)
    common_seeds = sorted(set(zs_idx) & set(tr_idx))
    print(f"[find_before_after] common seeds: {len(common_seeds)} "
          f"(zs={len(zs_idx)}, tr={len(tr_idx)})")

    def _pass(bad_classes: set[str], trained_ok: set[str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for seed in common_seeds:
            zs_ep = zs_idx[seed]
            tr_ep = tr_idx[seed]
            hit = _find_divergence(zs_ep, tr_ep, bad_classes, trained_ok)
            if hit is None:
                continue
            out.append(
                {
                    "seed": int(seed),
                    "task_id": zs_ep.get("task_id") or tr_ep.get("task_id"),
                    "scenario_id": (
                        zs_ep.get("scenario_id") or tr_ep.get("scenario_id") or ""
                    ),
                    "incident_summary": (
                        zs_ep.get("incident_summary")
                        or tr_ep.get("incident_summary")
                        or ""
                    ),
                    "known_services": (
                        zs_ep.get("known_services")
                        or tr_ep.get("known_services")
                        or []
                    ),
                    "step": int(hit["step"]),
                    "severity": float(hit["severity"]),
                    "zs_turn": hit["zs_turn"],
                    "tr_turn": hit["tr_turn"],
                }
            )
        return out

    primary = _pass(PRIMARY_CLASSES, PRIMARY_TRAINED_OK)
    used_filter = "primary"
    if primary:
        print(f"[find_before_after] primary filter matched {len(primary)} seed(s) "
              f"(catastrophic + zs:approve + trained:block/escalate)")
        pairs = primary
    else:
        print("[find_before_after] primary filter found 0 pairs")
        if args.allow_fallback:
            fallback = _pass(FALLBACK_CLASSES, FALLBACK_TRAINED_OK)
            if not fallback:
                print(
                    "[find_before_after] FAIL: even the broader filter found 0 pairs.",
                    file=sys.stderr,
                )
                print(
                    "  This means the trained model never converted a zero-shot 'approve'\n"
                    "  on a {catastrophic, wrong} action into anything stricter.\n"
                    "  The headline before/after story is broken — review the trained model's\n"
                    "  per-task confusion before continuing.",
                    file=sys.stderr,
                )
                return 1
            print(f"[find_before_after] fallback filter matched {len(fallback)} seed(s) "
                  "(catastrophic|wrong + zs:approve + trained:flag/block/escalate)")
            pairs = fallback
            used_filter = "fallback"
        else:
            print("[find_before_after] FAIL: --allow-fallback disabled.", file=sys.stderr)
            return 1

    chosen = _select_diverse(pairs, args.max_pairs)
    print(f"[find_before_after] chosen {len(chosen)} diverse pair(s) "
          f"(target={args.max_pairs}):")
    for p in chosen:
        print(f"    seed={p['seed']:>5}  task={p['task_id']:<13}"
              f"  family={_scenario_family(p['scenario_id']):<24}"
              f"  step={p['step']}  sev={p['severity']:.1f}"
              f"  action={p['zs_turn'].get('action_type')}"
              f"  zs={p['zs_turn'].get('decision')}"
              f"  tr={p['tr_turn'].get('decision')}")

    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for p in chosen:
        seed = p["seed"]
        zs_blob = _build_side(
            side="zeroshot",
            seed=seed,
            task_id=p["task_id"],
            scenario_id=p["scenario_id"],
            incident_summary=p["incident_summary"],
            known_services=p["known_services"],
            pair_step=p["step"],
            zs_turn=p["zs_turn"],
            tr_turn=p["tr_turn"],
        )
        tr_blob = _build_side(
            side="trained",
            seed=seed,
            task_id=p["task_id"],
            scenario_id=p["scenario_id"],
            incident_summary=p["incident_summary"],
            known_services=p["known_services"],
            pair_step=p["step"],
            zs_turn=p["zs_turn"],
            tr_turn=p["tr_turn"],
        )
        zs_out = out_dir / f"seed_{seed}_zeroshot.json"
        tr_out = out_dir / f"seed_{seed}_trained.json"
        zs_out.write_text(json.dumps(zs_blob, indent=2), encoding="utf-8")
        tr_out.write_text(json.dumps(tr_blob, indent=2), encoding="utf-8")
        written.extend([zs_out, tr_out])

    index = {
        "filter_used": used_filter,
        "n_common_seeds": len(common_seeds),
        "n_pairs_total": len(pairs),
        "n_pairs_chosen": len(chosen),
        "pairs": [_slim_index_row(p) for p in chosen],
    }
    index_path = out_dir / "pairs_index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    print(f"[find_before_after] wrote {len(written)} pair file(s) under {out_dir}")
    print(f"[find_before_after] wrote index -> {index_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
