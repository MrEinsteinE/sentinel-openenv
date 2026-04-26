"""
tools/regen_baseline_plot.py — regenerate training/plots/baseline_vs_trained.png
from current eval_data/baseline_*.json + training/run_summary.json.

Use this AFTER each new eval lands (whether zero-shot or trained) so the
headline plot reflects the latest numbers without waiting for an HF Job.

The script favours micro-F1 from JSON's `overall_f1` when available; for the
trained checkpoint it falls back to macro-mean of per-tier F1 from
`run_summary.json["f1_per_tier"]` and labels the value `~F1` to flag it as
approximate (the HF Job's verbose trained eval will overwrite with exact micro).

Usage:
    python tools/regen_baseline_plot.py
    python tools/regen_baseline_plot.py --tier overall --dpi 300
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "training"))
from plot_utils import plot_baseline_vs_trained  # noqa: E402

EVAL_DIR = REPO_ROOT / "eval_data"
PLOTS_DIR = REPO_ROOT / "training" / "plots"
RUN_SUMMARY = REPO_ROOT / "training" / "run_summary.json"


def _load_baselines() -> dict[str, dict[str, dict[str, float]]]:
    """{label: {tier: {f1, precision, recall}, 'overall': ...}}."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for p in sorted(EVAL_DIR.glob("baseline_*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[regen_baseline_plot] skip {p.name}: {e}", file=sys.stderr)
            continue
        per_task = dict(data.get("per_task_f1", {}))
        if isinstance(data.get("overall_f1"), dict):
            per_task["overall"] = data["overall_f1"]
        out[p.stem.removeprefix("baseline_")] = per_task
    return out


def _trained_from_run_summary() -> dict[str, dict[str, float]] | None:
    if not RUN_SUMMARY.exists():
        return None
    try:
        data = json.loads(RUN_SUMMARY.read_text(encoding="utf-8"))
    except Exception:
        return None
    per_tier = data.get("f1_per_tier") or {}
    if not isinstance(per_tier, dict) or not per_tier:
        return None
    out: dict[str, dict[str, float]] = dict(per_tier)
    if isinstance(data.get("trained_overall_f1"), dict):
        out["overall"] = data["trained_overall_f1"]
    else:
        f1s = [
            v.get("f1", 0.0) for v in per_tier.values() if isinstance(v, dict)
        ]
        if f1s:
            out["overall"] = {
                "f1": sum(f1s) / len(f1s),
                "precision": 0.0,
                "recall": 0.0,
            }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", default="overall",
                        choices=["overall", "action_screen", "war_room", "drift_ops"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--out",
                        default=str(PLOTS_DIR / "baseline_vs_trained.png"))
    args = parser.parse_args()

    baselines = _load_baselines()
    # Prefer the canonical micro-F1 from eval_data/baseline_qwen3_1_7b_trained.json
    # over the macro-mean computed from training/run_summary.json. The eval JSON is
    # the published-checkpoint number that the README and blog quote; run_summary
    # may reflect a later GRPO follow-up that didn't survive the auto-abort.
    eval_trained = baselines.get("qwen3_1_7b_trained")
    eval_has_overall = isinstance(eval_trained, dict) and isinstance(
        eval_trained.get("overall"), dict
    )
    if eval_has_overall:
        print(f"[regen_baseline_plot] using eval JSON micro-F1 for trained row "
              f"(overall_f1={eval_trained['overall'].get('f1'):.4f})")
    else:
        trained = _trained_from_run_summary()
        if trained is None:
            print("[regen_baseline_plot] WARN: no trained F1 in eval_data/ or "
                  "run_summary.json; plot will be missing the trained row.",
                  file=sys.stderr)
        else:
            print("[regen_baseline_plot] no eval JSON for trained model; "
                  "falling back to macro-mean from run_summary.json")
            baselines["qwen3_1_7b_trained"] = trained

    include = [
        "naive",
        "random",
        "qwen3_1_7b_zeroshot",
        "qwen2_5_7b",
        "llama3_1_8b",
        "qwen2_5_72b",
        "policy_aware",
        "qwen3_1_7b_trained",
    ]
    have = [k for k in include if k in baselines]
    missing = [k for k in include if k not in baselines]
    print(f"[regen_baseline_plot] tier={args.tier} dpi={args.dpi}")
    print(f"[regen_baseline_plot] including: {have}")
    if missing:
        print(f"[regen_baseline_plot] skipped (no eval JSON yet): {missing}")

    title = (
        "Overseer F1 on 50 held-out scenarios"
        if args.tier == "overall"
        else f"SENTINEL Overseer — {args.tier} F1 (held-out split)"
    )
    plot_baseline_vs_trained(
        baselines,
        trained_label="qwen3_1_7b_trained",
        out_path=args.out,
        tier=args.tier,
        include=have,
        title=title,
        orientation="vertical",
        dpi=args.dpi,
    )
    sz = Path(args.out).stat().st_size
    print(f"[regen_baseline_plot] wrote {args.out} ({sz} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
