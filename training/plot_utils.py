"""
training/plot_utils.py — Matplotlib helpers for the GRPO training pipeline.

Three plots, all written deterministically to disk so they're committable:

    plot_loss(steps, losses, out_path)
        Line chart, x: step, y: loss.

    plot_reward(steps, rewards, window, out_path)
        Per-step reward + rolling-average overlay.

    plot_baseline_vs_trained(baselines, trained_label, out_path, tier=...)
        Bar chart comparing per-tier F1 across all known baselines + trained.

The functions are deliberately dependency-light (only matplotlib + stdlib) so
they import cleanly inside the HF Jobs UV script and the Colab notebook alike.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _rolling_mean(xs: list[float], window: int) -> list[float]:
    if not xs or window <= 1:
        return list(xs)
    out: list[float] = []
    s = 0.0
    q: list[float] = []
    for x in xs:
        q.append(x)
        s += x
        if len(q) > window:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


def plot_loss(steps: Iterable[int], losses: Iterable[float], out_path: str) -> None:
    """GRPO loss curve. Empty inputs are tolerated — a placeholder image is written."""
    _ensure_parent(out_path)
    steps_l, losses_l = list(steps), list(losses)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=110)
    if steps_l:
        ax.plot(steps_l, losses_l, marker=".", linewidth=1.4, color="#3b6fff")
    else:
        ax.text(0.5, 0.5, "no loss logged yet", transform=ax.transAxes,
                ha="center", va="center", color="#888")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("SENTINEL Overseer — GRPO loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_reward(
    steps: Iterable[int],
    rewards: Iterable[float],
    window: int,
    out_path: str,
) -> None:
    """Reward curve with rolling-average overlay."""
    _ensure_parent(out_path)
    steps_l, rewards_l = list(steps), list(rewards)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=110)
    if steps_l:
        ax.plot(steps_l, rewards_l, marker=".", linewidth=0.8, alpha=0.4,
                color="#cc6600", label="per-log mean")
        rolling = _rolling_mean(rewards_l, window)
        ax.plot(steps_l, rolling, linewidth=2.2, color="#cc6600",
                label=f"{window}-step rolling avg")
        ax.legend(loc="best", frameon=False)
    else:
        ax.text(0.5, 0.5, "no rewards logged yet", transform=ax.transAxes,
                ha="center", va="center", color="#888")
    ax.set_xlabel("step")
    ax.set_ylabel("mean binary reward")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("SENTINEL Overseer — GRPO reward")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_baseline_vs_trained(
    baselines: dict[str, dict[str, dict[str, float]]],
    trained_label: str,
    out_path: str,
    tier: str = "action_screen",
) -> None:
    """Bar chart of overall F1 across baselines + trained checkpoint, for a given tier.

    `baselines` shape:
        {
          "policy_aware":          {"action_screen": {"f1": 1.0, "precision": 1.0, "recall": 1.0}, ...},
          "qwen3_1_7b_zeroshot":   {...},
          "trained_qwen3_1_7b":    {...},
          ...
        }
    """
    _ensure_parent(out_path)

    pretty_names = {
        "naive": "naive (always approve)",
        "random": "random",
        "policy_aware": "policy-aware heuristic",
        "qwen2_5_7b": "Qwen2.5-7B zero-shot",
        "qwen2_5_72b": "Qwen2.5-72B zero-shot",
        "llama3_1_8b": "Llama-3.1-8B zero-shot",
        "gpt_oss_20b": "GPT-OSS-20B zero-shot",
        "qwen3_1_7b_zeroshot": "Qwen3-1.7B zero-shot",
        "trained_qwen3_1_7b_grpo": "Qwen3-1.7B + SENTINEL (trained)",
        "trained_qwen3_1_7b": "Qwen3-1.7B + SENTINEL (trained)",
    }

    desired_order = [
        "naive",
        "random",
        "qwen2_5_7b",
        "qwen2_5_72b",
        "llama3_1_8b",
        "gpt_oss_20b",
        "qwen3_1_7b_zeroshot",
        "policy_aware",
        trained_label,
    ]
    seen: list[str] = []
    for k in desired_order:
        if k in baselines and k not in seen:
            seen.append(k)
    for k in baselines:
        if k not in seen:
            seen.append(k)

    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    for k in seen:
        per_tier = baselines.get(k, {}) or {}
        tier_data = per_tier.get(tier, {}) or {}
        f1 = float(tier_data.get("f1", 0.0))
        labels.append(pretty_names.get(k, k))
        values.append(f1)
        if k == trained_label:
            colors.append("#1f9d55")
        elif k == "policy_aware":
            colors.append("#2c7be5")
        elif k.endswith("_zeroshot") or k.startswith("qwen") or k.startswith("llama") or k.startswith("gpt"):
            colors.append("#e0a800")
        else:
            colors.append("#888888")

    fig, ax = plt.subplots(figsize=(10, max(4.0, 0.55 * len(labels) + 1.5)), dpi=110)
    y_pos = list(range(len(labels)))
    bars = ax.barh(y_pos, values, color=colors)
    for bar, v in zip(bars, values):
        ax.text(min(v + 0.01, 0.99), bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("F1")
    ax.invert_yaxis()
    ax.set_title(f"SENTINEL Overseer — {tier} F1 (held-out split)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    out_dir = Path(os.environ.get("SENTINEL_OUT", "training/plots"))
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_loss([], [], str(out_dir / "grpo_loss.png"))
    plot_reward([], [], 25, str(out_dir / "grpo_reward.png"))
    plot_baseline_vs_trained(
        {
            "naive": {"action_screen": {"f1": 0.0, "precision": 0.0, "recall": 0.0}},
            "random": {"action_screen": {"f1": 0.55, "precision": 0.41, "recall": 0.78}},
            "policy_aware": {"action_screen": {"f1": 1.0, "precision": 1.0, "recall": 1.0}},
            "qwen3_1_7b_zeroshot": {"action_screen": {"f1": 0.13, "precision": 0.42, "recall": 0.06}},
            "trained_qwen3_1_7b_grpo": {"action_screen": {"f1": 0.0, "precision": 0.0, "recall": 0.0}},
        },
        trained_label="trained_qwen3_1_7b_grpo",
        out_path=str(out_dir / "baseline_vs_trained.png"),
    )
    print(f"placeholder PNGs written to {out_dir}/")
