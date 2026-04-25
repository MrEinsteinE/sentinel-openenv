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
    ax.set_xlabel("training step")
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
    ax.set_xlabel("training step")
    ax.set_ylabel(f"mean binary reward (rolling {window})")
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
    tier: str = "overall",
    *,
    include: list[str] | None = None,
    title: str | None = None,
    orientation: str = "vertical",
    dpi: int = 300,
) -> None:
    """Bar chart of F1 across baselines + trained checkpoint.

    `baselines` shape:
        {
          "policy_aware":          {"action_screen": {"f1": 1.0, ...},
                                    "war_room":      {...},
                                    "drift_ops":     {...},
                                    "overall":       {"f1": ..., "precision": ..., "recall": ...}},
          "qwen3_1_7b_zeroshot":   {...},
          ...
        }

    `tier` defaults to "overall" — the macro-or-micro F1 across the full 50-ep
    held-out split. Pass "action_screen" / "war_room" / "drift_ops" for tier-
    specific charts.

    `include` (optional) restricts which keys are plotted, in the order given.
    Useful for the headline plot where we want a curated set (drop GPT-OSS).

    `orientation` is "vertical" (default — overseer names on x, F1 on y per
    the user's headline spec) or "horizontal" (legacy barh).
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
        "trained_qwen3_1_7b_grpo": "Qwen3-1.7B + SENTINEL GRPO",
        "trained_qwen3_1_7b": "Qwen3-1.7B + SENTINEL GRPO",
        "qwen3_1_7b_trained": "Qwen3-1.7B + SENTINEL GRPO",
    }

    desired_order = [
        "naive",
        "random",
        "qwen3_1_7b_zeroshot",
        "qwen2_5_7b",
        "llama3_1_8b",
        "gpt_oss_20b",
        "qwen2_5_72b",
        "policy_aware",
        trained_label,
    ]
    if include is not None:
        keys = [k for k in include if k in baselines]
    else:
        keys = []
        for k in desired_order:
            if k in baselines and k not in keys:
                keys.append(k)
        for k in baselines:
            if k not in keys:
                keys.append(k)

    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    for k in keys:
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

    if title is None:
        title = (
            "Overseer F1 on 50 held-out scenarios"
            if tier == "overall"
            else f"SENTINEL Overseer — {tier} F1 (held-out split)"
        )

    if orientation == "horizontal":
        fig, ax = plt.subplots(
            figsize=(10, max(4.0, 0.55 * len(labels) + 1.5)), dpi=dpi
        )
        y_pos = list(range(len(labels)))
        bars = ax.barh(y_pos, values, color=colors)
        for bar, v in zip(bars, values):
            ax.text(min(v + 0.01, 0.99), bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=9)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlim(0.0, 1.05)
        ax.set_xlabel("F1")
        ax.set_ylabel("Overseer")
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3)
    else:  # vertical (user's headline spec)
        fig, ax = plt.subplots(
            figsize=(11, 5.5), dpi=dpi
        )
        x_pos = list(range(len(labels)))
        bars = ax.bar(x_pos, values, color=colors)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    min(v + 0.015, 1.02), f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylim(0.0, 1.08)
        ax.set_ylabel("Overall F1")
        ax.set_xlabel("Overseer")
        ax.grid(True, axis="y", alpha=0.3)

    ax.set_title(title)
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
            "naive": {"overall": {"f1": 0.0, "precision": 0.0, "recall": 0.0}},
            "random": {"overall": {"f1": 0.55, "precision": 0.41, "recall": 0.78}},
            "policy_aware": {"overall": {"f1": 1.0, "precision": 1.0, "recall": 1.0}},
            "qwen3_1_7b_zeroshot": {"overall": {"f1": 0.0, "precision": 0.0, "recall": 0.0}},
            "qwen3_1_7b_trained": {"overall": {"f1": 0.98, "precision": 0.997, "recall": 0.964}},
        },
        trained_label="qwen3_1_7b_trained",
        out_path=str(out_dir / "baseline_vs_trained.png"),
    )
    print(f"placeholder PNGs written to {out_dir}/")
