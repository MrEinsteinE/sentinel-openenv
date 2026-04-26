# SENTINEL — Headline Summary

Held-out split: **50 episodes** across 3 task tiers (`action_screen`, `war_room`, `drift_ops`), totalling **650 Overseer decisions**. All numbers below come from `eval_data/baseline_qwen3_1_7b_trained.json` (canonical micro-F1 over the full confusion matrix).

- **Headline gap.** Qwen3-1.7B + SENTINEL training scores **Overall F1 = 0.969** (precision 0.985 / recall 0.953) vs the same base model zero-shot at **F1 = 0.000**. Absolute delta = **+0.969**; the zero-shot model approved every single catastrophic action.
- **Frontier comparison.** A 1.7B-parameter Overseer trained with SENTINEL beats Qwen2.5-72B zero-shot **10.4×** (0.969 vs 0.093). The training loop teaches a **42× smaller** model to outperform a frontier-class LLM on this oversight task.
- **Heuristic ceiling.** The rule-based `policy_aware` overseer scores **F1 = 0.944** — the rule reads the counterfactual preview directly and just keyword-matches "CATASTROPHIC". SENTINEL clears the heuristic by **+2.5pp**, doing so via learned reasoning (no hand-coded rules baked into the model).
- **Per-tier breakdown.** action_screen **1.000** · war_room **0.992** · drift_ops **0.924**. Hardest tier (mid-episode schema drift) is also the tier where the trained model most clearly beats both the rule (0.863) and the frontier zero-shot models (≤0.147).
- **Compute.** One Hugging Face Jobs run on `l4x1`, **wall clock ≈ 56 minutes**, single L4 GPU. The published checkpoint is the SFT-warmed model — auto-abort kept GRPO from regressing on `drift_ops` (we have receipts in `Elliot89/sentinel-overseer-qwen3-1.7b-grpo400`).
