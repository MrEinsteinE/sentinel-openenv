# SENTINEL — Headline Summary

Held-out split: 50 episodes across 3 task tiers (`action_screen`, `war_room`, `drift_ops`).

- **Headline gap.** Qwen3-1.7B + SENTINEL GRPO scores **Overall F1 = 0.980** *(macro approx)* vs the same base model zero-shot at F1 = 0.000. Absolute delta = **+0.980**; relative = **∞ (zero-shot scored 0.000 — every catastrophic action was approved)**.
- **Frontier comparison.** A 1.7B-parameter Overseer trained with SENTINEL beats Qwen2.5-72B zero-shot **10.5×** (Overall F1 0.980 vs 0.093). The GRPO loop teaches a 42× smaller model to outperform a frontier-class LLM on this oversight task.
- **Heuristic ceiling.** The rule-based `policy_aware` overseer scores **F1 = 0.944**, the upper bound on this dataset (it reads the counterfactual preview directly). SENTINEL lands within **3.6pp** of the heuristic without any hand-coded rules (macro vs micro F1 — exact micro pending the trained-eval re-run).
