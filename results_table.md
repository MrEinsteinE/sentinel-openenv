# SENTINEL — Overseer F1 on 50 held-out scenarios

Sorted by Overall F1 ascending. Trained checkpoint highlighted in **bold**.
All numbers are micro-F1 over the 650-decision confusion matrix (`eval_data/baseline_*.json`).

| Overseer | action_screen F1 | war_room F1 | drift_ops F1 | Overall F1 | P | R |
|---|---:|---:|---:|---:|---:|---:|
| Naive (always approve) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Qwen3-1.7B (zero-shot) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| GPT-OSS-20B (zero-shot) | 0.050 | 0.015 | 0.000 | 0.015 | 1.000 | 0.007 |
| Qwen2.5-72B (zero-shot) | 0.050 | 0.058 | 0.147 | 0.093 | 0.518 | 0.051 |
| Qwen2.5-7B (zero-shot) | 0.128 | 0.127 | 0.072 | 0.108 | 0.415 | 0.062 |
| Llama-3.1-8B (zero-shot) | 0.178 | 0.219 | 0.074 | 0.162 | 0.735 | 0.091 |
| Random | 0.553 | 0.504 | 0.575 | 0.539 | 0.412 | 0.782 |
| Policy-aware heuristic | 1.000 | 1.000 | 0.863 | 0.944 | 0.893 | 1.000 |
| **Qwen3-1.7B + SENTINEL** | **1.000** | **0.992** | **0.924** | **0.969** | **0.985** | **0.953** |
