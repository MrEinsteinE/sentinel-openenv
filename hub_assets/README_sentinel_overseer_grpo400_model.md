---
license: apache-2.0
language:
  - en
base_model: unsloth/qwen3-1.7b-unsloth-bnb-4bit
tags:
  - transformers
  - peft
  - trl
  - grpo
  - openenv
  - receipt
library_name: peft
pipeline_tag: text-generation
---

# SENTINEL Overseer — Qwen3-1.7B (GRPO-400 receipt)

This adapter is the **full 400-step curriculum GRPO** run published **after** disabling the training runner’s auto-abort gate. It exists as an **honest receipt**: under the same held-out 50-scenario eval, overall Overseer **F1 ≈ 0.934** vs **≈ 0.969** for the **production** checkpoint [`Elliot89/sentinel-overseer-qwen3-1.7b`](https://huggingface.co/Elliot89/sentinel-overseer-qwen3-1.7b). The regression concentrates in **`drift_ops`** (schema-drift tier): F1 **≈ 0.85** here vs **≈ 0.92** on the published model — false negatives on catastrophic / wrong actions roughly double.

| Tier | F1 (this checkpoint, held-out eval) |
|---|---:|
| action_screen | 0.987 |
| war_room | 0.981 |
| drift_ops | 0.848 |
| **Overall** | **0.934** |

**Use the `sentinel-overseer-qwen3-1.7b` repo for demos and the Live Space** unless you are reproducing the GRPO-ablation story. See the [long-form blog](https://huggingface.co/spaces/Elliot89/sentinel/blob/main/blog.md) §4.

## Baselines reminder

- **Random** Overseer: uniform random choice in {approve, flag, block, escalate} each step (non-strategic chance baseline).
- **Naive**: always approve → **F1 = 0** on this benchmark because every catastrophic/wrong block is missed (TP = 0 in the graded confusion matrix).
- **Qwen3-1.7B zero-shot** with the same JSON Overseer prompt: empirically **near-always-approve** on held-out seeds here → **F1 = 0** (same failure mode as naive on this suite).

## Load (same as production adapter)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE = "Qwen/Qwen3-1.7B"
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, "Elliot89/sentinel-overseer-qwen3-1.7b-grpo400")
```

## Links

- [SENTINEL Space](https://huggingface.co/spaces/Elliot89/sentinel) · [GitHub](https://github.com/MrEinsteinE/sentinel-openenv)
