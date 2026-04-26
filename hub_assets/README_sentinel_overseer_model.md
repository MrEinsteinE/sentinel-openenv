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
  - sft
  - openenv
  - ai-safety
  - scalable-oversight
  - qwen3
library_name: peft
pipeline_tag: text-generation
---

# SENTINEL Overseer — Qwen3-1.7B (LoRA)

**Role:** JSON **Overseer** policy for the [SENTINEL](https://huggingface.co/spaces/Elliot89/sentinel) multi-agent OpenEnv — given a proposed cloud / agent action, output  
`{"decision":"approve|flag|block|escalate","justification":"..."}` with evidence-grounded rationale.

| | |
|---|---|
| **Space (live env + API)** | [Elliot89/sentinel](https://huggingface.co/spaces/Elliot89/sentinel) |
| **Source code** | [MrEinsteinE/sentinel-openenv](https://github.com/MrEinsteinE/sentinel-openenv) |
| **Base model** | [`unsloth/qwen3-1.7b-unsloth-bnb-4bit`](https://huggingface.co/unsloth/qwen3-1.7b-unsloth-bnb-4bit) |
| **Training** | Stage B SFT (321-sample RFT) + Stage C curriculum GRPO on SENTINEL; published run uses this adapter as the production Overseer head. |
| **Held-out eval (50 scenarios)** | **Overall Overseer F1 ≈ 0.97** — see metrics below (per-tier breakdown matches [`eval_data/baseline_qwen3_1_7b_trained.json`](https://github.com/MrEinsteinE/sentinel-openenv/blob/main/eval_data/baseline_qwen3_1_7b_trained.json) on GitHub). |

## Evaluation (held-out seeds, micro-F1 over Overseer decisions)

| Tier | Task theme | F1 |
|---|--:|--:|
| Easy | `action_screen` | **1.000** |
| Medium | `war_room` | **0.992** |
| Hard | `drift_ops` (mid-episode schema drift) | **0.924** |
| **Overall** | 50 scenarios | **~0.969** |

On this **held-out 50-scenario** suite with the JSON Overseer prompt in `eval.py`, the same **Qwen3-1.7B** base without SENTINEL training scores **F1 = 0** (near-always-approve; see confusion matrices in the repo). This adapter is trained to fix that failure mode inside SENTINEL’s dual-reward, drift-aware loop.

## Intended use

- **Oversight / safety layer** for autonomous agents (call via SENTINEL [`POST /live/oversee`](https://github.com/MrEinsteinE/sentinel-openenv/blob/main/SENTINEL_LIVE.md) or load in-process with PEFT).
- **Research / reproducibility** for scalable oversight and OpenEnv-style training.

**Not for:** unconstrained chat, medical/legal advice, or deployment without your own safety review.

## How to load (Transformers + PEFT)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen3-1.7B"
ADAPTER = "Elliot89/sentinel-overseer-qwen3-1.7b"
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
)
model = PeftModel.from_pretrained(model, ADAPTER)
```

Use the same JSON-only Overseer prompt shape as in [`server/live_routes.py`](https://github.com/MrEinsteinE/sentinel-openenv/blob/main/server/live_routes.py) (`_build_trained_prompt`) for best alignment with training.

## Inference on Hugging Face

This repository is a **PEFT adapter**; it is **not** currently routed on [Inference Providers](https://huggingface.co/docs/inference-providers/en/index). The public **Space** loads base + LoRA in-process for the **trained** backend when configured.

## Training details

- **Framework:** Unsloth + TRL GRPO + SFT, three-stage pipeline (warmup GRPO → rejection fine-tuning → curriculum GRPO).  
- **Dataset:** Curated RFT JSONL + on-policy rollouts against the live SENTINEL HTTP env.  
- **Hardware:** Representative published job: single **L4**, ~56 minutes wall-clock (see project `training/run_summary.json` on GitHub).

## Citation

If you use this checkpoint, cite the **SENTINEL** OpenEnv project and the **Qwen3** base model per their respective model cards.

## Model card contact

OpenEnv Hackathon 2026 (Round 2) — links above.
