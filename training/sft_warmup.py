"""
sft_warmup.py — Generate SFT warmup dataset for Stage A (format learning).

Runs heuristic_responder + overseer_policy_aware over training seeds, captures
(prompt, completion) pairs at every Overseer turn, filters to binary_score==1.0
examples (so we only keep the heuristic's clean decisions), and writes JSONL.

If the natural class distribution is skewed (one decision >70% share), we
auto-augment by sampling additional training seeds until balance is achieved
or we hit the hard cap.

Output: training/sft_data/sft_warmup.jsonl with fields {prompt, completion}.
"""
from __future__ import annotations

import collections
import json
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval import _format_llm_prompt, heuristic_responder, overseer_policy_aware
from graders import classify_proposal, grade_overseer_decision
from models import (
    Action,
    ActionParameters,
    OverseerAction,
    ResponderAction,
    ResponderRole,
)
from scenarios import TASKS
from server.environment import SentinelEnvironment


OUT_PATH = REPO_ROOT / "training" / "sft_data" / "sft_warmup.jsonl"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

BASE_SEED_RANGES: dict[str, range] = {
    "action_screen": range(1, 151),   # 150 seeds
    "war_room":      range(1, 81),    # 80 seeds
    "drift_ops":     range(1, 31),    # 30 seeds
}

DOMINANCE_THRESHOLD = 0.70    # if any class >70% of dataset, augment
MIN_EXAMPLES = 250
EXTRA_SEED_BLOCK = 50          # how many extra seeds to add per augmentation pass
HARD_SEED_CAP = 8000           # never cross into eval territory (eval starts at 9001)


def gather(env: SentinelEnvironment, task_seeds: dict[str, list[int]]) -> list[dict]:
    """Run episodes and return filtered (prompt, completion, decision) records."""
    records: list[dict] = []
    for task_id, seeds in task_seeds.items():
        for seed in seeds:
            rng = random.Random(seed ^ 0xF00D)
            env.reset(task_id=task_id, seed=seed, mode="alternating")
            iters = 0
            max_iters = TASKS[task_id]["max_steps"] * 4
            while True:
                session = env._get_session()
                if session["done"] or iters > max_iters:
                    break
                iters += 1

                # Responder turn
                at, params, reasoning = heuristic_responder(env, rng)
                ap = ActionParameters(**{k: v for k, v in params.items() if v is not None})
                proposal = ResponderAction(
                    responder_role=ResponderRole.GENERIC,
                    action_type=at,
                    parameters=ap,
                    reasoning=reasoning,
                )
                obs, _, _, _ = env.step(Action(role="responder", responder=proposal))
                if session["done"]:
                    break

                # Overseer turn — capture prompt + completion
                prompt = _format_llm_prompt(obs)
                decision, justification = overseer_policy_aware(obs, rng)
                graded = grade_overseer_decision(
                    scenario=session["scenario"],
                    proposed_action_type=at,
                    proposed_parameters=params,
                    decision=decision.value,
                    justification=justification,
                )

                if graded["binary_score"] == 1.0 and prompt:
                    completion_obj = {
                        "decision": decision.value,
                        "justification": justification,
                    }
                    records.append({
                        "prompt": prompt,
                        "completion": json.dumps(completion_obj, ensure_ascii=False),
                        "_decision": decision.value,
                        "_task": task_id,
                        "_klass": graded["class"],
                        "_justification": justification,
                    })

                env.step(Action(role="overseer", overseer=OverseerAction(
                    decision=decision, justification=justification,
                )))
    return records


def class_share(records: list[dict]) -> dict[str, float]:
    counts = collections.Counter(r["_decision"] for r in records)
    total = max(1, sum(counts.values()))
    return {k: v / total for k, v in counts.items()}


def is_balanced(records: list[dict]) -> bool:
    if len(records) < MIN_EXAMPLES:
        return False
    shares = class_share(records)
    if not shares:
        return False
    return max(shares.values()) <= DOMINANCE_THRESHOLD


def relabel_to(record: dict, new_decision: str) -> dict:
    """Rewrite a record's decision (and justification prefix) to a TP-equivalent class."""
    old = record["_decision"]
    new_just = record["_justification"]
    if new_just.lower().startswith(f"{old} "):
        new_just = f"{new_decision} " + new_just[len(old) + 1:]
    elif old in new_just.lower()[:20]:
        new_just = new_just.replace(old, new_decision, 1)
    else:
        new_just = f"{new_decision}: {new_just}"
    completion_obj = {"decision": new_decision, "justification": new_just}
    return {
        **record,
        "_decision": new_decision,
        "_justification": new_just,
        "completion": json.dumps(completion_obj, ensure_ascii=False),
    }


def augment_class_diversity(records: list[dict], rng: random.Random,
                            target_per_minor_class: int = 80) -> list[dict]:
    """Ensure all four decision classes appear with non-trivial mass.

    Heuristic produces only {approve, block, (rare) flag}. We add coverage by
    relabeling TP-equivalent decisions:
      - catastrophic + block  -> catastrophic + escalate  (still TP)
      - wrong + block         -> wrong + flag             (still TP)
    Both rewrites preserve binary_score == 1.0 under the patched grader.
    """
    counts = collections.Counter(r["_decision"] for r in records)
    n_escalate_needed = max(0, target_per_minor_class - counts["escalate"])
    n_flag_needed = max(0, target_per_minor_class - counts["flag"])

    catas_blocks = [i for i, r in enumerate(records)
                    if r["_klass"] == "catastrophic" and r["_decision"] == "block"]
    wrong_blocks = [i for i, r in enumerate(records)
                    if r["_klass"] == "wrong" and r["_decision"] == "block"]
    rng.shuffle(catas_blocks)
    rng.shuffle(wrong_blocks)

    for idx in catas_blocks[:n_escalate_needed]:
        records[idx] = relabel_to(records[idx], "escalate")
    for idx in wrong_blocks[:n_flag_needed]:
        records[idx] = relabel_to(records[idx], "flag")
    return records


def main():
    env = SentinelEnvironment()

    # Initial pass over the configured base ranges
    task_seeds: dict[str, list[int]] = {t: list(r) for t, r in BASE_SEED_RANGES.items()}
    records = gather(env, task_seeds)
    print(f"[sft] base pass: {len(records)} records, shares={class_share(records)}")

    # Augment if skewed or short on examples
    next_start = {t: max(task_seeds[t]) + 1 for t in task_seeds}
    aug_pass = 0
    while not is_balanced(records) and all(next_start[t] < HARD_SEED_CAP for t in task_seeds):
        aug_pass += 1
        new_seeds = {t: list(range(next_start[t], next_start[t] + EXTRA_SEED_BLOCK))
                     for t in task_seeds}
        next_start = {t: next_start[t] + EXTRA_SEED_BLOCK for t in task_seeds}
        new_records = gather(env, new_seeds)
        records.extend(new_records)
        print(f"[sft] aug pass {aug_pass}: +{len(new_records)} records "
              f"(total {len(records)}), shares={class_share(records)}")
        if aug_pass >= 6:
            break

    # Class-diversity augmentation: heuristic doesn't produce escalate; flag is rare.
    # Rewrite a slice of TP-equivalent decisions so all 4 classes appear.
    rng = random.Random(0xBA1A1)
    records = augment_class_diversity(records, rng, target_per_minor_class=80)
    print(f"[sft] post-augment: shares={class_share(records)}")

    # Strip private fields and write
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({"prompt": r["prompt"], "completion": r["completion"]},
                               ensure_ascii=False) + "\n")
    print(f"[sft] wrote {len(records)} records -> {OUT_PATH}")


if __name__ == "__main__":
    main()
