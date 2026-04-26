"""
binary_sanity.py — Sanity check the Overseer binary reward signal.

Runs heuristic_responder + overseer_policy_aware over training seeds (NOT eval)
and verifies that grade_overseer_decision returns a non-degenerate binary signal.

Success criterion (printed at the end):
  mean binary >= 0.85  AND  decision-level binary==1.0 rate >= 0.80
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval import heuristic_responder, overseer_policy_aware
from graders import grade_overseer_decision
from models import (
    Action,
    ActionParameters,
    OverseerAction,
    ResponderAction,
    ResponderRole,
)
from scenarios import TASKS
from server.environment import SentinelEnvironment


TRAIN_SEEDS = list(range(1, 51))   # 50 training seeds, NOT eval (9001..)
TASK_IDS = list(TASKS.keys())       # action_screen, war_room, drift_ops


def run_one(env: SentinelEnvironment, task_id: str, seed: int) -> tuple[list[float], int]:
    """Run one episode, return (binary_scores_per_overseer_decision, n_decisions)."""
    rng = random.Random(seed ^ 0xF00D)
    env.reset(task_id=task_id, seed=seed, mode="alternating")

    binaries: list[float] = []
    max_iters = TASKS[task_id]["max_steps"] * 4

    iters = 0
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

        # Overseer turn — get decision + grade externally
        decision, justification = overseer_policy_aware(obs, rng)
        scenario = session["scenario"]
        result = grade_overseer_decision(
            scenario=scenario,
            proposed_action_type=at,
            proposed_parameters=params,
            decision=decision.value,
            justification=justification,
        )
        binaries.append(float(result["binary_score"]))

        obs, _, _, _ = env.step(
            Action(
                role="overseer",
                overseer=OverseerAction(decision=decision, justification=justification),
            )
        )

    return binaries, len(binaries)


def main():
    env = SentinelEnvironment()
    all_decisions: list[float] = []
    episode_means: list[float] = []
    n_episodes = 0

    for task_id in TASK_IDS:
        for seed in TRAIN_SEEDS:
            binaries, n = run_one(env, task_id, seed)
            if n == 0:
                continue
            n_episodes += 1
            mean_ep = sum(binaries) / n
            episode_means.append(mean_ep)
            all_decisions.extend(binaries)

    n_dec = len(all_decisions)
    mean_binary = sum(all_decisions) / max(1, n_dec)
    frac_eps_above = sum(1 for m in episode_means if m >= 0.5) / max(1, n_episodes)
    frac_dec_one = sum(1 for b in all_decisions if b == 1.0) / max(1, n_dec)

    print(f"[binary_sanity] tasks={TASK_IDS} seeds=1..{TRAIN_SEEDS[-1]}")
    print(f"[binary_sanity] episodes={n_episodes} decisions={n_dec}")
    print(f"[binary_sanity] mean_binary_reward          = {mean_binary:.4f}")
    print(f"[binary_sanity] frac_episodes_mean>=0.5     = {frac_eps_above:.4f}")
    print(f"[binary_sanity] frac_decisions_binary==1.0  = {frac_dec_one:.4f}")

    pass_mean = mean_binary >= 0.85
    pass_dec = frac_dec_one >= 0.80
    status = "PASS" if (pass_mean and pass_dec) else "FAIL"
    print(f"[binary_sanity] criterion: mean>=0.85 AND dec_rate>=0.80 -> {status}")

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
