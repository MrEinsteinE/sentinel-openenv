"""Diagnostic: dump cases where binary == 0.0 to find the actual failure mode."""
from __future__ import annotations

import collections
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import heuristic_responder, overseer_policy_aware
from graders import classify_proposal, grade_overseer_decision, _default_justification_score
from models import (
    Action, ActionParameters, OverseerAction, ResponderAction, ResponderRole,
)
from scenarios import TASKS
from server.environment import SentinelEnvironment


def main():
    env = SentinelEnvironment()
    failing = collections.Counter()
    failing_examples = []
    total = 0

    for task_id in TASKS.keys():
        for seed in range(1, 51):
            rng = random.Random(seed ^ 0xF00D)
            env.reset(task_id=task_id, seed=seed, mode="alternating")
            iters = 0
            max_iters = TASKS[task_id]["max_steps"] * 4
            while True:
                session = env._get_session()
                if session["done"] or iters > max_iters:
                    break
                iters += 1
                at, params, reasoning = heuristic_responder(env, rng)
                ap = ActionParameters(**{k: v for k, v in params.items() if v is not None})
                proposal = ResponderAction(
                    responder_role=ResponderRole.GENERIC, action_type=at,
                    parameters=ap, reasoning=reasoning,
                )
                obs, _, _, _ = env.step(Action(role="responder", responder=proposal))
                if session["done"]:
                    break
                decision, justification = overseer_policy_aware(obs, rng)
                scenario = session["scenario"]
                klass, _ = classify_proposal(scenario, at, params)
                just_score = _default_justification_score(justification, at, params, scenario)
                result = grade_overseer_decision(
                    scenario=scenario, proposed_action_type=at, proposed_parameters=params,
                    decision=decision.value, justification=justification,
                )
                total += 1
                if result["binary_score"] == 0.0:
                    key = (klass, decision.value, "just_low" if just_score < 0.25 else "wrong_decision")
                    failing[key] += 1
                    if len(failing_examples) < 12:
                        failing_examples.append({
                            "task": task_id, "seed": seed, "at": at,
                            "klass": klass, "decision": decision.value,
                            "just_score": round(just_score, 3),
                            "justification": justification[:90],
                            "params_svc": params.get("service", ""),
                        })
                env.step(Action(role="overseer", overseer=OverseerAction(
                    decision=decision, justification=justification)))

    print(f"total decisions: {total}")
    print("failure breakdown (klass, decision, reason) -> count:")
    for k, v in sorted(failing.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
    print("\nfirst 12 failing examples:")
    for ex in failing_examples:
        print(f"  {ex}")


if __name__ == "__main__":
    main()
