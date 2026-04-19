"""
client.py — Client-side EnvClient for SENTINEL.

Used by training scripts (TRL GRPOTrainer) to talk to a running SENTINEL
server. Wraps HTTP calls in typed Pydantic models.

Usage (training notebook):

    from sentinel.client import SentinelEnv, ResponderAction, OverseerAction
    env = SentinelEnv(base_url="https://Elliot89-sentinel.hf.space")
    obs = env.reset(task_id="war_room", seed=42).observation
    step = env.step(OverseerAction(decision="block", justification="rm -rf on prod db"))
"""
from __future__ import annotations

from typing import Any

try:
    from openenv.core.env_client import EnvClient  # type: ignore
except Exception:  # pragma: no cover — lets the repo import in dev without openenv-core
    EnvClient = object  # type: ignore

from models import (
    Action,
    DualReward,
    EpisodeState,
    Observation,
    OverseerAction,
    ResponderAction,
    StepResult,
)


class SentinelEnv(EnvClient):  # type: ignore[misc]
    """Typed client for the SENTINEL OpenEnv."""

    Action = Action
    Observation = Observation
    State = EpisodeState

    def _step_payload(self, action: Action | ResponderAction | OverseerAction) -> dict:
        if isinstance(action, ResponderAction):
            return {"role": "responder", "responder": action.model_dump()}
        if isinstance(action, OverseerAction):
            return {"role": "overseer", "overseer": action.model_dump()}
        return action.model_dump()

    def _parse_result(self, payload: dict[str, Any]) -> StepResult:
        return StepResult(
            observation=Observation(**payload["observation"]),
            reward=DualReward(**payload["reward"]),
            done=payload.get("done", False),
            info=payload.get("info", {}),
        )

    def _parse_state(self, payload: dict[str, Any]) -> EpisodeState:
        return EpisodeState(**payload)
