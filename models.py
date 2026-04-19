"""
models.py — Pydantic v2 data contracts for SENTINEL.

Multi-agent turn flow:
  RESPONDER_PROPOSE → OVERSEER_DECIDE → WORLD_EXECUTE → (next turn or done)

Two agents share one HTTP step() endpoint. The env's `turn_phase` tells the
caller which agent is on turn. Agents send Actions tagged with their role.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field


# ── Turn flow ───────────────────────────────────────────────────────────────


class TurnPhase(str, Enum):
    RESPONDER_PROPOSE = "responder_propose"
    OVERSEER_DECIDE = "overseer_decide"
    TERMINAL = "terminal"


class ResponderRole(str, Enum):
    NETWORK_OPS = "network_ops"
    DB_ADMIN = "db_admin"
    SEC_OPS = "sec_ops"
    GENERIC = "generic"


class OverseerDecision(str, Enum):
    APPROVE = "approve"
    FLAG = "flag"
    BLOCK = "block"
    ESCALATE = "escalate"


# ── Actions ─────────────────────────────────────────────────────────────────


class ActionParameters(BaseModel):
    service: str | None = None
    severity: str | None = None
    failure_mode: str | None = None
    summary: str | None = None
    target_version: str | None = None
    replicas: int | None = None
    flag: str | None = None
    runbook_action: str | None = None
    target: str | None = None
    reasoning: str | None = None

    model_config = {"extra": "allow"}


class ResponderAction(BaseModel):
    """Action proposed by the Responder agent."""
    role: Literal["responder"] = "responder"
    responder_role: ResponderRole = ResponderRole.GENERIC
    action_type: str
    parameters: ActionParameters = Field(default_factory=ActionParameters)
    reasoning: str | None = Field(
        default=None,
        description="Short plan-of-record. Used in explain_plan replies to Overseer flags.",
    )
    model_config = {"extra": "allow"}


class OverseerAction(BaseModel):
    """Decision by the Overseer agent on the Responder's proposed action."""
    role: Literal["overseer"] = "overseer"
    decision: OverseerDecision
    justification: str = Field(
        ...,
        description=(
            "Natural-language rationale for the decision. Judged for coherence "
            "by a frozen LLM. Richer justifications within the cap earn bonus."
        ),
        min_length=1,
    )
    suggested_alternative: ActionParameters | None = Field(
        default=None,
        description="If decision=BLOCK, an optional suggested alternative the Responder can adopt.",
    )
    model_config = {"extra": "allow"}


class Action(BaseModel):
    """Unified step() payload — discriminated on `role`."""
    role: Literal["responder", "overseer"]
    responder: ResponderAction | None = None
    overseer: OverseerAction | None = None
    model_config = {"extra": "allow"}


# ── Observation ─────────────────────────────────────────────────────────────


class ProposedActionView(BaseModel):
    """What the Overseer sees about the Responder's pending proposal."""
    action_type: str
    parameters: dict
    responder_role: str
    reasoning: str | None = None
    counterfactual: str | None = Field(
        default=None,
        description=(
            "Rule-based preview of what would happen if this action were executed. "
            "Always available to Overseer; absent for Responder."
        ),
    )
    severity_weight: float = Field(
        default=1.0,
        description="1.0 = normal, 3.0 = catastrophic-class action (rm -rf, rollback on prod, etc.).",
    )


class Observation(BaseModel):
    """Full observation returned after reset() or step().

    The agent on turn is identified by `turn_phase`. Responder sees the
    incident, tool responses, and Overseer feedback from prior turns. Overseer
    additionally sees the pending `proposed_action` with a counterfactual preview.
    """
    episode_id: str
    task_id: str
    scenario_id: str
    step_count: int
    max_steps: int

    # Shared context
    incident_summary: str
    alert: dict
    available_actions: list[str]
    queried_data: dict
    known_services: list[str] = Field(default_factory=list)
    policy_constraints: list[str] = Field(default_factory=list)

    # Multi-agent state
    turn_phase: TurnPhase
    proposed_action: ProposedActionView | None = None
    overseer_history: list[dict] = Field(default_factory=list)
    drift_events: list[dict] = Field(default_factory=list)

    # Rewards / progress
    cumulative_responder_reward: float = 0.0
    cumulative_overseer_reward: float = 0.0
    done: bool = False
    feedback: str = ""
    last_action_error: str | None = None

    # OpenEnv conventions — mirrored fields
    reward: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Reward ──────────────────────────────────────────────────────────────────


class DualReward(BaseModel):
    """Reward signal emitted by every step()."""
    responder_score: float
    overseer_score: float
    overseer_binary: float = Field(
        description=(
            "Binary version of the overseer score for GRPO training "
            "(1.0 if the decision was correct and justification non-trivial, else 0.0). "
            "TRL docs note binary rewards beat shaped for GRPO group normalization."
        )
    )
    reason: str
    responder_cumulative: float
    overseer_cumulative: float

    @computed_field
    @property
    def score(self) -> float:
        """Primary OpenEnv reward — scores the agent whose turn just ended."""
        return self.overseer_score if self.overseer_score != 0.0 else self.responder_score

    @computed_field
    @property
    def value(self) -> float:
        return self.score


class StepResult(BaseModel):
    """OpenEnv /step response envelope."""
    observation: Observation
    reward: DualReward
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


# ── State ───────────────────────────────────────────────────────────────────


class EpisodeState(BaseModel):
    """Full episode state returned by GET /state."""
    episode_id: str
    task_id: str
    scenario_id: str
    step_count: int
    max_steps: int
    turn_phase: TurnPhase
    action_history: list[dict]
    queried_data: dict
    submitted: bool
    resolved: bool
    done: bool
    cumulative_responder_reward: float
    cumulative_overseer_reward: float
    overseer_confusion: dict[str, int] = Field(
        default_factory=lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0},
        description="Per-episode confusion counts for Overseer's decisions — drives F1 reporting.",
    )
    drift_events: list[dict] = Field(default_factory=list)
    feedback: str
