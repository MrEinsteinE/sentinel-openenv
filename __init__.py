"""SENTINEL — Multi-Agent OpenEnv for Scalable LLM Oversight.

Top-level re-exports for downstream imports:

    from sentinel import SentinelEnv, ResponderAction, OverseerAction
"""
from __future__ import annotations

__version__ = "0.1.0"

from client import SentinelEnv
from models import (
    Action,
    ActionParameters,
    DualReward,
    EpisodeState,
    Observation,
    OverseerAction,
    OverseerDecision,
    ProposedActionView,
    ResponderAction,
    ResponderRole,
    StepResult,
    TurnPhase,
)

__all__ = [
    "__version__",
    "SentinelEnv",
    "Action",
    "ActionParameters",
    "DualReward",
    "EpisodeState",
    "Observation",
    "OverseerAction",
    "OverseerDecision",
    "ProposedActionView",
    "ResponderAction",
    "ResponderRole",
    "StepResult",
    "TurnPhase",
]
