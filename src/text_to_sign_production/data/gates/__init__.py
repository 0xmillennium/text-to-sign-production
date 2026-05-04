"""Structural gating and decision evaluation layer."""

from __future__ import annotations

from text_to_sign_production.data.gates.config import GatesConfig, load_gates_config
from text_to_sign_production.data.gates.evaluate import evaluate_sample_processing
from text_to_sign_production.data.gates.types import (
    GateResult,
    GateStatus,
    ProcessingDecision,
    ProcessingStatus,
)
from text_to_sign_production.data.gates.validate import validate_decision

__all__ = [
    "GatesConfig",
    "GateResult",
    "GateStatus",
    "ProcessingDecision",
    "ProcessingStatus",
    "evaluate_sample_processing",
    "load_gates_config",
    "validate_decision",
]
