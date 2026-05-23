"""Stable semantic-consistency objective identity and registry access."""

from __future__ import annotations

from text_to_sign_production.modeling.research import ObjectiveKey, ObjectiveSpec
from text_to_sign_production.modeling.registry import require_objective_spec

SEMANTIC_OBJECTIVE_KEY = ObjectiveKey.SEMANTIC_CONSISTENCY
SEMANTIC_OBJECTIVE_CANONICAL_ID = "text_pose_semantic_consistency_objective"
SEMANTIC_OBJECTIVE_PHASE_NUMBER = 9


def require_semantic_consistency_spec() -> ObjectiveSpec:
    """Return the registry-owned semantic objective specification."""

    return require_objective_spec(SEMANTIC_OBJECTIVE_KEY)


__all__ = [
    "SEMANTIC_OBJECTIVE_CANONICAL_ID",
    "SEMANTIC_OBJECTIVE_KEY",
    "SEMANTIC_OBJECTIVE_PHASE_NUMBER",
    "require_semantic_consistency_spec",
]
