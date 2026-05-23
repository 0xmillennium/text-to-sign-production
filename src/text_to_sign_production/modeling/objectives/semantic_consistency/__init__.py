"""Semantic-consistency auxiliary objective foundation contracts."""

from __future__ import annotations

from text_to_sign_production.modeling.objectives.semantic_consistency.config import (
    SemanticConsistencyObjectiveConfig,
    load_semantic_consistency_config,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.errors import (
    SemanticConsistencyError,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.spec import (
    SEMANTIC_OBJECTIVE_KEY,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.training import (
    SemanticTrainingLossResult,
    compute_semantic_training_loss,
    load_semantic_training_objective_for_request,
)

__all__ = [
    "SEMANTIC_OBJECTIVE_KEY",
    "SemanticConsistencyError",
    "SemanticConsistencyObjectiveConfig",
    "SemanticTrainingLossResult",
    "compute_semantic_training_loss",
    "load_semantic_consistency_config",
    "load_semantic_training_objective_for_request",
]
