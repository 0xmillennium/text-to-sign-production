"""Research-backed modeling domain for candidates, objectives, comparators, and protocols."""

from __future__ import annotations

from text_to_sign_production.modeling.registry import (
    get_comparator_spec,
    get_evaluation_protocol_spec,
    get_model_spec,
    get_objective_spec,
    list_comparator_specs,
    list_evaluation_protocol_specs,
    list_model_specs,
    list_objective_specs,
    require_comparator_spec,
    require_evaluation_protocol_spec,
    require_model_spec,
    require_objective_spec,
)
from text_to_sign_production.modeling.research import (
    ComparatorKey,
    EvaluationProtocolKey,
    ModelKey,
    ObjectiveKey,
)

__all__ = [
    "ComparatorKey",
    "EvaluationProtocolKey",
    "ModelKey",
    "ObjectiveKey",
    "get_comparator_spec",
    "get_evaluation_protocol_spec",
    "get_model_spec",
    "get_objective_spec",
    "list_comparator_specs",
    "list_evaluation_protocol_specs",
    "list_model_specs",
    "list_objective_specs",
    "require_comparator_spec",
    "require_evaluation_protocol_spec",
    "require_model_spec",
    "require_objective_spec",
]
