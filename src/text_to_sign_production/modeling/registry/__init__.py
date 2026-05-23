"""Accessors for research-backed modeling registry specifications."""

from __future__ import annotations

from text_to_sign_production.modeling.registry.access import ModelingRegistryError
from text_to_sign_production.modeling.registry.comparators import (
    get_comparator_spec,
    list_comparator_specs,
    require_comparator_spec,
)
from text_to_sign_production.modeling.registry.models import (
    get_model_spec,
    list_model_specs,
    require_model_spec,
)
from text_to_sign_production.modeling.registry.objectives import (
    get_objective_spec,
    list_objective_specs,
    require_objective_spec,
)
from text_to_sign_production.modeling.registry.protocols import (
    get_evaluation_protocol_spec,
    list_evaluation_protocol_specs,
    require_evaluation_protocol_spec,
)

__all__ = [
    "ModelingRegistryError",
    "get_model_spec",
    "require_model_spec",
    "list_model_specs",
    "get_objective_spec",
    "require_objective_spec",
    "list_objective_specs",
    "get_comparator_spec",
    "require_comparator_spec",
    "list_comparator_specs",
    "get_evaluation_protocol_spec",
    "require_evaluation_protocol_spec",
    "list_evaluation_protocol_specs",
]
