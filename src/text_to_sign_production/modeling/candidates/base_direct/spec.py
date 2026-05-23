"""Research-backed identity checks for the M0 direct baseline provider."""

from __future__ import annotations

from text_to_sign_production.modeling.candidates.errors import ModelCandidateError
from text_to_sign_production.modeling.registry import require_model_spec
from text_to_sign_production.modeling.research import ModelKey, ModelSpec, ResearchRole

BASE_DIRECT_MODEL_KEY = ModelKey.BASE_DIRECT
BASE_DIRECT_SPEC = require_model_spec(BASE_DIRECT_MODEL_KEY)


def require_base_direct_spec() -> ModelSpec:
    """Return the registered base-direct spec after provider-critical checks."""

    spec = require_model_spec(BASE_DIRECT_MODEL_KEY)
    if spec.key is not ModelKey.BASE_DIRECT:
        raise ModelCandidateError("base_direct provider spec key is invalid.")
    if spec.canonical_id != "m0_direct_text_to_pose":
        raise ModelCandidateError("base_direct provider canonical_id is invalid.")
    if spec.research_role is not ResearchRole.BASELINE_OR_ABLATION_FLOOR:
        raise ModelCandidateError("base_direct provider research role is invalid.")
    if spec.generated_pose_required is not True:
        raise ModelCandidateError("base_direct provider must produce generated pose.")
    if spec.default_stage_sequence != ("train", "generate", "export_generated_pose"):
        raise ModelCandidateError("base_direct provider stage sequence is invalid.")
    return spec


__all__ = [
    "BASE_DIRECT_MODEL_KEY",
    "BASE_DIRECT_SPEC",
    "require_base_direct_spec",
]
