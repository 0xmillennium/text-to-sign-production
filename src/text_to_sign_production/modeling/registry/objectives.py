"""Concrete registry entries for research-backed auxiliary objectives."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from text_to_sign_production.modeling.registry.access import (
    ModelingRegistryError,
    coerce_objective_key,
)
from text_to_sign_production.modeling.research import (
    MechanismTag,
    ModelKey,
    ObjectiveKey,
    ObjectiveSpec,
    RequiredReport,
    ResearchPhase,
    ResearchRole,
    ResearchTrace,
    RiskControl,
    ValidationRequirement,
)

_AUDIT_RESULT_PATH = Path("docs/research/contribution-audit/audit-result.md")
_LITERATURE_POSITIONING_PATH = Path("docs/research/literature-positioning.md")
_ROADMAP_PATH = Path("docs/research/roadmap.md")
_CANDIDATE_UNIVERSE_PATH = Path("docs/research/contribution-audit/candidate-universe/index.md")


def _reports(*identifiers: str) -> tuple[RequiredReport, ...]:
    return tuple(
        RequiredReport(identifier=identifier, title=identifier.replace("_", " ").title())
        for identifier in identifiers
    )


def _validations(*identifiers: str) -> tuple[ValidationRequirement, ...]:
    return tuple(
        ValidationRequirement(identifier=identifier, description=identifier.replace("_", " "))
        for identifier in identifiers
    )


SEMANTIC_CONSISTENCY_SPEC = ObjectiveSpec(
    key=ObjectiveKey.SEMANTIC_CONSISTENCY,
    canonical_id="text_pose_semantic_consistency_objective",
    display_name="Text-Pose Semantic Consistency Objective",
    phase=ResearchPhase.PHASE_9,
    phase_number=9,
    research_role=ResearchRole.AUXILIARY_ADDITIVE_OBJECTIVE,
    trace=ResearchTrace(
        candidate_id="CAND-TEXT-POSE-SEMANTIC-CONSISTENCY",
        candidate_name="Text-Pose Semantic Consistency",
        candidate_card_path=Path(
            "docs/research/contribution-audit/candidate-cards/text-pose-semantic-consistency.md"
        ),
        candidate_universe_path=_CANDIDATE_UNIVERSE_PATH,
        scorecard_path=Path(
            "docs/research/contribution-audit/scorecards/"
            "text-pose-semantic-consistency-scorecard.md"
        ),
        selection_decision_path=Path(
            "docs/research/contribution-audit/selection-decisions/"
            "text-pose-semantic-consistency-selection-decision.md"
        ),
        audit_result_path=_AUDIT_RESULT_PATH,
        literature_positioning_path=_LITERATURE_POSITIONING_PATH,
        roadmap_path=_ROADMAP_PATH,
        source_corpus_ids=(
            "SRC-LVMCN-2024",
            "SRC-MS2SL-2024",
            "SRC-POSE-EVAL-2025",
            "SRC-HOW2SIGN-2021",
        ),
    ),
    standalone_model_allowed=False,
    allowed_attachment_models=(
        ModelKey.LEARNED_POSE_TOKEN,
        ModelKey.LATENT_DIFFUSION,
        ModelKey.ARTICULATOR_AWARE,
    ),
    requires_ablation=True,
    allowed_mechanisms=(
        MechanismTag.SEMANTIC_CONSISTENCY_OBJECTIVE,
        MechanismTag.EMBEDDING_CHECKS,
    ),
    forbidden_mechanisms=(
        MechanismTag.RETRIEVAL_REUSE,
        MechanismTag.GLOSS_SUPERVISION,
        MechanismTag.DICTIONARY_STITCHING,
        MechanismTag.AVATAR_OR_3D,
        MechanismTag.RENDERING,
    ),
    required_reports=_reports(
        "semantic_attachment_decision",
        "semantic_objective_config",
        "with_without_ablation",
        "metric_limitations",
        "risk_controls",
    ),
    validation_requirements=_validations(
        "semantic_ablation_required",
        "same_split_policy",
        "same_artifact_schema",
        "same_evaluation_protocol",
        "metric_limitations_present",
    ),
    risk_controls=(
        RiskControl(
            identifier="semantic_objective_not_standalone_model",
            statement="Semantic consistency is not a standalone model.",
        ),
        RiskControl(
            identifier="embedding_similarity_not_intelligibility",
            statement="Embedding similarity is not proof of sign intelligibility.",
        ),
        RiskControl(
            identifier="with_without_ablation_required",
            statement="With/without ablation is required.",
        ),
    ),
)

_OBJECTIVE_SPEC_SEQUENCE = (SEMANTIC_CONSISTENCY_SPEC,)

OBJECTIVE_SPECS: Mapping[ObjectiveKey, ObjectiveSpec] = MappingProxyType(
    {spec.key: spec for spec in _OBJECTIVE_SPEC_SEQUENCE}
)


def list_objective_specs() -> tuple[ObjectiveSpec, ...]:
    """Return objective specs in registry order."""

    return _OBJECTIVE_SPEC_SEQUENCE


def get_objective_spec(key: ObjectiveKey | str) -> ObjectiveSpec | None:
    """Return an objective spec, or None when the key is unknown."""

    try:
        resolved = coerce_objective_key(key)
    except ModelingRegistryError:
        return None
    return OBJECTIVE_SPECS.get(resolved)


def require_objective_spec(key: ObjectiveKey | str) -> ObjectiveSpec:
    """Return an objective spec or raise for an unknown key."""

    resolved = coerce_objective_key(key)
    try:
        return OBJECTIVE_SPECS[resolved]
    except KeyError as exc:
        raise ModelingRegistryError(f"unknown objective key {resolved.value!r}") from exc


__all__ = [
    "OBJECTIVE_SPECS",
    "SEMANTIC_CONSISTENCY_SPEC",
    "get_objective_spec",
    "list_objective_specs",
    "require_objective_spec",
]
