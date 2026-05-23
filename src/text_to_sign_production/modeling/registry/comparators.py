"""Concrete registry entries for research-backed comparators."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from text_to_sign_production.modeling.registry.access import (
    ModelingRegistryError,
    coerce_comparator_key,
)
from text_to_sign_production.modeling.research import (
    ComparatorKey,
    ComparatorSpec,
    MechanismTag,
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


RETRIEVAL_POSE_SPEC = ComparatorSpec(
    key=ComparatorKey.RETRIEVAL_POSE,
    canonical_id="retrieval_augmented_pose_comparator",
    display_name="Retrieval-Augmented Pose Comparator",
    phase=ResearchPhase.PHASE_9,
    phase_number=9,
    research_role=ResearchRole.COUNTER_ALTERNATIVE_COMPARATOR,
    trace=ResearchTrace(
        candidate_id="CAND-RETRIEVAL-AUGMENTED-POSE-COMPARATOR",
        candidate_name="Retrieval-Augmented Pose Comparator",
        candidate_card_path=Path(
            "docs/research/contribution-audit/candidate-cards/"
            "retrieval-augmented-pose-comparator.md"
        ),
        candidate_universe_path=_CANDIDATE_UNIVERSE_PATH,
        scorecard_path=Path(
            "docs/research/contribution-audit/scorecards/"
            "retrieval-augmented-pose-comparator-scorecard.md"
        ),
        selection_decision_path=Path(
            "docs/research/contribution-audit/selection-decisions/"
            "retrieval-augmented-pose-comparator-selection-decision.md"
        ),
        audit_result_path=_AUDIT_RESULT_PATH,
        literature_positioning_path=_LITERATURE_POSITIONING_PATH,
        roadmap_path=_ROADMAP_PATH,
        source_corpus_ids=(
            "SRC-MIXED-SIGNALS-2021",
            "SRC-SIGN-STITCHING-2024",
            "SRC-HOW2SIGN-2021",
        ),
    ),
    primary_model_allowed=False,
    generated_pose_required=True,
    requires_leakage_policy=True,
    allowed_mechanisms=(MechanismTag.RETRIEVAL_REUSE,),
    forbidden_mechanisms=(
        MechanismTag.PRIMARY_LEARNED_GENERATOR,
        MechanismTag.GLOSS_SUPERVISION,
        MechanismTag.DICTIONARY_STITCHING,
        MechanismTag.MANUAL_GLOSS_SEGMENTATION,
        MechanismTag.LEARNED_POSE_TOKEN,
        MechanismTag.LATENT_DIFFUSION,
        MechanismTag.ARTICULATOR_AWARE_STRUCTURE,
        MechanismTag.SEMANTIC_CONSISTENCY_OBJECTIVE,
        MechanismTag.AVATAR_OR_3D,
        MechanismTag.RENDERING,
    ),
    required_reports=_reports(
        "retrieval_policy",
        "leakage_report",
        "retrieval_results",
        "comparison_against_models",
        "risk_controls",
    ),
    validation_requirements=_validations(
        "leakage_policy_enforced",
        "same_artifact_schema",
        "generated_pose_contract",
        "evaluation_harness_consumable",
        "metric_limitations_present",
    ),
    risk_controls=(
        RiskControl(
            identifier="retrieval_leakage_must_be_prevented",
            statement="Retrieval leakage must be prevented.",
        ),
        RiskControl(
            identifier="retrieval_realism_not_semantic_correctness",
            statement="Retrieval realism is not semantic correctness.",
        ),
        RiskControl(
            identifier="retrieval_not_primary_model",
            statement="Retrieval is not a primary model contribution.",
        ),
        RiskControl(
            identifier="dictionary_stitching_out_of_scope",
            statement="Gloss dictionary or stitching production is out of scope.",
        ),
    ),
)

_COMPARATOR_SPEC_SEQUENCE = (RETRIEVAL_POSE_SPEC,)

COMPARATOR_SPECS: Mapping[ComparatorKey, ComparatorSpec] = MappingProxyType(
    {spec.key: spec for spec in _COMPARATOR_SPEC_SEQUENCE}
)


def list_comparator_specs() -> tuple[ComparatorSpec, ...]:
    """Return comparator specs in registry order."""

    return _COMPARATOR_SPEC_SEQUENCE


def get_comparator_spec(key: ComparatorKey | str) -> ComparatorSpec | None:
    """Return a comparator spec, or None when the key is unknown."""

    try:
        resolved = coerce_comparator_key(key)
    except ModelingRegistryError:
        return None
    return COMPARATOR_SPECS.get(resolved)


def require_comparator_spec(key: ComparatorKey | str) -> ComparatorSpec:
    """Return a comparator spec or raise for an unknown key."""

    resolved = coerce_comparator_key(key)
    try:
        return COMPARATOR_SPECS[resolved]
    except KeyError as exc:
        raise ModelingRegistryError(f"unknown comparator key {resolved.value!r}") from exc


__all__ = [
    "COMPARATOR_SPECS",
    "RETRIEVAL_POSE_SPEC",
    "get_comparator_spec",
    "list_comparator_specs",
    "require_comparator_spec",
]
