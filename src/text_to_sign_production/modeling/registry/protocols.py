"""Concrete registry entries for research-backed evaluation protocols."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from text_to_sign_production.modeling.registry.access import (
    ModelingRegistryError,
    coerce_evaluation_protocol_key,
)
from text_to_sign_production.modeling.research import (
    EvaluationProtocolKey,
    EvaluationProtocolSpec,
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


PHASE5_CORE_PROTOCOL_SPEC = EvaluationProtocolSpec(
    key=EvaluationProtocolKey.PHASE5_CORE,
    canonical_id="how2sign_compatible_evaluation_core",
    display_name="How2Sign-Compatible Evaluation Core Protocol",
    phase=ResearchPhase.PHASE_5,
    phase_number=5,
    research_role=ResearchRole.EVALUATION_SUPPORT_SURFACE,
    trace=ResearchTrace(
        candidate_id="CAND-HOW2SIGN-EVALUATION-PROTOCOL",
        candidate_name="How2Sign Evaluation Protocol",
        candidate_card_path=Path(
            "docs/research/contribution-audit/candidate-cards/how2sign-evaluation-protocol.md"
        ),
        candidate_universe_path=_CANDIDATE_UNIVERSE_PATH,
        scorecard_path=None,
        selection_decision_path=None,
        audit_result_path=_AUDIT_RESULT_PATH,
        literature_positioning_path=_LITERATURE_POSITIONING_PATH,
        roadmap_path=_ROADMAP_PATH,
        source_corpus_ids=(
            "SRC-SLRTP2025-CHALLENGE",
            "SRC-POSE-EVAL-2025",
            "SRC-PROGTRANS-2020",
            "SRC-HOW2SIGN-2021",
        ),
    ),
    required_input_contracts=(
        "reference_prepared_pose",
        "t2sp-generated-pose-manifest-v1",
        "t2sp-generated-pose-v1",
    ),
    required_output_contracts=(
        "pairing_manifest",
        "metric_results",
        "aggregate_metrics",
        "qualitative_review_surface",
        "metric_limitations",
    ),
    required_metric_families=(
        "pose_keypoint_distance",
        "channel_level_metrics",
        "temporal_smoothness",
        "coverage_missingness",
        "confidence_coverage",
        "length_validity",
    ),
    optional_metric_families=(
        "embedding_checks",
        "cautious_recognition_or_back_translation",
    ),
    required_qualitative_surfaces=(
        "qualitative_checklist",
        "reference_vs_generated_visualization",
        "failure_mode_logging",
    ),
    required_limitation_statements=(
        "Automatic metrics are not proof of sign intelligibility.",
        "Visual plausibility is not proof of semantic correctness.",
        "Embedding similarity is not proof of linguistic adequacy.",
    ),
    required_reports=_reports(
        "evaluation_protocol",
        "metric_summary",
        "metric_limitations",
        "qualitative_checklist",
        "comparison_report",
    ),
    validation_requirements=_validations(
        "same_split_policy",
        "same_artifact_schema",
        "generated_pose_contract",
        "metric_limitations_present",
        "qualitative_surface_present",
    ),
    risk_controls=(
        RiskControl(
            identifier="automatic_metrics_not_intelligibility",
            statement="Automatic metrics are not proof of sign intelligibility.",
        ),
        RiskControl(
            identifier="visual_plausibility_not_semantic_correctness",
            statement="Visual plausibility is not proof of semantic correctness.",
        ),
        RiskControl(
            identifier="embedding_similarity_not_linguistic_adequacy",
            statement="Embedding similarity is not proof of linguistic adequacy.",
        ),
    ),
)

PHASE9_SEMANTIC_ABLATION_PROTOCOL_SPEC = EvaluationProtocolSpec(
    key=EvaluationProtocolKey.PHASE9_SEMANTIC_ABLATION,
    canonical_id="semantic_consistency_ablation_protocol",
    display_name="Semantic Consistency Ablation Protocol",
    phase=ResearchPhase.PHASE_9,
    phase_number=9,
    research_role=ResearchRole.EVALUATION_SUPPORT_SURFACE,
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
    required_input_contracts=(
        "generated_pose_without_semantic_objective",
        "generated_pose_with_semantic_objective",
        "same_split_policy",
        "same_evaluation_protocol",
    ),
    required_output_contracts=(
        "semantic_ablation_summary",
        "metric_limitations",
        "qualitative_review_surface",
    ),
    required_metric_families=(
        "phase5_core_metric_families",
        "semantic_alignment_diagnostics",
    ),
    optional_metric_families=("embedding_checks",),
    required_qualitative_surfaces=("with_without_comparison_review",),
    required_limitation_statements=(
        "Semantic embedding gain is not proof of sign intelligibility.",
        "Semantic consistency is not a standalone model contribution.",
    ),
    required_reports=_reports(
        "with_without_ablation",
        "metric_limitations",
        "semantic_attachment_decision",
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

PHASE9_RETRIEVAL_COMPARATOR_PROTOCOL_SPEC = EvaluationProtocolSpec(
    key=EvaluationProtocolKey.PHASE9_RETRIEVAL_COMPARATOR,
    canonical_id="retrieval_pose_comparator_protocol",
    display_name="Retrieval Pose Comparator Protocol",
    phase=ResearchPhase.PHASE_9,
    phase_number=9,
    research_role=ResearchRole.EVALUATION_SUPPORT_SURFACE,
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
    required_input_contracts=(
        "reference_surface",
        "retrieval_bank",
        "generated_pose_comparator_output",
        "leakage_policy",
    ),
    required_output_contracts=(
        "retrieval_results",
        "leakage_report",
        "comparator_metric_results",
        "metric_limitations",
    ),
    required_metric_families=(
        "phase5_core_metric_families",
        "retrieval_quality_diagnostics",
    ),
    optional_metric_families=("embedding_checks",),
    required_qualitative_surfaces=(
        "retrieval_failure_review",
        "reference_vs_retrieved_visualization",
    ),
    required_limitation_statements=(
        "Retrieval realism is not semantic correctness.",
        "Retrieval is not a primary model contribution.",
        "Retrieval leakage must be prevented before comparison.",
    ),
    required_reports=_reports(
        "retrieval_policy",
        "leakage_report",
        "comparator_report",
        "metric_limitations",
    ),
    validation_requirements=_validations(
        "leakage_policy_enforced",
        "same_artifact_schema",
        "generated_pose_contract",
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
    ),
)

_EVALUATION_PROTOCOL_SPEC_SEQUENCE = (
    PHASE5_CORE_PROTOCOL_SPEC,
    PHASE9_SEMANTIC_ABLATION_PROTOCOL_SPEC,
    PHASE9_RETRIEVAL_COMPARATOR_PROTOCOL_SPEC,
)

EVALUATION_PROTOCOL_SPECS: Mapping[EvaluationProtocolKey, EvaluationProtocolSpec] = (
    MappingProxyType({spec.key: spec for spec in _EVALUATION_PROTOCOL_SPEC_SEQUENCE})
)


def list_evaluation_protocol_specs() -> tuple[EvaluationProtocolSpec, ...]:
    """Return evaluation protocol specs in phase/key order."""

    return _EVALUATION_PROTOCOL_SPEC_SEQUENCE


def get_evaluation_protocol_spec(
    key: EvaluationProtocolKey | str,
) -> EvaluationProtocolSpec | None:
    """Return an evaluation protocol spec, or None when the key is unknown."""

    try:
        resolved = coerce_evaluation_protocol_key(key)
    except ModelingRegistryError:
        return None
    return EVALUATION_PROTOCOL_SPECS.get(resolved)


def require_evaluation_protocol_spec(
    key: EvaluationProtocolKey | str,
) -> EvaluationProtocolSpec:
    """Return an evaluation protocol spec or raise for an unknown key."""

    resolved = coerce_evaluation_protocol_key(key)
    try:
        return EVALUATION_PROTOCOL_SPECS[resolved]
    except KeyError as exc:
        raise ModelingRegistryError(f"unknown evaluation protocol key {resolved.value!r}") from exc


__all__ = [
    "EVALUATION_PROTOCOL_SPECS",
    "PHASE5_CORE_PROTOCOL_SPEC",
    "PHASE9_RETRIEVAL_COMPARATOR_PROTOCOL_SPEC",
    "PHASE9_SEMANTIC_ABLATION_PROTOCOL_SPEC",
    "get_evaluation_protocol_spec",
    "list_evaluation_protocol_specs",
    "require_evaluation_protocol_spec",
]
