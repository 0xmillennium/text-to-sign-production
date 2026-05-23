"""Honest deterministic reports for the semantic-consistency objective foundation."""

from __future__ import annotations

from pathlib import Path

from text_to_sign_production.modeling.objectives.semantic_consistency.attachments import (
    SemanticAblationPlan,
    SemanticObjectiveAttachmentDecision,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.ablation import (
    SemanticAblationReadinessResult,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.config import (
    SemanticConsistencyObjectiveConfig,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.errors import (
    SemanticConsistencyError,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.records import (
    SemanticAlignmentAggregate,
)

_REQUIRED_CAVEATS = (
    "Semantic consistency is an auxiliary objective, not a standalone model.",
    "Post-generation semantic artifacts are proxy diagnostics.",
    "A differentiable auxiliary training objective is available only when training_objective.enabled is true.",
    "With/without ablation is required.",
    "An executed baseline is required before contribution claims.",
    "Embedding similarity is not proof of sign intelligibility.",
    "Embedding similarity is not proof of linguistic semantic correctness.",
    "Deterministic hash text embeddings are proxy-only.",
    "Training objective text embeddings must come from the candidate conditioning path.",
    "Pose statistics projection remains a proxy alignment signal.",
    "No retrieval comparator is active.",
    "No gloss supervision is used.",
    "No dictionary stitching is used.",
    "No avatar/3D/rendering is active.",
)


def semantic_attachment_decision_report(
    decision: SemanticObjectiveAttachmentDecision,
) -> dict[str, object]:
    """Build a deterministic policy decision report for future integration."""

    _require(decision, SemanticObjectiveAttachmentDecision, "decision")
    return {
        "schema_version": "t2sp-semantic-attachment-decision-report-v1",
        "report_type": "semantic_attachment_decision",
        "objective_key": decision.objective_key.value,
        "model_key": decision.model_key.value,
        "attach_allowed": decision.attach_allowed,
        "requires_ablation": decision.requires_ablation,
        "reasons": list(decision.reasons),
        "blocking_issues": list(decision.blocking_issues),
        "caveats": list(_REQUIRED_CAVEATS),
    }


def semantic_objective_config_report(
    config: SemanticConsistencyObjectiveConfig,
) -> dict[str, object]:
    """Build a deterministic report of the validated foundation config."""

    _require(config, SemanticConsistencyObjectiveConfig, "config")
    return {
        "schema_version": "t2sp-semantic-objective-config-report-v1",
        "report_type": "semantic_objective_config",
        "objective_key": config.identity.objective_key.value,
        "phase_number": config.identity.phase_number,
        "config": config.to_dict(),
        "caveats": list(_REQUIRED_CAVEATS),
    }


def semantic_ablation_plan_report(plan: SemanticAblationPlan) -> dict[str, object]:
    """Build a deterministic report that never implies unexecuted completion."""

    _require(plan, SemanticAblationPlan, "plan")
    return {
        "schema_version": "t2sp-semantic-ablation-plan-report-v1",
        "report_type": "semantic_ablation_plan",
        "status": plan.status,
        "baseline_run_name": plan.baseline_run_name,
        "objective_run_name": plan.objective_run_name,
        "same_manifest_family_required": plan.same_manifest_family_required,
        "same_splits_required": plan.same_splits_required,
        "same_validation_protocol_required": plan.same_validation_protocol_required,
        "issues": list(plan.issues),
        "completed_ablation_exists": False,
        "caveats": list(_REQUIRED_CAVEATS),
    }


def semantic_ablation_readiness_report(
    readiness: SemanticAblationReadinessResult,
) -> dict[str, object]:
    """Render comparison readiness without implying an executed comparison."""

    _require(readiness, SemanticAblationReadinessResult, "readiness")
    return {
        "schema_version": "t2sp-semantic-ablation-readiness-report-v1",
        "report_type": "semantic_ablation_readiness",
        "objective_key": readiness.objective_key.value,
        "model_key": readiness.model_key.value,
        "objective_run_name": readiness.objective_run_name,
        "baseline_run_name": readiness.baseline_run_name,
        "ready_for_comparison": readiness.ready_for_comparison,
        "required_baseline_missing": readiness.required_baseline_missing,
        "same_model_required": readiness.same_model_required,
        "same_manifest_family_required": readiness.same_manifest_family_required,
        "same_splits_required": readiness.same_splits_required,
        "same_validation_protocol_required": readiness.same_validation_protocol_required,
        "issues": list(readiness.issues),
        "completed_ablation_exists": False,
        "caveats": list(readiness.caveats),
    }


def semantic_alignment_summary_report(
    aggregate: SemanticAlignmentAggregate,
    *,
    generated_payload_root: Path | None = None,
    payload_ref_policy: str | None = None,
) -> dict[str, object]:
    """Build the proxy-alignment summary with interpretation limits."""

    _require(aggregate, SemanticAlignmentAggregate, "aggregate")
    return {
        "schema_version": "t2sp-semantic-alignment-summary-report-v1",
        "report_type": "semantic_alignment_summary",
        "records_count": aggregate.records_count,
        "skipped_count": aggregate.skipped_count,
        "mean_cosine_similarity": aggregate.mean_cosine_similarity,
        "mean_cosine_distance": aggregate.mean_cosine_distance,
        "mean_loss_value": aggregate.mean_loss_value,
        "proxy_only": True,
        "candidate_policy": aggregate.candidate_policy,
        "generated_payload_root": (
            None if generated_payload_root is None else str(generated_payload_root)
        ),
        "payload_ref_policy": payload_ref_policy,
        "interpretation": "Proxy embedding alignment only; no linguistic or intelligibility claim.",
        "caveats": list(_REQUIRED_CAVEATS),
    }


def semantic_metric_limitations_report() -> dict[str, object]:
    """Build a report containing the mandatory metric limitations."""

    return {
        "schema_version": "t2sp-semantic-metric-limitations-report-v1",
        "report_type": "semantic_metric_limitations",
        "does_not_prove": [
            "sign intelligibility",
            "linguistic semantic correctness",
            "semantic objective contribution",
        ],
        "caveats": list(_REQUIRED_CAVEATS),
    }


def semantic_risk_controls_report(
    *,
    generated_payload_root: Path | None = None,
    payload_ref_policy: str | None = None,
) -> dict[str, object]:
    """Build a report of active controls and intentionally absent mechanisms."""

    return {
        "schema_version": "t2sp-semantic-risk-controls-report-v1",
        "report_type": "semantic_risk_controls",
        "generated_payload_root": (
            None if generated_payload_root is None else str(generated_payload_root)
        ),
        "payload_ref_policy": payload_ref_policy,
        "controls": {
            "auxiliary_only": True,
            "requires_with_without_ablation": True,
            "text_embedding_proxy_only": True,
            "pose_embedding_proxy_only": True,
            "zero_norm_fails": True,
            "non_finite_fails": True,
            "skipped_metrics_remain_absent": True,
            "single_candidate_only": True,
            "payload_refs_outside_current_final_validation_split_root_rejected": True,
        },
        "payload_ref_rejection_policy": (
            "payload refs outside the current final-validation split root are rejected"
        ),
        "required_ablation_state": "A compatible without-objective baseline is required; no completed ablation exists.",
        "scope_exclusions": [
            "retrieval comparator",
            "gloss supervision",
            "dictionary stitching",
            "avatar/3D/rendering",
        ],
        "caveats": list(_REQUIRED_CAVEATS),
    }


def _require(value: object, expected: type, name: str) -> None:
    if not isinstance(value, expected):
        raise SemanticConsistencyError(f"{name} must be a {expected.__name__}.")


__all__ = [
    "semantic_ablation_plan_report",
    "semantic_ablation_readiness_report",
    "semantic_alignment_summary_report",
    "semantic_attachment_decision_report",
    "semantic_metric_limitations_report",
    "semantic_objective_config_report",
    "semantic_risk_controls_report",
]
