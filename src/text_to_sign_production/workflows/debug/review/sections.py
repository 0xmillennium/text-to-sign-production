from __future__ import annotations

from enum import Enum
from pathlib import Path

from text_to_sign_production.workflows.debug.contracts import (
    DebugGateResult,
    DebugPublishPlan,
    DebugPublishResult,
    DebugPublishVerification,
    DebugReportResult,
    DebugRestorePlan,
    DebugRestorePlanValidation,
    DebugRestoreResult,
    DebugRuntimeVerification,
    DebugSampleDossier,
    DebugSampleRequest,
    DebugTargetResolution,
    DebugTierResult,
    DebugVisualizationResult,
)
from text_to_sign_production.workflows.debug.contracts.verdicts import (
    DebugLeakageContextMode,
    DebugRestoreOperationStatus,
)
from text_to_sign_production.workflows.foundation.review import (
    RenderableValue,
    WorkflowReviewSection,
    review_item,
    review_lines_section,
    review_section,
)


def review_request(request: DebugSampleRequest) -> tuple[WorkflowReviewSection, ...]:
    mode, production_like, warning = _split_context(request.debug_splits)
    return (
        review_section(
            "Operator request",
            (
                review_item(
                    "debug sample",
                    (
                        ("DEBUG_SPLITS", tuple(split.value for split in request.debug_splits)),
                        ("TARGET_SENTENCE_NAME", request.target_sentence_name),
                    ),
                ),
            ),
        ),
        review_section(
            "Derived mode",
            (
                review_item(
                    "split context",
                    (
                        ("mode", mode.value),
                        ("production-like leakage", production_like),
                        ("warning", warning),
                    ),
                ),
            ),
        ),
    )


def review_restore_plan(plan: DebugRestorePlan) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Restore plan summary",
            (
                review_item(
                    "debug restore",
                    (
                        ("DEBUG_SPLITS", tuple(split.value for split in plan.debug_splits)),
                        ("operation count", len(plan.operations)),
                        ("required groups", plan.required_groups),
                        ("optional groups", plan.optional_groups),
                    ),
                ),
            ),
        ),
        review_section(
            "Restore operations",
            tuple(
                review_item(
                    operation.description,
                    (
                        ("group", operation.group),
                        ("split", None if operation.split is None else operation.split.value),
                        ("required", operation.required),
                        ("source", operation.source),
                        ("target", operation.target),
                        ("operation kind", operation.workflow_operation.__class__.__name__),
                    ),
                )
                for operation in plan.operations
            ),
        ),
    )


def review_restore_plan_validation(
    validation: DebugRestorePlanValidation,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Restore plan validation",
            (
                review_item(
                    "validation",
                    (
                        ("succeeded", validation.succeeded),
                        ("valid", validation.valid),
                        ("error count", len(validation.blocking_errors)),
                        ("warning count", len(validation.warnings)),
                    ),
                ),
            ),
        ),
        review_section(
            "Restore plan validation errors",
            tuple(_issue_item(issue) for issue in validation.blocking_errors),
        ),
        review_section(
            "Restore plan validation warnings",
            tuple(_issue_item(issue) for issue in validation.warnings),
        ),
    )


def review_restore_result(result: DebugRestoreResult) -> tuple[WorkflowReviewSection, ...]:
    failures = tuple(item for item in result.operation_results if not item.succeeded)
    warnings = tuple(
        item
        for item in result.operation_results
        if item.status
        in {
            DebugRestoreOperationStatus.SKIPPED_OPTIONAL,
            DebugRestoreOperationStatus.FAILED_OPTIONAL,
        }
    )
    return (
        review_section(
            "Restore result summary",
            (
                review_item(
                    "restore",
                    (
                        ("succeeded", result.succeeded),
                        ("operation count", len(result.operation_results)),
                        ("failed required count", len(failures)),
                        ("optional warning count", len(warnings)),
                    ),
                ),
            ),
        ),
        review_section(
            "Restore result details",
            tuple(
                review_item(
                    item.operation.description,
                    (
                        ("status", item.status.value),
                        ("required", item.operation.required),
                        ("message", item.message),
                    ),
                )
                for item in result.operation_results
            ),
        ),
    )


def review_runtime_verification(
    verification: DebugRuntimeVerification,
) -> tuple[WorkflowReviewSection, ...]:
    failed_required = tuple(
        check
        for check in verification.checks
        if check.required and (not check.exists or not check.valid)
    )
    diagnostic_issues = tuple(
        check
        for check in verification.checks
        if not check.required and (not check.exists or not check.valid)
    )
    return (
        review_section(
            "Runtime verification summary",
            (
                review_item(
                    "readiness",
                    (
                        ("succeeded", verification.succeeded),
                        ("check count", len(verification.checks)),
                        ("failed required count", len(failed_required)),
                        ("diagnostic issue count", len(diagnostic_issues)),
                    ),
                ),
            ),
        ),
        review_section(
            "Runtime verification checks",
            tuple(
                review_item(
                    check.label,
                    (
                        ("required", check.required),
                        ("exists", check.exists),
                        ("valid", check.valid),
                        ("path", check.path),
                        ("message", check.message),
                    ),
                )
                for check in verification.checks
            ),
        ),
    )


def review_target_resolution(
    resolution: DebugTargetResolution,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Target resolution summary",
            (
                review_item(
                    "target",
                    (
                        ("status", resolution.status.value),
                        ("TARGET_SENTENCE_NAME", resolution.request.target_sentence_name),
                        ("match count", len(resolution.matches)),
                        ("errors", resolution.errors),
                        ("warnings", resolution.warnings),
                    ),
                ),
            ),
        ),
        review_section(
            "Target resolution matches",
            tuple(
                review_item(
                    match.sentence_name,
                    (
                        ("split", match.split.value),
                        ("VIDEO_ID", match.video_id),
                        ("VIDEO_NAME", match.video_name),
                        ("SENTENCE_ID", match.sentence_id),
                        ("START_REALIGNED", match.start_realigned),
                        ("END_REALIGNED", match.end_realigned),
                        ("translation_path", match.translation_path),
                        ("row_index", match.row_index),
                    ),
                )
                for match in resolution.matches
            ),
        ),
    )


def review_sample_dossier(dossier: DebugSampleDossier) -> tuple[WorkflowReviewSection, ...]:
    row = dossier.translation
    return (
        review_section(
            "How2Sign translation evidence",
            (
                review_item(
                    row.sentence_name,
                    (
                        ("split", row.split.value),
                        ("VIDEO_ID", row.video_id),
                        ("VIDEO_NAME", row.video_name),
                        ("SENTENCE_ID", row.sentence_id),
                        ("SENTENCE_NAME", row.sentence_name),
                        ("START_REALIGNED", row.start_realigned),
                        ("END_REALIGNED", row.end_realigned),
                        ("SENTENCE", row.sentence),
                        ("translation_path", row.translation_path),
                        ("row_index", row.row_index),
                    ),
                ),
            ),
        ),
        review_section(
            "Raw BFH/OpenPose source evidence",
            (
                review_item(
                    row.sentence_name,
                    (
                        ("keypoint JSON directory", dossier.source.keypoint_json_dir),
                        ("keypoint JSON directory exists", dossier.source.keypoint_json_dir_exists),
                        ("keypoint frame JSON count", dossier.source.keypoint_frame_json_count),
                        ("raw source video path", dossier.source.raw_video_path),
                        ("raw source video exists", dossier.source.raw_video_exists),
                        ("issues", dossier.source.source_issues),
                    ),
                ),
            ),
        ),
        review_section(
            "Existing gate output evidence",
            (
                review_item(
                    dossier.existing_gate.status.value,
                    (
                        ("passed manifest", dossier.existing_gate.passed_manifest_path),
                        ("dropped manifest", dossier.existing_gate.dropped_manifest_path),
                        ("passed payload path", dossier.existing_gate.passed_payload_path),
                        ("passed payload exists", dossier.existing_gate.passed_payload_exists),
                        ("dropped record path", dossier.existing_gate.dropped_record_path),
                        ("dropped record exists", dossier.existing_gate.dropped_record_exists),
                        ("errors", dossier.existing_gate.errors),
                        ("warnings", dossier.existing_gate.warnings),
                    ),
                ),
            ),
        ),
        review_section(
            "Existing tier output evidence",
            tuple(
                review_item(
                    tier.tier.value,
                    (
                        ("state", tier.state.value),
                        ("included manifest", tier.included_manifest_path),
                        ("excluded manifest", tier.excluded_manifest_path),
                        ("errors", tier.errors),
                        ("warnings", tier.warnings),
                    ),
                )
                for tier in dossier.existing_tiers
            ),
        ),
        review_lines_section("Initial consistency notes", dossier.consistency_notes),
    )


def review_gate_debug(result: DebugGateResult) -> tuple[WorkflowReviewSection, ...]:
    gate = result.computed_gate
    sample = result.computed_sample
    return (
        review_section(
            "Gate preflight",
            (
                review_item(
                    "preflight",
                    (
                        ("can compute", result.preflight.can_compute),
                        ("errors", result.preflight.errors),
                        ("warnings", result.preflight.warnings),
                    ),
                ),
            ),
        ),
        review_section(
            "Gate recompute",
            (
                review_item(
                    _enum_value(result.computed_status) or "not computed",
                    (
                        ("computed status", _enum_value(result.computed_status)),
                        ("drop stage", _drop_stage(result)),
                        ("terminal gate", _enum_value(None if gate is None else gate.terminal_gate)),
                        ("failed gates", _enum_tuple(() if gate is None else gate.failed_gates)),
                        ("issue codes", _gate_issue_codes(result)),
                        ("sample id", None if sample is None else sample.source.sample_id),
                        ("split", None if sample is None else sample.source.split.value),
                        ("passed payload ref", _passed_payload_ref(result)),
                        ("passed payload path", result.existing_gate.passed_payload_path),
                        ("dropped sample ref", _dropped_sample_ref(result)),
                        ("dropped record path", result.existing_gate.dropped_record_path),
                        ("frame count", result.frame_metrics.frame_count),
                        ("valid frame count", result.frame_metrics.valid_frame_count),
                        ("body nonzero frame count", result.frame_metrics.body_nonzero_frame_count),
                        ("face nonzero frame count", result.frame_metrics.face_nonzero_frame_count),
                        (
                            "left hand nonzero frame count",
                            result.frame_metrics.left_hand_nonzero_frame_count,
                        ),
                        (
                            "right hand nonzero frame count",
                            result.frame_metrics.right_hand_nonzero_frame_count,
                        ),
                    ),
                ),
            ),
        ),
        review_section(
            "Existing gate state",
            (
                review_item(
                    "existing gate",
                    (
                        ("status", result.existing_status.value),
                        ("passed manifest", result.existing_gate.passed_manifest_path),
                        ("dropped manifest", result.existing_gate.dropped_manifest_path),
                        ("passed payload path", result.existing_gate.passed_payload_path),
                        ("passed payload exists", result.existing_gate.passed_payload_exists),
                        ("dropped record path", result.existing_gate.dropped_record_path),
                        ("dropped record exists", result.existing_gate.dropped_record_exists),
                        ("warnings", result.comparison.warnings),
                        ("comparison available", result.comparison.compared),
                    ),
                ),
            ),
        ),
        review_section(
            "Gate comparison",
            (
                review_item(
                    "comparison",
                    (
                        ("compared", result.comparison.compared),
                        ("status matches", result.comparison.status_matches),
                        ("differences", result.comparison.differences),
                    ),
                ),
            ),
        ),
        review_section(
            "Gate verdict",
            (
                review_item(
                    result.verdict.value,
                    (
                        ("errors", result.errors),
                        ("warnings", result.warnings),
                    ),
                ),
            ),
        ),
    )


def review_tier_debug(result: DebugTierResult) -> tuple[WorkflowReviewSection, ...]:
    decision = result.computed_decision
    leakage = result.leakage_summary
    return (
        review_section(
            "Tier applicability",
            (
                review_item(
                    "applicability",
                    (
                        ("applicable", result.applicable),
                        ("reason", result.applicability_reason),
                    ),
                ),
            ),
        ),
        review_section(
            "Leakage context",
            (
                review_item(
                    result.leakage_context.mode.value,
                    (
                        ("DEBUG_SPLITS", tuple(split.value for split in result.leakage_context.debug_splits)),
                        ("production-like", result.leakage_context.production_like),
                        ("warning", result.leakage_context.warning),
                    ),
                ),
            ),
        ),
        review_section(
            "Tier recompute source",
            (
                review_item(
                    "existing passed payload",
                    (
                        ("payload path", result.recompute_source.payload_path),
                        ("payload exists", result.recompute_source.payload_exists),
                        ("can compute", result.recompute_source.can_compute),
                        ("reason", result.recompute_source.reason),
                    ),
                ),
            ),
        ),
        review_section(
            "Tier recompute",
            (
                review_item(
                    "decision",
                    (
                        ("computed status", None if decision is None else decision.status.value),
                        ("selected tier", _enum_value(None if decision is None else decision.selected_tier)),
                        ("family decision count", len(result.family_decisions)),
                        ("unsupported family count", _unsupported_family_count(result)),
                        ("issue codes", () if decision is None else tuple(issue.code.value for issue in decision.issues)),
                        ("quality metric count", len(result.quality_metrics)),
                        ("leakage max severity", None if leakage is None else leakage.max_severity.value),
                        ("same source sentence matches", None if leakage is None else leakage.same_source_sentence_match_count),
                        ("exact text matches", None if leakage is None else leakage.exact_text_match_count),
                        ("same source video matches", None if leakage is None else leakage.same_source_video_match_count),
                        ("has leakage", None if leakage is None else leakage.has_leakage),
                        ("matched samples", _matched_samples(leakage)),
                        ("matched sample count", None if leakage is None else len(leakage.matched_samples)),
                        ("matched sample list truncated", _matched_samples_truncated(leakage)),
                    ),
                ),
            ),
        ),
        review_section(
            "Tier family decisions",
            tuple(
                review_item(
                    family.family,
                    (
                        ("status", family.status.value),
                        ("supported tiers", tuple(tier.value for tier in family.supported_tiers)),
                        (
                            "best supported tier",
                            None
                            if family.best_supported_tier is None
                            else family.best_supported_tier.value,
                        ),
                        ("issue codes", family.issue_codes),
                        ("reasons", family.reasons),
                    ),
                )
                for family in result.family_decisions
            ),
        ),
        review_section(
            "Tier quality metrics",
            tuple(
                review_item(
                    metric.metric_name,
                    (
                        ("value", metric.value),
                        ("status", metric.status.value),
                        ("notes", metric.notes),
                    ),
                )
                for metric in result.quality_metrics[:30]
            )
            + (
                review_item(
                    "truncation",
                    (
                        ("shown", min(len(result.quality_metrics), 30)),
                        ("total", len(result.quality_metrics)),
                    ),
                ),
            ),
        ),
        review_section(
            "Existing tier state",
            tuple(
                review_item(
                    state.tier.value,
                    (
                        ("existing state", state.state.value),
                        ("included manifest", state.included_manifest_path),
                        ("excluded manifest", state.excluded_manifest_path),
                        ("included sample id", None if state.included_entry is None else state.included_entry.sample_id),
                        ("excluded sample id", None if state.excluded_entry is None else state.excluded_entry.sample_id),
                        ("errors", state.errors),
                        ("warnings", state.warnings),
                    ),
                )
                for state in result.existing_tier_states
            ),
        ),
        review_section(
            "Tier comparison",
            tuple(
                review_item(
                    membership.tier.value,
                    (
                        ("computed state", membership.computed_state.value),
                        ("existing state", membership.existing_state.value),
                        ("matches", membership.matches),
                    ),
                )
                for membership in result.comparison.memberships
            )
            + (
                review_item(
                    "comparison summary",
                    (
                        ("compared", result.comparison.compared),
                        ("differences", result.comparison.differences),
                        ("warnings", result.comparison.warnings),
                    ),
                ),
            ),
        ),
        review_section(
            "Tier verdict",
            (
                review_item(
                    result.verdict.value,
                    (
                        ("errors", result.errors),
                        ("warnings", result.warnings),
                    ),
                ),
            ),
        ),
    )


def review_visualization_debug(
    result: DebugVisualizationResult,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Visualization preflight",
            (
                review_item(
                    "inputs",
                    (
                        ("output root", result.output_root),
                        ("error count", len(result.errors)),
                        ("warning count", len(result.warnings)),
                    ),
                ),
            ),
        ),
        review_section(
            "Rendered artifacts",
            tuple(
                review_item(
                    artifact.label,
                    (
                        ("created", artifact.created),
                        ("path", artifact.path),
                        ("error", artifact.error),
                    ),
                )
                for artifact in result.artifacts
            ),
        ),
        review_section(
            "Visualization verdict",
            (
                review_item(
                    result.verdict.value,
                    (
                        ("errors", result.errors),
                        ("warnings", result.warnings),
                    ),
                ),
            ),
        ),
    )


def review_report_outputs(result: DebugReportResult) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Debug reports",
            (
                review_item(
                    "report outputs",
                    (
                        ("succeeded", result.succeeded),
                        ("run_id", result.run_id),
                        ("output root", result.output_root),
                        ("file count", len(result.files)),
                        ("errors", result.errors),
                        ("warnings", result.warnings),
                    ),
                ),
            ),
        ),
        review_lines_section("Debug report files", tuple(path.as_posix() for path in result.files)),
    )


def review_publish_plan(plan: DebugPublishPlan) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Publish plan summary",
            (
                review_item(
                    "debug output publish",
                    (
                        ("valid", plan.valid),
                        ("runtime output root", plan.runtime_output_root),
                        ("Drive output root", plan.drive_output_root),
                        ("expected file count", plan.expected_file_count),
                        ("operation count", len(plan.operations)),
                        ("blocking error count", len(plan.blocking_errors)),
                        ("warning count", len(plan.warnings)),
                    ),
                ),
            ),
        ),
        review_section(
            "Publish operations",
            tuple(
                review_item(
                    operation.description,
                    (
                        ("source", operation.source),
                        ("target", operation.target),
                        ("required", operation.required),
                        ("operation kind", operation.workflow_operation.__class__.__name__),
                    ),
                )
                for operation in plan.operations
            ),
        ),
        review_section(
            "Publish plan blocking errors",
            tuple(_issue_item(issue) for issue in plan.blocking_errors),
        ),
        review_section(
            "Publish plan warnings",
            tuple(_issue_item(issue) for issue in plan.warnings),
        ),
    )


def review_publish_result(result: DebugPublishResult) -> tuple[WorkflowReviewSection, ...]:
    skipped = tuple(
        item
        for item in result.operation_results
        if item.status.value == "skipped_missing_source"
    )
    failed = tuple(item for item in result.operation_results if item.status.value == "failed")
    return (
        review_section(
            "Publish result summary",
            (
                review_item(
                    "debug output publish",
                    (
                        ("succeeded", result.succeeded),
                        ("runtime output root", result.plan.runtime_output_root),
                        ("Drive output root", result.plan.drive_output_root),
                        ("expected file count", result.plan.expected_file_count),
                        ("published file count", result.published_file_count),
                        ("skipped missing source count", len(skipped)),
                        ("failed count", len(failed)),
                        ("blocking error count", len(result.blocking_errors)),
                        ("warning count", len(result.warnings)),
                    ),
                ),
            ),
        ),
        review_section(
            "Publish result details",
            tuple(
                review_item(
                    item.source.name,
                    (
                        ("status", item.status.value),
                        ("source", item.source),
                        ("target", item.target),
                        ("message", item.message),
                    ),
                )
                for item in result.operation_results
            ),
        ),
        review_section(
            "Publish result blocking errors",
            tuple(_issue_item(issue) for issue in result.blocking_errors),
        ),
        review_section(
            "Publish result warnings",
            tuple(_issue_item(issue) for issue in result.warnings),
        ),
    )


def review_publish_verification(
    verification: DebugPublishVerification,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Publish verification summary",
            (
                review_item(
                    verification.status.value,
                    (
                        ("succeeded", verification.succeeded),
                        ("Drive output root", verification.drive_output_root),
                        ("verified file count", verification.verified_file_count),
                        ("missing target count", len(verification.missing_targets)),
                        ("warning count", len(verification.warnings)),
                    ),
                ),
            ),
        ),
        review_lines_section(
            "Publish missing targets",
            tuple(path.as_posix() for path in verification.missing_targets),
        ),
        review_section(
            "Publish verification warnings",
            tuple(_issue_item(issue) for issue in verification.warnings),
        ),
    )


def _split_context(splits) -> tuple[DebugLeakageContextMode, bool, str | None]:
    split_set = {split.value for split in splits}
    if len(splits) == 1:
        return (
            DebugLeakageContextMode.SINGLE_SPLIT_CONTEXT,
            False,
            "cross-split leakage cannot be observed",
        )
    if split_set == {"train", "val", "test"}:
        return DebugLeakageContextMode.GLOBAL_SPLIT_CONTEXT, True, None
    if split_set.issubset({"train", "val", "test"}):
        return (
            DebugLeakageContextMode.MULTI_SPLIT_CONTEXT,
            False,
            "selected splits are a subset of train/val/test",
        )
    return DebugLeakageContextMode.MULTI_SPLIT_CONTEXT, False, None


def _drop_stage(result: DebugGateResult) -> str | None:
    if result.computed_dropped_entry is None:
        return None
    return result.computed_dropped_entry.drop_stage.value


def _gate_issue_codes(result: DebugGateResult) -> tuple[str, ...]:
    if result.computed_gate is not None:
        return tuple(
            code.value
            for decision in result.computed_gate.decisions
            for code in decision.issue_codes
        )
    if result.computed_dropped_entry is not None:
        return tuple(code.value for code in result.computed_dropped_entry.issue_codes)
    return ()


def _enum_value(value: Enum | None) -> str | None:
    if value is None:
        return None
    return value.value


def _enum_tuple(values) -> tuple[str, ...]:
    return tuple(value.value if isinstance(value, Enum) else str(value) for value in values)


def _issue_item(issue):
    return review_item(
        issue.code,
        (
            ("message", issue.message),
            ("path", issue.path),
            ("label", issue.label),
        ),
    )


def _passed_payload_ref(result: DebugGateResult) -> str | None:
    if result.computed_passed_entry is not None:
        return result.computed_passed_entry.payload_ref
    if result.existing_gate.passed_entry is not None:
        return result.existing_gate.passed_entry.payload_ref
    return None


def _dropped_sample_ref(result: DebugGateResult) -> str | None:
    if result.computed_dropped_entry is not None:
        return result.computed_dropped_entry.dropped_sample_ref
    if result.existing_gate.dropped_entry is not None:
        return result.existing_gate.dropped_entry.dropped_sample_ref
    return None


def _unsupported_family_count(result: DebugTierResult) -> int:
    return sum(1 for family in result.family_decisions if family.best_supported_tier is None)


def _matched_samples(leakage) -> tuple[str, ...]:
    if leakage is None:
        return ()
    return tuple(
        f"{ref.split.value}/{ref.sample_id}"
        for ref in sorted(
            leakage.matched_samples,
            key=lambda item: (item.split.value, item.sample_id),
        )[:20]
    )


def _matched_samples_truncated(leakage) -> bool | None:
    if leakage is None:
        return None
    return len(leakage.matched_samples) > 20


def _review_value(value: object) -> RenderableValue:
    if value is None or isinstance(value, str | int | float | bool | Path):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return tuple(_review_value(item) for item in value)
    return str(value)


__all__ = [
    "review_gate_debug",
    "review_report_outputs",
    "review_request",
    "review_restore_plan",
    "review_restore_plan_validation",
    "review_restore_result",
    "review_runtime_verification",
    "review_sample_dossier",
    "review_target_resolution",
    "review_tier_debug",
    "review_visualization_debug",
]
