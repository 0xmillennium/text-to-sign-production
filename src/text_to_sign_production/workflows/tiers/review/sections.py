from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any, cast

from text_to_sign_production.data.leakages import (
    build_leakage_affected_sample_coverage_records,
    build_leakage_sample_severity_distribution_records,
)
from text_to_sign_production.data.tiers import build_tier_inclusion_count_records
from text_to_sign_production.workflows.foundation.execution import (
    WorkflowOperation,
    operation_kind,
)
from text_to_sign_production.workflows.foundation.review import (
    WorkflowReviewItem,
    WorkflowReviewSection,
    review_item,
    review_section,
)
from text_to_sign_production.workflows.tiers.contracts import (
    TiersMembershipCountRow,
    TiersReportArtifactRow,
    TiersRuntimeAssetRow,
    TiersRuntimePlan,
    TiersRuntimeRestoreResult,
    TiersRuntimeVerification,
    TiersTieredManifestRow,
    TiersWorkflowInvariantError,
    TiersWorkflowResult,
)
from text_to_sign_production.workflows.tiers.processing import TiersExecutionBundle


def build_runtime_plan_sections(
    plan: TiersRuntimePlan,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Runtime restore plan",
            tuple(_operation_item(operation) for operation in plan.restore_operations),
        ),
    )


def build_runtime_restore_sections(
    result: TiersRuntimeRestoreResult,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Runtime restore execution",
            tuple(_execution_result_item(execution) for execution in result.execution.results),
        ),
    )


def build_runtime_verification_sections(
    verification: TiersRuntimeVerification,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Runtime verification",
            tuple(_runtime_asset_item(row) for row in _runtime_asset_rows(verification)),
        ),
    )


def build_processing_summary_sections(
    bundle: TiersExecutionBundle,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Processing overview",
            (
                review_item(
                    "tiers processing",
                    (
                        ("processed_count", bundle.processed_count),
                        ("metric bundle count", len(bundle.metric_bundles)),
                        (
                            "leakage summary count",
                            len(bundle.leakage_bundle.sample_summaries),
                        ),
                        ("tier policy count", len(bundle.tier_policies)),
                        (
                            "tiered manifest output count",
                            len(bundle.workflow_result.output_summary.tiered_manifest_outputs),
                        ),
                    ),
                ),
            ),
        ),
        review_section(
            "Membership counts by tier and split",
            tuple(_membership_count_item(row) for row in _membership_count_rows(bundle)),
        ),
    )


def build_calibration_sections(
    bundle: TiersExecutionBundle,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Filter config summary",
            (
                review_item(
                    "filters",
                    (
                        ("family count", len(_filter_config_projection(bundle))),
                        ("families", tuple(_filter_config_projection(bundle))),
                    ),
                ),
            ),
        ),
        review_section(
            "Tier policy summary",
            tuple(
                review_item(
                    str(_enum_value(policy.tier_name)),
                    (
                        ("max_allowed_leakage_severity", policy.max_allowed_leakage_severity),
                        ("family level count", len(policy.family_levels)),
                        ("family levels", _mapping_projection(dict(policy.family_levels))),
                    ),
                )
                for policy in bundle.tier_policies
            ),
        ),
        review_section(
            "Leakage summary",
            (
                review_item(
                    "leakage",
                    tuple(_leakage_summary_projection(bundle).items()),
                ),
            ),
        ),
        review_section(
            "Membership distribution summary",
            tuple(_membership_count_item(row) for row in _membership_count_rows(bundle)),
        ),
    )


def build_output_summary_sections(
    result: TiersWorkflowResult,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Tiered manifest outputs",
            tuple(_tiered_manifest_item(row) for row in _tiered_manifest_rows(result)),
        ),
        review_section(
            "Report artifacts",
            tuple(_report_artifact_item(row) for row in _report_artifact_rows(result)),
        ),
    )


def build_final_review_sections(
    bundle: TiersExecutionBundle,
) -> tuple[WorkflowReviewSection, ...]:
    artifacts = bundle.workflow_result.output_summary.report_artifacts
    return (
        review_section(
            "Final summary",
            (
                review_item(
                    "tiers",
                    (
                        ("workflow", "tiers"),
                        ("processed_count", bundle.processed_count),
                        ("tier policy count", len(bundle.tier_policies)),
                        (
                            "tiered manifest output count",
                            len(bundle.workflow_result.output_summary.tiered_manifest_outputs),
                        ),
                        ("summary markdown path", artifacts.summary_markdown_path),
                        ("calibration markdown path", artifacts.calibration_markdown_path),
                    ),
                ),
            ),
        ),
    )


def build_decision_detail_records(
    bundle: TiersExecutionBundle,
) -> tuple[Mapping[str, object], ...]:
    projection = _aligned_decision_projection(bundle)
    records: list[Mapping[str, object]] = []
    for membership in bundle.tier_bundle.memberships:
        key = (membership.split, membership.sample_id)
        tier_key = (membership.split, membership.sample_id, membership.tier_name)
        manifest = projection["manifests"][key]
        metric_bundle = projection["metrics"][key]
        leakage_summary = projection["leakages"][key]
        decision_detail = projection["decision_details"].get(tier_key)
        records.append(
            {
                "split": _enum_value(membership.split),
                "sample_id": membership.sample_id,
                "tier": _enum_value(membership.tier_name),
                "membership": _enum_value(membership.membership),
                "text": manifest.text,
                "num_frames": manifest.num_frames,
                "fps": manifest.fps,
                "leakage": _object_projection(leakage_summary),
                "metrics": _metric_bundle_projection(metric_bundle),
                "decision_detail": (
                    _object_projection(decision_detail) if decision_detail is not None else None
                ),
            }
        )
    return tuple(records)


def build_calibration_surfaces_payload(
    bundle: TiersExecutionBundle,
) -> Mapping[str, object]:
    return {
        "workflow": "tiers",
        "processed_count": bundle.processed_count,
        "membership_counts_by_tier_split": _membership_count_payload(bundle),
        "tier_policy_count": len(bundle.tier_policies),
        "leakage": _leakage_summary_projection(bundle),
    }


def build_calibration_detail_payload(
    bundle: TiersExecutionBundle,
) -> Mapping[str, object]:
    return {
        "workflow": "tiers",
        "processed_count": bundle.processed_count,
        "filter_config": _filter_config_projection(bundle),
        "tier_policies": _tier_policy_projection(bundle),
        "membership_counts_by_tier_split": _membership_count_payload(bundle),
        "leakage": _leakage_summary_projection(bundle),
        "outputs": {
            "tiered_manifest_outputs": tuple(
                {
                    "tier": output.tier,
                    "membership": output.membership,
                    "split": output.split,
                    "path": output.path,
                }
                for output in bundle.workflow_result.output_summary.tiered_manifest_outputs
            ),
        },
    }


def _operation_item(operation: WorkflowOperation) -> WorkflowReviewItem:
    fields: list[tuple[object, object]] = [
        ("operation kind", operation_kind(operation)),
        ("label", getattr(operation, "label", "")),
    ]
    for field_name in (
        "source_path",
        "target_path",
        "archive_path",
        "extraction_root",
        "expected_input_bytes",
    ):
        if hasattr(operation, field_name):
            value = getattr(operation, field_name)
            if value is not None:
                fields.append((field_name, value))
    progress = getattr(operation, "progress", None)
    if progress is not None:
        fields.append(("progress stage id", progress.stage.stage_id))
    return review_item(str(operation.label), fields)


def _execution_result_item(result: Any) -> WorkflowReviewItem:
    fields: list[tuple[object, object]] = [
        ("operation_kind", result.operation_kind),
        ("succeeded", result.succeeded),
        ("returncode", result.returncode),
        ("execution_mode", result.execution_mode),
    ]
    if result.observed_outputs:
        fields.append(("observed_outputs", result.observed_outputs))
    return review_item(str(result.label), fields)


def _runtime_asset_rows(
    verification: TiersRuntimeVerification,
) -> tuple[TiersRuntimeAssetRow, ...]:
    return tuple(
        TiersRuntimeAssetRow(label=check.label, path=check.path, exists=check.exists)
        for check in verification.checks
    )


def _runtime_asset_item(row: TiersRuntimeAssetRow) -> WorkflowReviewItem:
    return review_item(
        row.label,
        (
            ("path", row.path),
            ("exists", row.exists),
        ),
    )


def _membership_count_rows(
    bundle: TiersExecutionBundle,
) -> tuple[TiersMembershipCountRow, ...]:
    return tuple(
        TiersMembershipCountRow(
            tier=_enum_text(record.tier_name),
            split=_enum_text(record.split),
            included_count=record.included_count,
            excluded_count=record.excluded_count,
        )
        for record in build_tier_inclusion_count_records(bundle.tier_bundle.memberships)
    )


def _membership_count_item(row: TiersMembershipCountRow) -> WorkflowReviewItem:
    return review_item(
        f"{row.tier} [{row.split}]",
        (
            ("tier", row.tier),
            ("split", row.split),
            ("included_count", row.included_count),
            ("excluded_count", row.excluded_count),
        ),
    )


def _membership_count_payload(
    bundle: TiersExecutionBundle,
) -> tuple[Mapping[str, object], ...]:
    return tuple(
        {
            "tier": row.tier,
            "split": row.split,
            "included_count": row.included_count,
            "excluded_count": row.excluded_count,
        }
        for row in _membership_count_rows(bundle)
    )


def _tiered_manifest_rows(
    result: TiersWorkflowResult,
) -> tuple[TiersTieredManifestRow, ...]:
    return tuple(
        TiersTieredManifestRow(
            tier=output.tier,
            membership=output.membership,
            split=output.split,
            path=output.path,
        )
        for output in result.output_summary.tiered_manifest_outputs
    )


def _tiered_manifest_item(row: TiersTieredManifestRow) -> WorkflowReviewItem:
    return review_item(
        f"{row.tier}/{row.membership} [{row.split}]",
        (
            ("tier", row.tier),
            ("membership", row.membership),
            ("split", row.split),
            ("path", row.path),
        ),
    )


def _report_artifact_rows(
    result: TiersWorkflowResult,
) -> tuple[TiersReportArtifactRow, ...]:
    artifacts = result.output_summary.report_artifacts
    return (
        TiersReportArtifactRow("summary markdown", artifacts.summary_markdown_path),
        TiersReportArtifactRow("calibration markdown", artifacts.calibration_markdown_path),
        TiersReportArtifactRow("decision detail JSONL", artifacts.decision_detail_jsonl_path),
        TiersReportArtifactRow(
            "calibration surfaces JSON",
            artifacts.calibration_surfaces_json_path,
        ),
        TiersReportArtifactRow(
            "calibration detail JSON",
            artifacts.calibration_detail_json_path,
        ),
        TiersReportArtifactRow("report index JSON", artifacts.index_json_path),
    )


def _report_artifact_item(row: TiersReportArtifactRow) -> WorkflowReviewItem:
    return review_item(row.label, (("path", row.path),))


def _aligned_decision_projection(
    bundle: TiersExecutionBundle,
) -> dict[str, Any]:
    manifests = {
        (manifest.split, manifest.sample_id): manifest
        for manifest in bundle.catalog_bundle.manifests
    }
    metrics = {
        (metric_bundle.split, metric_bundle.sample_id): metric_bundle
        for metric_bundle in bundle.metric_bundles
    }
    leakages = {
        (summary.split, summary.sample_id): summary
        for summary in bundle.leakage_bundle.sample_summaries
    }
    memberships = {
        (membership.split, membership.sample_id) for membership in bundle.tier_bundle.memberships
    }
    expected_keys = set(manifests)
    if set(metrics) != expected_keys or set(leakages) != expected_keys:
        raise TiersWorkflowInvariantError("tiers decision detail inputs are misaligned")
    if memberships != expected_keys:
        raise TiersWorkflowInvariantError("tiers membership samples are misaligned")
    decision_details = {
        (detail.split, detail.sample_id, detail.tier_name): detail
        for detail in bundle.tier_bundle.decision_details
    }
    return {
        "manifests": manifests,
        "metrics": metrics,
        "leakages": leakages,
        "decision_details": decision_details,
    }


def _filter_config_projection(
    bundle: TiersExecutionBundle,
) -> Mapping[str, object]:
    return cast(Mapping[str, object], _object_projection(bundle.filter_config))


def _tier_policy_projection(
    bundle: TiersExecutionBundle,
) -> tuple[Mapping[str, object], ...]:
    return tuple(_object_projection(policy) for policy in bundle.tier_policies)


def _leakage_summary_projection(
    bundle: TiersExecutionBundle,
) -> Mapping[str, object]:
    severity_counts: dict[str, int] = {}
    for record in build_leakage_sample_severity_distribution_records(bundle.leakage_bundle):
        severity = _enum_text(record.severity)
        severity_counts[severity] = severity_counts.get(severity, 0) + record.sample_count
    affected_count = sum(
        record.affected_sample_count
        for record in build_leakage_affected_sample_coverage_records(bundle.leakage_bundle)
    )
    return {
        "sample_summary_count": len(bundle.leakage_bundle.sample_summaries),
        "pair_fact_count": len(bundle.leakage_bundle.pair_facts),
        "affected_sample_count": affected_count,
        "max_severity_counts": dict(sorted(severity_counts.items())),
    }


def _metric_bundle_projection(metric_bundle: object) -> Mapping[str, object]:
    return cast(Mapping[str, object], _object_projection(metric_bundle))


def _mapping_projection(value: Mapping[Any, Any]) -> Mapping[str, object]:
    return {
        str(_enum_value(key)): _object_projection(mapping_value)
        for key, mapping_value in value.items()
    }


def _object_projection(value: object) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _object_projection(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, Mapping):
        return _mapping_projection(value)
    if isinstance(value, (tuple, list)):
        return tuple(_object_projection(item) for item in value)
    return _enum_value(value)


def _enum_value(value: Any) -> object:
    if hasattr(value, "value"):
        return value.value
    return value


def _enum_text(value: Any) -> str:
    return str(_enum_value(value))
