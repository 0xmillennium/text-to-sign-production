from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any, Protocol, cast

from text_to_sign_production.core.ids import TierName
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
    TiersWorkflowResult,
)
from text_to_sign_production.workflows.tiers.processing import (
    TiersExecutionBundle,
    TiersReportBundle,
)


class _ReportLike(Protocol):
    summary: Any
    tables: Any


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
    summary = bundle.workflow_result.output_summary
    return (
        review_section(
            "Processing overview",
            (
                review_item(
                    "tiers processing",
                    (
                        ("loaded passed samples", summary.loaded_passed_sample_count),
                        ("quality facts", summary.quality_fact_count),
                        ("quality contexts", summary.quality_context_count),
                        ("quality metric bundles", summary.quality_metric_bundle_count),
                        ("leakage sample summaries", summary.leakage_sample_summary_count),
                        ("tier decisions", summary.tier_decision_count),
                        ("quality reports", summary.quality_report_count),
                        ("tiered manifest outputs", len(summary.tiered_manifest_outputs)),
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
                    tuple(_object_projection(bundle.filter_config).items()),
                ),
            ),
        ),
        review_section(
            "Tier policy summary",
            (
                review_item(
                    "policies",
                    tuple(_object_projection(bundle.tier_policies).items())
                    if isinstance(_object_projection(bundle.tier_policies), Mapping)
                    else (("policies", _object_projection(bundle.tier_policies)),),
                ),
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
            "Quality report summaries",
            tuple(_quality_report_summary_item(report) for report in bundle.quality_reports),
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
                        ("tiered manifest output count", len(bundle.tiered_manifest_outputs)),
                        ("quality report count", len(bundle.quality_reports)),
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
    records: list[Mapping[str, object]] = []
    report_lookup = {
        (report.manifest.split, report.manifest.sample_id): report
        for report in bundle.quality_reports
    }
    for decision_bundle in bundle.decision_bundles:
        key = (decision_bundle.manifest.split, decision_bundle.manifest.sample_id)
        report_bundle = report_lookup[key]
        report = cast(_ReportLike, report_bundle.report)
        records.append(
            {
                "split": decision_bundle.manifest.split.value,
                "sample_id": decision_bundle.manifest.sample_id,
                "selected_tier": _enum_value(
                    getattr(decision_bundle.decision, "selected_tier", None)
                ),
                "tier_status": _enum_value(getattr(decision_bundle.decision, "status", None)),
                "report_summary": _object_projection(getattr(report, "summary", None)),
                "metric_rows": _object_projection(getattr(report.tables, "metric_rows", ())),
                "tier_rows": _object_projection(getattr(report.tables, "tier_rows", ())),
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
        "leakage": _leakage_summary_projection(bundle),
        "quality_report_count": len(bundle.quality_reports),
    }


def build_calibration_detail_payload(
    bundle: TiersExecutionBundle,
) -> Mapping[str, object]:
    return {
        "workflow": "tiers",
        "processed_count": bundle.processed_count,
        "filter_config": _object_projection(bundle.filter_config),
        "tier_policies": _object_projection(bundle.tier_policies),
        "membership_counts_by_tier_split": _membership_count_payload(bundle),
        "leakage": _leakage_summary_projection(bundle),
        "quality_reports": tuple(
            _object_projection(cast(_ReportLike, report.report).summary)
            for report in bundle.quality_reports
        ),
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
    operation_fields: list[tuple[object, object]] = [
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
                operation_fields.append((field_name, value))
    progress = getattr(operation, "progress", None)
    if progress is not None:
        operation_fields.append(("progress stage id", progress.stage.stage_id))
    return review_item(str(operation.label), operation_fields)


def _execution_result_item(result: Any) -> WorkflowReviewItem:
    result_fields: list[tuple[object, object]] = [
        ("operation_kind", result.operation_kind),
        ("succeeded", result.succeeded),
        ("returncode", result.returncode),
        ("execution_mode", result.execution_mode),
    ]
    if result.observed_outputs:
        result_fields.append(("observed_outputs", result.observed_outputs))
    return review_item(str(result.label), result_fields)


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
    rows: list[TiersMembershipCountRow] = []
    for tier in TierName:
        for split in bundle.workflow_result.config.splits:
            included = 0
            excluded = 0
            for decision_bundle in bundle.decision_bundles:
                if decision_bundle.manifest.split.value != split:
                    continue
                if _is_included(
                    selected_tier=_selected_tier(decision_bundle.decision),
                    tier=tier,
                ):
                    included += 1
                else:
                    excluded += 1
            rows.append(
                TiersMembershipCountRow(
                    tier=tier.value,
                    split=split,
                    included_count=included,
                    excluded_count=excluded,
                )
            )
    return tuple(rows)


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


def _quality_report_summary_item(report_bundle: TiersReportBundle) -> WorkflowReviewItem:
    report = cast(_ReportLike, report_bundle.report)
    summary = report.summary
    return review_item(
        f"{summary.split.value}/{summary.sample_id}",
        tuple(_object_projection(summary).items()),
    )


def _leakage_summary_projection(
    bundle: TiersExecutionBundle,
) -> Mapping[str, object]:
    severity_counts: dict[str, int] = {}
    affected_sample_count = 0
    for summary in bundle.leakage_bundle.sample_summaries:
        severity = _enum_text(summary.max_severity)
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
        if summary.has_leakage:
            affected_sample_count += 1
    return {
        "sample_summary_count": len(bundle.leakage_bundle.sample_summaries),
        "pair_fact_count": len(bundle.leakage_bundle.pair_facts),
        "affected_sample_count": affected_sample_count,
        "max_severity_counts": dict(sorted(severity_counts.items())),
    }


def _selected_tier(decision: object) -> TierName | None:
    selected_tier = getattr(decision, "selected_tier", None)
    if selected_tier is None:
        return None
    return TierName(str(selected_tier))


def _is_included(*, selected_tier: TierName | None, tier: TierName) -> bool:
    if selected_tier is None:
        return False
    tier_order = tuple(TierName)
    return tier_order.index(tier) <= tier_order.index(selected_tier)


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
