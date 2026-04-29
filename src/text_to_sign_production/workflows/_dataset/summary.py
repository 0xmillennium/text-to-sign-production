"""Notebook-facing Dataset workflow and publish summaries."""

from __future__ import annotations

from pathlib import Path

from text_to_sign_production.workflows._dataset.publish import verify_dataset_publish_plan
from text_to_sign_production.workflows._dataset.types import (
    DatasetPublishPlan,
    DatasetPublishSummary,
    DatasetQualityReadinessSummary,
    DatasetWorkflowOutputSummary,
    DatasetWorkflowResult,
)
from text_to_sign_production.workflows._dataset.verify import verify_dataset_outputs


def summarize_dataset_workflow_outputs(
    result: DatasetWorkflowResult,
) -> DatasetWorkflowOutputSummary:
    """Verify and summarize completed Dataset workflow outputs for notebook display."""

    (
        processed_manifest_paths,
        processed_sample_archive_member_counts,
        interim_report_paths,
        processed_report_paths,
    ) = verify_dataset_outputs(result)

    validation_error_count = sum(
        1
        for issues in result.validation_issues.values()
        for issue in issues
        if issue.severity == "error"
    )
    validation_warning_count = sum(
        1
        for issues in result.validation_issues.values()
        for issue in issues
        if issue.severity == "warning"
    )
    return DatasetWorkflowOutputSummary(
        splits=result.splits,
        processed_manifest_paths=processed_manifest_paths,
        processed_sample_archive_member_counts=processed_sample_archive_member_counts,
        validation_error_count=validation_error_count,
        validation_warning_count=validation_warning_count,
        validation_issue_counts=_validation_issue_counts(result),
        raw_source_issue_counts=_raw_source_issue_counts(result),
        dropped_sample_counts=_dropped_sample_counts(result),
        top_drop_reasons=_top_drop_reasons(result),
        report_paths=_combined_report_paths(
            interim_report_paths=interim_report_paths,
            processed_report_paths=processed_report_paths,
        ),
        processed_report_paths=processed_report_paths,
        interim_report_paths=interim_report_paths,
        quality_report_paths=result.quality_report_paths,
        leakage_report_paths=result.leakage_report_paths,
        tier_manifest_paths=result.tier_manifest_paths,
        tier_counts=result.tier_counts,
        blocking_leakage=result.blocking_leakage,
        quality_config_hash=result.quality_config_hash,
    )


def summarize_dataset_publish_plan(plan: DatasetPublishPlan) -> DatasetPublishSummary:
    """Verify and summarize Dataset publish outputs for notebook display."""

    return verify_dataset_publish_plan(plan)


def _validation_issue_counts(result: DatasetWorkflowResult) -> dict[str, int]:
    return {stage_key: len(issues) for stage_key, issues in result.validation_issues.items()}


def _raw_source_issue_counts(result: DatasetWorkflowResult) -> dict[str, int]:
    splits_report = dict(result.assumption_report.get("splits", {}))
    counts: dict[str, int] = {}
    for split in result.splits:
        split_report = dict(splits_report.get(split, {}))
        video_metadata = dict(split_report.get("video_metadata", {}))
        openpose_schema = dict(split_report.get("openpose_schema", {}))
        deviation_counts = dict(openpose_schema.get("deviation_counts", {}))
        counts[split] = (
            int(split_report.get("unmatched_sample_count", 0) or 0)
            + int(video_metadata.get("unreadable_count", 0) or 0)
            + sum(int(value) for value in deviation_counts.values())
        )
    return counts


def _dropped_sample_counts(result: DatasetWorkflowResult) -> dict[str, int]:
    splits_report = dict(result.filter_report.get("splits", {}))
    counts: dict[str, int] = {}
    for split in result.splits:
        split_report = dict(splits_report.get(split, {}))
        counts[split] = int(split_report.get("dropped_samples", 0) or 0)
    return counts


def _top_drop_reasons(
    result: DatasetWorkflowResult,
    *,
    limit: int = 3,
) -> dict[str, tuple[tuple[str, int], ...]]:
    splits_report = dict(result.filter_report.get("splits", {}))
    reasons_by_split: dict[str, tuple[tuple[str, int], ...]] = {}
    for split in result.splits:
        split_report = dict(splits_report.get(split, {}))
        counts = {
            str(reason): int(count)
            for reason, count in dict(split_report.get("drop_reason_counts", {})).items()
        }
        reasons_by_split[split] = tuple(
            sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
        )
    return reasons_by_split


def _combined_report_paths(
    *,
    interim_report_paths: dict[str, Path],
    processed_report_paths: dict[str, Path],
) -> dict[str, Path]:
    report_paths: dict[str, Path] = {}
    report_paths.update({f"interim_{label}": path for label, path in interim_report_paths.items()})
    report_paths.update(
        {f"processed_{label}": path for label, path in processed_report_paths.items()}
    )
    return report_paths


def build_dataset_quality_readiness_summary(
    summary: DatasetWorkflowOutputSummary,
) -> DatasetQualityReadinessSummary:
    """Build operator-facing quality readiness summary for notebook display.

    The notebook prints this to confirm all quality, leakage, and tier
    artifacts are present before proceeding to Base runs.
    """

    # Quality config path/hash
    quality_config_path: str | None = None
    quality_config_hash: str | None = summary.quality_config_hash
    # Derive from quality report paths
    # The config path is not directly on the summary, but the hash is.
    # We expose it through the workflow config if available.

    # Quality report paths
    quality_reports: dict[str, str] = {k: str(v) for k, v in summary.quality_report_paths.items()}

    # Leakage report paths
    leakage_reports: dict[str, str] = {k: str(v) for k, v in summary.leakage_report_paths.items()}

    # Tier manifest paths
    tier_manifests: dict[str, dict[str, str]] = {}
    for split_or_key, tier_paths in summary.tier_manifest_paths.items():
        tier_manifests[split_or_key] = {tier: str(path) for tier, path in tier_paths.items()}

    # Tier counts
    tier_counts: dict[str, dict[str, int]] = {
        split: dict(counts) for split, counts in summary.tier_counts.items()
    }

    # Complete-mode dataset gate
    has_blocking = summary.blocking_leakage
    has_quality_train = False
    train_counts = tier_counts.get("train", {})
    if train_counts.get("quality", 0) > 0:
        has_quality_train = True

    gate_pass = (not has_blocking) and has_quality_train
    if gate_pass:
        gate_status = "PASS"
    else:
        reasons = []
        if has_blocking:
            reasons.append("blocking leakage detected")
        if not has_quality_train:
            reasons.append("quality tier train count is 0")
        gate_status = f"FAIL ({'; '.join(reasons)})"

    return DatasetQualityReadinessSummary(
        quality_config_path=quality_config_path,
        quality_config_hash=quality_config_hash,
        quality_report_paths=quality_reports,
        leakage_report_paths=leakage_reports,
        blocking_for_complete=has_blocking,
        tier_manifest_paths=tier_manifests,
        tier_counts=tier_counts,
        complete_dataset_gate=gate_status,
        complete_dataset_gate_pass=gate_pass,
    )


__all__ = [
    "build_dataset_quality_readiness_summary",
    "summarize_dataset_publish_plan",
    "summarize_dataset_workflow_outputs",
]
