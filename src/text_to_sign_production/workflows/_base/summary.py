"""Notebook-facing Base workflow summary projections."""

from __future__ import annotations

from pathlib import Path

from text_to_sign_production.workflows._base.constants import (
    BASE_RUN_MODE_POLICIES,
)
from text_to_sign_production.workflows._base.publish import (
    _base_file_publish_spec,
    verify_base_checkpoint_publish,
    verify_base_file_publish,
    verify_base_prediction_sample_archive_file,
    verify_base_prediction_sample_archive_members,
    verify_base_qualitative_sample_archive_file,
    verify_base_qualitative_sample_archive_members,
)
from text_to_sign_production.workflows._base.types import (
    BasePredictionSampleArchiveSpec,
    BasePublishedArchiveSummary,
    BasePublishPlan,
    BasePublishSummary,
    BaseQualitativeSampleArchiveSpec,
    BaseRuntimePlan,
    BaseRuntimeReadinessSummary,
    BaseWorkflowError,
    BaseWorkflowOutputSummary,
    BaseWorkflowResult,
)
from text_to_sign_production.workflows._shared.metadata import verify_portable_json_file


def summarize_base_workflow_outputs(result: BaseWorkflowResult) -> BaseWorkflowOutputSummary:
    """Return a notebook-safe summary of verified Base workflow outputs."""

    verification = result.verification
    return BaseWorkflowOutputSummary(
        run_root=verification.run_root,
        config_snapshot_path=verification.config_snapshot_path,
        baseline_config_copy_path=verification.baseline_config_copy_path,
        training_metrics_path=verification.training_metrics_path,
        training_live_log_path=verification.training_live_log_path,
        training_metric_record_count=verification.training_metric_record_count,
        training_summary_path=verification.training_summary_path,
        best_checkpoint_path=verification.best_checkpoint_path,
        last_checkpoint_path=verification.last_checkpoint_path,
        prediction_splits=verification.prediction_splits,
        baseline_report_json_path=verification.baseline_report_json_path,
        baseline_report_markdown_path=verification.baseline_report_markdown_path,
        failure_modes_json_path=verification.failure_modes_json_path,
        failure_modes_markdown_path=verification.failure_modes_markdown_path,
        qualitative=verification.qualitative,
        run_summary_path=verification.run_summary_path,
        verification_status="verified",
    )


def build_base_runtime_readiness_summary(
    plan: BaseRuntimePlan,
) -> BaseRuntimeReadinessSummary:
    """Build operator-facing readiness summary from a runtime plan.

    The notebook prints this before training to confirm all readiness
    surfaces are visible.  No report parsing or policy computation
    happens in the notebook.
    """

    run_mode = plan.run_mode
    policy = BASE_RUN_MODE_POLICIES.get(run_mode)

    # Recovery
    recovery_enabled = plan.periodic_persistence
    recovery_root: str | None = None
    recovery_manifest_path: str | None = None
    if recovery_enabled and plan.workflow_config.drive_project_root is not None:
        from text_to_sign_production.core.paths import ProjectLayout

        drive_layout = ProjectLayout(Path(plan.workflow_config.drive_project_root))
        rr = drive_layout.base_m0_run_root(plan.run_name)
        recovery_root = str(rr)
        recovery_manifest_path = str(rr / "recovery_manifest.json")

    # Quality tier config
    quality_config_path = plan.run_layout.project_root / "configs" / "data" / "quality-tier-v1.yaml"
    quality_config_path_str: str | None = (
        str(quality_config_path) if quality_config_path.is_file() else None
    )
    quality_config_hash: str | None = None
    if quality_config_path.is_file():
        from text_to_sign_production.core.files import sha256_file

        quality_config_hash = sha256_file(quality_config_path)

    # Tier counts (from runtime plan processed manifest files)
    tier_counts = _collect_tier_counts(plan)

    # Target standardization
    ts_enabled = run_mode == "complete"
    ts_fit_surface = "quality train tier" if ts_enabled else "disabled"
    ts_metadata_path: str | None = (
        "will be produced after target standardization fit" if ts_enabled else None
    )

    # Duration estimate
    duration_estimate = "unknown until first epoch completes"

    # Warnings
    complete_warning: str | None = None
    check_warning: str | None = None
    if run_mode == "complete" and policy is not None:
        complete_warning = policy.run_mode_statement
    elif run_mode == "check" and policy is not None:
        check_warning = policy.run_mode_statement

    return BaseRuntimeReadinessSummary(
        run_mode=run_mode,
        run_name=plan.run_name,
        live_log_path=str(plan.live_log_path),
        recovery_persistence_enabled=recovery_enabled,
        recovery_root=recovery_root,
        recovery_manifest_path=recovery_manifest_path,
        training_surface=plan.training_surface,
        validation_surface=plan.validation_surface,
        quality_tier_config_path=quality_config_path_str,
        quality_tier_config_hash=quality_config_hash,
        tier_counts=tier_counts if tier_counts else None,
        primary_metric_name="val_masked_l2",
        target_standardization_enabled=ts_enabled,
        target_standardization_fit_surface=ts_fit_surface,
        target_standardization_leakage_policy="train-only fit",
        target_standardization_metadata_path=ts_metadata_path,
        target_standardization_inverse_prediction=ts_enabled,
        max_epochs=plan.epoch_count,
        min_epochs=plan.min_epochs,
        early_stopping_patience=plan.early_stopping_patience,
        progress_expectation=(
            f"{plan.epoch_count} epochs max, "
            f"{plan.min_epochs} min, "
            f"patience {plan.early_stopping_patience}"
        ),
        output_policy=str(plan.workflow_config.output_policy),
        resume_supported=plan.resume_supported,
        duration_estimate=duration_estimate,
        complete_mode_warning=complete_warning,
        check_mode_warning=check_warning,
        risk_control_reminders=(
            "M0 is a comparison floor.",
            "M0 is not a contribution-strength claim.",
            "M0 is not a sign-intelligibility claim.",
        ),
    )


def _collect_tier_counts(
    plan: BaseRuntimePlan,
) -> dict[str, dict[str, int]] | None:
    """Read tier counts from dataset artifact layout if available."""

    from text_to_sign_production.artifacts.layout import DatasetArtifactLayout
    from text_to_sign_production.core.paths import ProjectLayout
    from text_to_sign_production.data.jsonl import iter_jsonl

    layout = ProjectLayout(plan.project_root)
    artifact_layout = DatasetArtifactLayout(layout)
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        split_counts: dict[str, int] = {}
        for tier in ("broad", "quality", "audit_low_quality", "dropped"):
            path = artifact_layout.processed_tier_manifest_path(tier, split)
            if path.is_file():
                split_counts[tier] = sum(1 for _ in iter_jsonl(path))
        if split_counts:
            counts[split] = split_counts
    return counts if counts else None


def summarize_base_publish_plan(plan: BasePublishPlan) -> BasePublishSummary:
    """Verify and summarize a notebook-executed Base publish plan."""

    direct_files = tuple(verify_base_file_publish(spec) for spec in plan.direct_file_specs)
    checkpoints = tuple(verify_base_checkpoint_publish(spec) for spec in plan.checkpoint_specs)
    checkpoint_manifest = verify_base_file_publish(
        _base_file_publish_spec(
            label="checkpoint manifest",
            group="checkpoints",
            split=None,
            source_path=plan.checkpoint_manifest_source_path,
            target_path=plan.checkpoint_manifest_target_path,
        )
    )
    _verify_portable_json_file(
        checkpoint_manifest.target_path,
        project_root=plan.drive_project_root,
    )
    prediction_archives = tuple(
        _summarize_prediction_archive(spec) for spec in plan.prediction_sample_archive_specs
    )
    qualitative_archive = (
        None
        if plan.qualitative_sample_archive_spec is None
        else _summarize_qualitative_archive(plan.qualitative_sample_archive_spec)
    )
    return BasePublishSummary(
        direct_files=direct_files,
        checkpoints=checkpoints,
        checkpoint_manifest=checkpoint_manifest,
        prediction_archives=prediction_archives,
        qualitative_archive=qualitative_archive,
        missing_count=0,
        invalid_count=0,
        verification_status="verified",
    )


def _summarize_prediction_archive(
    spec: BasePredictionSampleArchiveSpec,
) -> BasePublishedArchiveSummary:
    archive_path = verify_base_prediction_sample_archive_file(spec)
    observed_members = verify_base_prediction_sample_archive_members(spec)
    return BasePublishedArchiveSummary(
        label=f"{spec.split} prediction samples",
        target_archive_path=archive_path,
        archive_size=archive_path.stat().st_size,
        observed_member_list_path=spec.observed_member_list_path,
        expected_member_count=spec.expected_member_count,
        observed_member_count=len(observed_members),
    )


def _summarize_qualitative_archive(
    spec: BaseQualitativeSampleArchiveSpec,
) -> BasePublishedArchiveSummary:
    archive_path = verify_base_qualitative_sample_archive_file(spec)
    observed_members = verify_base_qualitative_sample_archive_members(spec)
    return BasePublishedArchiveSummary(
        label="qualitative samples",
        target_archive_path=archive_path,
        archive_size=archive_path.stat().st_size,
        observed_member_list_path=spec.observed_member_list_path,
        expected_member_count=spec.expected_member_count,
        observed_member_count=len(observed_members),
    )


def _verify_portable_json_file(path: Path, *, project_root: Path) -> None:
    verify_portable_json_file(path, project_root=project_root, error_factory=BaseWorkflowError)


__all__ = [
    "build_base_runtime_readiness_summary",
    "summarize_base_publish_plan",
    "summarize_base_workflow_outputs",
]
