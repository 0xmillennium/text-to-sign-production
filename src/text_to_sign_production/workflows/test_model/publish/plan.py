"""Build digest-bearing publish plans for test_model outputs."""

from __future__ import annotations

from pathlib import Path

from text_to_sign_production.core.integrity import sha256_file
from text_to_sign_production.workflows.foundation.execution import (
    FileCopyOperation,
    OperationProgressSpec,
)
from text_to_sign_production.workflows.test_model.constants import TEST_MODEL_STAGE_PUBLISH
from text_to_sign_production.workflows.test_model.contracts import (
    TestModelInferenceResult,
    TestModelIssue,
    TestModelPublishPlan,
    TestModelPublishTarget,
    TestModelReportResult,
    TestModelVisualizationResult,
)
from text_to_sign_production.workflows.test_model.layout import (
    TestModelLayout,
    drive_test_model_sample_run_root,
)
from text_to_sign_production.workflows.test_model.progress import test_model_progress_stage


def build_test_model_publish_plan(
    layout: TestModelLayout,
    *,
    report_result: TestModelReportResult,
    inference_result: TestModelInferenceResult,
    visualization_result: TestModelVisualizationResult,
) -> TestModelPublishPlan:
    runtime_output_root = report_result.output_root
    blocking_errors: list[TestModelIssue] = []
    warnings: list[TestModelIssue] = []
    relative_root: Path | None = None
    try:
        relative_root = runtime_output_root.relative_to(
            layout.config.runtime_root / "reports" / "test_model"
        )
    except ValueError:
        blocking_errors.append(
            TestModelIssue(
                "publish_root_outside_runtime",
                f"runtime output root is outside test_model runtime reports: {runtime_output_root}",
                path=runtime_output_root,
            )
        )
    if relative_root is None:
        model_run_name = target_sentence_slug = execution_id = "invalid"
    elif len(relative_root.parts) != 3:
        blocking_errors.append(
            TestModelIssue(
                "publish_root_invalid",
                "runtime output root must have model/sample/execution shape",
                path=runtime_output_root,
            )
        )
        model_run_name = target_sentence_slug = execution_id = "invalid"
    else:
        model_run_name, target_sentence_slug, execution_id = relative_root.parts
    drive_output_root = drive_test_model_sample_run_root(
        layout,
        model_run_name,
        target_sentence_slug,
        execution_id,
    )
    raw_sources = (
        *report_result.files,
        *(receipt.path for receipt in inference_result.receipts),
        *(receipt.path for receipt in visualization_result.receipts),
    )
    targets: list[TestModelPublishTarget] = []
    seen_targets: set[Path] = set()
    for source in raw_sources:
        if source.is_dir():
            warnings.append(
                TestModelIssue(
                    "publish_directory_skipped",
                    f"directory source is not published as a file: {source}",
                    path=source,
                )
            )
            continue
        if not source.is_file():
            warnings.append(
                TestModelIssue(
                    "publish_missing_source_skipped",
                    f"runtime output source is missing and was not published: {source}",
                    path=source,
                )
            )
            continue
        try:
            relative = source.resolve(strict=False).relative_to(
                runtime_output_root.resolve(strict=False)
            )
        except ValueError:
            blocking_errors.append(
                TestModelIssue(
                    "publish_source_outside_runtime_output_root",
                    f"publish source is outside runtime output root: {source}",
                    path=source,
                )
            )
            continue
        target_path = drive_output_root / relative
        resolved_target = target_path.resolve(strict=False)
        if resolved_target in seen_targets:
            blocking_errors.append(
                TestModelIssue(
                    "duplicate_publish_target",
                    f"multiple runtime outputs map to publish target: {target_path}",
                    path=target_path,
                )
            )
            continue
        seen_targets.add(resolved_target)
        targets.append(
            TestModelPublishTarget(
                label=f"publish test_model {relative.as_posix()}",
                source_path=source,
                target_path=target_path,
                source_sha256=sha256_file(source),
            )
        )
    if not targets:
        warnings.append(TestModelIssue("publish_no_sources", "no runtime output files to publish"))
    operations = tuple(_copy_operation(target) for target in targets)
    return TestModelPublishPlan(
        runtime_output_root=runtime_output_root,
        drive_output_root=drive_output_root,
        targets=tuple(targets),
        operations=operations,
        blocking_errors=tuple(blocking_errors),
        warnings=tuple(warnings),
    )


def _copy_operation(target: TestModelPublishTarget) -> FileCopyOperation:
    expected_bytes = target.source_path.stat().st_size if target.source_path.is_file() else None
    return FileCopyOperation(
        label=target.label,
        source_path=target.source_path,
        target_path=target.target_path,
        failure_message=f"Failed to {target.label}",
        overwrite_policy="atomic_replace",
        expected_input_bytes=expected_bytes,
        progress=OperationProgressSpec(
            stage=test_model_progress_stage(
                stage_id=TEST_MODEL_STAGE_PUBLISH,
                label="Publish test_model outputs",
                unit="bytes",
                owner_module=__name__,
                operation_kind="publish",
                total_semantics="publish input bytes when known",
                bar_eligible=False,
            ),
            expected_total=expected_bytes,
            live_owner="shell",
        ),
    )


__all__ = ["build_test_model_publish_plan"]
