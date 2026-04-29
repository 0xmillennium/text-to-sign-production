"""Base workflow runtime input restore verification."""

from __future__ import annotations

from pathlib import Path

from text_to_sign_production.core.files import require_dir, require_dir_contains, verify_output_file
from text_to_sign_production.core.paths import resolve_manifest_path
from text_to_sign_production.workflows._base.types import (
    BaseProcessedManifestRestoreSpec,
    BaseProcessedRestoreVerificationSummary,
    BaseProcessedSampleArchiveRestoreSpec,
    BaseProcessedSplitInputVerification,
    BaseRuntimePlan,
    BaseWorkflowError,
)
from text_to_sign_production.workflows._shared.metadata import (
    count_jsonl_records,
    iter_jsonl_records,
)


def verify_base_processed_manifest_restore(
    spec: BaseProcessedManifestRestoreSpec,
) -> Path:
    """Verify a notebook-executed direct processed manifest restore."""

    target_path = verify_output_file(spec.target_path, label=spec.label)
    observed_size = target_path.stat().st_size
    if observed_size != spec.expected_input_bytes:
        raise BaseWorkflowError(
            f"{spec.label} byte count mismatch: expected "
            f"{spec.expected_input_bytes}, observed {observed_size}: {target_path}"
        )
    return target_path


def verify_base_processed_sample_restore(
    spec: BaseProcessedSampleArchiveRestoreSpec,
    *,
    manifest_sample_check_limit: int = 5,
) -> Path:
    """Verify a notebook-executed processed sample archive restore."""

    split_root = require_dir(spec.expected_split_root, label=f"{spec.split} processed samples")
    require_dir_contains(split_root, "*.npz", label=f"{spec.split} processed samples")
    if spec.target_manifest_path.is_file():
        _verify_manifest_sample_subset(
            spec.target_manifest_path,
            split=spec.split,
            data_root=spec.data_root,
            limit=manifest_sample_check_limit,
        )
    return split_root


def verify_base_processed_restore(plan: BaseRuntimePlan) -> BaseProcessedRestoreVerificationSummary:
    """Verify all processed restore artifacts in a Base runtime plan."""

    return verify_base_runtime_inputs(plan)


def verify_base_runtime_inputs(
    plan: BaseRuntimePlan,
    *,
    manifest_sample_check_limit: int = 5,
) -> BaseProcessedRestoreVerificationSummary:
    """Verify restored Base processed inputs and return a notebook-safe summary."""

    split_inputs: dict[str, BaseProcessedSplitInputVerification] = {}
    for split in plan.required_splits:
        try:
            manifest_spec = plan.processed_manifest_files[split]
            archive_spec = plan.processed_sample_archives[split]
        except KeyError as exc:
            raise BaseWorkflowError(
                f"Base runtime plan is missing restore specs for required split {split!r}."
            ) from exc

        manifest_path = verify_base_processed_manifest_restore(manifest_spec)
        manifest_record_count = count_jsonl_records(manifest_path)
        if manifest_record_count <= 0:
            raise BaseWorkflowError(f"Processed {split} manifest is empty: {manifest_path}")

        samples_dir = require_dir(
            archive_spec.expected_split_root,
            label=f"{split} processed samples",
        )
        require_dir_contains(samples_dir, "*.npz", label=f"{split} processed samples")
        sample_count = sum(1 for _path in samples_dir.glob("*.npz"))
        checked_manifest_sample_count = _verify_manifest_sample_subset(
            manifest_path,
            split=split,
            data_root=archive_spec.data_root,
            limit=manifest_sample_check_limit,
        )
        split_inputs[split] = BaseProcessedSplitInputVerification(
            split=split,
            manifest_path=manifest_path,
            manifest_record_count=manifest_record_count,
            samples_dir=samples_dir,
            sample_count=sample_count,
            checked_manifest_sample_count=checked_manifest_sample_count,
        )

    return BaseProcessedRestoreVerificationSummary(
        required_splits=plan.required_splits,
        split_inputs=split_inputs,
    )


def _verify_manifest_sample_subset(
    manifest_path: Path,
    *,
    split: str,
    data_root: Path,
    limit: int,
) -> int:
    checked = 0
    for record in iter_jsonl_records(manifest_path):
        if checked >= limit:
            break
        if record.get("split") != split:
            raise BaseWorkflowError(
                f"Processed manifest subset has wrong split in {manifest_path}: {record}"
            )
        sample_path = record.get("sample_path")
        if not isinstance(sample_path, str):
            raise BaseWorkflowError(
                f"Processed manifest row is missing sample_path in {manifest_path}: {record}"
            )
        resolved_sample_path = resolve_manifest_path(sample_path, data_root=data_root)
        if not resolved_sample_path.is_file():
            raise BaseWorkflowError(
                f"Processed sample referenced by {manifest_path} is missing: {sample_path}"
            )
        checked += 1
    if checked <= 0:
        raise BaseWorkflowError(f"Processed manifest has no sample rows: {manifest_path}")
    return checked


__all__ = ["verify_base_runtime_inputs"]
