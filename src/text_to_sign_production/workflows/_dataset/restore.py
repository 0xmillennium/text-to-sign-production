"""Dataset workflow runtime input restore verification."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from text_to_sign_production.core.files import (
    require_dir,
    require_dir_contains,
    verify_output_file,
)
from text_to_sign_production.workflows._dataset.types import (
    DatasetFileTransferSpec,
    DatasetRawArchiveRestoreSpec,
    DatasetRuntimePlan,
    DatasetWorkflowError,
)


def verify_dataset_runtime_inputs(plan: DatasetRuntimePlan) -> None:
    """Verify notebook-executed Dataset runtime restores for all planned inputs."""

    for spec in plan.raw_translation_specs.values():
        _verify_transfer_file(spec)
    _verify_raw_archive_restore_specs(plan.raw_bfh_archive_specs)


def _verify_transfer_file(spec: DatasetFileTransferSpec) -> Path:
    target_path = verify_output_file(spec.target_path, label=spec.label)
    observed_size = target_path.stat().st_size
    if observed_size != spec.expected_input_bytes:
        raise DatasetWorkflowError(
            f"{spec.label} byte count mismatch: expected "
            f"{spec.expected_input_bytes}, observed {observed_size}: {target_path}"
        )
    return target_path


def _verify_raw_archive_restore_specs(
    specs: Mapping[str, DatasetRawArchiveRestoreSpec],
) -> None:
    for split, spec in specs.items():
        require_dir(spec.expected_split_root, label=f"Runtime {split} raw BFH split root")
        require_dir(spec.expected_openpose_root, label=f"Runtime {split} openpose_output root")
        require_dir_contains(
            spec.expected_json_root,
            "*",
            label=f"Runtime {split} keypoint JSON root",
        )
        require_dir_contains(
            spec.expected_video_root,
            "*.mp4",
            label=f"Runtime {split} source video root",
        )


__all__ = ["verify_dataset_runtime_inputs"]
