"""Stage-level orchestration for the Dataset Build workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from text_to_sign_production.data.constants import REPO_ROOT, SPLITS
from text_to_sign_production.data.filtering import filter_all_splits
from text_to_sign_production.data.manifests import export_final_manifests
from text_to_sign_production.data.normalize import normalize_all_splits
from text_to_sign_production.data.raw import build_raw_manifests
from text_to_sign_production.ops.colab_workflow import (
    package_dataset_build_outputs,
    publish_colab_outputs,
    stage_colab_inputs,
)

DatasetBuildInputMode = Literal["existing_raw", "fixed_colab_drive"]
DatasetBuildOutputMode = Literal["none", "local_archives", "fixed_colab_drive"]

DEFAULT_FILTER_CONFIG_PATH = REPO_ROOT / "configs" / "data" / "filter-v1.yaml"


@dataclass(frozen=True, slots=True)
class DatasetBuildResult:
    """Summary of one Dataset Build run."""

    splits: tuple[str, ...]
    staged_inputs: tuple[dict[str, str], ...]
    assumption_report: dict[str, Any]
    filter_report: dict[str, Any]
    export_report: dict[str, Any]
    output_paths: tuple[Path, ...]


def run_dataset_build(
    *,
    splits: tuple[str, ...] = SPLITS,
    filter_config_path: Path = DEFAULT_FILTER_CONFIG_PATH,
    input_mode: DatasetBuildInputMode = "existing_raw",
    output_mode: DatasetBuildOutputMode = "local_archives",
) -> DatasetBuildResult:
    """Run the full Dataset Build stage using existing reusable pipeline functions."""

    requested_splits = _validate_splits(splits)
    staged_inputs: tuple[dict[str, str], ...] = ()

    if input_mode == "fixed_colab_drive":
        staged_inputs = tuple(stage_colab_inputs(splits=requested_splits))
    elif input_mode != "existing_raw":
        raise ValueError(f"Unsupported Dataset Build input mode: {input_mode!r}")

    assumption_report = build_raw_manifests(splits=requested_splits)
    normalize_all_splits(splits=requested_splits)
    filter_report = filter_all_splits(
        filter_config_path,
        splits=requested_splits,
    )
    export_report = export_final_manifests(
        assumption_report=assumption_report,
        filter_report=filter_report,
        splits=requested_splits,
    )

    if output_mode == "none":
        output_paths: tuple[Path, ...] = ()
    elif output_mode == "local_archives":
        output_paths = tuple(package_dataset_build_outputs(splits=requested_splits))
    elif output_mode == "fixed_colab_drive":
        output_paths = tuple(publish_colab_outputs(splits=requested_splits))
    else:
        raise ValueError(f"Unsupported Dataset Build output mode: {output_mode!r}")

    return DatasetBuildResult(
        splits=requested_splits,
        staged_inputs=staged_inputs,
        assumption_report=assumption_report,
        filter_report=filter_report,
        export_report=export_report,
        output_paths=output_paths,
    )


def _validate_splits(splits: tuple[str, ...]) -> tuple[str, ...]:
    if not splits:
        raise ValueError("At least one split must be requested for Dataset Build.")
    unsupported_splits = [split for split in splits if split not in SPLITS]
    if unsupported_splits:
        expected = ", ".join(SPLITS)
        observed = ", ".join(unsupported_splits)
        raise ValueError(f"Unsupported Dataset Build split(s): {observed}; expected: {expected}")
    seen_splits: set[str] = set()
    duplicate_splits: list[str] = []
    for split in splits:
        if split in seen_splits and split not in duplicate_splits:
            duplicate_splits.append(split)
        seen_splits.add(split)
    if duplicate_splits:
        observed = ", ".join(duplicate_splits)
        raise ValueError(f"Duplicate Dataset Build split(s): {observed}")
    return tuple(splits)
