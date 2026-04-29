"""Dataset workflow output verification helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from text_to_sign_production.core.files import verify_output_file
from text_to_sign_production.workflows._dataset.types import (
    DatasetWorkflowError,
    DatasetWorkflowResult,
)


def verify_dataset_outputs(
    result: DatasetWorkflowResult,
) -> tuple[dict[str, Path], dict[str, int], dict[str, Path], dict[str, Path]]:
    """Verify completed Dataset workflow artifacts and return summary inputs."""

    processed_manifest_paths: dict[str, Path] = {}
    processed_sample_archive_member_counts: dict[str, int] = {}

    try:
        for split in result.splits:
            try:
                manifest_path = result.processed_manifest_paths[split]
            except KeyError as exc:
                raise DatasetWorkflowError(
                    f"Processed manifest path is missing from workflow result for {split}."
                ) from exc
            processed_manifest_paths[split] = verify_output_file(
                manifest_path,
                label=f"Processed manifest {split}",
            )

            try:
                sample_members = result.processed_sample_archive_members[split]
            except KeyError as exc:
                raise DatasetWorkflowError(
                    "Processed sample archive members are missing from workflow result "
                    f"for {split}."
                ) from exc
            if not sample_members:
                raise DatasetWorkflowError(
                    f"No final processed sample archive members were produced for {split}."
                )
            processed_sample_archive_member_counts[split] = len(sample_members)

        interim_report_paths = _verified_report_paths(
            result.interim_report_paths,
            label="Interim report",
        )
        processed_report_paths = _verified_report_paths(
            result.processed_report_paths,
            label="Processed report",
        )
    except (FileNotFoundError, IsADirectoryError, ValueError) as exc:
        raise DatasetWorkflowError(str(exc)) from exc

    return (
        processed_manifest_paths,
        processed_sample_archive_member_counts,
        interim_report_paths,
        processed_report_paths,
    )


def _verified_report_paths(paths: Mapping[str, Path], *, label: str) -> dict[str, Path]:
    return {key: verify_output_file(path, label=f"{label} {key}") for key, path in paths.items()}


__all__: list[str] = []
