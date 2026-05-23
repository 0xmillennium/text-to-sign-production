from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from text_to_sign_production.workflows.model.contracts import (
    ModelWorkflowInvariantError,
)


@dataclass(frozen=True, slots=True)
class CalibrationReaderRuntimeOptions:
    num_workers: int
    prefetch_factor: int | None
    persistent_workers: bool


def calibration_reader_runtime_options(
    effective_config: Mapping[str, object],
) -> CalibrationReaderRuntimeOptions:
    application = effective_config.get("compute_profile_application")
    if not isinstance(application, Mapping):
        return CalibrationReaderRuntimeOptions(
            num_workers=0,
            prefetch_factor=None,
            persistent_workers=False,
        )

    dataloader_applied = application.get("dataloader_applied")
    if not isinstance(dataloader_applied, Mapping):
        return CalibrationReaderRuntimeOptions(
            num_workers=0,
            prefetch_factor=None,
            persistent_workers=False,
        )

    num_workers = dataloader_applied.get("num_workers", 0)
    if (
        not isinstance(num_workers, int)
        or isinstance(num_workers, bool)
        or num_workers < 0
    ):
        raise ModelWorkflowInvariantError(
            "calibration dataloader_applied.num_workers must be a non-negative integer."
        )

    prefetch_factor = dataloader_applied.get("prefetch_factor")
    if prefetch_factor is not None and (
        not isinstance(prefetch_factor, int)
        or isinstance(prefetch_factor, bool)
        or prefetch_factor <= 0
    ):
        raise ModelWorkflowInvariantError(
            "calibration dataloader_applied.prefetch_factor must be a positive integer or null."
        )

    persistent_workers = dataloader_applied.get("persistent_workers", False)
    if not isinstance(persistent_workers, bool):
        raise ModelWorkflowInvariantError(
            "calibration dataloader_applied.persistent_workers must be boolean."
        )

    return CalibrationReaderRuntimeOptions(
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )
