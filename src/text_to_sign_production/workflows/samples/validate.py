"""Static input validation for the samples workflow."""

from __future__ import annotations

from text_to_sign_production.data._shared.identities import SampleSplit
from text_to_sign_production.data.gates.config import load_gates_config
from text_to_sign_production.data.pose.types import PersonSelectionPolicy
from text_to_sign_production.workflows.samples.types import (
    SamplesWorkflowConfig,
    SamplesWorkflowInputError,
)


def validate_samples_inputs(config: SamplesWorkflowConfig) -> None:
    errors: list[str] = []

    if not config.project_root.is_absolute():
        errors.append(f"project_root must be absolute: {config.project_root}")
    if not config.drive_project_root.is_absolute():
        errors.append(f"drive_project_root must be absolute: {config.drive_project_root}")
    if not config.splits:
        errors.append("splits must be non-empty")

    parsed_splits: list[SampleSplit] = []
    for split in config.splits:
        try:
            parsed_splits.append(SampleSplit(split))
        except ValueError:
            errors.append(f"invalid split: {split!r}")

    if len(set(parsed_splits)) != len(parsed_splits):
        errors.append("splits must contain unique values")

    if config.gates_config_relative_path.is_absolute():
        errors.append(
            "gates_config_relative_path must remain relative: "
            f"{config.gates_config_relative_path}"
        )

    try:
        PersonSelectionPolicy(config.person_selection_policy)
    except ValueError:
        errors.append(
            "person_selection_policy must be a valid PersonSelectionPolicy: "
            f"{config.person_selection_policy!r}"
        )

    if errors:
        raise SamplesWorkflowInputError("; ".join(errors))

    try:
        load_gates_config(config.project_root / config.gates_config_relative_path)
    except Exception as exc:
        raise SamplesWorkflowInputError(f"gates config is invalid: {exc}") from exc


__all__ = ["validate_samples_inputs"]
