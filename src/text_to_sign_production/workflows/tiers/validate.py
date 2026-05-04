"""Static input validation for the tiers workflow."""

from __future__ import annotations

from text_to_sign_production.data._shared.identities import SampleSplit
from text_to_sign_production.data.tiers.filters import load_filter_config
from text_to_sign_production.data.tiers.policies import load_tier_policies
from text_to_sign_production.workflows.tiers.types import (
    TiersWorkflowConfig,
    TiersWorkflowInputError,
)


def validate_tiers_inputs(config: TiersWorkflowConfig) -> None:
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

    if config.filters_config_relative_path.is_absolute():
        errors.append(
            "filters_config_relative_path must remain relative: "
            f"{config.filters_config_relative_path}"
        )
    if config.tiers_config_relative_path.is_absolute():
        errors.append(
            "tiers_config_relative_path must remain relative: "
            f"{config.tiers_config_relative_path}"
        )

    if errors:
        raise TiersWorkflowInputError("; ".join(errors))

    try:
        load_filter_config(config.project_root / config.filters_config_relative_path)
    except Exception as exc:
        raise TiersWorkflowInputError(f"filters config is invalid: {exc}") from exc

    try:
        load_tier_policies(config.project_root / config.tiers_config_relative_path)
    except Exception as exc:
        raise TiersWorkflowInputError(f"tiers config is invalid: {exc}") from exc


__all__ = ["validate_tiers_inputs"]
