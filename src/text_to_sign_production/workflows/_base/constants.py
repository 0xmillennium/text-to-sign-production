"""Constants for the private Base workflow implementation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

BaseRunMode = Literal["smoke", "check", "complete"]


@dataclass(frozen=True, slots=True)
class BaseRunModePolicy:
    """Code-owned runtime policy for one operator-selected Base run mode."""

    run_mode: BaseRunMode
    purpose: str
    claim: str
    run_mode_statement: str
    limit_train_samples: int | None
    limit_validation_samples: int | None
    limit_prediction_samples: int | None
    epoch_count: int
    min_epochs: int
    early_stopping_patience: int
    shuffle_train: bool
    run_qualitative_export: bool
    panel_size: int
    prediction_splits: tuple[str, ...]


BASELINE_REPORT_JSON_FILENAME = "baseline-report.json"


BASELINE_REPORT_MARKDOWN_FILENAME = "baseline-report.md"


FAILURE_MODES_JSON_FILENAME = "failure-modes.json"


FAILURE_MODES_MARKDOWN_FILENAME = "failure-modes.md"


BASELINE_REPORT_SCHEMA_VERSION = "t2sp-baseline-report-v1"


FAILURE_MODE_REPORT_SCHEMA_VERSION = "t2sp-baseline-failure-modes-v1"


CONFIG_SNAPSHOT_SCHEMA_VERSION = "t2sp-base-run-config-v1"


RUN_SUMMARY_SCHEMA_VERSION = "t2sp-base-run-summary-v1"


CHECKPOINT_MANIFEST_SCHEMA_VERSION = "t2sp-base-checkpoint-manifest-v1"


BASELINE_CONFIG_COPY_FILENAME = "baseline.yaml"


BASE_RUN_SUMMARY_FILENAME = "run-summary.json"


BASE_CHECKPOINT_MANIFEST_FILENAME = "checkpoint-manifest.json"


BASE_PREDICTION_SAMPLES_ARCHIVE_FILENAME = "samples.tar.zst"


BASE_PREDICTION_SAMPLES_OBSERVED_MEMBERS_FILENAME = "samples_observed_members.txt"


BASE_QUALITATIVE_PANEL_SAMPLES_ARCHIVE_FILENAME = "panel_samples.tar.zst"


BASE_QUALITATIVE_PANEL_SAMPLES_OBSERVED_MEMBERS_FILENAME = "panel_samples_observed_members.txt"


METRIC_LIMITATION_NOTE = (
    "Automatic metrics are useful diagnostics, but they are not sufficient evidence of sign "
    "intelligibility or strong task-solving quality."
)


LENGTH_POLICY_WARNING = (
    "Predictions are generated at the reference sequence length; this does not evaluate duration "
    "or timing generation."
)


CONFIDENCE_POLICY_WARNING = (
    "Prediction confidence arrays are synthetic validity masks and are not model uncertainty."
)


BASE_DEFAULT_PREDICTION_SPLITS = ("val",)


BASE_ALLOWED_SPLITS = ("train", "val", "test")


BASE_RUN_QUALITATIVE_EXPORT = True


BASE_RUN_MODES: Final[tuple[BaseRunMode, ...]] = ("smoke", "check", "complete")


BASE_DEFAULT_RUN_MODE: Final[BaseRunMode] = "smoke"


BASE_RUN_MODE_POLICIES: Final[Mapping[str, BaseRunModePolicy]] = {
    "smoke": BaseRunModePolicy(
        run_mode="smoke",
        purpose="fast notebook/workflow/publish smoke",
        claim="no model quality claim",
        run_mode_statement=(
            "This is a tiny smoke run for pipeline validation and carries no model quality claim."
        ),
        limit_train_samples=1,
        limit_validation_samples=1,
        limit_prediction_samples=1,
        epoch_count=1,
        min_epochs=1,
        early_stopping_patience=1,
        shuffle_train=False,
        run_qualitative_export=BASE_RUN_QUALITATIVE_EXPORT,
        panel_size=1,
        prediction_splits=BASE_DEFAULT_PREDICTION_SPLITS,
    ),
    "check": BaseRunModePolicy(
        run_mode="check",
        purpose="full-data integration check",
        claim="not final M0 baseline training",
        run_mode_statement=("This is a full-data check run, not final M0 baseline training."),
        limit_train_samples=None,
        limit_validation_samples=None,
        limit_prediction_samples=None,
        epoch_count=1,
        min_epochs=1,
        early_stopping_patience=1,
        shuffle_train=True,
        run_qualitative_export=BASE_RUN_QUALITATIVE_EXPORT,
        panel_size=8,
        prediction_splits=BASE_DEFAULT_PREDICTION_SPLITS,
    ),
    "complete": BaseRunModePolicy(
        run_mode="complete",
        purpose="final M0 comparison-floor baseline training intent",
        claim="M0 comparison floor only",
        run_mode_statement=(
            "This is the final M0 comparison-floor baseline training mode. It is not a "
            "sign-intelligibility or contribution-strength claim."
        ),
        limit_train_samples=None,
        limit_validation_samples=None,
        limit_prediction_samples=None,
        epoch_count=50,
        min_epochs=10,
        early_stopping_patience=8,
        shuffle_train=True,
        run_qualitative_export=BASE_RUN_QUALITATIVE_EXPORT,
        panel_size=16,
        prediction_splits=BASE_DEFAULT_PREDICTION_SPLITS,
    ),
}

__all__ = [
    "BaseRunMode",
    "BaseRunModePolicy",
    "BASELINE_REPORT_JSON_FILENAME",
    "BASELINE_REPORT_MARKDOWN_FILENAME",
    "FAILURE_MODES_JSON_FILENAME",
    "FAILURE_MODES_MARKDOWN_FILENAME",
    "BASELINE_REPORT_SCHEMA_VERSION",
    "FAILURE_MODE_REPORT_SCHEMA_VERSION",
    "CONFIG_SNAPSHOT_SCHEMA_VERSION",
    "RUN_SUMMARY_SCHEMA_VERSION",
    "CHECKPOINT_MANIFEST_SCHEMA_VERSION",
    "BASELINE_CONFIG_COPY_FILENAME",
    "BASE_RUN_SUMMARY_FILENAME",
    "BASE_CHECKPOINT_MANIFEST_FILENAME",
    "BASE_PREDICTION_SAMPLES_ARCHIVE_FILENAME",
    "BASE_PREDICTION_SAMPLES_OBSERVED_MEMBERS_FILENAME",
    "BASE_QUALITATIVE_PANEL_SAMPLES_ARCHIVE_FILENAME",
    "BASE_QUALITATIVE_PANEL_SAMPLES_OBSERVED_MEMBERS_FILENAME",
    "METRIC_LIMITATION_NOTE",
    "LENGTH_POLICY_WARNING",
    "CONFIDENCE_POLICY_WARNING",
    "BASE_DEFAULT_PREDICTION_SPLITS",
    "BASE_ALLOWED_SPLITS",
    "BASE_RUN_QUALITATIVE_EXPORT",
    "BASE_RUN_MODES",
    "BASE_DEFAULT_RUN_MODE",
    "BASE_RUN_MODE_POLICIES",
]
