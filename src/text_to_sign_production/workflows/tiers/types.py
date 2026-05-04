"""Tiers workflow-specific data transfer objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from text_to_sign_production.data._shared.identities import SampleSplit
from text_to_sign_production.data.tiers.types import TierName
from text_to_sign_production.workflows._shared.archives import ArchiveExtractSpec
from text_to_sign_production.workflows._shared.transfers import FileTransferSpec


class TiersWorkflowError(RuntimeError):
    """Base error for tiers workflow failures."""


class TiersWorkflowInputError(TiersWorkflowError):
    """Raised when tiers workflow inputs are invalid or incomplete."""


@dataclass(frozen=True, slots=True)
class TiersWorkflowConfig:
    project_root: Path
    drive_project_root: Path
    splits: tuple[SampleSplit, ...]
    filters_config_relative_path: Path
    tiers_config_relative_path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_root", Path(self.project_root))
        object.__setattr__(self, "drive_project_root", Path(self.drive_project_root))
        object.__setattr__(
            self,
            "splits",
            tuple(SampleSplit(split) for split in self.splits),
        )
        filters_config_relative_path = Path(self.filters_config_relative_path)
        if filters_config_relative_path.is_absolute():
            raise ValueError(
                "filters_config_relative_path must remain relative: "
                f"{filters_config_relative_path}"
            )
        tiers_config_relative_path = Path(self.tiers_config_relative_path)
        if tiers_config_relative_path.is_absolute():
            raise ValueError(
                "tiers_config_relative_path must remain relative: "
                f"{tiers_config_relative_path}"
            )
        object.__setattr__(
            self,
            "filters_config_relative_path",
            filters_config_relative_path,
        )
        object.__setattr__(
            self,
            "tiers_config_relative_path",
            tiers_config_relative_path,
        )


@dataclass(frozen=True, slots=True)
class TiersRuntimePlan:
    project_root: Path
    drive_project_root: Path
    restore_file_transfers: tuple[FileTransferSpec, ...]
    restore_archive_extracts: tuple[ArchiveExtractSpec, ...]


@dataclass(frozen=True, slots=True)
class TiersWorkflowOutputSummary:
    included_counts_by_tier_by_split: Mapping[TierName, Mapping[SampleSplit, int]]
    excluded_counts_by_tier_by_split: Mapping[TierName, Mapping[SampleSplit, int]]
    report_paths: tuple[Path, ...]
    tiered_manifest_paths: tuple[Path, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "included_counts_by_tier_by_split",
            _freeze_nested_counts(self.included_counts_by_tier_by_split),
        )
        object.__setattr__(
            self,
            "excluded_counts_by_tier_by_split",
            _freeze_nested_counts(self.excluded_counts_by_tier_by_split),
        )
        object.__setattr__(self, "report_paths", tuple(self.report_paths))
        object.__setattr__(
            self,
            "tiered_manifest_paths",
            tuple(self.tiered_manifest_paths),
        )


@dataclass(frozen=True, slots=True)
class TiersWorkflowResult:
    output_summary: TiersWorkflowOutputSummary


@dataclass(frozen=True, slots=True)
class TiersPublishPlan:
    file_transfers: tuple[FileTransferSpec, ...]


@dataclass(frozen=True, slots=True)
class TiersPublishPlanSummary:
    planned_file_count: int
    target_paths: tuple[Path, ...]


def _freeze_nested_counts(
    counts: Mapping[TierName, Mapping[SampleSplit, int]],
) -> Mapping[TierName, Mapping[SampleSplit, int]]:
    return MappingProxyType(
        {
            TierName(tier): MappingProxyType(
                {SampleSplit(split): int(count) for split, count in split_counts.items()}
            )
            for tier, split_counts in counts.items()
        }
    )


__all__ = [
    "TiersPublishPlan",
    "TiersPublishPlanSummary",
    "TiersRuntimePlan",
    "TiersWorkflowConfig",
    "TiersWorkflowError",
    "TiersWorkflowInputError",
    "TiersWorkflowOutputSummary",
    "TiersWorkflowResult",
]
