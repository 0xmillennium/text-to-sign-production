"""Samples workflow-specific data transfer objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from text_to_sign_production.data._shared.identities import SampleSplit
from text_to_sign_production.data.pose.types import PersonSelectionPolicy
from text_to_sign_production.workflows._shared.archives import (
    ArchiveCreateSpec as _ArchiveCreateSpec,
)
from text_to_sign_production.workflows._shared.archives import (
    ArchiveExtractSpec as _ArchiveExtractSpec,
)
from text_to_sign_production.workflows._shared.archives import (
    ArchiveMemberListSpec as _ArchiveMemberListSpec,
)
from text_to_sign_production.workflows._shared.archives import (
    ArchiveVerifySpec as _ArchiveVerifySpec,
)
from text_to_sign_production.workflows._shared.transfers import (
    FileTransferSpec as _FileTransferSpec,
)


class SamplesWorkflowError(RuntimeError):
    """Base error for samples workflow failures."""


class SamplesWorkflowInputError(SamplesWorkflowError):
    """Raised when samples workflow inputs are invalid or incomplete."""


@dataclass(frozen=True, slots=True)
class SamplesWorkflowConfig:
    project_root: Path
    drive_project_root: Path
    splits: tuple[SampleSplit, ...]
    gates_config_relative_path: Path
    person_selection_policy: PersonSelectionPolicy

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_root", Path(self.project_root))
        object.__setattr__(self, "drive_project_root", Path(self.drive_project_root))
        object.__setattr__(
            self,
            "splits",
            tuple(SampleSplit(split) for split in self.splits),
        )
        gates_config_relative_path = Path(self.gates_config_relative_path)
        if gates_config_relative_path.is_absolute():
            raise ValueError(
                "gates_config_relative_path must remain relative: "
                f"{gates_config_relative_path}"
            )
        object.__setattr__(
            self,
            "gates_config_relative_path",
            gates_config_relative_path,
        )
        object.__setattr__(
            self,
            "person_selection_policy",
            PersonSelectionPolicy(self.person_selection_policy),
        )


@dataclass(frozen=True, slots=True)
class SamplesRuntimePlan:
    project_root: Path
    drive_project_root: Path
    restore_file_transfers: tuple[_FileTransferSpec, ...]
    restore_archive_extracts: tuple[_ArchiveExtractSpec, ...]


@dataclass(frozen=True, slots=True)
class SamplesWorkflowOutputSummary:
    passed_counts_by_split: Mapping[SampleSplit, int]
    dropped_counts_by_split: Mapping[SampleSplit, int]
    report_paths: tuple[Path, ...]
    untiered_manifest_paths: tuple[Path, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "passed_counts_by_split",
            MappingProxyType(dict(self.passed_counts_by_split)),
        )
        object.__setattr__(
            self,
            "dropped_counts_by_split",
            MappingProxyType(dict(self.dropped_counts_by_split)),
        )
        object.__setattr__(self, "report_paths", tuple(self.report_paths))
        object.__setattr__(
            self,
            "untiered_manifest_paths",
            tuple(self.untiered_manifest_paths),
        )


@dataclass(frozen=True, slots=True)
class SamplesWorkflowResult:
    output_summary: SamplesWorkflowOutputSummary


@dataclass(frozen=True, slots=True)
class SamplesPublishPlan:
    file_transfers: tuple[_FileTransferSpec, ...]
    archive_member_lists: tuple[_ArchiveMemberListSpec, ...]
    archive_creates: tuple[_ArchiveCreateSpec, ...]
    archive_verifies: tuple[_ArchiveVerifySpec, ...]


@dataclass(frozen=True, slots=True)
class SamplesPublishPlanSummary:
    planned_file_count: int
    planned_archive_count: int
    target_paths: tuple[Path, ...]


__all__ = [
    "SamplesPublishPlan",
    "SamplesPublishPlanSummary",
    "SamplesRuntimePlan",
    "SamplesWorkflowConfig",
    "SamplesWorkflowError",
    "SamplesWorkflowInputError",
    "SamplesWorkflowOutputSummary",
    "SamplesWorkflowResult",
]
