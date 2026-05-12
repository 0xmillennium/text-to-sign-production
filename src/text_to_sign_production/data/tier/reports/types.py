"""Type system for PreparedSample-based tier reports."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.core.ids import SampleSplit, TierName
from text_to_sign_production.core.models import GateName, GateStatus
from text_to_sign_production.data.tier.families.types import BindingQualityFamily
from text_to_sign_production.data.tier.leakages import LeakageSeverity
from text_to_sign_production.data.tier.policies.types import TierStatus

ReportCellValue = int | float | str | bool | None


class ReportSectionName(enum.StrEnum):
    """Authoritative report section names and order labels."""

    SAMPLE = "sample"
    METRICS = "metrics"
    LEAKAGE = "leakage"
    TIER = "tier"
    GATES = "gates"


class ReportValidationIssueCode(enum.StrEnum):
    """Stable report-domain validation issue codes."""

    BUILD_INPUT_INCOMPLETE = "build_input_incomplete"
    SECTION_MISSING = "section_missing"
    SUMMARY_CONTRADICTION = "summary_contradiction"
    INVALID_REPORT = "invalid_report"


@dataclass(frozen=True, slots=True)
class ReportValidationIssue:
    """Structured report-domain validation issue."""

    code: ReportValidationIssueCode
    message: str
    field_path: str | None = None


@dataclass(frozen=True, slots=True)
class TierReportSummary:
    """Top-level deterministic tier report summary."""

    sample_id: str
    split: SampleSplit
    frame_count: int
    duration_seconds: float
    checkpoint_admission_status: str
    source_manifest_path: Path | None
    source_manifest_sha256: str | None
    payload_ref: str | None
    gate_detail_available: bool
    gate_detail_status: str
    tier_status: TierStatus
    selected_tier: TierName | None
    max_leakage_severity: LeakageSeverity
    diagnostic_highlights: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TierReportSection:
    """Base typed report section."""

    name: ReportSectionName
    title: str


@dataclass(frozen=True, slots=True)
class SampleReportSection(TierReportSection):
    """Projection of PreparedSample identity and pose state."""

    sample_id: str
    split: SampleSplit
    source_video_id: str
    source_sentence_id: str
    frame_count: int
    valid_frame_count: int
    source_complete: bool
    pose_complete: bool
    validation_issue_count: int


@dataclass(frozen=True, slots=True)
class MetricsReportSection(TierReportSection):
    """Projection of quality-family metric coverage."""

    binding_family_count: int
    diagnostic_family_count: int
    metric_row_count: int


@dataclass(frozen=True, slots=True)
class LeakageReportSection(TierReportSection):
    """Projection of leakage outcomes."""

    pair_count: int
    affected_sample_count: int
    max_severity: LeakageSeverity


@dataclass(frozen=True, slots=True)
class TierOutcomesSection(TierReportSection):
    """Projection of tier outcomes."""

    status: TierStatus
    selected_tier: TierName | None
    family_decision_count: int
    leakage_observed_max_severity: str | None
    leakage_admissible_tiers: tuple[TierName, ...]
    leakage_rejected_tiers: tuple[TierName, ...]
    issue_count: int


@dataclass(frozen=True, slots=True)
class GatesReportSection(TierReportSection):
    """Projection of checkpoint admission with explicit gate-detail availability."""

    sample_id: str
    split: SampleSplit
    checkpoint_admission_status: str
    source_manifest_path: Path | None
    source_manifest_sha256: str | None
    payload_ref: str | None
    gate_detail_available: bool
    gate_detail_status: str
    failed_gate_names: tuple[GateName, ...]
    gate_count: int


@dataclass(frozen=True, slots=True)
class MetricReportRow:
    """Machine-readable metric row projection."""

    family: str
    metric_name: str
    value: ReportCellValue


@dataclass(frozen=True, slots=True)
class GateReportRow:
    """Machine-readable gate row projection."""

    gate_name: GateName
    status: GateStatus
    issue_count: int


@dataclass(frozen=True, slots=True)
class TierReportRow:
    """Machine-readable binding-family tier row projection."""

    family: BindingQualityFamily
    status: TierStatus
    best_supported_tier: TierName | None
    supported_tiers: tuple[TierName, ...]
    issue_count: int


@dataclass(frozen=True, slots=True)
class TierReportTables:
    """Typed machine-readable table surfaces for a tier report."""

    metric_rows: tuple[MetricReportRow, ...]
    gate_rows: tuple[GateReportRow, ...]
    tier_rows: tuple[TierReportRow, ...]


@dataclass(frozen=True, slots=True)
class TierReportBundle:
    """Structured projection-only tier report."""

    summary: TierReportSummary
    sections: tuple[TierReportSection, ...]
    tables: TierReportTables


__all__ = [
    "GateReportRow",
    "GatesReportSection",
    "LeakageReportSection",
    "MetricReportRow",
    "MetricsReportSection",
    "TierReportBundle",
    "TierReportSection",
    "TierReportSummary",
    "TierReportTables",
    "ReportCellValue",
    "ReportSectionName",
    "ReportValidationIssue",
    "ReportValidationIssueCode",
    "SampleReportSection",
    "TierReportRow",
    "TierOutcomesSection",
]
