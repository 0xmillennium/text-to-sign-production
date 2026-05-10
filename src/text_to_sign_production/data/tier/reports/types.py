"""Type system for PreparedSample-based quality reports."""

from __future__ import annotations

import enum
from dataclasses import dataclass

from text_to_sign_production.core.ids import SampleSplit, TierName
from text_to_sign_production.core.models import GateName, GateStatus
from text_to_sign_production.data.tier.families import BindingQualityFamily
from text_to_sign_production.data.tier.leakages import LeakageSeverity
from text_to_sign_production.data.tier.policies import TierStatus

ReportCellValue = int | float | str | bool | None


class ReportSectionName(enum.StrEnum):
    """Authoritative report section names and order labels."""

    SAMPLE = "sample"
    METRICS = "metrics"
    LEAKAGE = "leakage"
    TIERS = "tiers"
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
class QualityReportSummary:
    """Top-level deterministic quality report summary."""

    sample_id: str
    split: SampleSplit
    frame_count: int
    duration_seconds: float
    gate_status: str | None
    tier_status: TierStatus
    selected_tier: TierName | None
    max_leakage_severity: LeakageSeverity
    diagnostic_highlights: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class QualityReportSection:
    """Base typed report section."""

    name: ReportSectionName
    title: str


@dataclass(frozen=True, slots=True)
class SampleReportSection(QualityReportSection):
    """Projection of PreparedSample identity and pose state."""

    sample_id: str
    split: SampleSplit
    source_video_id: str
    source_sentence_id: str
    canonical_normalized_text_present: bool
    frame_count: int
    valid_frame_count: int


@dataclass(frozen=True, slots=True)
class MetricsReportSection(QualityReportSection):
    """Projection of quality-family metric coverage."""

    binding_family_count: int
    diagnostic_family_count: int
    metric_row_count: int


@dataclass(frozen=True, slots=True)
class LeakageReportSection(QualityReportSection):
    """Projection of leakage outcomes."""

    pair_count: int
    affected_sample_count: int
    max_severity: LeakageSeverity


@dataclass(frozen=True, slots=True)
class TiersReportSection(QualityReportSection):
    """Projection of tier outcomes."""

    status: TierStatus
    selected_tier: TierName | None
    family_decision_count: int
    issue_count: int


@dataclass(frozen=True, slots=True)
class GatesReportSection(QualityReportSection):
    """Projection of samples admission gate outcomes."""

    final_status: str
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
class QualityReportTables:
    """Typed machine-readable table surfaces for a quality report."""

    metric_rows: tuple[MetricReportRow, ...]
    gate_rows: tuple[GateReportRow, ...]
    tier_rows: tuple[TierReportRow, ...]


@dataclass(frozen=True, slots=True)
class QualityReportBundle:
    """Structured projection-only quality report."""

    summary: QualityReportSummary
    sections: tuple[QualityReportSection, ...]
    tables: QualityReportTables


__all__ = [
    "GateReportRow",
    "GatesReportSection",
    "LeakageReportSection",
    "MetricReportRow",
    "MetricsReportSection",
    "QualityReportBundle",
    "QualityReportSection",
    "QualityReportSummary",
    "QualityReportTables",
    "ReportCellValue",
    "ReportSectionName",
    "ReportValidationIssue",
    "ReportValidationIssueCode",
    "SampleReportSection",
    "TierReportRow",
    "TiersReportSection",
]
