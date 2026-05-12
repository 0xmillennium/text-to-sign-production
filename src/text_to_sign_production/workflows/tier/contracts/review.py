from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.workflows.tier.contracts.config import (
    TierWorkflowInputError,
)

ReviewValue = str | int | float | bool | None | tuple[str, ...]
ReviewItems = tuple[tuple[str, ReviewValue], ...]


@dataclass(frozen=True, slots=True)
class TierRuntimeAssetRow:
    label: str
    path: Path
    exists: bool
    valid: bool = True
    message: str | None = None
    scope: str = "asset"

    def __post_init__(self) -> None:
        _validate_non_empty_text("label", self.label)
        object.__setattr__(self, "label", self.label.strip())


@dataclass(frozen=True, slots=True)
class TierMembershipCountRow:
    tier: str
    split: str
    included_count: int
    excluded_count: int

    def __post_init__(self) -> None:
        _validate_non_empty_text("tier", self.tier)
        _validate_non_empty_text("split", self.split)
        _ensure_non_negative("included_count", self.included_count)
        _ensure_non_negative("excluded_count", self.excluded_count)
        object.__setattr__(self, "tier", self.tier.strip())
        object.__setattr__(self, "split", self.split.strip())


@dataclass(frozen=True, slots=True)
class TierPlannedReportOutputRow:
    label: str
    path: Path

    def __post_init__(self) -> None:
        _validate_non_empty_text("label", self.label)
        object.__setattr__(self, "label", self.label.strip())


@dataclass(frozen=True, slots=True)
class TierWrittenReportArtifactRow:
    label: str
    path: Path
    sha256: str
    execution_id: str

    def __post_init__(self) -> None:
        _validate_non_empty_text("label", self.label)
        _validate_non_empty_text("sha256", self.sha256)
        _validate_non_empty_text("execution_id", self.execution_id)
        object.__setattr__(self, "label", self.label.strip())


TierReportArtifactRow = TierPlannedReportOutputRow


@dataclass(frozen=True, slots=True)
class TieredManifestRow:
    tier: str
    membership: str
    split: str
    path: Path

    def __post_init__(self) -> None:
        _validate_non_empty_text("tier", self.tier)
        _validate_non_empty_text("membership", self.membership)
        _validate_non_empty_text("split", self.split)
        object.__setattr__(self, "tier", self.tier.strip())
        object.__setattr__(self, "membership", self.membership.strip())
        object.__setattr__(self, "split", self.split.strip())


@dataclass(frozen=True, slots=True)
class TierPublishTargetRow:
    label: str
    kind: str
    source_path: Path
    target_path: Path
    tier: str | None = None
    membership: str | None = None
    split: str | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_text("label", self.label)
        _validate_non_empty_text("kind", self.kind)
        object.__setattr__(self, "label", self.label.strip())
        object.__setattr__(self, "kind", self.kind.strip())


@dataclass(frozen=True, slots=True)
class TierPublishedArtifactRow:
    label: str
    kind: str
    target_path: Path
    target_sha256: str | None
    source_file_sha256: str | None
    file_digest_match: bool | None
    coherent: bool
    tier: str | None = None
    membership: str | None = None
    split: str | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_text("label", self.label)
        _validate_non_empty_text("kind", self.kind)
        object.__setattr__(self, "label", self.label.strip())
        object.__setattr__(self, "kind", self.kind.strip())


@dataclass(frozen=True, slots=True)
class TierReportSummaryReview:
    """Presentation DTO for one tier report summary."""

    split: str
    sample_id: str
    frame_count: int
    duration_seconds: float
    checkpoint_admission_status: str
    gate_detail_status: str
    tier_status: str
    selected_tier: str | None
    max_leakage_severity: str

    @property
    def label(self) -> str:
        return f"{self.split}/{self.sample_id}"

    def items(self) -> ReviewItems:
        return (
            ("split", self.split),
            ("sample_id", self.sample_id),
            ("frame_count", self.frame_count),
            ("duration_seconds", self.duration_seconds),
            ("checkpoint_admission_status", self.checkpoint_admission_status),
            ("gate_detail_status", self.gate_detail_status),
            ("tier_status", self.tier_status),
            ("selected_tier", self.selected_tier),
            ("max_leakage_severity", self.max_leakage_severity),
        )


@dataclass(frozen=True, slots=True)
class MetricRowReview:
    """Presentation DTO for one tier report metric row."""

    family: str
    metric_name: str
    value: str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class TierRowReview:
    """Presentation DTO for one tier family decision row."""

    family: str
    status: str
    best_supported_tier: str | None
    supported_tiers: tuple[str, ...]
    issue_count: int


@dataclass(frozen=True, slots=True)
class TierDecisionReviewRecord:
    """Presentation DTO for one machine-readable tier decision row."""

    split: str
    sample_id: str
    selected_tier: str | None
    tier_status: str
    leakage_observed_max_severity: str | None
    leakage_admissible_tiers: tuple[str, ...]
    leakage_rejected_tiers: tuple[str, ...]
    report_summary: TierReportSummaryReview
    metric_rows: tuple[MetricRowReview, ...]
    tier_rows: tuple[TierRowReview, ...]


@dataclass(frozen=True, slots=True)
class NamedCountReview:
    """Presentation DTO for one named aggregate count."""

    name: str
    count: int

    @property
    def label(self) -> str:
        return f"{self.name}={self.count}"


@dataclass(frozen=True, slots=True)
class LeakageReviewSummary:
    """Presentation DTO for aggregate leakage review fields."""

    sample_summary_count: int
    pair_fact_count: int
    affected_sample_count: int
    max_severity_counts: tuple[NamedCountReview, ...]

    def items(self) -> ReviewItems:
        return (
            ("sample_summary_count", self.sample_summary_count),
            ("pair_fact_count", self.pair_fact_count),
            ("affected_sample_count", self.affected_sample_count),
            ("max_severity_counts", tuple(row.label for row in self.max_severity_counts)),
        )


@dataclass(frozen=True, slots=True)
class FamilyFilterLevelReview:
    """Presentation DTO for one family-to-filter-level binding."""

    family: str
    level: str


@dataclass(frozen=True, slots=True)
class TierPolicyReview:
    """Presentation DTO for one configured tier policy."""

    tier: str
    max_allowed_leakage_severity: str
    family_filter_levels: tuple[FamilyFilterLevelReview, ...]

    @property
    def label(self) -> str:
        return f"{self.tier}:{self.max_allowed_leakage_severity}"


@dataclass(frozen=True, slots=True)
class TierPolicyReviewSummary:
    """Presentation DTO for tier policy configuration."""

    tier_order: tuple[str, ...]
    policy_count: int
    policies: tuple[TierPolicyReview, ...]

    def items(self) -> ReviewItems:
        return (
            ("tier_order", self.tier_order),
            ("policy_count", self.policy_count),
            ("policies", tuple(policy.label for policy in self.policies)),
        )


@dataclass(frozen=True, slots=True)
class TierFilterFamilyReview:
    """Presentation DTO for one tier filter family section."""

    family: str
    threshold_level_count: int
    levels: tuple[str, ...]

    @property
    def label(self) -> str:
        return f"{self.family}:{self.threshold_level_count}"


@dataclass(frozen=True, slots=True)
class TierFilterReviewSummary:
    """Presentation DTO for tier filter configuration."""

    family_count: int
    families: tuple[TierFilterFamilyReview, ...]

    def items(self) -> ReviewItems:
        return (
            ("family_count", self.family_count),
            ("families", tuple(family.label for family in self.families)),
        )


@dataclass(frozen=True, slots=True)
class FamilyPassSurfaceReview:
    """Calibration pass counts for one binding family."""

    family: str
    loose_pass_count: int
    loose_pass_ratio: float
    clean_pass_count: int
    clean_pass_ratio: float
    tight_pass_count: int
    tight_pass_ratio: float

    @property
    def label(self) -> str:
        return (
            f"{self.family}:"
            f"loose={self.loose_pass_count},"
            f"clean={self.clean_pass_count},"
            f"tight={self.tight_pass_count}"
        )


@dataclass(frozen=True, slots=True)
class DistributionSummaryReview:
    """Calibration distribution summary for one numeric surface."""

    family: str
    metric_name: str
    minimum: float
    p05: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    p95: float
    maximum: float

    @property
    def label(self) -> str:
        return f"{self.family}.{self.metric_name}:p50={self.p50:g}"


@dataclass(frozen=True, slots=True)
class FamilyWaterfallStepReview:
    """Calibration cascading waterfall row for one family."""

    tier: str
    family: str
    remaining_before_count: int
    remaining_after_count: int
    dropped_count: int

    @property
    def label(self) -> str:
        return (
            f"{self.tier}.{self.family}:"
            f"{self.remaining_before_count}->{self.remaining_after_count}"
        )


@dataclass(frozen=True, slots=True)
class FamilyWaterfallsReview:
    """Calibration cascading waterfall rows for all supported tiers."""

    loose: tuple[FamilyWaterfallStepReview, ...]
    clean: tuple[FamilyWaterfallStepReview, ...]
    tight: tuple[FamilyWaterfallStepReview, ...]


@dataclass(frozen=True, slots=True)
class ActiveSpanDerivationReview:
    """Calibration summary for staged active-span derivation."""

    count_distributions: tuple[DistributionSummaryReview, ...]
    fallback_used_count: int
    fallback_used_ratio: float

    def items(self) -> ReviewItems:
        return (
            (
                "count_distributions",
                tuple(distribution.label for distribution in self.count_distributions),
            ),
            ("fallback_used_count", self.fallback_used_count),
            ("fallback_used_ratio", self.fallback_used_ratio),
        )


@dataclass(frozen=True, slots=True)
class TierManifestOutputReview:
    """Presentation DTO for one planned tiered manifest output."""

    tier: str
    membership: str
    split: str
    path: Path


@dataclass(frozen=True, slots=True)
class TierDecisionDetailReviewPayload:
    """Typed DTO for the decision-detail JSONL payload."""

    records: tuple[TierDecisionReviewRecord, ...]


@dataclass(frozen=True, slots=True)
class MembershipCountReview:
    """Presentation DTO for one tier/membership aggregate count row."""

    tier: str
    split: str
    included_count: int
    excluded_count: int


@dataclass(frozen=True, slots=True)
class CalibrationSurfacesReviewPayload:
    """Typed DTO for the calibration surfaces JSON payload."""

    workflow: str
    processed_count: int
    membership_counts_by_tier_split: tuple[MembershipCountReview, ...]
    family_pass_surfaces: tuple[FamilyPassSurfaceReview, ...]
    binding_metric_distributions: tuple[DistributionSummaryReview, ...]
    family_waterfalls: FamilyWaterfallsReview
    active_span_derivation: ActiveSpanDerivationReview
    leakage: LeakageReviewSummary
    tier_report_count: int


@dataclass(frozen=True, slots=True)
class CalibrationDetailReviewPayload:
    """Typed DTO for the calibration detail JSON payload."""

    workflow: str
    processed_count: int
    filter_config: TierFilterReviewSummary
    tier_policies: TierPolicyReviewSummary
    membership_counts_by_tier_split: tuple[MembershipCountReview, ...]
    family_pass_surfaces: tuple[FamilyPassSurfaceReview, ...]
    binding_metric_distributions: tuple[DistributionSummaryReview, ...]
    family_waterfalls: FamilyWaterfallsReview
    active_span_derivation: ActiveSpanDerivationReview
    leakage: LeakageReviewSummary
    tier_reports: tuple[TierReportSummaryReview, ...]
    planned_tiered_manifest_outputs: tuple[TierManifestOutputReview, ...]


@dataclass(frozen=True, slots=True)
class TierReportIndexPayload:
    """Typed DTO for the tier report index JSON payload."""

    summary_markdown_path: Path
    calibration_markdown_path: Path
    decision_detail_jsonl_path: Path
    calibration_surfaces_json_path: Path
    calibration_detail_json_path: Path
    index_json_path: Path
    planned_tiered_manifest_outputs: tuple[TierManifestOutputReview, ...]
    config_provenance: tuple[tuple[str, Path, Path, str], ...]


def _validate_non_empty_text(field_name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise TierWorkflowInputError(f"{field_name} must be a non-empty string")


def _ensure_non_negative(field_name: str, value: int) -> None:
    if value < 0:
        raise TierWorkflowInputError(f"{field_name} must be >= 0")
