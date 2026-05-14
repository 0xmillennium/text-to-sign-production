"""Typed calibration report surfaces for tier workflow review."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Callable, TypeVar

from text_to_sign_production.core.ids import TierName
from text_to_sign_production.core.models import TierDecisionBundle, TierFamilyDecision
from text_to_sign_production.core.progress import ProgressSession, ProgressStageSpec
from text_to_sign_production.data.tier.context.types import QualityContext
from text_to_sign_production.data.tier.families.analysis import binding_metric_summary
from text_to_sign_production.data.tier.families.types import (
    BindingQualityFamily,
    QualityMetricBundle,
)

_BINDING_METRIC_NAMES_BY_FAMILY: dict[BindingQualityFamily, tuple[str, ...]] = {
    BindingQualityFamily.OOB: (
        "body_out_of_bounds_frame_ratio",
        "left_hand_out_of_bounds_frame_ratio",
        "right_hand_out_of_bounds_frame_ratio",
        "face_out_of_bounds_frame_ratio",
        "max_channel_out_of_bounds_frame_ratio",
    ),
    BindingQualityFamily.UPPER_BODY_SUPPORT: (
        "active_span_upper_body_available_frame_ratio",
        "active_span_upper_body_landmark_coverage_ratio",
        "max_active_span_upper_body_dropout_run_ratio",
    ),
    BindingQualityFamily.MANUAL_VISIBILITY: (
        "active_span_any_hand_available_frame_ratio",
        "active_span_both_hands_unavailable_frame_ratio",
        "max_active_span_any_hand_dropout_run_ratio",
    ),
    BindingQualityFamily.NON_MANUAL_VISIBILITY: (
        "active_span_face_available_frame_ratio",
        "max_active_span_face_unavailable_run_ratio",
    ),
    BindingQualityFamily.CONFIDENCE: (
        "active_span_body_observed_mean_confidence",
        "active_span_hand_observed_mean_confidence",
    ),
    BindingQualityFamily.KINEMATIC_NATURALNESS: (
        "comparable_transition_abrupt_ratio",
        "comparable_transition_discontinuity_ratio",
        "max_active_span_frozen_run_ratio",
    ),
    BindingQualityFamily.TRACKING_QUALITY: (
        "tracked_target_missing_frame_ratio",
        "person_tracking_continuity_break_ratio",
        "person_tracking_reanchor_ratio",
    ),
    BindingQualityFamily.MANUAL_DETAIL: (
        "active_span_representative_hand_landmark_coverage_ratio",
        "active_span_representative_hand_fingertip_coverage_ratio",
        "active_span_representative_hand_distal_chain_coverage_ratio",
        "max_active_span_representative_hand_detail_dropout_run_ratio",
    ),
    BindingQualityFamily.NON_MANUAL_QUALITY: (
        "active_span_face_landmark_coverage_ratio",
        "active_span_upper_face_landmark_coverage_ratio",
        "active_span_lower_face_landmark_coverage_ratio",
        "max_active_span_face_detail_dropout_run_ratio",
        "active_span_face_available_given_manual_frame_ratio",
    ),
    BindingQualityFamily.GEOMETRY: (
        "active_span_upper_body_bone_length_outlier_frame_ratio",
        "active_span_cross_channel_scale_outlier_frame_ratio",
    ),
}

_ProgressResult = TypeVar("_ProgressResult")


@dataclass(frozen=True, slots=True)
class TierFamilyPassSurface:
    """Calibration pass counts for one binding family."""

    family: BindingQualityFamily
    loose_pass_count: int
    loose_pass_ratio: float
    clean_pass_count: int
    clean_pass_ratio: float
    tight_pass_count: int
    tight_pass_ratio: float


@dataclass(frozen=True, slots=True)
class TierMetricDistributionSummary:
    """Calibration distribution summary for one numeric metric surface."""

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


@dataclass(frozen=True, slots=True)
class TierFamilyWaterfallStep:
    """Calibration cascading waterfall row for one family."""

    tier: TierName
    family: BindingQualityFamily
    remaining_before_count: int
    remaining_after_count: int
    dropped_count: int


@dataclass(frozen=True, slots=True)
class TierFamilyWaterfalls:
    """Calibration cascading waterfall rows for all supported tiers."""

    loose: tuple[TierFamilyWaterfallStep, ...]
    clean: tuple[TierFamilyWaterfallStep, ...]
    tight: tuple[TierFamilyWaterfallStep, ...]


@dataclass(frozen=True, slots=True)
class TierActiveSpanDerivationSummary:
    """Calibration summary for staged active-span derivation."""

    count_distributions: tuple[TierMetricDistributionSummary, ...]
    fallback_used_count: int
    fallback_used_ratio: float


@dataclass(frozen=True, slots=True)
class TierCalibrationSurfaces:
    """Typed calibration surfaces used by tier review/report projection."""

    family_pass_surfaces: tuple[TierFamilyPassSurface, ...]
    binding_metric_distributions: tuple[TierMetricDistributionSummary, ...]
    family_waterfalls: TierFamilyWaterfalls
    active_span_derivation: TierActiveSpanDerivationSummary


@dataclass(frozen=True, slots=True)
class TierCalibrationProgressSpecs:
    """Progress stages for tier calibration aggregate report computation."""

    family_pass_surfaces: ProgressStageSpec | None = None
    binding_metric_distributions: ProgressStageSpec | None = None
    family_waterfalls: ProgressStageSpec | None = None
    active_span_derivation: ProgressStageSpec | None = None


def build_tier_calibration_surfaces(
    *,
    quality_metrics: tuple[QualityMetricBundle, ...],
    quality_contexts: tuple[QualityContext, ...],
    tier_decisions: tuple[TierDecisionBundle, ...],
    progress_session: ProgressSession | None = None,
    progress_specs: TierCalibrationProgressSpecs | None = None,
) -> TierCalibrationSurfaces:
    """Build typed tier calibration report surfaces from domain/data inputs."""
    if progress_session is None or progress_specs is None:
        return TierCalibrationSurfaces(
            family_pass_surfaces=_family_pass_surfaces(tier_decisions),
            binding_metric_distributions=_binding_metric_distributions(quality_metrics),
            family_waterfalls=_family_waterfalls(tier_decisions),
            active_span_derivation=_active_span_derivation_summary(quality_contexts),
        )

    return TierCalibrationSurfaces(
        family_pass_surfaces=_compute_with_optional_progress(
            progress_session,
            progress_specs.family_pass_surfaces,
            lambda: _family_pass_surfaces(tier_decisions),
        ),
        binding_metric_distributions=_compute_with_optional_progress(
            progress_session,
            progress_specs.binding_metric_distributions,
            lambda: _binding_metric_distributions(quality_metrics),
        ),
        family_waterfalls=_compute_with_optional_progress(
            progress_session,
            progress_specs.family_waterfalls,
            lambda: _family_waterfalls(tier_decisions),
        ),
        active_span_derivation=_compute_with_optional_progress(
            progress_session,
            progress_specs.active_span_derivation,
            lambda: _active_span_derivation_summary(quality_contexts),
        ),
    )


def _compute_with_optional_progress(
    progress_session: ProgressSession,
    progress_spec: ProgressStageSpec | None,
    compute: Callable[[], _ProgressResult],
) -> _ProgressResult:
    if progress_spec is None:
        return compute()
    with progress_session.task(progress_spec, total=1) as task:
        result = compute()
        task.advance(counters={"projected": 1})
        return result


def family_decision_for(
    decisions: tuple[TierFamilyDecision, ...],
    family: BindingQualityFamily,
) -> TierFamilyDecision | None:
    """Return the typed decision for a binding family when present."""
    for decision in decisions:
        if decision.family == family:
            return decision
    return None


def _family_pass_surfaces(
    tier_decisions: tuple[TierDecisionBundle, ...],
) -> tuple[TierFamilyPassSurface, ...]:
    total = len(tier_decisions)
    rows: list[TierFamilyPassSurface] = []
    for family in BindingQualityFamily:
        pass_counts: dict[TierName, int] = {tier: 0 for tier in TierName}
        for tier_decision in tier_decisions:
            decision = family_decision_for(tier_decision.family_decisions, family)
            if decision is None:
                continue
            for tier in TierName:
                if tier in decision.supported_tiers:
                    pass_counts[tier] += 1
        rows.append(
            TierFamilyPassSurface(
                family=family,
                loose_pass_count=pass_counts[TierName.LOOSE],
                loose_pass_ratio=_ratio(pass_counts[TierName.LOOSE], total),
                clean_pass_count=pass_counts[TierName.CLEAN],
                clean_pass_ratio=_ratio(pass_counts[TierName.CLEAN], total),
                tight_pass_count=pass_counts[TierName.TIGHT],
                tight_pass_ratio=_ratio(pass_counts[TierName.TIGHT], total),
            )
        )
    return tuple(rows)


def _binding_metric_distributions(
    quality_metrics: tuple[QualityMetricBundle, ...],
) -> tuple[TierMetricDistributionSummary, ...]:
    values: dict[tuple[BindingQualityFamily, str], list[float]] = {
        (family, metric_name): []
        for family, metric_names in _BINDING_METRIC_NAMES_BY_FAMILY.items()
        for metric_name in metric_names
    }
    for metrics in quality_metrics:
        for family_summary in binding_metric_summary(metrics):
            if not isinstance(family_summary.family, BindingQualityFamily):
                continue
            allowed_names = _BINDING_METRIC_NAMES_BY_FAMILY[family_summary.family]
            for metric in family_summary.metrics:
                if metric.name in allowed_names:
                    values[(family_summary.family, metric.name)].append(metric.value)
    return tuple(
        _distribution_summary(family.value, metric_name, values[(family, metric_name)])
        for family in BindingQualityFamily
        for metric_name in _BINDING_METRIC_NAMES_BY_FAMILY[family]
    )


def _family_waterfalls(
    tier_decisions: tuple[TierDecisionBundle, ...],
) -> TierFamilyWaterfalls:
    return TierFamilyWaterfalls(
        loose=_family_waterfall(tier_decisions, TierName.LOOSE),
        clean=_family_waterfall(tier_decisions, TierName.CLEAN),
        tight=_family_waterfall(tier_decisions, TierName.TIGHT),
    )


def _family_waterfall(
    tier_decisions: tuple[TierDecisionBundle, ...],
    tier: TierName,
) -> tuple[TierFamilyWaterfallStep, ...]:
    remaining = tuple(range(len(tier_decisions)))
    rows: list[TierFamilyWaterfallStep] = []
    for family in BindingQualityFamily:
        before = len(remaining)
        after_remaining = tuple(
            index
            for index in remaining
            if (
                decision := family_decision_for(
                    tier_decisions[index].family_decisions,
                    family,
                )
            )
            is not None
            and tier in decision.supported_tiers
        )
        after = len(after_remaining)
        rows.append(
            TierFamilyWaterfallStep(
                tier=tier,
                family=family,
                remaining_before_count=before,
                remaining_after_count=after,
                dropped_count=before - after,
            )
        )
        remaining = after_remaining
    return tuple(rows)


def _active_span_derivation_summary(
    quality_contexts: tuple[QualityContext, ...],
) -> TierActiveSpanDerivationSummary:
    spans = tuple(context.active_span for context in quality_contexts)
    fallback_count = sum(1 for span in spans if span.fallback_used)
    return TierActiveSpanDerivationSummary(
        count_distributions=(
            _distribution_summary(
                "active_span_derivation",
                "raw_evidence_frame_count",
                [float(span.raw_evidence_frame_count) for span in spans],
            ),
            _distribution_summary(
                "active_span_derivation",
                "stabilized_evidence_frame_count",
                [float(span.stabilized_evidence_frame_count) for span in spans],
            ),
            _distribution_summary(
                "active_span_derivation",
                "bridged_evidence_frame_count",
                [float(span.bridged_evidence_frame_count) for span in spans],
            ),
            _distribution_summary(
                "active_span_derivation",
                "padded_active_frame_count",
                [float(span.padded_active_frame_count) for span in spans],
            ),
        ),
        fallback_used_count=fallback_count,
        fallback_used_ratio=_ratio(fallback_count, len(spans)),
    )


def _distribution_summary(
    family: str,
    metric_name: str,
    values: Iterable[float],
) -> TierMetricDistributionSummary:
    ordered = sorted(float(value) for value in values)
    return TierMetricDistributionSummary(
        family=family,
        metric_name=metric_name,
        minimum=_percentile(ordered, 0.0),
        p05=_percentile(ordered, 0.05),
        p10=_percentile(ordered, 0.10),
        p25=_percentile(ordered, 0.25),
        p50=_percentile(ordered, 0.50),
        p75=_percentile(ordered, 0.75),
        p90=_percentile(ordered, 0.90),
        p95=_percentile(ordered, 0.95),
        maximum=_percentile(ordered, 1.0),
    )


def _percentile(ordered_values: list[float], percentile: float) -> float:
    if not ordered_values:
        return 0.0
    if len(ordered_values) == 1:
        return ordered_values[0]
    position = (len(ordered_values) - 1) * percentile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered_values) - 1)
    fraction = position - lower_index
    lower = ordered_values[lower_index]
    upper = ordered_values[upper_index]
    return lower + ((upper - lower) * fraction)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


__all__ = [
    "TierActiveSpanDerivationSummary",
    "TierCalibrationProgressSpecs",
    "TierCalibrationSurfaces",
    "TierFamilyPassSurface",
    "TierFamilyWaterfallStep",
    "TierFamilyWaterfalls",
    "TierMetricDistributionSummary",
    "build_tier_calibration_surfaces",
    "family_decision_for",
]
