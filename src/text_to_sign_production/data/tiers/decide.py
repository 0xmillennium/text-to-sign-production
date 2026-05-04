"""Deterministic tier decision composition."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

from text_to_sign_production.data.leakages.types import (
    LeakageSampleSummary,
    LeakageSeverity,
)
from text_to_sign_production.data.metrics.types import MetricBundle
from text_to_sign_production.data.samples.types import PassedManifestEntry
from text_to_sign_production.data.tiers.types import (
    ConfidenceThresholds,
    FaceThresholds,
    FilterConfig,
    FilterLevel,
    HandThresholds,
    LengthThresholds,
    MetricFamily,
    OobThresholds,
    TextThresholds,
    TierBundle,
    TierDecision,
    TierLeakageFailure,
    TierMetricFailure,
    TierPolicy,
    ValidThresholds,
)

SampleKey = tuple[str, str]
ThresholdT = TypeVar("ThresholdT")

_TIER_ORDER = ("loose", "clean", "tight")
_FAMILY_ORDER = tuple(family.value for family in MetricFamily)
_SEVERITY_RANK = {
    LeakageSeverity.NONE: 0,
    LeakageSeverity.LOW: 1,
    LeakageSeverity.MEDIUM: 2,
    LeakageSeverity.HIGH: 3,
}


def build_tier_bundle(
    manifests: Sequence[PassedManifestEntry],
    metric_bundles: Sequence[MetricBundle],
    leakage_summaries: Sequence[LeakageSampleSummary],
    filter_config: FilterConfig,
    tier_policies: Sequence[TierPolicy],
) -> TierBundle:
    """Compose one deterministic decision for every sample and tier."""
    _require_filter_config_levels(filter_config)
    manifest_by_key = _build_manifest_lookup(manifests)
    metrics_by_key = _build_metric_lookup(metric_bundles)
    leakage_by_key = _build_leakage_lookup(leakage_summaries)
    _require_matching_keys(manifest_by_key, metrics_by_key, leakage_by_key)

    policy_by_name = _build_policy_lookup(tier_policies)
    decisions: list[TierDecision] = []

    for split, sample_id in sorted(manifest_by_key):
        key = (split, sample_id)
        metric_bundle = metrics_by_key[key]
        leakage_summary = leakage_by_key[key]

        for tier_name in _TIER_ORDER:
            policy = policy_by_name[tier_name]
            metric_failures = _evaluate_metric_failures(
                metric_bundle,
                filter_config,
                policy.family_levels,
            )
            leakage_failure = _evaluate_leakage_failure(leakage_summary, policy)
            decisions.append(
                TierDecision(
                    sample_id=sample_id,
                    split=split,
                    tier_name=tier_name,
                    included=not metric_failures and leakage_failure is None,
                    metric_failures=metric_failures,
                    leakage_failure=leakage_failure,
                    max_leakage_severity=leakage_summary.max_severity,
                    applied_family_levels=dict(policy.family_levels),
                )
            )

    return TierBundle(decisions=tuple(decisions))


def _build_manifest_lookup(
    manifests: Sequence[PassedManifestEntry],
) -> dict[SampleKey, PassedManifestEntry]:
    lookup: dict[SampleKey, PassedManifestEntry] = {}
    for manifest in manifests:
        key = (manifest.split, manifest.sample_id)
        if key in lookup:
            raise ValueError(f"Duplicate manifest entry for sample key {key}")
        lookup[key] = manifest
    return lookup


def _build_metric_lookup(metric_bundles: Sequence[MetricBundle]) -> dict[SampleKey, MetricBundle]:
    lookup: dict[SampleKey, MetricBundle] = {}
    for bundle in metric_bundles:
        key = (bundle.split, bundle.sample_id)
        if key in lookup:
            raise ValueError(f"Duplicate metric bundle for sample key {key}")
        lookup[key] = bundle
    return lookup


def _build_leakage_lookup(
    leakage_summaries: Sequence[LeakageSampleSummary],
) -> dict[SampleKey, LeakageSampleSummary]:
    lookup: dict[SampleKey, LeakageSampleSummary] = {}
    for summary in leakage_summaries:
        key = (summary.split, summary.sample_id)
        if key in lookup:
            raise ValueError(f"Duplicate leakage summary for sample key {key}")
        lookup[key] = summary
    return lookup


def _require_matching_keys(
    manifests: dict[SampleKey, PassedManifestEntry],
    metrics: dict[SampleKey, MetricBundle],
    leakages: dict[SampleKey, LeakageSampleSummary],
) -> None:
    manifest_keys = set(manifests)
    metric_keys = set(metrics)
    leakage_keys = set(leakages)
    if manifest_keys == metric_keys == leakage_keys:
        return

    details: list[str] = []
    if manifest_keys - metric_keys:
        details.append(f"missing_metrics={sorted(manifest_keys - metric_keys)}")
    if manifest_keys - leakage_keys:
        details.append(f"missing_leakage_summaries={sorted(manifest_keys - leakage_keys)}")
    if metric_keys - manifest_keys:
        details.append(f"metrics_without_manifest={sorted(metric_keys - manifest_keys)}")
    if leakage_keys - manifest_keys:
        details.append(f"leakage_summaries_without_manifest={sorted(leakage_keys - manifest_keys)}")
    raise ValueError(
        f"Tier inputs must have identical (split, sample_id) keys: {', '.join(details)}"
    )


def _build_policy_lookup(tier_policies: Sequence[TierPolicy]) -> dict[str, TierPolicy]:
    lookup: dict[str, TierPolicy] = {}
    for policy in tier_policies:
        if policy.tier_name in lookup:
            raise ValueError(f"Duplicate tier policy {policy.tier_name!r}")
        _require_exact_family_levels(policy.family_levels, f"tier policy {policy.tier_name!r}")
        lookup[policy.tier_name] = policy

    actual_names = set(lookup)
    expected_names = set(_TIER_ORDER)
    if actual_names != expected_names:
        raise ValueError(
            "Tier policies must contain exactly "
            f"{list(_TIER_ORDER)} (missing={sorted(expected_names - actual_names)}, "
            f"unknown={sorted(actual_names - expected_names)})"
        )
    return lookup


def _require_exact_family_levels(family_levels: dict[str, FilterLevel], name: str) -> None:
    actual = set(family_levels)
    expected = set(_FAMILY_ORDER)
    if actual != expected:
        raise ValueError(
            f"{name} family levels must contain exactly {list(_FAMILY_ORDER)} "
            f"(missing={sorted(expected - actual)}, unknown={sorted(actual - expected)})"
        )


def _require_filter_config_levels(filter_config: FilterConfig) -> None:
    _require_exact_filter_levels(filter_config.oob, "oob")
    _require_exact_filter_levels(filter_config.hand, "hand")
    _require_exact_filter_levels(filter_config.face, "face")
    _require_exact_filter_levels(filter_config.valid, "valid")
    _require_exact_filter_levels(filter_config.confidence, "confidence")
    _require_exact_filter_levels(filter_config.text, "text")
    _require_exact_filter_levels(filter_config.length, "length")


def _require_exact_filter_levels(
    family_thresholds: dict[FilterLevel, ThresholdT],
    family_name: str,
) -> None:
    actual: set[FilterLevel] = set(family_thresholds)
    expected: set[FilterLevel] = {level for level in FilterLevel}
    if actual != expected:
        missing = [level.value for level in FilterLevel if level not in actual]
        unknown = sorted(str(level) for level in actual if level not in expected)
        raise ValueError(
            f"Filter config family {family_name!r} must contain exactly "
            f"{[level.value for level in FilterLevel]} levels "
            f"(missing={missing}, unknown={unknown})"
        )


def _evaluate_metric_failures(
    bundle: MetricBundle,
    filter_config: FilterConfig,
    family_levels: dict[str, FilterLevel],
) -> tuple[TierMetricFailure, ...]:
    _require_exact_family_levels(family_levels, "applied policy")

    failures: list[TierMetricFailure] = []
    oob_level = family_levels["oob"]
    hand_level = family_levels["hand"]
    face_level = family_levels["face"]
    valid_level = family_levels["valid"]
    confidence_level = family_levels["confidence"]
    text_level = family_levels["text"]
    length_level = family_levels["length"]

    failures.extend(_evaluate_oob(bundle, filter_config.oob[oob_level], oob_level))
    failures.extend(_evaluate_hand(bundle, filter_config.hand[hand_level], hand_level))
    failures.extend(_evaluate_face(bundle, filter_config.face[face_level], face_level))
    failures.extend(_evaluate_valid(bundle, filter_config.valid[valid_level], valid_level))
    failures.extend(
        _evaluate_confidence(bundle, filter_config.confidence[confidence_level], confidence_level)
    )
    failures.extend(_evaluate_text(bundle, filter_config.text[text_level], text_level))
    failures.extend(_evaluate_length(bundle, filter_config.length[length_level], length_level))
    return tuple(failures)


def _evaluate_oob(
    bundle: MetricBundle,
    thresholds: OobThresholds,
    applied_level: FilterLevel,
) -> tuple[TierMetricFailure, ...]:
    if bundle.oob.out_of_bounds_ratio <= thresholds.max_out_of_bounds_ratio:
        return ()
    return (
        _failure(
            family="oob",
            metric_key="out_of_bounds_ratio",
            reason_code="max_out_of_bounds_ratio_exceeded",
            actual_value=bundle.oob.out_of_bounds_ratio,
            expected_value=thresholds.max_out_of_bounds_ratio,
            comparison="<=",
            applied_level=applied_level,
        ),
    )


def _evaluate_hand(
    bundle: MetricBundle,
    thresholds: HandThresholds,
    applied_level: FilterLevel,
) -> tuple[TierMetricFailure, ...]:
    if bundle.hand.any_hand_nonzero_frame_ratio >= thresholds.min_any_hand_nonzero_frame_ratio:
        return ()
    return (
        _failure(
            family="hand",
            metric_key="any_hand_nonzero_frame_ratio",
            reason_code="min_any_hand_nonzero_frame_ratio_not_met",
            actual_value=bundle.hand.any_hand_nonzero_frame_ratio,
            expected_value=thresholds.min_any_hand_nonzero_frame_ratio,
            comparison=">=",
            applied_level=applied_level,
        ),
    )


def _evaluate_face(
    bundle: MetricBundle,
    thresholds: FaceThresholds,
    applied_level: FilterLevel,
) -> tuple[TierMetricFailure, ...]:
    failures: list[TierMetricFailure] = []
    if bundle.face.face_nonzero_frame_ratio < thresholds.min_face_nonzero_frame_ratio:
        failures.append(
            _failure(
                family="face",
                metric_key="face_nonzero_frame_ratio",
                reason_code="min_face_nonzero_frame_ratio_not_met",
                actual_value=bundle.face.face_nonzero_frame_ratio,
                expected_value=thresholds.min_face_nonzero_frame_ratio,
                comparison=">=",
                applied_level=applied_level,
            )
        )
    if bundle.face.face_missing_frame_ratio > thresholds.max_face_missing_frame_ratio:
        failures.append(
            _failure(
                family="face",
                metric_key="face_missing_frame_ratio",
                reason_code="max_face_missing_frame_ratio_exceeded",
                actual_value=bundle.face.face_missing_frame_ratio,
                expected_value=thresholds.max_face_missing_frame_ratio,
                comparison="<=",
                applied_level=applied_level,
            )
        )
    return tuple(failures)


def _evaluate_valid(
    bundle: MetricBundle,
    thresholds: ValidThresholds,
    applied_level: FilterLevel,
) -> tuple[TierMetricFailure, ...]:
    failures: list[TierMetricFailure] = []
    if bundle.valid.valid_frame_ratio < thresholds.min_valid_frame_ratio:
        failures.append(
            _failure(
                family="valid",
                metric_key="valid_frame_ratio",
                reason_code="min_valid_frame_ratio_not_met",
                actual_value=bundle.valid.valid_frame_ratio,
                expected_value=thresholds.min_valid_frame_ratio,
                comparison=">=",
                applied_level=applied_level,
            )
        )
    if (
        bundle.valid.zeroed_canonical_joint_frame_ratio
        > thresholds.max_zeroed_canonical_joint_frame_ratio
    ):
        failures.append(
            _failure(
                family="valid",
                metric_key="zeroed_canonical_joint_frame_ratio",
                reason_code="max_zeroed_canonical_joint_frame_ratio_exceeded",
                actual_value=bundle.valid.zeroed_canonical_joint_frame_ratio,
                expected_value=thresholds.max_zeroed_canonical_joint_frame_ratio,
                comparison="<=",
                applied_level=applied_level,
            )
        )
    return tuple(failures)


def _evaluate_confidence(
    bundle: MetricBundle,
    thresholds: ConfidenceThresholds,
    applied_level: FilterLevel,
) -> tuple[TierMetricFailure, ...]:
    failures: list[TierMetricFailure] = []
    if bundle.confidence.overall_mean_confidence < thresholds.min_overall_mean_confidence:
        failures.append(
            _failure(
                family="confidence",
                metric_key="overall_mean_confidence",
                reason_code="min_overall_mean_confidence_not_met",
                actual_value=bundle.confidence.overall_mean_confidence,
                expected_value=thresholds.min_overall_mean_confidence,
                comparison=">=",
                applied_level=applied_level,
            )
        )
    if (
        bundle.confidence.overall_nonzero_confidence_ratio
        < thresholds.min_overall_nonzero_confidence_ratio
    ):
        failures.append(
            _failure(
                family="confidence",
                metric_key="overall_nonzero_confidence_ratio",
                reason_code="min_overall_nonzero_confidence_ratio_not_met",
                actual_value=bundle.confidence.overall_nonzero_confidence_ratio,
                expected_value=thresholds.min_overall_nonzero_confidence_ratio,
                comparison=">=",
                applied_level=applied_level,
            )
        )
    return tuple(failures)


def _evaluate_text(
    bundle: MetricBundle,
    thresholds: TextThresholds,
    applied_level: FilterLevel,
) -> tuple[TierMetricFailure, ...]:
    failures: list[TierMetricFailure] = []
    if bundle.text.character_count < thresholds.min_character_count:
        failures.append(
            _failure(
                family="text",
                metric_key="character_count",
                reason_code="min_character_count_not_met",
                actual_value=bundle.text.character_count,
                expected_value=thresholds.min_character_count,
                comparison=">=",
                applied_level=applied_level,
            )
        )
    if bundle.text.token_count < thresholds.min_token_count:
        failures.append(
            _failure(
                family="text",
                metric_key="token_count",
                reason_code="min_token_count_not_met",
                actual_value=bundle.text.token_count,
                expected_value=thresholds.min_token_count,
                comparison=">=",
                applied_level=applied_level,
            )
        )
    return tuple(failures)


def _evaluate_length(
    bundle: MetricBundle,
    thresholds: LengthThresholds,
    applied_level: FilterLevel,
) -> tuple[TierMetricFailure, ...]:
    failures: list[TierMetricFailure] = []
    if bundle.length.num_frames < thresholds.min_num_frames:
        failures.append(
            _failure(
                family="length",
                metric_key="num_frames",
                reason_code="min_num_frames_not_met",
                actual_value=bundle.length.num_frames,
                expected_value=thresholds.min_num_frames,
                comparison=">=",
                applied_level=applied_level,
            )
        )
    if bundle.length.duration_seconds is None:
        failures.append(
            _failure(
                family="length",
                metric_key="duration_seconds",
                reason_code="duration_seconds_missing",
                actual_value=None,
                expected_value=thresholds.min_duration_seconds,
                comparison=">=",
                applied_level=applied_level,
            )
        )
    elif bundle.length.duration_seconds < thresholds.min_duration_seconds:
        failures.append(
            _failure(
                family="length",
                metric_key="duration_seconds",
                reason_code="min_duration_seconds_not_met",
                actual_value=bundle.length.duration_seconds,
                expected_value=thresholds.min_duration_seconds,
                comparison=">=",
                applied_level=applied_level,
            )
        )
    return tuple(failures)


def _failure(
    *,
    family: str,
    metric_key: str,
    reason_code: str,
    actual_value: float | int | None,
    expected_value: float | int | None,
    comparison: str,
    applied_level: FilterLevel,
) -> TierMetricFailure:
    return TierMetricFailure(
        family=family,
        metric_key=metric_key,
        reason_code=reason_code,
        actual_value=actual_value,
        expected_value=expected_value,
        comparison=comparison,
        applied_level=applied_level,
    )


def _evaluate_leakage_failure(
    summary: LeakageSampleSummary,
    policy: TierPolicy,
) -> TierLeakageFailure | None:
    if _SEVERITY_RANK[summary.max_severity] <= _SEVERITY_RANK[policy.max_allowed_leakage_severity]:
        return None
    return TierLeakageFailure(
        reason_code="max_allowed_leakage_severity_exceeded",
        actual_max_severity=summary.max_severity,
        allowed_max_severity=policy.max_allowed_leakage_severity,
        matched_samples=summary.matched_samples,
    )
