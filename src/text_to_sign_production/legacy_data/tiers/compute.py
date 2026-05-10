"""Deterministic tier decision composition."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, TypeVar, cast

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.legacy_data.leakages.types import LeakageSampleSummary
from text_to_sign_production.legacy_data.metrics.types import MetricBundle
from text_to_sign_production.legacy_data.samples.types import PassedManifestEntry
from text_to_sign_production.legacy_data.tiers.coherence import (
    evaluate_temporal_coherence_family,
)
from text_to_sign_production.legacy_data.tiers.confidence import evaluate_confidence_family
from text_to_sign_production.legacy_data.tiers.face import evaluate_face_family
from text_to_sign_production.legacy_data.tiers.geometry import evaluate_geometry_family
from text_to_sign_production.legacy_data.tiers.hand import evaluate_hand_family
from text_to_sign_production.legacy_data.tiers.leakage import evaluate_leakage_policy
from text_to_sign_production.legacy_data.tiers.manual_detail import evaluate_manual_detail_family
from text_to_sign_production.legacy_data.tiers.non_manual_quality import (
    evaluate_non_manual_quality_family,
)
from text_to_sign_production.legacy_data.tiers.oob import evaluate_oob_family
from text_to_sign_production.legacy_data.tiers.roles import BINDING_TIER_FAMILIES
from text_to_sign_production.legacy_data.tiers.support import (
    evaluate_upper_body_support_family,
)
from text_to_sign_production.legacy_data.tiers.tracking import evaluate_tracking_quality_family
from text_to_sign_production.legacy_data.tiers.types import (
    BindingTierFamily,
    FilterConfig,
    FilterLevel,
    TierBundle,
    TierDecisionDetail,
    TierMembership,
    TierMembershipRecord,
    TierMetricFailure,
    TierName,
    TierPolicy,
    binding_tier_family_display_label,
)

SampleKey = tuple[SampleSplit, str]
ThresholdT = TypeVar("ThresholdT")
FamilyEvaluator = Callable[[MetricBundle, object, FilterLevel], tuple[TierMetricFailure, ...]]


@dataclass(frozen=True, slots=True)
class TierDecisionProgressEvent:
    split: SampleSplit
    sample_id: str
    tier_name: TierName
    membership: TierMembership


class TierDecisionProgressSink(Protocol):
    def update_tier_decision_progress(self, event: TierDecisionProgressEvent) -> None: ...


_TIER_ORDER: tuple[TierName, ...] = tuple(TierName)
_FAMILY_ORDER: tuple[BindingTierFamily, ...] = BINDING_TIER_FAMILIES
_FAMILY_DISPLAY_ORDER: tuple[str, ...] = tuple(
    binding_tier_family_display_label(family) for family in _FAMILY_ORDER
)
_FAMILY_EVALUATORS: Mapping[BindingTierFamily, FamilyEvaluator] = {
    BindingTierFamily.OOB: cast(FamilyEvaluator, evaluate_oob_family),
    BindingTierFamily.UPPER_BODY_SUPPORT: cast(
        FamilyEvaluator,
        evaluate_upper_body_support_family,
    ),
    BindingTierFamily.MANUAL_VISIBILITY: cast(FamilyEvaluator, evaluate_hand_family),
    BindingTierFamily.NON_MANUAL_VISIBILITY: cast(FamilyEvaluator, evaluate_face_family),
    BindingTierFamily.CONFIDENCE: cast(FamilyEvaluator, evaluate_confidence_family),
    BindingTierFamily.KINEMATIC_NATURALNESS: cast(
        FamilyEvaluator,
        evaluate_temporal_coherence_family,
    ),
    BindingTierFamily.TRACKING_QUALITY: cast(FamilyEvaluator, evaluate_tracking_quality_family),
    BindingTierFamily.MANUAL_DETAIL: cast(FamilyEvaluator, evaluate_manual_detail_family),
    BindingTierFamily.NON_MANUAL_QUALITY: cast(
        FamilyEvaluator,
        evaluate_non_manual_quality_family,
    ),
    BindingTierFamily.GEOMETRY: cast(FamilyEvaluator, evaluate_geometry_family),
}


def build_tier_bundle(
    manifests: Sequence[PassedManifestEntry],
    metric_bundles: Sequence[MetricBundle],
    leakage_summaries: Sequence[LeakageSampleSummary],
    filter_config: FilterConfig,
    tier_policies: Sequence[TierPolicy],
    *,
    progress_sink: TierDecisionProgressSink | None = None,
) -> TierBundle:
    """Compose thin memberships and explicit decision details for every sample and tier."""
    _require_filter_config_levels(filter_config)
    manifest_by_key = _build_manifest_lookup(manifests)
    metrics_by_key = _build_metric_lookup(metric_bundles)
    leakage_by_key = _build_leakage_lookup(leakage_summaries)
    _require_matching_keys(manifest_by_key, metrics_by_key, leakage_by_key)

    policy_by_name = _build_policy_lookup(tier_policies)
    memberships: list[TierMembershipRecord] = []
    decision_details: list[TierDecisionDetail] = []

    sample_keys = sorted(manifest_by_key)
    for split, sample_id in sample_keys:
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
            leakage_failure = evaluate_leakage_policy(leakage_summary, policy)
            membership = (
                TierMembership.INCLUDED
                if not metric_failures and leakage_failure is None
                else TierMembership.EXCLUDED
            )
            memberships.append(
                TierMembershipRecord(
                    sample_id=sample_id,
                    split=split,
                    tier_name=tier_name,
                    membership=membership,
                )
            )
            decision_details.append(
                TierDecisionDetail(
                    sample_id=sample_id,
                    split=split,
                    tier_name=tier_name,
                    membership=membership,
                    metric_failures=metric_failures,
                    leakage_failure=leakage_failure,
                    max_leakage_severity=leakage_summary.max_severity,
                    applied_family_levels=dict(policy.family_levels),
                )
            )
            if progress_sink is not None:
                progress_sink.update_tier_decision_progress(
                    TierDecisionProgressEvent(
                        split=split,
                        sample_id=sample_id,
                        tier_name=tier_name,
                        membership=membership,
                    )
                )

    return TierBundle(
        memberships=tuple(memberships),
        decision_details=tuple(decision_details),
    )


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


def _build_policy_lookup(tier_policies: Sequence[TierPolicy]) -> dict[TierName, TierPolicy]:
    lookup: dict[TierName, TierPolicy] = {}
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
            f"{[tier.value for tier in _TIER_ORDER]} "
            f"(missing={sorted(tier.value for tier in expected_names - actual_names)}, "
            f"unknown={sorted(tier.value for tier in actual_names - expected_names)})"
        )
    return lookup


def _require_exact_family_levels(
    family_levels: Mapping[BindingTierFamily, FilterLevel], name: str
) -> None:
    actual = set(family_levels)
    expected = set(_FAMILY_ORDER)
    if actual != expected:
        raise ValueError(
            f"{name} family levels must contain exactly {list(_FAMILY_DISPLAY_ORDER)} "
            f"(missing={_display_families(expected - actual)}, "
            f"unknown={_display_families(actual - expected)})"
        )


def _require_filter_config_levels(filter_config: FilterConfig) -> None:
    for family in _FAMILY_ORDER:
        _require_exact_filter_levels(getattr(filter_config, family.value), family.value)


def _require_exact_filter_levels(
    family_thresholds: Mapping[FilterLevel, ThresholdT],
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
    family_levels: Mapping[BindingTierFamily, FilterLevel],
) -> tuple[TierMetricFailure, ...]:
    _require_exact_family_levels(family_levels, "applied policy")

    failures: list[TierMetricFailure] = []
    for family in _FAMILY_ORDER:
        level = family_levels[family]
        thresholds = getattr(filter_config, family.value)[level]
        failures.extend(_FAMILY_EVALUATORS[family](bundle, thresholds, level))
    return tuple(failures)


def _display_families(families: set[BindingTierFamily]) -> list[str]:
    return sorted(binding_tier_family_display_label(family) for family in families)
