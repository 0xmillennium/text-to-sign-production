"""Deterministic tier decision composition."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TypeVar

from text_to_sign_production.data._shared.identities import SampleSplit
from text_to_sign_production.data.leakages.types import LeakageSampleSummary
from text_to_sign_production.data.metrics.types import MetricBundle
from text_to_sign_production.data.samples.types import PassedManifestEntry
from text_to_sign_production.data.tiers.confidence import evaluate_confidence_family
from text_to_sign_production.data.tiers.coverage import evaluate_coverage_family
from text_to_sign_production.data.tiers.face import evaluate_face_family
from text_to_sign_production.data.tiers.hand import evaluate_hand_family
from text_to_sign_production.data.tiers.leakage import evaluate_leakage_policy
from text_to_sign_production.data.tiers.length import evaluate_length_family
from text_to_sign_production.data.tiers.oob import evaluate_oob_family
from text_to_sign_production.data.tiers.text import evaluate_text_family
from text_to_sign_production.data.tiers.types import (
    BindingTierFamily,
    FilterConfig,
    FilterLevel,
    TierBundle,
    TierDecision,
    TierMetricFailure,
    TierName,
    TierPolicy,
)

SampleKey = tuple[SampleSplit, str]
ThresholdT = TypeVar("ThresholdT")
TierDecisionProgressCallback = Callable[..., None]

_TIER_ORDER: tuple[TierName, ...] = tuple(TierName)
_FAMILY_ORDER: tuple[BindingTierFamily, ...] = tuple(BindingTierFamily)
_FAMILY_DISPLAY_ORDER: tuple[str, ...] = tuple(family.value for family in BindingTierFamily)


def build_tier_bundle(
    manifests: Sequence[PassedManifestEntry],
    metric_bundles: Sequence[MetricBundle],
    leakage_summaries: Sequence[LeakageSampleSummary],
    filter_config: FilterConfig,
    tier_policies: Sequence[TierPolicy],
    *,
    progress_callback: TierDecisionProgressCallback | None = None,
) -> TierBundle:
    """Compose one deterministic decision for every sample and tier."""
    _require_filter_config_levels(filter_config)
    manifest_by_key = _build_manifest_lookup(manifests)
    metrics_by_key = _build_metric_lookup(metric_bundles)
    leakage_by_key = _build_leakage_lookup(leakage_summaries)
    _require_matching_keys(manifest_by_key, metrics_by_key, leakage_by_key)

    policy_by_name = _build_policy_lookup(tier_policies)
    decisions: list[TierDecision] = []
    included_count = 0
    excluded_count = 0

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
            included = not metric_failures and leakage_failure is None
            if included:
                included_count += 1
            else:
                excluded_count += 1
            decisions.append(
                TierDecision(
                    sample_id=sample_id,
                    split=split,
                    tier_name=tier_name,
                    included=included,
                    metric_failures=metric_failures,
                    leakage_failure=leakage_failure,
                    max_leakage_severity=leakage_summary.max_severity,
                    applied_family_levels=dict(policy.family_levels),
                )
            )
            if progress_callback is not None:
                progress_callback(
                    split=split.value,
                    tier=tier_name.value,
                    included=included_count,
                    excluded=excluded_count,
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
    _require_exact_filter_levels(filter_config.oob, "oob")
    _require_exact_filter_levels(filter_config.coverage, "coverage")
    _require_exact_filter_levels(filter_config.hand, "hand")
    _require_exact_filter_levels(filter_config.face, "face")
    _require_exact_filter_levels(filter_config.confidence, "confidence")
    _require_exact_filter_levels(filter_config.text, "text")
    _require_exact_filter_levels(filter_config.length, "length")


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

    oob_level = family_levels[BindingTierFamily.OOB]
    coverage_level = family_levels[BindingTierFamily.COVERAGE]
    hand_level = family_levels[BindingTierFamily.HAND]
    face_level = family_levels[BindingTierFamily.FACE]
    confidence_level = family_levels[BindingTierFamily.CONFIDENCE]
    text_level = family_levels[BindingTierFamily.TEXT]
    length_level = family_levels[BindingTierFamily.LENGTH]

    return (
        *evaluate_oob_family(bundle, filter_config.oob[oob_level], oob_level),
        *evaluate_coverage_family(
            bundle,
            filter_config.coverage[coverage_level],
            coverage_level,
        ),
        *evaluate_hand_family(bundle, filter_config.hand[hand_level], hand_level),
        *evaluate_face_family(bundle, filter_config.face[face_level], face_level),
        *evaluate_confidence_family(
            bundle,
            filter_config.confidence[confidence_level],
            confidence_level,
        ),
        *evaluate_text_family(bundle, filter_config.text[text_level], text_level),
        *evaluate_length_family(bundle, filter_config.length[length_level], length_level),
    )


def _display_families(families: set[BindingTierFamily]) -> list[str]:
    return sorted(family.value for family in families)
