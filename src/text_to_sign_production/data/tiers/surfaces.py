"""Typed, file-free surfaces for included and excluded tier decisions."""

from __future__ import annotations

from text_to_sign_production.data.tiers.types import (
    ExcludedTierSurfaceEntry,
    IncludedTierSurfaceEntry,
    TierBundle,
)


def build_included_surface_entries(
    bundle: TierBundle,
    *,
    tier_name: str | None = None,
) -> tuple[IncludedTierSurfaceEntry, ...]:
    """Build deterministic included entries, preserving bundle decision order."""
    return tuple(
        IncludedTierSurfaceEntry(
            sample_id=decision.sample_id,
            split=decision.split,
            tier_name=decision.tier_name,
        )
        for decision in bundle.decisions
        if decision.included and (tier_name is None or decision.tier_name == tier_name)
    )


def build_excluded_surface_entries(
    bundle: TierBundle,
    *,
    tier_name: str | None = None,
) -> tuple[ExcludedTierSurfaceEntry, ...]:
    """Build deterministic excluded entries, preserving bundle decision order."""
    return tuple(
        ExcludedTierSurfaceEntry(
            sample_id=decision.sample_id,
            split=decision.split,
            tier_name=decision.tier_name,
            metric_failures=decision.metric_failures,
            leakage_failure=decision.leakage_failure,
        )
        for decision in bundle.decisions
        if not decision.included and (tier_name is None or decision.tier_name == tier_name)
    )
