"""Typed, file-free selection surfaces for tier decisions."""

from __future__ import annotations

from text_to_sign_production.data.tiers.types import (
    ExcludedTierSurfaceEntry,
    IncludedTierSurfaceEntry,
    TierBundle,
    TierName,
)


def build_included_surface_entries(
    bundle: TierBundle,
    *,
    tier_name: TierName | str | None = None,
) -> tuple[IncludedTierSurfaceEntry, ...]:
    """Build deterministic included entries, preserving bundle decision order."""
    selected_tier = TierName(tier_name) if tier_name is not None else None
    return tuple(
        IncludedTierSurfaceEntry(
            sample_id=decision.sample_id,
            split=decision.split,
            tier_name=decision.tier_name,
        )
        for decision in bundle.decisions
        if decision.included and (selected_tier is None or decision.tier_name == selected_tier)
    )


def build_excluded_surface_entries(
    bundle: TierBundle,
    *,
    tier_name: TierName | str | None = None,
) -> tuple[ExcludedTierSurfaceEntry, ...]:
    """Build deterministic excluded selection entries, preserving bundle decision order."""
    selected_tier = TierName(tier_name) if tier_name is not None else None
    return tuple(
        ExcludedTierSurfaceEntry(
            sample_id=decision.sample_id,
            split=decision.split,
            tier_name=decision.tier_name,
        )
        for decision in bundle.decisions
        if not decision.included and (selected_tier is None or decision.tier_name == selected_tier)
    )
