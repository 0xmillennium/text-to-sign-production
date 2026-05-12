"""Tier-owned projection of tier decisions into manifest membership surfaces."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from text_to_sign_production.core.ids import SampleSplit, TierMembership, TierName
from text_to_sign_production.core.models import PassedManifestEntry, TierDecisionBundle
from text_to_sign_production.data.dataset.manifests import write_passed_manifest_jsonl
from text_to_sign_production.data.tier.policies.analysis import tier_membership_for_decision

TierManifestBucketKey: TypeAlias = tuple[TierName, TierMembership, SampleSplit]


@dataclass(frozen=True, slots=True)
class TierManifestProduction:
    """Tier-owned result of writing one tier manifest surface."""

    path: Path
    entries: tuple[PassedManifestEntry, ...]


@dataclass(frozen=True, slots=True)
class TierManifestDecisionInput:
    """Passed manifest row plus tier decision used for physical tier projections."""

    manifest: PassedManifestEntry
    decision: TierDecisionBundle


@dataclass(frozen=True, slots=True)
class TierManifestOutputTarget:
    """Resolved physical output target for one tier manifest projection."""

    tier: TierName
    membership: TierMembership
    split: SampleSplit
    path: Path


@dataclass(frozen=True, slots=True)
class TierManifestPlanEntry:
    """One planned tier manifest file and the entries it must contain."""

    target: TierManifestOutputTarget
    entries: tuple[PassedManifestEntry, ...]


@dataclass(frozen=True, slots=True)
class TierManifestWritePlan:
    """Auditable tier manifest materialization plan."""

    entries: tuple[TierManifestPlanEntry, ...]


def bucket_tier_manifest_entries(
    inputs: Iterable[TierManifestDecisionInput],
) -> Mapping[TierManifestBucketKey, tuple[PassedManifestEntry, ...]]:
    """Project passed rows into tier/membership/split manifest buckets."""
    mutable: dict[TierManifestBucketKey, list[PassedManifestEntry]] = {}
    for item in inputs:
        for tier in TierName:
            membership = tier_membership_for_decision(item.decision, tier)
            key = (tier, membership, item.manifest.split)
            mutable.setdefault(key, []).append(item.manifest)
    return {key: tuple(entries) for key, entries in mutable.items()}


def plan_tier_manifest_outputs(
    *,
    targets: Sequence[TierManifestOutputTarget],
    buckets: Mapping[TierManifestBucketKey, tuple[PassedManifestEntry, ...]],
) -> TierManifestWritePlan:
    """Plan physical tier manifest outputs from already-projected membership buckets."""
    return TierManifestWritePlan(
        entries=tuple(
            TierManifestPlanEntry(
                target=target,
                entries=buckets.get((target.tier, target.membership, target.split), ()),
            )
            for target in targets
        )
    )


def write_tier_manifest_output(
    *,
    path: str | Path,
    entries: Iterable[PassedManifestEntry],
) -> TierManifestProduction:
    """Write one tier included/excluded manifest using passed-row semantics."""
    materialized_entries = tuple(entries)
    manifest_path = Path(path)
    write_passed_manifest_jsonl(manifest_path, materialized_entries)
    return TierManifestProduction(path=manifest_path, entries=materialized_entries)


def write_tier_manifest_plan(plan: TierManifestWritePlan) -> tuple[TierManifestProduction, ...]:
    """Materialize a precomputed tier manifest write plan."""
    return tuple(
        write_tier_manifest_output(
            path=entry.target.path,
            entries=entry.entries,
        )
        for entry in plan.entries
    )


def write_tier_manifest_outputs(
    *,
    targets: Sequence[TierManifestOutputTarget],
    inputs: Iterable[TierManifestDecisionInput],
) -> tuple[TierManifestProduction, ...]:
    """Compatibility helper that projects, plans, then writes tier manifests."""
    buckets = bucket_tier_manifest_entries(inputs)
    plan = plan_tier_manifest_outputs(targets=targets, buckets=buckets)
    return write_tier_manifest_plan(plan)


__all__ = [
    "TierManifestBucketKey",
    "TierManifestDecisionInput",
    "TierManifestPlanEntry",
    "TierManifestOutputTarget",
    "TierManifestProduction",
    "TierManifestWritePlan",
    "bucket_tier_manifest_entries",
    "plan_tier_manifest_outputs",
    "write_tier_manifest_output",
    "write_tier_manifest_outputs",
    "write_tier_manifest_plan",
]
