from __future__ import annotations

from collections.abc import Mapping

from text_to_sign_production.core.ids import SampleSplit, TierMembership, TierName
from text_to_sign_production.core.models import PassedManifestEntry
from text_to_sign_production.core.progress import ProgressSession, ProgressStageSpec
from text_to_sign_production.data.dataset.manifests import write_passed_manifest_jsonl
from text_to_sign_production.workflows.tiers.constants import (
    TIERS_STAGE_MANIFEST_WRITE,
    TIERS_WORKFLOW_NAME,
)
from text_to_sign_production.workflows.tiers.contracts import (
    TiersTieredManifestOutput,
    TiersWorkflowInvariantError,
)
from text_to_sign_production.workflows.tiers.layout import TiersLayout
from text_to_sign_production.workflows.tiers.processing.models import TiersDecisionBundle


def write_tiered_manifests(
    *,
    layout: TiersLayout,
    decision_bundles: tuple[TiersDecisionBundle, ...],
    progress_session: ProgressSession | None = None,
) -> tuple[TiersTieredManifestOutput, ...]:
    buckets = _bucket_tier_memberships(decision_bundles)
    outputs: list[TiersTieredManifestOutput] = []
    total = sum(len(entries) for entries in buckets.values())
    if progress_session is not None:
        with progress_session.task(
            _tiered_manifest_progress_spec(),
            total=total,
        ) as progress_task:
            _write_tiered_manifest_outputs(
                layout=layout,
                buckets=buckets,
                outputs=outputs,
            )
            if total:
                progress_task.advance(total)
    else:
        _write_tiered_manifest_outputs(
            layout=layout,
            buckets=buckets,
            outputs=outputs,
        )
    return tuple(outputs)


def _bucket_tier_memberships(
    decision_bundles: tuple[TiersDecisionBundle, ...],
) -> Mapping[tuple[TierName, TierMembership, SampleSplit], tuple[PassedManifestEntry, ...]]:
    mutable: dict[tuple[TierName, TierMembership, SampleSplit], list[PassedManifestEntry]] = {}
    for decision_bundle in decision_bundles:
        selected_tier = _selected_tier(decision_bundle.decision)
        for tier in TierName:
            membership = (
                TierMembership.INCLUDED
                if _is_included(selected_tier=selected_tier, tier=tier)
                else TierMembership.EXCLUDED
            )
            key = (tier, membership, decision_bundle.manifest.split)
            mutable.setdefault(key, []).append(decision_bundle.manifest)
    return {key: tuple(entries) for key, entries in mutable.items()}


def _write_tiered_manifest_outputs(
    *,
    layout: TiersLayout,
    buckets: Mapping[tuple[TierName, TierMembership, SampleSplit], tuple[PassedManifestEntry, ...]],
    outputs: list[TiersTieredManifestOutput],
) -> None:
    configured_splits = tuple(SampleSplit(split) for split in layout.config.splits)
    for tier in TierName:
        for membership in TierMembership:
            for split in configured_splits:
                path = layout.stores.runtime.manifests.tiered_manifest(
                    tier,
                    membership,
                    split,
                ).path
                entries = buckets.get((tier, membership, split), ())
                write_passed_manifest_jsonl(path, entries)
                outputs.append(
                    TiersTieredManifestOutput(
                        tier=tier.value,
                        membership=membership.value,
                        split=split.value,
                        path=path,
                    )
                )


def _selected_tier(decision: object) -> TierName | None:
    selected_tier = getattr(decision, "selected_tier", None)
    if selected_tier is None:
        return None
    try:
        return TierName(str(selected_tier))
    except ValueError as exc:
        raise TiersWorkflowInvariantError(
            f"Tier decision selected_tier is not a known tier: {selected_tier!r}"
        ) from exc


def _is_included(*, selected_tier: TierName | None, tier: TierName) -> bool:
    if selected_tier is None:
        return False
    tier_order = tuple(TierName)
    return tier_order.index(tier) <= tier_order.index(selected_tier)


def _tiered_manifest_progress_spec() -> ProgressStageSpec:
    return ProgressStageSpec(
        workflow_id=TIERS_WORKFLOW_NAME,
        stage_id=TIERS_STAGE_MANIFEST_WRITE,
        label="tiered manifest write",
        unit="entry",
        owner_module=__name__,
        split_behavior="global",
        operation_kind="tiered_manifest_write",
        total_semantics="passed manifest rows projected into tier/membership/split files",
        bar_eligible=True,
    )
