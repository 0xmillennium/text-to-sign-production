from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path

from text_to_sign_production.core.progress import (
    ProgressSession,
    ProgressStageSpec,
    ProgressTaskHandle,
)
from text_to_sign_production.data.samples import (
    ManifestWriteProgressEvent,
    ManifestWriteProgressSink,
    PassedManifestEntry,
    write_manifest_jsonl,
)
from text_to_sign_production.data.tiers import (
    TierBundle,
    TierMembership,
    TierName,
)
from text_to_sign_production.workflows.tiers.constants import (
    TIERS_STAGE_MANIFEST_WRITE,
    TIERS_WORKFLOW_NAME,
)
from text_to_sign_production.workflows.tiers.contracts import (
    TiersTieredManifestOutput,
    TiersWorkflowInvariantError,
)
from text_to_sign_production.workflows.tiers.layout import TiersLayout


def write_tiered_manifests(
    *,
    layout: TiersLayout,
    manifests: tuple[PassedManifestEntry, ...],
    tier_bundle: TierBundle,
    progress_session: ProgressSession | None = None,
) -> tuple[TiersTieredManifestOutput, ...]:
    manifest_lookup = _build_passed_manifest_lookup(manifests)
    buckets = _bucket_tier_memberships(tier_bundle, manifest_lookup)
    outputs: list[TiersTieredManifestOutput] = []
    entries_by_tier = _tiered_manifest_entry_counts(buckets, layout)
    if progress_session is not None:
        with ExitStack() as stack:
            progress_adapters = {
                tier: _ManifestWriteProgressAdapter(
                    stack.enter_context(
                        progress_session.task(
                            _tiered_manifest_progress_spec(tier),
                            total=entries_by_tier[tier],
                        )
                    )
                )
                for tier in TierName
            }
            _write_canonical_tiered_manifests(
                layout=layout,
                buckets=buckets,
                outputs=outputs,
                progress_adapters=progress_adapters,
            )
    else:
        _write_canonical_tiered_manifests(
            layout=layout,
            buckets=buckets,
            outputs=outputs,
            progress_adapters=None,
        )
    return tuple(outputs)


def _write_canonical_tiered_manifests(
    *,
    layout: TiersLayout,
    buckets: dict[tuple[object, object, str], tuple[PassedManifestEntry, ...]],
    outputs: list[TiersTieredManifestOutput],
    progress_adapters: Mapping[TierName, ManifestWriteProgressSink] | None,
) -> None:
    for tier in TierName:
        progress_adapter = progress_adapters[tier] if progress_adapters is not None else None
        for membership in TierMembership:
            for split in layout.config.splits:
                path = layout.stores.runtime.manifests.tiered_manifest(
                    tier,
                    membership,
                    split,
                ).path
                entries = tuple(buckets.get((tier, membership, split), ()))
                write_manifest_jsonl(path, entries, progress_sink=progress_adapter)
                outputs.append(
                    TiersTieredManifestOutput(
                        tier=tier.value,
                        membership=membership.value,
                        split=split,
                        path=path,
                    )
                )


def _build_passed_manifest_lookup(
    manifests: tuple[PassedManifestEntry, ...],
) -> dict[tuple[object, str], PassedManifestEntry]:
    lookup: dict[tuple[object, str], PassedManifestEntry] = {}
    for manifest in manifests:
        key = (manifest.split, manifest.sample_id)
        if key in lookup:
            raise TiersWorkflowInvariantError(
                "Duplicate passed manifest entry for split/sample: "
                f"{manifest.split}/{manifest.sample_id}"
            )
        lookup[key] = manifest
    return lookup


def _bucket_tier_memberships(
    tier_bundle: TierBundle,
    manifest_lookup: dict[tuple[object, str], PassedManifestEntry],
) -> dict[tuple[object, object, str], tuple[PassedManifestEntry, ...]]:
    mutable_buckets: dict[tuple[object, object, str], list[PassedManifestEntry]] = {}
    for membership_record in tier_bundle.memberships:
        manifest_key = (membership_record.split, membership_record.sample_id)
        manifest = manifest_lookup.get(manifest_key)
        if manifest is None:
            raise TiersWorkflowInvariantError(
                "tier membership references a sample without a passed manifest"
            )
        bucket_key = (
            membership_record.tier_name,
            membership_record.membership,
            membership_record.split.value,
        )
        mutable_buckets.setdefault(bucket_key, []).append(manifest)
    return {key: tuple(entries) for key, entries in mutable_buckets.items()}


def _tiered_manifest_entry_counts(
    buckets: dict[tuple[object, object, str], tuple[PassedManifestEntry, ...]],
    layout: TiersLayout,
) -> dict[TierName, int]:
    return {
        tier: sum(
            len(buckets.get((tier, membership, split), ()))
            for membership in TierMembership
            for split in layout.config.splits
        )
        for tier in TierName
    }


def _tiered_manifest_progress_spec(tier: TierName) -> ProgressStageSpec:
    return ProgressStageSpec(
        workflow_id=TIERS_WORKFLOW_NAME,
        stage_id=TIERS_STAGE_MANIFEST_WRITE,
        label=f"tiered manifest write [{tier.value}]",
        unit="entry",
        owner_module=__name__,
        split_behavior="global",
        operation_kind="tiered_manifest_write",
        total_semantics="manifest entries written for tier",
        bar_eligible=True,
    )


class _ManifestWriteProgressAdapter(ManifestWriteProgressSink):
    def __init__(self, progress_task: ProgressTaskHandle) -> None:
        self._progress_task = progress_task
        self._completed_by_path: dict[Path, int] = {}

    def update_manifest_write_progress(self, event: ManifestWriteProgressEvent) -> None:
        path = Path(event.path)
        completed = int(event.completed)
        previous = self._completed_by_path.get(path, 0)
        delta = completed - previous
        if delta > 0:
            self._progress_task.advance(delta)
        self._completed_by_path[path] = max(previous, completed)
