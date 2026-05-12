from __future__ import annotations

from text_to_sign_production.core.ids import SampleSplit, TierMembership, TierName
from text_to_sign_production.core.progress import ProgressSession, ProgressStageSpec
from text_to_sign_production.data.tier.manifests import (
    TierManifestDecisionInput,
    TierManifestOutputTarget,
    TierManifestWritePlan,
    bucket_tier_manifest_entries,
    plan_tier_manifest_outputs,
    write_tier_manifest_plan,
)
from text_to_sign_production.workflows.foundation.provenance import written_file_receipt
from text_to_sign_production.workflows.tier.constants import (
    TIER_STAGE_MANIFEST_WRITE,
    TIER_WORKFLOW_NAME,
)
from text_to_sign_production.workflows.tier.contracts import (
    TieredWrittenManifestArtifact,
)
from text_to_sign_production.workflows.tier.layout import TierLayout
from text_to_sign_production.workflows.tier.processing.models import TierDecisionResult


def write_tiered_manifests(
    *,
    layout: TierLayout,
    decision_bundles: tuple[TierDecisionResult, ...],
    execution_id: str,
    progress_session: ProgressSession | None = None,
) -> tuple[TieredWrittenManifestArtifact, ...]:
    targets = _tiered_manifest_targets(layout)
    inputs = tuple(
        TierManifestDecisionInput(
            manifest=decision_bundle.manifest,
            decision=decision_bundle.decision,
        )
        for decision_bundle in decision_bundles
    )
    plan = _plan_tiered_manifests(targets=targets, inputs=inputs)
    total = len(decision_bundles) * len(tuple(TierName))
    if progress_session is not None:
        with progress_session.task(
            _tiered_manifest_progress_spec(),
            total=total,
        ) as progress_task:
            outputs = _write_tiered_manifest_outputs(plan, execution_id=execution_id)
            if total:
                progress_task.advance(total)
    else:
        outputs = _write_tiered_manifest_outputs(plan, execution_id=execution_id)
    return outputs


def _plan_tiered_manifests(
    *,
    targets: tuple[TierManifestOutputTarget, ...],
    inputs: tuple[TierManifestDecisionInput, ...],
) -> TierManifestWritePlan:
    buckets = bucket_tier_manifest_entries(inputs)
    return plan_tier_manifest_outputs(targets=targets, buckets=buckets)


def _write_tiered_manifest_outputs(
    plan: TierManifestWritePlan,
    *,
    execution_id: str,
) -> tuple[TieredWrittenManifestArtifact, ...]:
    write_tier_manifest_plan(plan)
    return tuple(
        _tiered_manifest_output(entry.target, execution_id=execution_id)
        for entry in plan.entries
    )


def _tiered_manifest_output(
    target: TierManifestOutputTarget,
    *,
    execution_id: str,
) -> TieredWrittenManifestArtifact:
    return TieredWrittenManifestArtifact(
        tier=target.tier.value,
        membership=target.membership.value,
        split=target.split.value,
        receipt=written_file_receipt(
            f"tiered manifest [{target.tier.value}/{target.membership.value}/{target.split.value}]",
            target.path,
            execution_id=execution_id,
            kind="tiered_manifest",
        ),
    )


def _tiered_manifest_targets(layout: TierLayout) -> tuple[TierManifestOutputTarget, ...]:
    configured_splits = tuple(SampleSplit(split) for split in layout.config.splits)
    targets: list[TierManifestOutputTarget] = []
    for tier in TierName:
        for membership in TierMembership:
            for split in configured_splits:
                targets.append(
                    TierManifestOutputTarget(
                        tier=tier,
                        membership=membership,
                        split=split,
                        path=layout.stores.runtime.manifests.tiered_manifest(
                            tier,
                            membership,
                            split,
                        ).path,
                    )
                )
    return tuple(targets)


def _tiered_manifest_progress_spec() -> ProgressStageSpec:
    return ProgressStageSpec(
        workflow_id=TIER_WORKFLOW_NAME,
        stage_id=TIER_STAGE_MANIFEST_WRITE,
        label="tiered manifest write",
        unit="entry",
        owner_module=__name__,
        split_behavior="global",
        operation_kind="tiered_manifest_write",
        total_semantics="passed manifest rows projected into tier/membership/split files",
        bar_eligible=True,
    )
