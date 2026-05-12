from __future__ import annotations

from text_to_sign_production.core.models import (
    CheckpointAdmission,
)
from text_to_sign_production.core.progress import ProgressSession, ProgressStageSpec
from text_to_sign_production.data.tier.leakages import LeakageBundle, sample_leakage_summary
from text_to_sign_production.data.tier.policies.compute import evaluate_tier_decision
from text_to_sign_production.data.tier.policies.filters import (
    TierFiltersConfig,
    load_tier_filters_config,
)
from text_to_sign_production.data.tier.policies.policies import (
    TierPoliciesConfig,
    load_tier_policies_config,
)
from text_to_sign_production.data.tier.policies.validate import validate_tier_decision_bundle
from text_to_sign_production.workflows.tier.constants import (
    TIER_STAGE_DECISION_COMPUTE,
    TIER_WORKFLOW_NAME,
)
from text_to_sign_production.workflows.tier.contracts import TierWorkflowInvariantError
from text_to_sign_production.workflows.tier.layout import TierLayout
from text_to_sign_production.workflows.tier.processing.models import (
    TierDecisionResult,
    TierQualityBundle,
)


def build_tier_decision_bundles(
    *,
    layout: TierLayout,
    quality_bundles: tuple[TierQualityBundle, ...],
    leakage_bundle: LeakageBundle,
    progress_session: ProgressSession | None = None,
) -> tuple[TierFiltersConfig, TierPoliciesConfig, tuple[TierDecisionResult, ...]]:
    filter_config = load_tier_filters_config(layout.runtime.filters_config_path)
    tier_policies = load_tier_policies_config(layout.runtime.tier_config_path)

    decision_bundles: list[TierDecisionResult] = []
    if progress_session is not None and quality_bundles:
        with progress_session.task(
            _tier_decision_progress_spec(),
            total=len(quality_bundles),
        ) as progress_task:
            for quality_bundle in quality_bundles:
                decision_bundles.append(
                    _evaluate_quality_bundle(
                        quality_bundle=quality_bundle,
                        leakage_bundle=leakage_bundle,
                        filter_config=filter_config,
                        tier_policies=tier_policies,
                    )
                )
                progress_task.advance()
    else:
        decision_bundles = [
            _evaluate_quality_bundle(
                quality_bundle=quality_bundle,
                leakage_bundle=leakage_bundle,
                filter_config=filter_config,
                tier_policies=tier_policies,
            )
            for quality_bundle in quality_bundles
        ]
    return filter_config, tier_policies, tuple(decision_bundles)


def _evaluate_quality_bundle(
    *,
    quality_bundle: TierQualityBundle,
    leakage_bundle: LeakageBundle,
    filter_config: TierFiltersConfig,
    tier_policies: TierPoliciesConfig,
) -> TierDecisionResult:
    checkpoint_admission = _checkpoint_admission(quality_bundle)
    leakage_summary = sample_leakage_summary(
        leakage_bundle,
        split=quality_bundle.manifest.split,
        sample_id=quality_bundle.manifest.sample_id,
    )
    decision = evaluate_tier_decision(
        quality_bundle.sample,
        quality_bundle.facts,
        quality_bundle.context,
        quality_bundle.metrics,
        leakage_summary,
        checkpoint_admission,
        filter_config,
        tier_policies,
    )
    issues = validate_tier_decision_bundle(decision, checkpoint_admission)
    if issues:
        raise TierWorkflowInvariantError(
            "Tier decision validation failed for "
            f"{quality_bundle.manifest.split.value}/{quality_bundle.manifest.sample_id}: "
            f"{issues}"
        )
    return TierDecisionResult(
        sample=quality_bundle.sample,
        manifest=quality_bundle.manifest,
        checkpoint_admission=checkpoint_admission,
        decision=decision,
    )


def _checkpoint_admission(quality_bundle: TierQualityBundle) -> CheckpointAdmission:
    """Carry checkpoint admission without fabricating gate predicate truth."""
    return CheckpointAdmission(
        sample_id=quality_bundle.sample.source.sample_id,
        split=quality_bundle.sample.source.split,
        source_manifest_path=quality_bundle.source_manifest_path,
        source_manifest_sha256=quality_bundle.source_manifest_sha256,
        payload_ref=quality_bundle.manifest.payload_ref,
        gate_detail_available=False,
    )


def _tier_decision_progress_spec() -> ProgressStageSpec:
    return ProgressStageSpec(
        workflow_id=TIER_WORKFLOW_NAME,
        stage_id=TIER_STAGE_DECISION_COMPUTE,
        label="tier decision",
        unit="sample",
        owner_module=__name__,
        split_behavior="global",
        operation_kind="tier_decision_compute",
        total_semantics="PreparedSample-based tier decisions",
        bar_eligible=True,
    )
