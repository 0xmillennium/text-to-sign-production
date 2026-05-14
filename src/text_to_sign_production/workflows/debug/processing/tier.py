from __future__ import annotations

from text_to_sign_production.core.ids import TierMembership
from text_to_sign_production.core.integrity import sha256_file
from text_to_sign_production.core.models import CheckpointAdmission
from text_to_sign_production.core.progress import ProgressSession, ProgressStageSpec
from text_to_sign_production.data.dataset.payloads import load_prepared_sample_payload
from text_to_sign_production.data.dataset.validate import validate_payload_manifest_coherence
from text_to_sign_production.data.tier.policies.analysis import tier_membership_for_decision
from text_to_sign_production.data.tier.policies.compute import evaluate_tier_decision
from text_to_sign_production.data.tier.policies.filters import load_tier_filters_config
from text_to_sign_production.data.tier.policies.policies import load_tier_policies_config
from text_to_sign_production.data.tier.policies.validate import validate_tier_decision_bundle
from text_to_sign_production.data.tier.families.analysis import (
    binding_metric_summary,
    diagnostic_metric_summary,
)
from text_to_sign_production.data.tier.quality import compute_tier_quality
from text_to_sign_production.workflows.debug.contracts import (
    DebugGateResult,
    DebugExistingTierStateSummary,
    DebugLeakageContext,
    DebugSampleDossier,
    DebugTierComparison,
    DebugTierComputedMembership,
    DebugTierFamilyDecisionSummary,
    DebugTierQualityMetricSummary,
    DebugTierRecomputeSource,
    DebugTierResult,
    DebugWorkflowConfig,
    ExistingGateStatus,
    ExistingTierMembershipState,
)
from text_to_sign_production.workflows.debug.contracts.verdicts import (
    DebugQualityMetricStatus,
    DebugTierVerdict,
)
from text_to_sign_production.workflows.debug.constants import (
    DEBUG_STAGE_TIER_DECISION,
    DEBUG_STAGE_TIER_QUALITY,
)
from text_to_sign_production.workflows.debug.layout import DebugLayout
from text_to_sign_production.workflows.debug.processing.leakage import (
    build_leakage_context,
    manifest_only_sample_leakage_summary,
)
from text_to_sign_production.workflows.debug.progress import debug_progress_stage


def debug_tier(
    config: DebugWorkflowConfig,
    layout: DebugLayout,
    dossier: DebugSampleDossier,
    gate_result: DebugGateResult,
    *,
    progress_session: ProgressSession | None = None,
) -> DebugTierResult:
    del gate_result
    leakage_context = build_leakage_context(dossier.resolution.request.debug_splits)
    if dossier.existing_gate.status is not ExistingGateStatus.PASSED:
        return _not_applicable(
            leakage_context,
            "production tier pipeline only processes passed samples",
            existing_tier_states=_existing_tier_states(dossier),
        )
    source = _recompute_source(dossier)
    if not source.can_compute:
        return DebugTierResult(
            applicable=True,
            applicability_reason="existing gate status is passed",
            leakage_context=leakage_context,
            recompute_source=source,
            existing_tier_states=_existing_tier_states(dossier),
            leakage_summary=None,
            computed_decision=None,
            family_decisions=(),
            quality_metrics=(),
            comparison=DebugTierComparison(False, (), (), (source.reason or "cannot compute",)),
            verdict=DebugTierVerdict.COMPUTE_FAILED,
            errors=(source.reason or "tier recompute source unavailable",),
            warnings=_context_warnings(leakage_context),
        )
    try:
        assert source.payload_path is not None
        assert source.manifest_entry is not None
        sample = load_prepared_sample_payload(source.payload_path)
        coherence_issues = validate_payload_manifest_coherence(sample, source.manifest_entry)
        if coherence_issues:
            raise ValueError(
                f"PreparedSample payload and passed manifest differ: {coherence_issues}"
            )
        quality = _compute_quality(sample, progress_session=progress_session)
        leakage_summary = manifest_only_sample_leakage_summary(
            layout,
            debug_splits=dossier.resolution.request.debug_splits,
            split=source.manifest_entry.split,
            sample_id=source.manifest_entry.sample_id,
            progress_session=progress_session,
        )
        checkpoint_admission = CheckpointAdmission(
            sample_id=source.manifest_entry.sample_id,
            split=source.manifest_entry.split,
            source_manifest_path=dossier.existing_gate.passed_manifest_path,
            source_manifest_sha256=sha256_file(dossier.existing_gate.passed_manifest_path),
            payload_ref=source.manifest_entry.payload_ref,
            gate_detail_available=False,
        )
        decision = _compute_decision(
            sample=sample,
            quality=quality,
            leakage_summary=leakage_summary,
            checkpoint_admission=checkpoint_admission,
            filter_config=load_tier_filters_config(config.filters_config_runtime_path),
            tier_policies=load_tier_policies_config(config.tiers_config_runtime_path),
            progress_session=progress_session,
        )
        decision_issues = validate_tier_decision_bundle(decision, checkpoint_admission)
        if decision_issues:
            raise ValueError(f"Tier decision validation failed: {decision_issues}")
        family_decisions = _family_decision_summaries(decision)
        quality_metrics = _quality_metric_summaries(quality)
    except (OSError, TypeError, ValueError, KeyError) as exc:
        return DebugTierResult(
            applicable=True,
            applicability_reason="existing gate status is passed",
            leakage_context=leakage_context,
            recompute_source=source,
            existing_tier_states=_existing_tier_states(dossier),
            leakage_summary=None,
            computed_decision=None,
            family_decisions=(),
            quality_metrics=(),
            comparison=DebugTierComparison(False, (), (), (str(exc),)),
            verdict=DebugTierVerdict.COMPUTE_FAILED,
            errors=(str(exc),),
            warnings=_context_warnings(leakage_context),
        )
    comparison = _compare_tier(dossier, decision, leakage_context)
    verdict = _tier_verdict(comparison, leakage_context)
    return DebugTierResult(
        applicable=True,
        applicability_reason="existing gate status is passed",
        leakage_context=leakage_context,
        recompute_source=source,
        existing_tier_states=_existing_tier_states(dossier),
        leakage_summary=leakage_summary,
        computed_decision=decision,
        family_decisions=family_decisions,
        quality_metrics=quality_metrics,
        comparison=comparison,
        verdict=verdict,
        errors=(),
        warnings=(
            *_context_warnings(leakage_context),
            *_context_limited_difference_warning(verdict),
            *comparison.warnings,
        ),
    )


def _not_applicable(
    leakage_context: DebugLeakageContext,
    reason: str,
    *,
    existing_tier_states: tuple[DebugExistingTierStateSummary, ...],
) -> DebugTierResult:
    return DebugTierResult(
        applicable=False,
        applicability_reason=reason,
        leakage_context=leakage_context,
        recompute_source=DebugTierRecomputeSource(None, False, None, False, reason),
        existing_tier_states=existing_tier_states,
        leakage_summary=None,
        computed_decision=None,
        family_decisions=(),
        quality_metrics=(),
        comparison=DebugTierComparison(False, (), (), (reason,)),
        verdict=DebugTierVerdict.NOT_APPLICABLE,
        errors=(),
        warnings=_context_warnings(leakage_context),
    )


def _recompute_source(dossier: DebugSampleDossier) -> DebugTierRecomputeSource:
    payload_path = dossier.existing_gate.passed_payload_path
    entry = dossier.existing_gate.passed_entry
    if entry is None:
        return DebugTierRecomputeSource(
            payload_path,
            False,
            None,
            False,
            "existing gate says passed but passed manifest entry is unavailable",
        )
    if payload_path is None or not payload_path.is_file():
        return DebugTierRecomputeSource(
            payload_path,
            False,
            entry,
            False,
            "existing gate says passed but payload is unavailable",
        )
    return DebugTierRecomputeSource(payload_path, True, entry, True, None)


def _existing_tier_states(
    dossier: DebugSampleDossier,
) -> tuple[DebugExistingTierStateSummary, ...]:
    return tuple(
        DebugExistingTierStateSummary(
            tier=evidence.tier,
            state=evidence.state,
            included_manifest_path=evidence.included_manifest_path,
            excluded_manifest_path=evidence.excluded_manifest_path,
            included_entry=evidence.included_entry,
            excluded_entry=evidence.excluded_entry,
            warnings=evidence.warnings,
            errors=evidence.errors,
        )
        for evidence in dossier.existing_tiers
    )


def _compare_tier(
    dossier: DebugSampleDossier,
    decision,
    leakage_context: DebugLeakageContext,
) -> DebugTierComparison:
    del leakage_context
    memberships: list[DebugTierComputedMembership] = []
    differences: list[str] = []
    warnings: list[str] = []
    for evidence in dossier.existing_tiers:
        computed_membership = tier_membership_for_decision(decision, evidence.tier)
        computed_state = (
            ExistingTierMembershipState.INCLUDED
            if computed_membership is TierMembership.INCLUDED
            else ExistingTierMembershipState.EXCLUDED
        )
        matches = computed_state is evidence.state
        memberships.append(
            DebugTierComputedMembership(
                tier=evidence.tier,
                computed_state=computed_state,
                existing_state=evidence.state,
                matches=matches,
            )
        )
        if evidence.state is ExistingTierMembershipState.AMBIGUOUS:
            differences.append(f"{evidence.tier.value}: existing tier membership is ambiguous")
        elif evidence.state is ExistingTierMembershipState.MISSING:
            differences.append(f"{evidence.tier.value}: existing tier membership is missing")
        elif not matches:
            differences.append(
                f"{evidence.tier.value}: computed={computed_state.value}, "
                f"existing={evidence.state.value}"
            )
    return DebugTierComparison(
        compared=True,
        memberships=tuple(memberships),
        differences=tuple(differences),
        warnings=tuple(warnings),
    )


def _family_decision_summaries(decision) -> tuple[DebugTierFamilyDecisionSummary, ...]:
    return tuple(
        DebugTierFamilyDecisionSummary(
            family=_family_label(item.family),
            status=item.status,
            supported_tiers=tuple(item.supported_tiers),
            best_supported_tier=item.best_supported_tier,
            issue_codes=tuple(sorted(issue.code.value for issue in item.issues)),
            reasons=tuple(issue.message for issue in item.issues),
        )
        for item in sorted(decision.family_decisions, key=lambda entry: _family_label(entry.family))
    )


def _quality_metric_summaries(quality) -> tuple[DebugTierQualityMetricSummary, ...]:
    summaries: list[DebugTierQualityMetricSummary] = []
    family_summaries = (
        *binding_metric_summary(quality.metrics),
        *diagnostic_metric_summary(quality.metrics),
    )
    for family_summary in family_summaries:
        family = _family_label(family_summary.family)
        for metric in family_summary.metrics:
            summaries.append(
                DebugTierQualityMetricSummary(
                    metric_name=f"{family}.{metric.name}",
                    value=metric.value,
                    status=DebugQualityMetricStatus.OBSERVED,
                    notes=(),
                )
            )
    return tuple(sorted(summaries, key=lambda item: item.metric_name))


def _family_label(family) -> str:
    return family.value if hasattr(family, "value") else str(family)


def _tier_verdict(
    comparison: DebugTierComparison,
    leakage_context: DebugLeakageContext,
) -> DebugTierVerdict:
    if any(
        item.existing_state is ExistingTierMembershipState.AMBIGUOUS
        for item in comparison.memberships
    ):
        return DebugTierVerdict.EXISTING_AMBIGUOUS
    if any(
        item.existing_state is ExistingTierMembershipState.MISSING
        for item in comparison.memberships
    ):
        return DebugTierVerdict.EXISTING_MISSING
    if not comparison.differences:
        return DebugTierVerdict.MATCH
    only_context_limited = all(
        item.computed_state is ExistingTierMembershipState.INCLUDED
        and item.existing_state is ExistingTierMembershipState.EXCLUDED
        for item in comparison.memberships
        if not item.matches
    )
    if only_context_limited and not leakage_context.production_like:
        return DebugTierVerdict.CONTEXT_LIMITED_DIFFERENCE
    return DebugTierVerdict.MISMATCH


def _context_warnings(context: DebugLeakageContext) -> tuple[str, ...]:
    return (context.warning,) if context.warning else ()


def _context_limited_difference_warning(verdict: DebugTierVerdict) -> tuple[str, ...]:
    if verdict is not DebugTierVerdict.CONTEXT_LIMITED_DIFFERENCE:
        return ()
    return (
        "This sample satisfies tier criteria in the selected debug context. "
        "However, the existing tier output excludes it. "
        "Because DEBUG_SPLITS does not include the full train/val/test context, "
        "cross-split leakage may not be observable. "
        'Rerun with: DEBUG_SPLITS = ("train", "val", "test") '
        "to verify leakage-driven exclusion.",
    )


def _compute_quality(sample, *, progress_session: ProgressSession | None):
    if progress_session is not None:
        with progress_session.task(_tier_quality_progress_spec(), total=1) as progress_task:
            quality = compute_tier_quality(sample)
            progress_task.advance()
            return quality
    return compute_tier_quality(sample)


def _compute_decision(
    *,
    sample,
    quality,
    leakage_summary,
    checkpoint_admission,
    filter_config,
    tier_policies,
    progress_session: ProgressSession | None,
):
    if progress_session is not None:
        with progress_session.task(_tier_decision_progress_spec(), total=1) as progress_task:
            decision = evaluate_tier_decision(
                sample,
                quality.facts,
                quality.context,
                quality.metrics,
                leakage_summary,
                checkpoint_admission,
                filter_config,
                tier_policies,
            )
            progress_task.advance()
            return decision
    return evaluate_tier_decision(
        sample,
        quality.facts,
        quality.context,
        quality.metrics,
        leakage_summary,
        checkpoint_admission,
        filter_config,
        tier_policies,
    )


def _tier_quality_progress_spec() -> ProgressStageSpec:
    return debug_progress_stage(
        stage_id=DEBUG_STAGE_TIER_QUALITY,
        label="tier quality",
        unit="sample",
        owner_module=__name__,
        split_behavior="per_split",
        operation_kind="tier_quality_compute",
        total_semantics="target sample tier quality metrics computed",
    )


def _tier_decision_progress_spec() -> ProgressStageSpec:
    return debug_progress_stage(
        stage_id=DEBUG_STAGE_TIER_DECISION,
        label="tier decision",
        unit="sample",
        owner_module=__name__,
        split_behavior="per_split",
        operation_kind="tier_decision_compute",
        total_semantics="target sample tier decision computed",
    )


__all__ = ["debug_tier"]
