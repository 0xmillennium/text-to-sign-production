from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any, cast

from text_to_sign_production.core.ids import SampleStatus
from text_to_sign_production.core.models import (
    GateDecision,
    GateDecisionBundle,
    GateName,
    GateStatus,
    PreparedSample,
)
from text_to_sign_production.core.progress import ProgressSession, ProgressStageSpec
from text_to_sign_production.workflows.tiers.constants import (
    TIERS_STAGE_DECISION_COMPUTE,
    TIERS_WORKFLOW_NAME,
)
from text_to_sign_production.workflows.tiers.contracts import TiersWorkflowInvariantError
from text_to_sign_production.workflows.tiers.layout import TiersLayout
from text_to_sign_production.workflows.tiers.processing.models import (
    TiersDecisionBundle,
    TiersQualityBundle,
)


def build_tiers_decision_bundles(
    *,
    layout: TiersLayout,
    quality_bundles: tuple[TiersQualityBundle, ...],
    leakage_bundle: object,
    progress_session: ProgressSession | None = None,
) -> tuple[object, object, tuple[TiersDecisionBundle, ...]]:
    owner_api = _load_tier_owner_api()
    filter_config = owner_api["load_tier_filters_config"](layout.runtime.filters_config_path)
    tier_policies = owner_api["load_tier_policies_config"](layout.runtime.tiers_config_path)

    decision_bundles: list[TiersDecisionBundle] = []
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
                        owner_api=owner_api,
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
                owner_api=owner_api,
            )
            for quality_bundle in quality_bundles
        ]
    return filter_config, tier_policies, tuple(decision_bundles)


def _evaluate_quality_bundle(
    *,
    quality_bundle: TiersQualityBundle,
    leakage_bundle: object,
    filter_config: object,
    tier_policies: object,
    owner_api: dict[str, Callable[..., Any]],
) -> TiersDecisionBundle:
    gate_context = _passed_gate_admission_context(quality_bundle.sample)
    decision = owner_api["evaluate_quality_tiers"](
        quality_bundle.sample,
        quality_bundle.facts,
        quality_bundle.context,
        quality_bundle.metrics,
        leakage_bundle,
        gate_context,
        filter_config,
        tier_policies,
    )
    issues = owner_api["validate_tier_decision_bundle"](decision, gate_context)
    if issues:
        raise TiersWorkflowInvariantError(
            "Tier decision validation failed for "
            f"{quality_bundle.manifest.split.value}/{quality_bundle.manifest.sample_id}: "
            f"{issues}"
        )
    return TiersDecisionBundle(
        sample=quality_bundle.sample,
        manifest=quality_bundle.manifest,
        decision=decision,
    )


def _passed_gate_admission_context(sample: PreparedSample) -> GateDecisionBundle:
    """Carry checkpoint admission as context without rerunning gate predicates."""
    return GateDecisionBundle(
        sample_id=sample.source.sample_id,
        split=sample.source.split,
        final_status=SampleStatus.PASSED,
        terminal_gate=None,
        decisions=tuple(
            GateDecision(gate=gate, status=GateStatus.PASS, issue_codes=()) for gate in GateName
        ),
        failed_gates=(),
    )


def _load_tier_owner_api() -> dict[str, Callable[..., Any]]:
    try:
        tiers = import_module("text_to_sign_production.data.quality.policies.tiers")
    except ModuleNotFoundError as exc:
        raise TiersWorkflowInvariantError(
            "Unable to import data.quality.policies.tiers owner package."
        ) from exc
    return {
        "evaluate_quality_tiers": cast(Callable[..., Any], tiers.evaluate_quality_tiers),
        "load_tier_filters_config": cast(Callable[..., Any], tiers.load_tier_filters_config),
        "load_tier_policies_config": cast(Callable[..., Any], tiers.load_tier_policies_config),
        "validate_tier_decision_bundle": cast(
            Callable[..., Any],
            tiers.validate_tier_decision_bundle,
        ),
    }


def _tier_decision_progress_spec() -> ProgressStageSpec:
    return ProgressStageSpec(
        workflow_id=TIERS_WORKFLOW_NAME,
        stage_id=TIERS_STAGE_DECISION_COMPUTE,
        label="tier decision",
        unit="sample",
        owner_module=__name__,
        split_behavior="global",
        operation_kind="tier_decision_compute",
        total_semantics="PreparedSample-based tier decisions",
        bar_eligible=True,
    )
