from __future__ import annotations

from contextlib import ExitStack

from text_to_sign_production.core.progress import (
    ProgressSession,
    ProgressStageSpec,
    ProgressTaskHandle,
)
from text_to_sign_production.data.leakages import LeakageBundle
from text_to_sign_production.data.metrics import MetricBundle
from text_to_sign_production.data.samples import PassedManifestEntry
from text_to_sign_production.data.tiers import (
    FilterConfig,
    TierBundle,
    TierDecisionProgressEvent,
    TierDecisionProgressSink,
    TierMembership,
    TierName,
    TierPolicy,
    build_tier_bundle,
    load_filter_config,
    load_tier_policies,
    validate_tier_bundle,
)
from text_to_sign_production.workflows.tiers.constants import (
    TIERS_STAGE_DECISION_COMPUTE,
    TIERS_WORKFLOW_NAME,
)
from text_to_sign_production.workflows.tiers.contracts import TiersWorkflowInvariantError
from text_to_sign_production.workflows.tiers.layout import TiersLayout


def build_tiers_decision_bundle(
    *,
    layout: TiersLayout,
    manifests: tuple[PassedManifestEntry, ...],
    metric_bundles: tuple[MetricBundle, ...],
    leakage_bundle: LeakageBundle,
    progress_session: ProgressSession | None = None,
) -> tuple[FilterConfig, tuple[TierPolicy, ...], TierBundle]:
    filter_config = load_filter_config(layout.runtime.filters_config_path)
    tier_policies = tuple(load_tier_policies(layout.runtime.tiers_config_path))
    sample_total = len(manifests)
    if progress_session is not None and sample_total > 0 and tier_policies:
        with ExitStack() as stack:
            progress_tasks = {
                policy.tier_name: stack.enter_context(
                    progress_session.task(
                        _tier_decision_progress_spec(policy.tier_name),
                        total=sample_total,
                    )
                )
                for policy in tier_policies
            }
            tier_bundle = build_tier_bundle(
                manifests,
                metric_bundles,
                leakage_bundle.sample_summaries,
                filter_config,
                tier_policies,
                progress_sink=_TierDecisionProgressAdapter(
                    progress_tasks=progress_tasks,
                    total=sample_total,
                ),
            )
    else:
        tier_bundle = build_tier_bundle(
            manifests,
            metric_bundles,
            leakage_bundle.sample_summaries,
            filter_config,
            tier_policies,
        )
    issues = validate_tier_bundle(tier_bundle)
    if issues:
        raise TiersWorkflowInvariantError("Tier bundle validation failed")
    return filter_config, tier_policies, tier_bundle


class _TierDecisionProgressAdapter(TierDecisionProgressSink):
    def __init__(
        self,
        *,
        progress_tasks: dict[TierName, ProgressTaskHandle],
        total: int,
    ) -> None:
        self._progress_tasks = progress_tasks
        self._total = total
        self._included = {tier_name: 0 for tier_name in progress_tasks}
        self._excluded = {tier_name: 0 for tier_name in progress_tasks}

    def update_tier_decision_progress(self, event: TierDecisionProgressEvent) -> None:
        progress_task = self._progress_tasks[event.tier_name]
        if event.membership is TierMembership.INCLUDED:
            self._included[event.tier_name] += 1
        elif event.membership is TierMembership.EXCLUDED:
            self._excluded[event.tier_name] += 1
        else:
            raise TiersWorkflowInvariantError(f"Unsupported tier membership: {event.membership}")
        progress_task.advance(
            counters=_tier_decision_progress_counters(
                included=self._included[event.tier_name],
                excluded=self._excluded[event.tier_name],
                total=self._total,
            )
        )


def _tier_decision_progress_spec(tier_name: TierName) -> ProgressStageSpec:
    return ProgressStageSpec(
        workflow_id=TIERS_WORKFLOW_NAME,
        stage_id=TIERS_STAGE_DECISION_COMPUTE,
        label=f"tier decision [{tier_name.value}]",
        unit="sample",
        owner_module=__name__,
        split_behavior="global",
        operation_kind="tier_decision_compute",
        total_semantics="processed samples for tier",
        bar_eligible=True,
        allowed_counters=("included", "excluded", "include_rate", "exclude_rate"),
    )


def _tier_decision_progress_counters(
    *,
    included: int,
    excluded: int,
    total: int,
) -> dict[str, object]:
    return {
        "included": included,
        "excluded": excluded,
        "include_rate": _format_percent(included, total),
        "exclude_rate": _format_percent(excluded, total),
    }


def _format_percent(count: int, total: int) -> str:
    if total <= 0:
        return "0.0%"
    return f"{count / total * 100:.1f}%"
