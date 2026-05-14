from __future__ import annotations

from text_to_sign_production.core.progress import ProgressSession, ProgressStageSpec
from text_to_sign_production.data.tier.leakages import (
    LeakageBundle,
    LeakageProgressSpecs,
    build_leakage_bundle,
    validate_leakage_bundle,
)
from text_to_sign_production.workflows.tier.constants import (
    TIER_STAGE_LEAKAGE_DUPLICATES,
    TIER_STAGE_LEAKAGE_RELATIONS,
    TIER_STAGE_LEAKAGE_SUMMARIES,
    TIER_WORKFLOW_NAME,
)
from text_to_sign_production.workflows.tier.contracts import TierWorkflowInvariantError
from text_to_sign_production.workflows.tier.processing.models import TierCatalogBundle


def build_tier_leakage_bundle(
    catalog_bundle: TierCatalogBundle,
    *,
    progress_session: ProgressSession | None = None,
) -> LeakageBundle:
    """Build leakage facts from passed checkpoint/sample authority."""
    leakage_bundle = build_leakage_bundle(
        catalog_bundle.payloads,
        manifests=catalog_bundle.manifests,
        progress_session=progress_session,
        progress_specs=LeakageProgressSpecs(
            duplicate_check=_leakage_duplicate_progress_spec(),
            relation_scan=_leakage_relation_progress_spec(),
            summary_build=_leakage_summary_progress_spec(),
        ),
    )
    issues = validate_leakage_bundle(leakage_bundle)
    if issues:
        raise TierWorkflowInvariantError(f"Leakage bundle validation failed: {issues}")
    return leakage_bundle


def _leakage_duplicate_progress_spec() -> ProgressStageSpec:
    return ProgressStageSpec(
        workflow_id=TIER_WORKFLOW_NAME,
        stage_id=TIER_STAGE_LEAKAGE_DUPLICATES,
        label="leakage duplicate check",
        unit="sample",
        owner_module=__name__,
        split_behavior="global",
        operation_kind="leakage_duplicate_check",
        total_semantics="leakage input keys checked for duplicates",
        bar_eligible=True,
    )


def _leakage_relation_progress_spec() -> ProgressStageSpec:
    return ProgressStageSpec(
        workflow_id=TIER_WORKFLOW_NAME,
        stage_id=TIER_STAGE_LEAKAGE_RELATIONS,
        label="leakage relations",
        unit="pair",
        owner_module=__name__,
        split_behavior="global",
        operation_kind="leakage_relation_scan",
        total_semantics="accepted sample pairs scanned for leakage relations",
        bar_eligible=True,
    )


def _leakage_summary_progress_spec() -> ProgressStageSpec:
    return ProgressStageSpec(
        workflow_id=TIER_WORKFLOW_NAME,
        stage_id=TIER_STAGE_LEAKAGE_SUMMARIES,
        label="leakage summaries",
        unit="sample",
        owner_module=__name__,
        split_behavior="global",
        operation_kind="leakage_summary_build",
        total_semantics="accepted sample leakage summaries built",
        bar_eligible=True,
    )
