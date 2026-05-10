from __future__ import annotations

from text_to_sign_production.data.tier.leakages import (
    LeakageBundle,
    build_leakage_bundle,
    validate_leakage_bundle,
)
from text_to_sign_production.workflows.tiers.contracts import TiersWorkflowInvariantError
from text_to_sign_production.workflows.tiers.processing.models import TiersCatalogBundle


def build_tiers_leakage_bundle(catalog_bundle: TiersCatalogBundle) -> LeakageBundle:
    """Build leakage facts from passed checkpoint/sample authority."""
    leakage_bundle = build_leakage_bundle(
        catalog_bundle.payloads,
        manifests=catalog_bundle.manifests,
    )
    issues = validate_leakage_bundle(leakage_bundle)
    if issues:
        raise TiersWorkflowInvariantError(f"Leakage bundle validation failed: {issues}")
    return leakage_bundle
