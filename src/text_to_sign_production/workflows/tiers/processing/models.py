from __future__ import annotations

from dataclasses import dataclass

from text_to_sign_production.artifacts.catalog import SampleHandle, SamplesCatalog
from text_to_sign_production.core.models import PassedManifestEntry, PreparedSample
from text_to_sign_production.data.tier.context import QualityContext
from text_to_sign_production.data.tier.facts import QualityFacts
from text_to_sign_production.data.tier.families import QualityMetricBundle
from text_to_sign_production.data.tier.leakages import LeakageBundle
from text_to_sign_production.workflows.tiers.contracts import TiersWorkflowResult
from text_to_sign_production.workflows.tiers.contracts.results import (
    TiersTieredManifestOutput,
)


@dataclass(frozen=True, slots=True)
class TiersSampleBundle:
    """One passed checkpoint row aligned with its PreparedSample payload."""

    handle: SampleHandle
    manifest: PassedManifestEntry
    sample: PreparedSample


@dataclass(frozen=True, slots=True)
class TiersCatalogBundle:
    """Checkpoint-only catalog and payload load result for tiers processing."""

    catalog: SamplesCatalog
    samples: tuple[TiersSampleBundle, ...]

    @property
    def handles(self) -> tuple[SampleHandle, ...]:
        return tuple(sample.handle for sample in self.samples)

    @property
    def manifests(self) -> tuple[PassedManifestEntry, ...]:
        return tuple(sample.manifest for sample in self.samples)

    @property
    def payloads(self) -> tuple[PreparedSample, ...]:
        return tuple(sample.sample for sample in self.samples)

    @property
    def processed_count(self) -> int:
        return len(self.samples)


@dataclass(frozen=True, slots=True)
class TiersQualityBundle:
    """PreparedSample-derived quality continuation for one accepted sample."""

    sample: PreparedSample
    manifest: PassedManifestEntry
    facts: QualityFacts
    context: QualityContext
    metrics: QualityMetricBundle


@dataclass(frozen=True, slots=True)
class TiersDecisionBundle:
    """Tier decision attached to the sample identity owned by checkpoint rows."""

    sample: PreparedSample
    manifest: PassedManifestEntry
    decision: object


@dataclass(frozen=True, slots=True)
class TiersReportBundle:
    """Quality report bundle attached to checkpoint sample identity."""

    sample: PreparedSample
    manifest: PassedManifestEntry
    report: object


@dataclass(frozen=True, slots=True)
class TiersExecutionBundle:
    workflow_result: TiersWorkflowResult
    catalog_bundle: TiersCatalogBundle
    filter_config: object
    tier_policies: object
    quality_bundles: tuple[TiersQualityBundle, ...]
    leakage_bundle: LeakageBundle
    decision_bundles: tuple[TiersDecisionBundle, ...]
    tiered_manifest_outputs: tuple[TiersTieredManifestOutput, ...]
    quality_reports: tuple[TiersReportBundle, ...]

    @property
    def processed_count(self) -> int:
        return self.catalog_bundle.processed_count


__all__ = [
    "TiersCatalogBundle",
    "TiersDecisionBundle",
    "TiersExecutionBundle",
    "TiersQualityBundle",
    "TiersReportBundle",
    "TiersSampleBundle",
]
