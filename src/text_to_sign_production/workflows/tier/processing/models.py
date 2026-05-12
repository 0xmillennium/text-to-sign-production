from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.artifacts.catalog import SampleHandle, SamplesCatalog
from text_to_sign_production.core.models import (
    CheckpointAdmission,
    PassedManifestEntry,
    PreparedSample,
)
from text_to_sign_production.core.models import (
    TierDecisionBundle as DomainTierDecisionBundle,
)
from text_to_sign_production.data.tier.context import QualityContext
from text_to_sign_production.data.tier.facts import QualityFacts
from text_to_sign_production.data.tier.families.types import QualityMetricBundle
from text_to_sign_production.data.tier.leakages import LeakageBundle
from text_to_sign_production.data.tier.policies.filters import TierFiltersConfig
from text_to_sign_production.data.tier.policies.policies import TierPoliciesConfig
from text_to_sign_production.data.tier.reports import (
    TierReportBundle as DomainTierReportBundle,
)
from text_to_sign_production.workflows.tier.contracts import TierWorkflowResult
from text_to_sign_production.workflows.tier.contracts.results import (
    TieredWrittenManifestArtifact,
)


@dataclass(frozen=True, slots=True)
class TierSampleBundle:
    """One passed checkpoint row aligned with its PreparedSample payload."""

    handle: SampleHandle
    manifest: PassedManifestEntry
    source_manifest_path: Path
    source_manifest_sha256: str
    sample: PreparedSample


@dataclass(frozen=True, slots=True)
class TierCatalogBundle:
    """Checkpoint-only catalog and payload load result for tier processing."""

    catalog: SamplesCatalog
    samples: tuple[TierSampleBundle, ...]

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
class TierQualityBundle:
    """PreparedSample-derived quality continuation for one accepted sample."""

    sample: PreparedSample
    manifest: PassedManifestEntry
    source_manifest_path: Path
    source_manifest_sha256: str
    facts: QualityFacts
    context: QualityContext
    metrics: QualityMetricBundle


@dataclass(frozen=True, slots=True)
class TierDecisionResult:
    """Tier decision attached to the sample identity owned by checkpoint rows."""

    sample: PreparedSample
    manifest: PassedManifestEntry
    checkpoint_admission: CheckpointAdmission
    decision: DomainTierDecisionBundle


@dataclass(frozen=True, slots=True)
class TierReportResult:
    """Tier report bundle attached to checkpoint sample identity."""

    sample: PreparedSample
    manifest: PassedManifestEntry
    report: DomainTierReportBundle


@dataclass(frozen=True, slots=True)
class TierExecutionBundle:
    workflow_result: TierWorkflowResult
    catalog_bundle: TierCatalogBundle
    filter_config: TierFiltersConfig
    tier_policies: TierPoliciesConfig
    quality_bundles: tuple[TierQualityBundle, ...]
    leakage_bundle: LeakageBundle
    decision_bundles: tuple[TierDecisionResult, ...]
    written_tiered_manifest_artifacts: tuple[TieredWrittenManifestArtifact, ...]
    tier_reports: tuple[TierReportResult, ...]

    @property
    def processed_count(self) -> int:
        return self.catalog_bundle.processed_count


__all__ = [
    "TierCatalogBundle",
    "TierDecisionResult",
    "TierExecutionBundle",
    "TierQualityBundle",
    "TierReportResult",
    "TierSampleBundle",
]
