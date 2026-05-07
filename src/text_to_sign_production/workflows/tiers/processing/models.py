from __future__ import annotations

from dataclasses import dataclass

from text_to_sign_production.artifacts.catalog import (
    SampleHandle,
    SamplesCatalog,
)
from text_to_sign_production.data.leakages import LeakageBundle
from text_to_sign_production.data.metrics import MetricBundle
from text_to_sign_production.data.samples import PassedManifestEntry
from text_to_sign_production.data.tiers import (
    FilterConfig,
    TierBundle,
    TierPolicy,
)
from text_to_sign_production.workflows.tiers.contracts import TiersWorkflowResult


@dataclass(frozen=True, slots=True)
class TiersCatalogBundle:
    catalog: SamplesCatalog
    handles: tuple[SampleHandle, ...]
    manifests: tuple[PassedManifestEntry, ...]

    @property
    def processed_count(self) -> int:
        return len(self.handles)


@dataclass(frozen=True, slots=True)
class TiersExecutionBundle:
    workflow_result: TiersWorkflowResult
    catalog_bundle: TiersCatalogBundle
    filter_config: FilterConfig
    tier_policies: tuple[TierPolicy, ...]
    metric_bundles: tuple[MetricBundle, ...]
    leakage_bundle: LeakageBundle
    tier_bundle: TierBundle

    @property
    def processed_count(self) -> int:
        return self.catalog_bundle.processed_count
