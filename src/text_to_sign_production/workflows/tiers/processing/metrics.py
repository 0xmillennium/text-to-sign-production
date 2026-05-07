from __future__ import annotations

from text_to_sign_production.core.progress import ProgressSession, ProgressStageSpec
from text_to_sign_production.data.metrics import (
    MetricBundle,
    build_metric_bundle,
    validate_metric_bundle,
)
from text_to_sign_production.data.samples import load_processed_sample_payload
from text_to_sign_production.workflows.tiers.constants import (
    TIERS_STAGE_METRIC_COMPUTE,
    TIERS_WORKFLOW_NAME,
)
from text_to_sign_production.workflows.tiers.contracts import TiersWorkflowInvariantError
from text_to_sign_production.workflows.tiers.processing.models import TiersCatalogBundle


def build_tiers_metric_bundles(
    catalog_bundle: TiersCatalogBundle,
    *,
    progress_session: ProgressSession | None = None,
) -> tuple[MetricBundle, ...]:
    metric_bundles: list[MetricBundle] = []
    if progress_session is not None and catalog_bundle.processed_count > 0:
        with progress_session.task(
            _metric_compute_progress_spec(),
            total=catalog_bundle.processed_count,
        ) as progress_task:
            for handle, manifest in zip(
                catalog_bundle.handles,
                catalog_bundle.manifests,
                strict=True,
            ):
                metric_bundles.append(_build_metric_bundle(handle, manifest))
                progress_task.advance()
    else:
        for handle, manifest in zip(
            catalog_bundle.handles,
            catalog_bundle.manifests,
            strict=True,
        ):
            metric_bundles.append(_build_metric_bundle(handle, manifest))
    return tuple(metric_bundles)


def _build_metric_bundle(handle: object, manifest: object) -> MetricBundle:
    if handle.runtime_sample is None:
        raise TiersWorkflowInvariantError(
            "Missing runtime sample for passed sample: "
            f"{handle.ref.split}/{handle.ref.sample_id}"
        )
    payload = load_processed_sample_payload(handle.runtime_sample.path)
    metric_bundle = build_metric_bundle(payload, manifest)
    issues = validate_metric_bundle(metric_bundle)
    if issues:
        raise TiersWorkflowInvariantError(
            f"Metric bundle validation failed for {manifest.split}/{manifest.sample_id}"
        )
    return metric_bundle


def _metric_compute_progress_spec() -> ProgressStageSpec:
    return ProgressStageSpec(
        workflow_id=TIERS_WORKFLOW_NAME,
        stage_id=TIERS_STAGE_METRIC_COMPUTE,
        label="metric compute",
        unit="sample",
        owner_module=__name__,
        split_behavior="global",
        operation_kind="metric_compute",
        total_semantics="canonical passed samples with validated metric bundles",
        bar_eligible=True,
    )
