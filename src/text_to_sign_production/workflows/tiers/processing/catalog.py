from __future__ import annotations

from text_to_sign_production.artifacts.catalog import (
    SampleHandle,
    iter_samples,
    load_passed_samples_catalog,
)
from text_to_sign_production.core.models import PassedManifestEntry, PreparedSample
from text_to_sign_production.core.progress import ProgressSession, ProgressStageSpec
from text_to_sign_production.data.dataset.manifests import read_passed_manifest_jsonl
from text_to_sign_production.data.dataset.payloads import load_prepared_sample_payload
from text_to_sign_production.data.dataset.validate import validate_payload_manifest_coherence
from text_to_sign_production.workflows.tiers.constants import (
    TIERS_STAGE_CATALOG_LOAD,
    TIERS_WORKFLOW_NAME,
)
from text_to_sign_production.workflows.tiers.contracts import (
    TiersWorkflowExecutionInputs,
    TiersWorkflowInvariantError,
)
from text_to_sign_production.workflows.tiers.layout import TiersLayout
from text_to_sign_production.workflows.tiers.processing.models import (
    TiersCatalogBundle,
    TiersSampleBundle,
)


def load_tiers_catalog_bundle(
    *,
    layout: TiersLayout,
    execution_inputs: TiersWorkflowExecutionInputs,
    splits: tuple[str, ...],
    progress_session: ProgressSession | None = None,
) -> TiersCatalogBundle:
    catalog = load_passed_samples_catalog(layout.stores, splits=splits)
    handles = tuple(iter_samples(catalog))
    manifest_lookup = _read_passed_manifest_lookup(execution_inputs)
    handle_keys = tuple((handle.ref.split, handle.ref.sample_id) for handle in handles)
    handle_key_set = set(handle_keys)
    manifest_key_set = set(manifest_lookup)
    if handle_key_set != manifest_key_set:
        raise TiersWorkflowInvariantError(
            "passed samples catalog and runtime manifests are misaligned"
        )
    samples: list[TiersSampleBundle] = []
    if progress_session is not None and handles:
        with progress_session.task(
            _catalog_alignment_progress_spec(),
            total=len(handles),
        ) as progress_task:
            for handle in handles:
                samples.append(
                    _load_aligned_sample(
                        handle=handle,
                        manifest=manifest_lookup[(handle.ref.split, handle.ref.sample_id)],
                    )
                )
                progress_task.advance()
    else:
        samples = [
            _load_aligned_sample(
                handle=handle,
                manifest=manifest_lookup[(handle.ref.split, handle.ref.sample_id)],
            )
            for handle in handles
        ]
    return TiersCatalogBundle(
        catalog=catalog,
        samples=tuple(samples),
    )


def _read_passed_manifest_lookup(
    execution_inputs: TiersWorkflowExecutionInputs,
) -> dict[tuple[object, str], PassedManifestEntry]:
    manifest_lookup: dict[tuple[object, str], PassedManifestEntry] = {}
    for split_input in execution_inputs.split_inputs:
        for entry in read_passed_manifest_jsonl(split_input.passed_manifest_path):
            key = (entry.split, entry.sample_id)
            if key in manifest_lookup:
                raise TiersWorkflowInvariantError(
                    "Duplicate passed manifest entry for split/sample: "
                    f"{entry.split}/{entry.sample_id}"
                )
            manifest_lookup[key] = entry
    return manifest_lookup


def _load_aligned_sample(
    *,
    handle: SampleHandle,
    manifest: PassedManifestEntry,
) -> TiersSampleBundle:
    if handle.runtime_sample is None:
        raise TiersWorkflowInvariantError(
            "Passed catalog handle must resolve a PreparedSample payload: "
            f"{handle.ref.split.value}/{handle.ref.sample_id}"
        )
    sample = load_prepared_sample_payload(handle.runtime_sample.path)
    _validate_identity_alignment(sample, manifest)
    issues = validate_payload_manifest_coherence(sample, manifest)
    if issues:
        raise TiersWorkflowInvariantError(
            "PreparedSample payload and passed manifest are misaligned: "
            f"{manifest.split.value}/{manifest.sample_id}: {issues}"
        )
    return TiersSampleBundle(handle=handle, manifest=manifest, sample=sample)


def _validate_identity_alignment(
    sample: PreparedSample,
    manifest: PassedManifestEntry,
) -> None:
    if sample.source.sample_id != manifest.sample_id or sample.source.split is not manifest.split:
        raise TiersWorkflowInvariantError(
            "PreparedSample identity does not match passed manifest: "
            f"payload={sample.source.split.value}/{sample.source.sample_id}, "
            f"manifest={manifest.split.value}/{manifest.sample_id}"
        )


def _catalog_alignment_progress_spec() -> ProgressStageSpec:
    return ProgressStageSpec(
        workflow_id=TIERS_WORKFLOW_NAME,
        stage_id=TIERS_STAGE_CATALOG_LOAD,
        label="catalog alignment",
        unit="sample",
        owner_module=__name__,
        split_behavior="global",
        operation_kind="catalog_alignment",
        total_semantics="canonical sample handles aligned with runtime passed manifests",
        bar_eligible=True,
    )
