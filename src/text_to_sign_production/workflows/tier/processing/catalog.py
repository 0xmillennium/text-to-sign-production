from __future__ import annotations

from pathlib import Path

from text_to_sign_production.artifacts.catalog import (
    SampleHandle,
    iter_samples,
    iter_samples_split,
    load_passed_samples_catalog,
)
from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.core.integrity import sha256_file
from text_to_sign_production.core.models import PassedManifestEntry, PreparedSample
from text_to_sign_production.core.progress import ProgressSession, ProgressStageSpec
from text_to_sign_production.data.dataset.manifests import read_passed_manifest_json
from text_to_sign_production.data.dataset.payloads import load_prepared_sample_payload
from text_to_sign_production.data.dataset.validate import validate_payload_manifest_coherence
from text_to_sign_production.workflows.tier.constants import (
    TIER_STAGE_CATALOG_LOAD,
    TIER_WORKFLOW_NAME,
)
from text_to_sign_production.workflows.tier.contracts import (
    TierWorkflowExecutionInputs,
    TierWorkflowInvariantError,
)
from text_to_sign_production.workflows.tier.layout import TierLayout
from text_to_sign_production.workflows.tier.processing.models import (
    TierCatalogBundle,
    TierSampleBundle,
)


def load_tier_catalog_bundle(
    *,
    layout: TierLayout,
    execution_inputs: TierWorkflowExecutionInputs,
    splits: tuple[str, ...],
    progress_session: ProgressSession | None = None,
) -> TierCatalogBundle:
    catalog = load_passed_samples_catalog(layout.stores, splits=splits)
    handles = tuple(iter_samples(catalog))
    manifest_lookup = _read_passed_manifest_lookup(execution_inputs)
    handle_keys = tuple((handle.ref.split, handle.ref.sample_id) for handle in handles)
    handle_key_set = set(handle_keys)
    manifest_key_set = set(manifest_lookup)
    if handle_key_set != manifest_key_set:
        raise TierWorkflowInvariantError(
            "passed samples catalog and runtime manifests are misaligned"
        )
    samples: list[TierSampleBundle] = []
    configured_splits = tuple(SampleSplit(split) for split in splits)
    if progress_session is not None and handles:
        for split in configured_splits:
            split_handles = tuple(iter_samples_split(catalog, split))
            if not split_handles:
                continue
            with progress_session.task(
                _catalog_alignment_progress_spec(split),
                total=len(split_handles),
            ) as progress_task:
                for handle in split_handles:
                    samples.append(
                        _load_aligned_sample(
                            handle=handle,
                            manifest_record=manifest_lookup[
                                (handle.ref.split, handle.ref.sample_id)
                            ],
                        )
                    )
                    progress_task.advance()
    else:
        samples = [
            _load_aligned_sample(
                handle=handle,
                manifest_record=manifest_lookup[(handle.ref.split, handle.ref.sample_id)],
            )
            for handle in handles
        ]
    return TierCatalogBundle(
        catalog=catalog,
        samples=tuple(samples),
    )


def _read_passed_manifest_lookup(
    execution_inputs: TierWorkflowExecutionInputs,
) -> dict[tuple[object, str], tuple[PassedManifestEntry, Path, str]]:
    manifest_lookup: dict[tuple[object, str], tuple[PassedManifestEntry, Path, str]] = {}
    for split_input in execution_inputs.split_inputs:
        manifest_sha256 = sha256_file(split_input.passed_manifest_path)
        for entry in read_passed_manifest_json(split_input.passed_manifest_path):
            key = (entry.split, entry.sample_id)
            if key in manifest_lookup:
                raise TierWorkflowInvariantError(
                    "Duplicate passed manifest entry for split/sample: "
                    f"{entry.split}/{entry.sample_id}"
                )
            manifest_lookup[key] = (
                entry,
                split_input.passed_manifest_path,
                manifest_sha256,
            )
    return manifest_lookup


def _load_aligned_sample(
    *,
    handle: SampleHandle,
    manifest_record: tuple[PassedManifestEntry, Path, str],
) -> TierSampleBundle:
    manifest, source_manifest_path, source_manifest_sha256 = manifest_record
    if handle.runtime_sample is None:
        raise TierWorkflowInvariantError(
            "Passed catalog handle must resolve a PreparedSample payload: "
            f"{handle.ref.split.value}/{handle.ref.sample_id}"
        )
    sample = load_prepared_sample_payload(handle.runtime_sample.path)
    _validate_identity_alignment(sample, manifest)
    issues = validate_payload_manifest_coherence(sample, manifest)
    if issues:
        raise TierWorkflowInvariantError(
            "PreparedSample payload and passed manifest are misaligned: "
            f"{manifest.split.value}/{manifest.sample_id}: {issues}"
        )
    return TierSampleBundle(
        handle=handle,
        manifest=manifest,
        source_manifest_path=source_manifest_path,
        source_manifest_sha256=source_manifest_sha256,
        sample=sample,
    )


def _validate_identity_alignment(
    sample: PreparedSample,
    manifest: PassedManifestEntry,
) -> None:
    if sample.source.sample_id != manifest.sample_id or sample.source.split is not manifest.split:
        raise TierWorkflowInvariantError(
            "PreparedSample identity does not match passed manifest: "
            f"payload={sample.source.split.value}/{sample.source.sample_id}, "
            f"manifest={manifest.split.value}/{manifest.sample_id}"
        )


def _catalog_alignment_progress_spec(split: SampleSplit) -> ProgressStageSpec:
    return ProgressStageSpec(
        workflow_id=TIER_WORKFLOW_NAME,
        stage_id=f"{TIER_STAGE_CATALOG_LOAD}.{split.value}",
        label=f"catalog alignment [{split.value}]",
        unit="sample",
        owner_module=__name__,
        split_behavior="per_split",
        operation_kind="catalog_alignment",
        total_semantics="canonical sample handles aligned with runtime passed manifests",
        bar_eligible=True,
    )
