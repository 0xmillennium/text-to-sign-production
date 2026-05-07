from __future__ import annotations

from text_to_sign_production.artifacts.catalog import (
    iter_samples,
    load_passed_samples_catalog,
)
from text_to_sign_production.core.progress import ProgressSession, ProgressStageSpec
from text_to_sign_production.data.samples import (
    PassedManifestEntry,
    read_manifest_jsonl,
)
from text_to_sign_production.workflows.tiers.constants import (
    TIERS_STAGE_CATALOG_LOAD,
    TIERS_WORKFLOW_NAME,
)
from text_to_sign_production.workflows.tiers.contracts import (
    TiersWorkflowExecutionInputs,
    TiersWorkflowInvariantError,
)
from text_to_sign_production.workflows.tiers.layout import TiersLayout
from text_to_sign_production.workflows.tiers.processing.models import TiersCatalogBundle


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
    manifests: list[PassedManifestEntry] = []
    if progress_session is not None and handles:
        with progress_session.task(
            _catalog_alignment_progress_spec(),
            total=len(handles),
        ) as progress_task:
            for key in handle_keys:
                manifests.append(manifest_lookup[key])
                progress_task.advance()
    else:
        manifests = [manifest_lookup[key] for key in handle_keys]
    return TiersCatalogBundle(
        catalog=catalog,
        handles=handles,
        manifests=tuple(manifests),
    )


def _read_passed_manifest_lookup(
    execution_inputs: TiersWorkflowExecutionInputs,
) -> dict[tuple[object, str], PassedManifestEntry]:
    manifest_lookup: dict[tuple[object, str], PassedManifestEntry] = {}
    for split_input in execution_inputs.split_inputs:
        for entry in read_manifest_jsonl(split_input.passed_manifest_path):
            if not isinstance(entry, PassedManifestEntry):
                raise TiersWorkflowInvariantError(
                    f"Expected passed manifest entry in {split_input.passed_manifest_path}"
                )
            key = (entry.split, entry.sample_id)
            if key in manifest_lookup:
                raise TiersWorkflowInvariantError(
                    "Duplicate passed manifest entry for split/sample: "
                    f"{entry.split}/{entry.sample_id}"
                )
            manifest_lookup[key] = entry
    return manifest_lookup


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
