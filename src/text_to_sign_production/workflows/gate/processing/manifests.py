from __future__ import annotations

from pathlib import Path

from text_to_sign_production.data.dataset.build import write_prepared_sample_payload_plan
from text_to_sign_production.data.dataset.dropped_payloads import write_dropped_sample_payload
from text_to_sign_production.data.dataset.manifests import (
    write_dropped_manifest_json,
    write_passed_manifest_json,
)
from text_to_sign_production.data.dataset.types import (
    DatasetDroppedSampleProduction,
    DatasetPayloadProduction,
)
from text_to_sign_production.workflows.foundation.provenance import written_file_receipt
from text_to_sign_production.workflows.gate.contracts import GateWorkflowInvariantError
from text_to_sign_production.workflows.gate.contracts.results import GateWrittenManifestArtifact
from text_to_sign_production.workflows.gate.layout import GateLayout
from text_to_sign_production.workflows.gate.processing.models import (
    GateSplitProcessingResult,
)


def write_gate_payloads(
    *,
    prepared_payloads: tuple[DatasetPayloadProduction, ...],
    dropped_sample_payloads: tuple[DatasetDroppedSampleProduction, ...],
) -> None:
    """Persist already-planned gate payloads."""
    for payload in prepared_payloads:
        write_prepared_sample_payload_plan(payload)
    for payload in dropped_sample_payloads:
        write_dropped_sample_payload(payload.path, payload.sample)


def write_gate_manifests(
    *,
    layout: GateLayout,
    config_splits: tuple[str, ...],
    split_results: tuple[GateSplitProcessingResult, ...],
    execution_id: str,
) -> tuple[GateWrittenManifestArtifact, ...]:
    artifacts: list[GateWrittenManifestArtifact] = []
    for split_result in _split_results_by_config_order(
        config_splits=config_splits,
        split_results=split_results,
    ):
        _validate_manifest_identity_uniqueness(split_result)
        write_passed_manifest_json(
            _manifest_path(layout, "passed", split_result.split),
            split_result.passed_entries,
            split=split_result.split,
        )
        write_dropped_manifest_json(
            _manifest_path(layout, "dropped", split_result.split),
            split_result.dropped_entries,
            split=split_result.split,
        )
        artifacts.append(
            GateWrittenManifestArtifact(
                partition="passed",
                split=split_result.split,
                receipt=written_file_receipt(
                    f"passed manifest [{split_result.split}]",
                    _manifest_path(layout, "passed", split_result.split),
                    execution_id=execution_id,
                    kind="gate_manifest",
                ),
            )
        )
        artifacts.append(
            GateWrittenManifestArtifact(
                partition="dropped",
                split=split_result.split,
                receipt=written_file_receipt(
                    f"dropped manifest [{split_result.split}]",
                    _manifest_path(layout, "dropped", split_result.split),
                    execution_id=execution_id,
                    kind="gate_manifest",
                ),
            )
        )
    return tuple(artifacts)


def _split_results_by_config_order(
    *,
    config_splits: tuple[str, ...],
    split_results: tuple[GateSplitProcessingResult, ...],
) -> tuple[GateSplitProcessingResult, ...]:
    by_split = {result.split: result for result in split_results}
    ordered_results = []
    for split in config_splits:
        result = by_split.get(split)
        if result is None:
            raise GateWorkflowInvariantError(f"Missing split processing result: {split}")
        ordered_results.append(result)
    if len(by_split) != len(split_results):
        raise GateWorkflowInvariantError("Duplicate split processing results are not allowed")
    return tuple(ordered_results)


def _validate_manifest_identity_uniqueness(split_result: GateSplitProcessingResult) -> None:
    _ensure_unique(
        tuple((entry.split.value, entry.sample_id) for entry in split_result.passed_entries),
        "passed manifest physical sample ids",
    )
    _ensure_unique(
        tuple((entry.split.value, entry.sample_id) for entry in split_result.dropped_entries),
        "dropped manifest physical sample ids",
    )


def _ensure_unique(values: tuple[object, ...], label: str) -> None:
    if len(set(values)) != len(values):
        raise GateWorkflowInvariantError(f"Duplicate {label} are not allowed: {values}.")


def _manifest_path(layout: GateLayout, partition: str, split: str) -> Path:
    for manifest in layout.outputs.manifest_outputs:
        if manifest.partition == partition and manifest.split == split:
            return manifest.path
    raise GateWorkflowInvariantError(f"Missing {partition} manifest output path for split: {split}")


__all__ = [
    "write_gate_manifests",
    "write_gate_payloads",
]
