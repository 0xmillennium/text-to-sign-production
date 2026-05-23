"""Generated-surface inventory for evaluation protocol variants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.artifacts.store import ArtifactTopology
from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.data import ModelingDataError, load_generated_pose_surface
from text_to_sign_production.modeling.evaluation.config import EvaluationProtocolConfig
from text_to_sign_production.modeling.evaluation.variants import (
    EvaluationVariant,
    evaluation_variant_from_config,
)

SURFACE_READY = "ready"
SURFACE_MISSING = "missing"
SURFACE_EMPTY = "empty"
SURFACE_INVALID = "invalid"
SURFACE_HAS_FAILURES = "has_failures"


@dataclass(frozen=True, slots=True)
class EvaluationSurfaceInventoryEntry:
    variant: EvaluationVariant
    variant_id: str
    surface_exists: bool
    manifest_path: Path
    payload_root: Path
    entry_count: int
    success_count: int
    failure_count: int
    paired_candidate_count: int
    producer_type: str
    producer_key: str
    run_name: str
    status: str
    missing_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "variant": self.variant.to_dict(),
            "variant_id": self.variant_id,
            "surface_exists": self.surface_exists,
            "manifest_path": str(self.manifest_path),
            "payload_root": str(self.payload_root),
            "entry_count": self.entry_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "paired_candidate_count": self.paired_candidate_count,
            "producer_type": self.producer_type,
            "producer_key": self.producer_key,
            "run_name": self.run_name,
            "status": self.status,
            "missing_reason": self.missing_reason,
        }


@dataclass(frozen=True, slots=True)
class EvaluationSurfaceInventory:
    entries: tuple[EvaluationSurfaceInventoryEntry, ...]

    @property
    def ready_entries(self) -> tuple[EvaluationSurfaceInventoryEntry, ...]:
        return tuple(entry for entry in self.entries if entry.status in {SURFACE_READY, SURFACE_HAS_FAILURES})

    @property
    def missing_entries(self) -> tuple[EvaluationSurfaceInventoryEntry, ...]:
        return tuple(entry for entry in self.entries if entry.status == SURFACE_MISSING)

    def by_variant_id(self) -> dict[str, EvaluationSurfaceInventoryEntry]:
        return {entry.variant_id: entry for entry in self.entries}

    def to_dict(self) -> dict[str, object]:
        return {"entries": [entry.to_dict() for entry in self.entries]}


def build_surface_inventory(
    *,
    topology: ArtifactTopology,
    config: EvaluationProtocolConfig,
    manifest_family_id: str | None,
) -> EvaluationSurfaceInventory:
    entries = tuple(
        _inventory_entry(
            topology=topology,
            variant_config=variant_config,
            split=config.split,
            manifest_family_id=manifest_family_id,
        )
        for variant_config in config.variants
    )
    return EvaluationSurfaceInventory(entries=entries)


def build_planned_run_request(entry: EvaluationSurfaceInventoryEntry) -> dict[str, object] | None:
    if entry.status != SURFACE_MISSING:
        return None
    variant = entry.variant
    record: dict[str, object] = {
        "variant_id": variant.variant_id,
        "producer_type": variant.producer_type,
        "config_path": variant.config_path,
        "manifest_family": variant.manifest_family,
        "split": variant.split.value,
        "run_name": variant.run_name,
        "status": "missing_generated_surface",
        "executed": False,
    }
    if variant.producer_type == "model":
        record.update(
            {
                "model_key": variant.producer_key,
                "auxiliary_objectives": list(variant.auxiliary_objectives),
                "objective_config_paths": dict(variant.objective_config_paths),
            }
        )
    else:
        record["comparator_key"] = variant.producer_key
    return record


def build_planned_run_requests(
    inventory: EvaluationSurfaceInventory,
) -> tuple[dict[str, object], ...]:
    return tuple(
        record
        for entry in inventory.entries
        for record in (build_planned_run_request(entry),)
        if record is not None
    )


def _inventory_entry(
    *,
    topology: ArtifactTopology,
    variant_config: object,
    split: SampleSplit,
    manifest_family_id: str | None,
) -> EvaluationSurfaceInventoryEntry:
    manifest_path = topology.evaluations.generated_pose_manifest(
        variant_config.producer_key,
        variant_config.run_name,
        split,
    ).path
    payload_root = topology.evaluations.generated_pose_samples_root(
        variant_config.producer_key,
        variant_config.run_name,
        split,
    ).path
    status = SURFACE_READY
    missing_reason: str | None = None
    surface_exists = manifest_path.exists()
    entry_count = 0
    success_count = 0
    failure_count = 0
    paired_candidate_count = 0
    if not surface_exists:
        status = SURFACE_MISSING
        missing_reason = (
            f"required evaluation variant {variant_config.variant_id!r} is missing its generated-pose surface."
            if variant_config.required
            else "generated-pose manifest is missing"
        )
    else:
        try:
            samples = load_generated_pose_surface(
                topology,
                producer_key=variant_config.producer_key,
                run_name=variant_config.run_name,
                split=split,
            )
        except (OSError, ModelingDataError, ValueError) as exc:
            samples = ()
            status = SURFACE_INVALID
            missing_reason = str(exc)
        if status != SURFACE_INVALID:
            entries = tuple(sample.entry for sample in samples)
            entry_count = len(entries)
            success_count = sum(entry.failure_reason is None for entry in entries)
            failure_count = sum(entry.failure_reason is not None for entry in entries)
            paired_candidate_count = success_count
            if entry_count == 0:
                status = SURFACE_EMPTY
                missing_reason = "generated-pose manifest has no entries"
            elif failure_count > 0:
                status = SURFACE_HAS_FAILURES
                missing_reason = "generated-pose manifest includes failed entries"
    variant = evaluation_variant_from_config(
        variant_config,
        expected_manifest_path=manifest_path,
        surface_status=status,
        split=split,
        manifest_family=manifest_family_id,
    )
    return EvaluationSurfaceInventoryEntry(
        variant=variant,
        variant_id=variant.variant_id,
        surface_exists=surface_exists,
        manifest_path=manifest_path,
        payload_root=payload_root,
        entry_count=entry_count,
        success_count=success_count,
        failure_count=failure_count,
        paired_candidate_count=paired_candidate_count,
        producer_type=variant.producer_type,
        producer_key=variant.producer_key,
        run_name=variant.run_name,
        status=status,
        missing_reason=missing_reason,
    )


__all__ = [
    "SURFACE_EMPTY",
    "SURFACE_HAS_FAILURES",
    "SURFACE_INVALID",
    "SURFACE_MISSING",
    "SURFACE_READY",
    "EvaluationSurfaceInventory",
    "EvaluationSurfaceInventoryEntry",
    "build_planned_run_request",
    "build_planned_run_requests",
    "build_surface_inventory",
]
