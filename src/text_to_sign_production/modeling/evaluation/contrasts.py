"""Contrast and claim-readiness logic for evaluation protocol runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from text_to_sign_production.modeling.evaluation.config import (
    EvaluationContrastConfig,
    EvaluationProtocolConfig,
)
from text_to_sign_production.modeling.evaluation.inventory import (
    SURFACE_HAS_FAILURES,
    SURFACE_READY,
    EvaluationSurfaceInventory,
    EvaluationSurfaceInventoryEntry,
)

READINESS_READY = "ready"
READINESS_MISSING_SURFACE = "not_ready_missing_surface"
READINESS_INSUFFICIENT_SHARED_SUBSET = "not_ready_insufficient_shared_subset"
READINESS_PROTOCOL_VIOLATION = "not_ready_protocol_violation"
READINESS_MISSING_BASELINE = "not_ready_missing_baseline"
READINESS_COMPARISON_UNAVAILABLE = "not_ready_comparison_unavailable"


@dataclass(frozen=True, slots=True)
class EvaluationContrastReadiness:
    contrast_id: str
    contrast_type: str
    control_variant_id: str
    treatment_variant_id: str
    intended_difference: str
    control_status: str
    treatment_status: str
    shared_pairing_count: int | None
    shared_pairing_keys: tuple[tuple[str, int], ...]
    shared_pairing_sample_ids: tuple[str, ...]
    protocol_violations: tuple[str, ...]
    claim_ready: bool
    readiness_status: str

    def to_dict(self) -> dict[str, object]:
        return {
            "contrast_id": self.contrast_id,
            "contrast_type": self.contrast_type,
            "control_variant_id": self.control_variant_id,
            "treatment_variant_id": self.treatment_variant_id,
            "intended_difference": self.intended_difference,
            "control_status": self.control_status,
            "treatment_status": self.treatment_status,
            "shared_pairing_count": self.shared_pairing_count,
            "shared_pairing_keys": [
                {"sample_id": sample_id, "generation_index": generation_index}
                for sample_id, generation_index in self.shared_pairing_keys
            ],
            "shared_pairing_sample_ids": list(self.shared_pairing_sample_ids),
            "protocol_violations": list(self.protocol_violations),
            "claim_ready": self.claim_ready,
            "readiness_status": self.readiness_status,
        }


def evaluate_contrast_readiness(
    *,
    config: EvaluationProtocolConfig,
    inventory: EvaluationSurfaceInventory,
    validation_comparison: object | None,
    paired_keys_by_variant_id: Mapping[str, set[tuple[str, int]]] | None = None,
) -> tuple[EvaluationContrastReadiness, ...]:
    entries = inventory.by_variant_id()
    paired_keys_by_variant_id = paired_keys_by_variant_id or {}
    return tuple(
        _contrast_readiness(
            contrast=contrast,
            control=entries[contrast.control_variant_id],
            treatment=entries[contrast.treatment_variant_id],
            minimum_shared_paired_count=config.validation_engine.minimum_shared_paired_count,
            require_shared_pairing_subset=config.validation_engine.require_shared_pairing_subset,
            validation_comparison=validation_comparison,
            shared_pairing_keys=_contrast_shared_keys(
                contrast.control_variant_id,
                contrast.treatment_variant_id,
                paired_keys_by_variant_id,
            ),
        )
        for contrast in config.contrasts
    )


def _contrast_readiness(
    *,
    contrast: EvaluationContrastConfig,
    control: EvaluationSurfaceInventoryEntry,
    treatment: EvaluationSurfaceInventoryEntry,
    minimum_shared_paired_count: int,
    require_shared_pairing_subset: bool,
    validation_comparison: object | None,
    shared_pairing_keys: tuple[tuple[str, int], ...],
) -> EvaluationContrastReadiness:
    violations = _protocol_violations(contrast, control, treatment)
    control_ready = _surface_ready(control)
    treatment_ready = _surface_ready(treatment)
    shared_pairing_count = len(shared_pairing_keys)
    if not control_ready or not treatment_ready:
        status = (
            READINESS_MISSING_BASELINE
            if contrast.contrast_type == "comparator_baseline"
            and (
                _is_retrieval_variant(control) and not control_ready
                or _is_retrieval_variant(treatment) and not treatment_ready
            )
            else READINESS_MISSING_SURFACE
        )
    elif violations:
        status = READINESS_PROTOCOL_VIOLATION
    elif validation_comparison is None or getattr(validation_comparison, "status", None) != "ok":
        status = READINESS_COMPARISON_UNAVAILABLE
    elif (
        require_shared_pairing_subset
        and shared_pairing_count < minimum_shared_paired_count
    ):
        status = READINESS_INSUFFICIENT_SHARED_SUBSET
    else:
        status = READINESS_READY
    return EvaluationContrastReadiness(
        contrast_id=contrast.contrast_id,
        contrast_type=contrast.contrast_type,
        control_variant_id=contrast.control_variant_id,
        treatment_variant_id=contrast.treatment_variant_id,
        intended_difference=contrast.intended_difference,
        control_status=control.status,
        treatment_status=treatment.status,
        shared_pairing_count=shared_pairing_count,
        shared_pairing_keys=shared_pairing_keys,
        shared_pairing_sample_ids=tuple(sample_id for sample_id, _ in shared_pairing_keys),
        protocol_violations=violations,
        claim_ready=status == READINESS_READY,
        readiness_status=status,
    )


def _surface_ready(entry: EvaluationSurfaceInventoryEntry) -> bool:
    return entry.status in {SURFACE_READY, SURFACE_HAS_FAILURES}


def _contrast_shared_keys(
    control_variant_id: str,
    treatment_variant_id: str,
    paired_keys_by_variant_id: Mapping[str, set[tuple[str, int]]],
) -> tuple[tuple[str, int], ...]:
    control_keys = paired_keys_by_variant_id.get(control_variant_id)
    treatment_keys = paired_keys_by_variant_id.get(treatment_variant_id)
    if control_keys is None or treatment_keys is None:
        return ()
    return tuple(sorted(control_keys & treatment_keys, key=lambda key: (key[0], key[1])))


def _protocol_violations(
    contrast: EvaluationContrastConfig,
    control: EvaluationSurfaceInventoryEntry,
    treatment: EvaluationSurfaceInventoryEntry,
) -> tuple[str, ...]:
    violations: list[str] = []
    if not contrast.intended_difference.strip():
        violations.append("contrast intended_difference must be explicit.")
    if contrast.contrast_type == "ablation":
        if control.producer_type != "model" or treatment.producer_type != "model":
            violations.append("ablation contrasts require model control and treatment variants.")
        if _is_retrieval_variant(control) or _is_retrieval_variant(treatment):
            violations.append("retrieval baseline comparisons are comparator baselines, not ablations.")
    if contrast.contrast_type == "comparator_baseline":
        if not (_is_retrieval_variant(control) or _is_retrieval_variant(treatment)):
            violations.append("comparator_baseline contrast should include a comparator baseline variant.")
    if control.variant.manifest_family != treatment.variant.manifest_family:
        violations.append("contrast variants must share the same manifest family.")
    if control.variant.split is not treatment.variant.split:
        violations.append("contrast variants must share the same validation split.")
    if _is_semantic_objective_contrast(control, treatment):
        violations.extend(_semantic_objective_violations(contrast, control, treatment))
    return tuple(violations)


def _is_retrieval_variant(entry: EvaluationSurfaceInventoryEntry) -> bool:
    return entry.variant_id == "retrieval_pose_baseline" or entry.producer_key == "retrieval_pose"


def _is_semantic_objective_contrast(
    control: EvaluationSurfaceInventoryEntry,
    treatment: EvaluationSurfaceInventoryEntry,
) -> bool:
    return (
        "semantic_consistency" in control.variant.auxiliary_objectives
        or "semantic_consistency" in treatment.variant.auxiliary_objectives
    )


def _semantic_objective_violations(
    contrast: EvaluationContrastConfig,
    control: EvaluationSurfaceInventoryEntry,
    treatment: EvaluationSurfaceInventoryEntry,
) -> tuple[str, ...]:
    violations: list[str] = []
    if control.producer_key != treatment.producer_key:
        violations.append("semantic objective ablation requires the same producer_key.")
    if (
        control.variant.config_path != treatment.variant.config_path
        and not contrast.allow_config_path_mismatch
    ):
        violations.append(
            "semantic objective ablation requires control and treatment to share the same model config_path."
        )
    if "semantic_consistency" in control.variant.auxiliary_objectives:
        violations.append("semantic objective ablation control must not include semantic_consistency.")
    if "semantic_consistency" not in treatment.variant.auxiliary_objectives:
        violations.append("semantic objective ablation treatment must include semantic_consistency.")
    if "semantic_consistency" not in treatment.variant.objective_config_paths:
        violations.append("semantic objective ablation treatment requires objective config path.")
    return tuple(violations)


__all__ = [
    "READINESS_COMPARISON_UNAVAILABLE",
    "READINESS_INSUFFICIENT_SHARED_SUBSET",
    "READINESS_MISSING_BASELINE",
    "READINESS_MISSING_SURFACE",
    "READINESS_PROTOCOL_VIOLATION",
    "READINESS_READY",
    "EvaluationContrastReadiness",
    "evaluate_contrast_readiness",
]
