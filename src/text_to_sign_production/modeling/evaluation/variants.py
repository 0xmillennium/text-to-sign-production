"""Evaluation protocol variant contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.evaluation.config import EvaluationVariantConfig


@dataclass(frozen=True, slots=True)
class EvaluationVariant:
    variant_id: str
    label: str
    producer_type: str
    producer_key: str
    run_name: str
    config_path: str
    auxiliary_objectives: tuple[str, ...]
    objective_config_paths: Mapping[str, str]
    tags: tuple[str, ...]
    required: bool
    expected_manifest_path: Path
    surface_status: str
    split: SampleSplit = SampleSplit.VAL
    manifest_family: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "variant_id": self.variant_id,
            "label": self.label,
            "producer_type": self.producer_type,
            "producer_key": self.producer_key,
            "run_name": self.run_name,
            "config_path": self.config_path,
            "auxiliary_objectives": list(self.auxiliary_objectives),
            "objective_config_paths": dict(self.objective_config_paths),
            "tags": list(self.tags),
            "required": self.required,
            "expected_manifest_path": str(self.expected_manifest_path),
            "surface_status": self.surface_status,
            "split": self.split.value,
            "manifest_family": self.manifest_family,
        }


def evaluation_variant_from_config(
    config: EvaluationVariantConfig,
    *,
    expected_manifest_path: Path,
    surface_status: str,
    split: SampleSplit,
    manifest_family: str | None,
) -> EvaluationVariant:
    return EvaluationVariant(
        variant_id=config.variant_id,
        label=config.label,
        producer_type=config.producer_type,
        producer_key=config.producer_key,
        run_name=config.run_name,
        config_path=config.config_path,
        auxiliary_objectives=config.auxiliary_objectives,
        objective_config_paths=config.objective_config_paths,
        tags=config.tags,
        required=config.required,
        expected_manifest_path=expected_manifest_path,
        surface_status=surface_status,
        split=split,
        manifest_family=manifest_family,
    )


__all__ = ["EvaluationVariant", "evaluation_variant_from_config"]
