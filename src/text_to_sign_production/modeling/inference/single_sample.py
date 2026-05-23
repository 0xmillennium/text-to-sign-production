"""Provider-neutral metadata for one normalized generated-pose inference output."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from text_to_sign_production.core.ids import SampleSplit

SINGLE_SAMPLE_INFERENCE_SCHEMA_VERSION = "t2sp-single-sample-inference-v1"


@dataclass(frozen=True, slots=True)
class SingleSampleInferenceMetadata:
    """Serializable identity and artifact references for one inference request."""

    schema_version: str
    model_key: str
    run_name: str
    checkpoint_role: str
    checkpoint_path: Path
    sample_id: str
    sentence_name: str | None
    split: SampleSplit
    generated_pose_manifest_path: Path
    generated_pose_sample_path: Path
    provider_metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.schema_version != SINGLE_SAMPLE_INFERENCE_SCHEMA_VERSION:
            raise ValueError("single-sample inference schema_version is unsupported.")
        for value, name in (
            (self.model_key, "model_key"),
            (self.run_name, "run_name"),
            (self.checkpoint_role, "checkpoint_role"),
            (self.sample_id, "sample_id"),
        ):
            _require_text(value, name)
        if self.sentence_name is not None:
            _require_text(self.sentence_name, "sentence_name")
        object.__setattr__(self, "split", SampleSplit(self.split))
        for name in (
            "checkpoint_path",
            "generated_pose_manifest_path",
            "generated_pose_sample_path",
        ):
            object.__setattr__(self, name, _path(getattr(self, name), name))
        object.__setattr__(
            self,
            "provider_metadata",
            _json_mapping(self.provider_metadata, "provider_metadata"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model_key": self.model_key,
            "run_name": self.run_name,
            "checkpoint_role": self.checkpoint_role,
            "checkpoint_path": str(self.checkpoint_path),
            "sample_id": self.sample_id,
            "sentence_name": self.sentence_name,
            "split": self.split.value,
            "generated_pose_manifest_path": str(self.generated_pose_manifest_path),
            "generated_pose_sample_path": str(self.generated_pose_sample_path),
            "provider_metadata": dict(self.provider_metadata),
        }


def _require_text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty.")


def _path(value: object, name: str) -> Path:
    if not isinstance(value, str | Path) or not str(value).strip():
        raise ValueError(f"{name} must be a non-empty path.")
    return Path(value)


def _json_mapping(value: Mapping[str, object], name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a mapping with string keys.")
    resolved = dict(value)
    try:
        json.dumps(resolved, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain JSON-serializable values.") from exc
    return MappingProxyType(resolved)


__all__ = ["SINGLE_SAMPLE_INFERENCE_SCHEMA_VERSION", "SingleSampleInferenceMetadata"]
