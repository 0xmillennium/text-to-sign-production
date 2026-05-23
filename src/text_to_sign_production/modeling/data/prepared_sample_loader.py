"""PreparedSample loading adapters for modeling manifest families."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from text_to_sign_production.artifacts.store import ArtifactTopology, resolve_samples_relative
from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.core.models import PassedManifestEntry, PreparedSample
from text_to_sign_production.data.dataset import load_prepared_sample_payload
from text_to_sign_production.data.dataset.validate import validate_payload_manifest_coherence
from text_to_sign_production.modeling.data.bfh_schema import (
    BfhPoseArrays,
    pose_arrays_from_prepared_sample,
)
from text_to_sign_production.modeling.data.errors import ModelingDataError
from text_to_sign_production.modeling.data.manifest_families import (
    ModelingManifestFamily,
    parse_modeling_manifest_family,
    read_modeling_manifest,
)


@dataclass(frozen=True, slots=True)
class ModelingManifestSample:
    """A modeling-ready prepared sample with canonical BFH pose arrays."""

    manifest_family: ModelingManifestFamily
    manifest_path: Path
    entry: PassedManifestEntry
    payload_path: Path
    sample: PreparedSample
    pose: BfhPoseArrays

    @property
    def sample_id(self) -> str:
        return self.entry.sample_id

    @property
    def split(self) -> SampleSplit:
        return self.entry.split

    @property
    def text(self) -> str:
        return self.entry.text

    @property
    def frame_count(self) -> int:
        return self.pose.frame_count

    @property
    def source_video_id(self) -> str:
        return self.entry.source_video_id

    @property
    def source_sentence_id(self) -> str:
        return self.entry.source_sentence_id

    @property
    def source_sentence_name(self) -> str:
        return self.entry.source_sentence_name

    @property
    def payload_ref(self) -> str:
        return self.entry.payload_ref


def resolve_prepared_payload_path(
    topology: ArtifactTopology,
    entry: PassedManifestEntry,
) -> Path:
    """Resolve a prepared payload path from a passed manifest entry."""

    try:
        return resolve_samples_relative(topology, entry.payload_ref).path
    except ValueError as exc:
        raise ModelingDataError(
            f"prepared payload_ref is not a safe samples-relative path: {entry.payload_ref!r}"
        ) from exc


def load_manifest_sample(
    topology: ArtifactTopology,
    manifest_family: ModelingManifestFamily | str,
    manifest_path: str | Path,
    entry: PassedManifestEntry,
) -> ModelingManifestSample:
    """Load and adapt one prepared sample referenced by a modeling manifest."""

    resolved_family = (
        manifest_family
        if isinstance(manifest_family, ModelingManifestFamily)
        else parse_modeling_manifest_family(manifest_family)
    )
    payload_path = resolve_prepared_payload_path(topology, entry)
    try:
        sample = load_prepared_sample_payload(payload_path)
    except ValueError as exc:
        raise ModelingDataError(f"prepared payload could not be loaded: {payload_path}: {exc}") from exc
    issues = validate_payload_manifest_coherence(sample, entry)
    if issues:
        messages = "; ".join(issue.message for issue in issues)
        raise ModelingDataError(f"prepared payload and manifest entry mismatch: {messages}")
    try:
        pose = pose_arrays_from_prepared_sample(sample)
    except ModelingDataError as exc:
        split_value = entry.split.value if hasattr(entry.split, "value") else entry.split
        raise ModelingDataError(
            "prepared payload violates BFH modeling contract: "
            f"sample_id={entry.sample_id!r}, "
            f"split={split_value!r}, "
            f"payload_ref={entry.payload_ref!r}, "
            f"payload_path={payload_path}: {exc}"
        ) from exc
    if pose.frame_count != entry.frame_count:
        raise ModelingDataError("prepared payload frame_count does not match manifest entry.")
    if int(np.count_nonzero(pose.valid_frame_mask)) != entry.valid_frame_count:
        raise ModelingDataError("prepared payload valid frame count does not match manifest entry.")
    return ModelingManifestSample(
        manifest_family=resolved_family,
        manifest_path=Path(manifest_path),
        entry=entry,
        payload_path=payload_path,
        sample=sample,
        pose=pose,
    )


def load_manifest_samples(
    topology: ArtifactTopology,
    manifest_family: ModelingManifestFamily | str,
    split: SampleSplit | str,
    *,
    limit: int | None = None,
) -> tuple[ModelingManifestSample, ...]:
    """Load modeling-ready prepared samples from a family in manifest order."""

    if limit is not None and limit <= 0:
        raise ModelingDataError("limit must be positive when provided.")
    manifest = read_modeling_manifest(
        topology,
        manifest_family
        if isinstance(manifest_family, ModelingManifestFamily)
        else parse_modeling_manifest_family(manifest_family),
        split,
    )
    entries = manifest.entries if limit is None else manifest.entries[:limit]
    return tuple(
        load_manifest_sample(
            topology,
            manifest.manifest_family,
            manifest.manifest_path,
            entry,
        )
        for entry in entries
    )


__all__ = [
    "ModelingManifestSample",
    "load_manifest_sample",
    "load_manifest_samples",
    "resolve_prepared_payload_path",
]
