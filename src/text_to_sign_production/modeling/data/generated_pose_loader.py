"""Generated-pose surface loading for later evaluation workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.artifacts.store import ArtifactTopology, resolve_repo_relative
from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.artifacts import (
    GeneratedPoseArtifactError,
    GeneratedPoseManifestEntry,
    GeneratedPoseSample,
    load_generated_pose_payload,
    read_generated_pose_manifest_jsonl,
)
from text_to_sign_production.modeling.data.errors import ModelingDataError


@dataclass(frozen=True, slots=True)
class GeneratedPoseSurfaceSample:
    """A generated-pose manifest entry plus optional loaded payload."""

    entry: GeneratedPoseManifestEntry
    payload_path: Path | None
    sample: GeneratedPoseSample | None


def resolve_generated_payload_path(
    topology: ArtifactTopology,
    entry: GeneratedPoseManifestEntry,
) -> Path | None:
    """Resolve a generated-pose payload path from a manifest entry."""

    if entry.generated_payload_ref is None:
        return None
    try:
        path = resolve_repo_relative(topology, entry.generated_payload_ref).path
    except ValueError as exc:
        raise ModelingDataError(
            f"generated payload ref is not a safe repo-relative path: {entry.generated_payload_ref!r}"
        ) from exc
    root = topology.repo_root.resolve(strict=False)
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ModelingDataError(f"generated payload path escapes repo root: {path}") from exc
    return path


def load_generated_pose_surface(
    topology: ArtifactTopology,
    *,
    producer_key: str,
    run_name: str,
    split: SampleSplit | str,
) -> tuple[GeneratedPoseSurfaceSample, ...]:
    """Load one generated-pose manifest surface and its available payloads."""

    resolved_split = SampleSplit(split)
    manifest_path = topology.evaluations.generated_pose_manifest(
        producer_key,
        run_name,
        resolved_split,
    ).path
    try:
        entries = read_generated_pose_manifest_jsonl(manifest_path, expected_split=resolved_split)
    except GeneratedPoseArtifactError as exc:
        raise ModelingDataError(f"generated-pose manifest is invalid: {manifest_path}: {exc}") from exc
    loaded: list[GeneratedPoseSurfaceSample] = []
    for entry in entries:
        _validate_entry_identity(entry, producer_key=producer_key, run_name=run_name, split=resolved_split)
        payload_path = resolve_generated_payload_path(topology, entry)
        if payload_path is None:
            loaded.append(GeneratedPoseSurfaceSample(entry=entry, payload_path=None, sample=None))
            continue
        try:
            sample = load_generated_pose_payload(payload_path)
        except GeneratedPoseArtifactError as exc:
            raise ModelingDataError(
                f"generated-pose payload is invalid: {payload_path}: {exc}"
            ) from exc
        _validate_payload_identity(entry, sample)
        loaded.append(
            GeneratedPoseSurfaceSample(
                entry=entry,
                payload_path=payload_path,
                sample=sample,
            )
        )
    return tuple(loaded)


def _validate_entry_identity(
    entry: GeneratedPoseManifestEntry,
    *,
    producer_key: str,
    run_name: str,
    split: SampleSplit,
) -> None:
    if entry.producer_key != producer_key:
        raise ModelingDataError("generated manifest entry producer_key does not match request.")
    if entry.run_name != run_name:
        raise ModelingDataError("generated manifest entry run_name does not match request.")
    if entry.split is not split:
        raise ModelingDataError("generated manifest entry split does not match request.")


def _validate_payload_identity(
    entry: GeneratedPoseManifestEntry,
    sample: GeneratedPoseSample,
) -> None:
    checks = {
        "sample_id": sample.sample_id == entry.sample_id,
        "split": sample.split is entry.split,
        "producer_key": sample.producer_key == entry.producer_key,
        "run_name": sample.run_name == entry.run_name,
        "generation_index": sample.generation_index == entry.generation_index,
    }
    for field_name, passed in checks.items():
        if not passed:
            raise ModelingDataError(f"generated payload {field_name} does not match manifest.")
    if entry.frame_count is not None and sample.pose.frame_count != entry.frame_count:
        raise ModelingDataError("generated payload frame_count does not match manifest.")


__all__ = [
    "GeneratedPoseSurfaceSample",
    "load_generated_pose_surface",
    "resolve_generated_payload_path",
]
