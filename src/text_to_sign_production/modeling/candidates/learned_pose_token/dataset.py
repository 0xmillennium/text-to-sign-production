"""Representation dataset builder for pose-token training units."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from text_to_sign_production.artifacts.store import ArtifactTopology
from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.backbones.bfh_vectorization import (
    BfhVectorizedPose,
    flatten_bfh_vectorized_pose,
    vectorize_bfh_pose_arrays,
)
from text_to_sign_production.modeling.backbones.bfh_standardization import (
    BfhStandardizationAccumulator,
    write_bfh_standardization_stats_json,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.errors import (
    LearnedPoseTokenError,
)
from text_to_sign_production.modeling.data import (
    ModelingManifestFamily,
    TemporalWindowSpec,
    extract_bfh_pose_windows,
    flatten_bfh_pose_window_validity,
    flatten_bfh_pose_windows,
    load_manifest_sample,
    read_modeling_manifest,
)
from text_to_sign_production.modeling.data_surfaces import (
    ModelDataSurface,
    ModelDataSurfaceWriter,
)


@dataclass(frozen=True, slots=True)
class PoseTokenSourceSample:
    sample_id: str
    source_sentence_name: str
    text: str
    source_video_id: str
    source_sentence_id: str
    reference_payload_ref: str
    split: SampleSplit
    frame_count: int
    vectorized_pose: BfhVectorizedPose

    def __post_init__(self) -> None:
        for value, name in (
            (self.sample_id, "sample_id"),
            (self.source_sentence_name, "source_sentence_name"),
            (self.text, "text"),
            (self.source_video_id, "source_video_id"),
            (self.source_sentence_id, "source_sentence_id"),
            (self.reference_payload_ref, "reference_payload_ref"),
        ):
            _require_text(value, name)
        object.__setattr__(self, "split", SampleSplit(self.split))
        if not isinstance(self.frame_count, int) or isinstance(self.frame_count, bool) or self.frame_count <= 0:
            raise LearnedPoseTokenError("frame_count must be a positive integer.")
        if not isinstance(self.vectorized_pose, BfhVectorizedPose):
            raise LearnedPoseTokenError("vectorized_pose must be a BfhVectorizedPose.")
        if self.vectorized_pose.frame_count != self.frame_count:
            raise LearnedPoseTokenError("source frame_count must match vectorized_pose.")
        if self.vectorized_pose.source_sample_id != self.sample_id:
            raise LearnedPoseTokenError("source sample_id must match vectorized_pose source_sample_id.")


@dataclass(frozen=True, slots=True)
class PoseTokenTrainingSample:
    source: PoseTokenSourceSample
    sample_id: str
    source_sentence_name: str
    split: SampleSplit
    frame_index: int
    values: np.ndarray
    validity_mask: np.ndarray
    token_index: int | None = None
    source_frame_indices: tuple[int, ...] | None = None
    real_frame_mask: tuple[bool, ...] | None = None
    temporal_granularity: str = "frame"
    window_size: int = 1
    stride: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.source, PoseTokenSourceSample):
            raise LearnedPoseTokenError("source must be a PoseTokenSourceSample.")
        _require_text(self.sample_id, "sample_id")
        _require_text(self.source_sentence_name, "source_sentence_name")
        object.__setattr__(self, "split", SampleSplit(self.split))
        if self.sample_id != self.source.sample_id:
            raise LearnedPoseTokenError("sample_id must match source.sample_id.")
        if self.source_sentence_name != self.source.source_sentence_name:
            raise LearnedPoseTokenError(
                "source_sentence_name must match source.source_sentence_name."
            )
        if self.split is not self.source.split:
            raise LearnedPoseTokenError("split must match source.split.")
        if not isinstance(self.frame_index, int) or isinstance(self.frame_index, bool) or self.frame_index < 0:
            raise LearnedPoseTokenError("frame_index must be a non-negative integer.")
        if self.frame_index >= self.source.frame_count:
            raise LearnedPoseTokenError("frame_index must be less than source.frame_count.")
        token_index = self.frame_index if self.token_index is None else self.token_index
        if not isinstance(token_index, int) or isinstance(token_index, bool) or token_index < 0:
            raise LearnedPoseTokenError("token_index must be a non-negative integer.")
        if self.temporal_granularity not in {"frame", "window"}:
            raise LearnedPoseTokenError(
                "tokenizer.temporal_granularity must be one of {'frame', 'window'}."
            )
        if not isinstance(self.window_size, int) or isinstance(self.window_size, bool) or self.window_size <= 0:
            raise LearnedPoseTokenError("window_size must be a positive integer.")
        if not isinstance(self.stride, int) or isinstance(self.stride, bool) or self.stride <= 0:
            raise LearnedPoseTokenError("stride must be a positive integer.")
        if self.temporal_granularity == "frame" and (
            self.window_size != 1 or self.stride != 1
        ):
            raise LearnedPoseTokenError(
                "frame tokenizer requires window_size=1 and stride=1."
            )
        if self.temporal_granularity == "window" and self.window_size <= 1:
            raise LearnedPoseTokenError(
                "window tokenizer requires window_size > 1 and stride >= 1."
            )
        source_indices = (
            (self.frame_index,)
            if self.source_frame_indices is None
            else tuple(self.source_frame_indices)
        )
        real_mask = (
            (True,)
            if self.real_frame_mask is None
            else tuple(bool(value) for value in self.real_frame_mask)
        )
        if len(source_indices) != self.window_size or len(real_mask) != self.window_size:
            raise LearnedPoseTokenError(
                "source_frame_indices and real_frame_mask must match window_size."
            )
        for index, is_real in zip(source_indices, real_mask, strict=True):
            if not isinstance(index, int | np.integer) or isinstance(index, bool):
                raise LearnedPoseTokenError("source_frame_indices must contain integers.")
            if is_real and not 0 <= int(index) < self.source.frame_count:
                raise LearnedPoseTokenError("real source_frame_indices must be in range.")
            if not is_real and int(index) != -1:
                raise LearnedPoseTokenError("padded source_frame_indices must be -1.")
        values = np.asarray(self.values, dtype=np.float32).copy()
        mask = np.asarray(self.validity_mask, dtype=np.bool_).copy()
        if values.ndim != 1 or values.shape[0] <= 0:
            raise LearnedPoseTokenError("values must be a non-empty 1D array.")
        if mask.shape != values.shape:
            raise LearnedPoseTokenError("validity_mask must match values shape.")
        if not np.all(np.isfinite(values[mask])):
            raise LearnedPoseTokenError("values must be finite for valid pose features.")
        values.setflags(write=False)
        mask.setflags(write=False)
        object.__setattr__(self, "token_index", int(token_index))
        object.__setattr__(self, "source_frame_indices", tuple(int(v) for v in source_indices))
        object.__setattr__(self, "real_frame_mask", real_mask)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "validity_mask", mask)


def build_pose_token_training_samples(
    *,
    topology: ArtifactTopology,
    manifest_family: ModelingManifestFamily,
    split: SampleSplit,
    max_samples: int | None = None,
    temporal_spec: TemporalWindowSpec | None = None,
    on_sample_loaded: Callable[[int, int, str], None] | None = None,
) -> tuple[PoseTokenTrainingSample, ...]:
    """Build one token-unit sample per manifest payload in manifest order."""

    if not isinstance(topology, ArtifactTopology):
        raise LearnedPoseTokenError("topology must be an ArtifactTopology.")
    if not isinstance(manifest_family, ModelingManifestFamily):
        raise LearnedPoseTokenError("manifest_family must be a ModelingManifestFamily.")
    resolved_split = SampleSplit(split)
    if max_samples is not None and (
        not isinstance(max_samples, int) or isinstance(max_samples, bool) or max_samples <= 0
    ):
        raise LearnedPoseTokenError("max_samples must be positive when provided.")
    spec = TemporalWindowSpec.frame() if temporal_spec is None else temporal_spec
    if not isinstance(spec, TemporalWindowSpec):
        raise LearnedPoseTokenError("temporal_spec must be a TemporalWindowSpec.")
    manifest = read_modeling_manifest(topology, manifest_family, resolved_split)
    entries = manifest.entries if max_samples is None else manifest.entries[:max_samples]
    samples: list[PoseTokenTrainingSample] = []
    total = len(entries)
    for index, entry in enumerate(entries, start=1):
        loaded = load_manifest_sample(
            topology,
            manifest.manifest_family,
            manifest.manifest_path,
            entry,
        )
        if on_sample_loaded is not None:
            on_sample_loaded(index, total, loaded.sample_id)
        vectorized = vectorize_bfh_pose_arrays(loaded.pose, sample_id=loaded.sample_id)
        source = PoseTokenSourceSample(
            sample_id=loaded.sample_id,
            source_sentence_name=loaded.source_sentence_name,
            text=loaded.text,
            source_video_id=loaded.source_video_id,
            source_sentence_id=loaded.source_sentence_id,
            reference_payload_ref=loaded.payload_ref,
            split=resolved_split,
            frame_count=vectorized.frame_count,
            vectorized_pose=vectorized,
        )
        if spec.temporal_granularity == "frame":
            flattened = flatten_bfh_vectorized_pose(vectorized)
            feature_mask = np.repeat(
                vectorized.validity_mask,
                vectorized.layout.coordinate_dimensions,
                axis=1,
            )
            for frame_index in range(vectorized.frame_count):
                samples.append(
                    PoseTokenTrainingSample(
                        source=source,
                        sample_id=loaded.sample_id,
                        source_sentence_name=loaded.source_sentence_name,
                        split=resolved_split,
                        frame_index=frame_index,
                        values=flattened[frame_index],
                        validity_mask=feature_mask[frame_index],
                        token_index=frame_index,
                        source_frame_indices=(frame_index,),
                        real_frame_mask=(True,),
                        temporal_granularity=spec.temporal_granularity,
                        window_size=spec.window_size,
                        stride=spec.stride,
                    )
                )
            continue
        windows = extract_bfh_pose_windows(vectorized, spec=spec)
        flat_values = flatten_bfh_pose_windows(windows).reshape(
            windows.values.shape[0],
            spec.window_size * vectorized.layout.total_feature_dim,
        )
        flat_masks = flatten_bfh_pose_window_validity(windows).reshape(flat_values.shape)
        for window_index in range(windows.values.shape[0]):
            real_indices = tuple(
                int(value)
                for value, is_real in zip(
                    windows.source_frame_indices[window_index],
                    windows.real_frame_mask[window_index],
                    strict=True,
                )
                if bool(is_real)
            )
            frame_index = real_indices[0] if real_indices else 0
            samples.append(
                PoseTokenTrainingSample(
                    source=source,
                    sample_id=loaded.sample_id,
                    source_sentence_name=loaded.source_sentence_name,
                    split=resolved_split,
                    frame_index=frame_index,
                    values=flat_values[window_index],
                    validity_mask=flat_masks[window_index],
                    token_index=window_index,
                    source_frame_indices=tuple(
                        int(value) for value in windows.source_frame_indices[window_index]
                    ),
                    real_frame_mask=tuple(
                        bool(value) for value in windows.real_frame_mask[window_index]
                    ),
                    temporal_granularity=spec.temporal_granularity,
                    window_size=spec.window_size,
                    stride=spec.stride,
                )
            )
    return tuple(samples)


def build_pose_token_training_surface(
    *,
    topology: ArtifactTopology,
    manifest_family: ModelingManifestFamily,
    split: SampleSplit,
    surface_root: Path,
    source_manifest_sha256: str,
    provider_config_sha256: str,
    cache_key: str,
    run_mode: str,
    manifest_entry_count: int,
    max_samples: int | None = None,
    temporal_spec: TemporalWindowSpec | None = None,
    max_units_per_shard: int = 50_000,
    max_source_samples_per_shard: int = 512,
    standardization_missing_observation_policy: str = "raise",
    on_sample_loaded: Callable[[int, int, str], None] | None = None,
) -> ModelDataSurface:
    """Build a sharded tensor surface for pose-token training units."""

    if not isinstance(topology, ArtifactTopology):
        raise LearnedPoseTokenError("topology must be an ArtifactTopology.")
    if not isinstance(manifest_family, ModelingManifestFamily):
        raise LearnedPoseTokenError("manifest_family must be a ModelingManifestFamily.")
    resolved_split = SampleSplit(split)
    if max_samples is not None and (
        not isinstance(max_samples, int) or isinstance(max_samples, bool) or max_samples <= 0
    ):
        raise LearnedPoseTokenError("max_samples must be positive when provided.")
    spec = TemporalWindowSpec.frame() if temporal_spec is None else temporal_spec
    if not isinstance(spec, TemporalWindowSpec):
        raise LearnedPoseTokenError("temporal_spec must be a TemporalWindowSpec.")
    manifest = read_modeling_manifest(topology, manifest_family, resolved_split)
    entries = manifest.entries if max_samples is None else manifest.entries[:max_samples]
    root = Path(surface_root)
    root.mkdir(parents=True, exist_ok=True)
    sources_path = root / "sources.jsonl"
    writer = ModelDataSurfaceWriter(
        root=root,
        provider_key="learned_pose_token",
        surface_kind="pose_token_units",
        split=resolved_split.value,
        manifest_family=manifest.manifest_family.family_id,
        source_manifest_path=manifest.manifest_path,
        source_manifest_sha256=source_manifest_sha256,
        provider_config_sha256=provider_config_sha256,
        cache_key=cache_key,
        manifest_entry_count=manifest_entry_count,
        run_mode=run_mode,
        data_version=(
            f"{spec.temporal_granularity}:window={spec.window_size}:stride={spec.stride}"
        ),
        max_units_per_shard=max_units_per_shard,
        max_source_samples_per_shard=max_source_samples_per_shard,
    )
    accumulator = BfhStandardizationAccumulator()
    total = len(entries)
    with sources_path.open("w", encoding="utf-8") as sources_handle:
        for source_index, entry in enumerate(entries):
            loaded = load_manifest_sample(
                topology,
                manifest.manifest_family,
                manifest.manifest_path,
                entry,
            )
            if on_sample_loaded is not None:
                on_sample_loaded(source_index + 1, total, loaded.sample_id)
            vectorized = vectorize_bfh_pose_arrays(loaded.pose, sample_id=loaded.sample_id)
            accumulator.update(vectorized)
            sources_handle.write(
                json.dumps(
                    {
                        "source_index": source_index,
                        "sample_id": loaded.sample_id,
                        "source_sentence_name": loaded.source_sentence_name,
                        "text": loaded.text,
                        "source_video_id": loaded.source_video_id,
                        "source_sentence_id": loaded.source_sentence_id,
                        "reference_payload_ref": loaded.payload_ref,
                        "frame_count": vectorized.frame_count,
                        "split": resolved_split.value,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
                + "\n"
            )
            fields = _pose_token_surface_fields(
                vectorized=vectorized,
                source_index=source_index,
                temporal_spec=spec,
            )
            writer.append_units(
                fields,
                sample_count=1,
                frame_count=vectorized.frame_count,
            )
    stats = accumulator.finalize(
        epsilon=1e-6,
        missing_observation_policy=standardization_missing_observation_policy,
    )
    write_bfh_standardization_stats_json(root / "standardization_stats.json", stats)
    return writer.close()


def _pose_token_surface_fields(
    *,
    vectorized: BfhVectorizedPose,
    source_index: int,
    temporal_spec: TemporalWindowSpec,
) -> dict[str, np.ndarray]:
    if temporal_spec.temporal_granularity == "frame":
        values = flatten_bfh_vectorized_pose(vectorized)
        mask = np.repeat(
            vectorized.validity_mask,
            vectorized.layout.coordinate_dimensions,
            axis=1,
        )
        frame_indices = np.arange(vectorized.frame_count, dtype=np.int64)
        return {
            "values": values.astype(np.float32, copy=False),
            "validity_mask": mask.astype(np.bool_, copy=False),
            "source_index": np.full((vectorized.frame_count,), source_index, dtype=np.int64),
            "frame_index": frame_indices,
            "token_index": frame_indices.copy(),
            "source_frame_indices": frame_indices.reshape(-1, 1),
            "real_frame_mask": np.ones((vectorized.frame_count, 1), dtype=np.bool_),
        }
    windows = extract_bfh_pose_windows(vectorized, spec=temporal_spec)
    flat_values = flatten_bfh_pose_windows(windows).reshape(
        windows.values.shape[0],
        temporal_spec.window_size * vectorized.layout.total_feature_dim,
    )
    flat_masks = flatten_bfh_pose_window_validity(windows).reshape(flat_values.shape)
    frame_indices = np.asarray(
        [
            next((int(value) for value, real in zip(indices, mask, strict=True) if bool(real)), 0)
            for indices, mask in zip(
                windows.source_frame_indices,
                windows.real_frame_mask,
                strict=True,
            )
        ],
        dtype=np.int64,
    )
    token_indices = np.arange(flat_values.shape[0], dtype=np.int64)
    return {
        "values": flat_values.astype(np.float32, copy=False),
        "validity_mask": flat_masks.astype(np.bool_, copy=False),
        "source_index": np.full((flat_values.shape[0],), source_index, dtype=np.int64),
        "frame_index": frame_indices,
        "token_index": token_indices,
        "source_frame_indices": windows.source_frame_indices.astype(np.int64, copy=False),
        "real_frame_mask": windows.real_frame_mask.astype(np.bool_, copy=False),
    }


def _require_text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise LearnedPoseTokenError(f"{name} must be non-empty.")


__all__ = [
    "PoseTokenSourceSample",
    "PoseTokenTrainingSample",
    "build_pose_token_training_surface",
    "build_pose_token_training_samples",
]
