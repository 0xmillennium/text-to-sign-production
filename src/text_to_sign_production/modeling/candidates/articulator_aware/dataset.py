"""Manifest-backed dataset builders for the articulator-aware provider."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Callable
from types import MappingProxyType
from pathlib import Path

import numpy as np

from text_to_sign_production.artifacts.store import ArtifactTopology
from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.data.gate.pose.types import PoseChannel
from text_to_sign_production.modeling.backbones.bfh_vectorization import (
    BfhVectorizedPose,
    flatten_bfh_vectorized_pose,
    vectorize_bfh_pose_arrays,
)
from text_to_sign_production.modeling.candidates.articulator_aware.config import (
    ChannelMaskStrategyConfig,
)
from text_to_sign_production.modeling.candidates.articulator_aware.errors import (
    ArticulatorAwareError,
)
from text_to_sign_production.modeling.candidates.articulator_aware.masks import (
    build_channel_loss_masks,
    summarize_channel_masks,
)
from text_to_sign_production.modeling.candidates.articulator_aware.partitions import (
    ArticulatorChannelPartitionPolicy,
)
from text_to_sign_production.modeling.data import (
    ModelingManifestFamily,
    load_manifest_sample,
    read_modeling_manifest,
)
from text_to_sign_production.modeling.data_surfaces import (
    ModelDataSurface,
    ModelDataSurfaceWriter,
)


@dataclass(frozen=True, slots=True)
class ArticulatorSourceSample:
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
        if not isinstance(self.frame_count, int) or isinstance(self.frame_count, bool) or self.frame_count < 1:
            raise ArticulatorAwareError("frame_count must be a positive integer.")
        if not isinstance(self.vectorized_pose, BfhVectorizedPose):
            raise ArticulatorAwareError("vectorized_pose must be a BfhVectorizedPose.")
        if self.vectorized_pose.frame_count != self.frame_count:
            raise ArticulatorAwareError("frame_count must match vectorized_pose.frame_count.")
        if self.vectorized_pose.source_sample_id != self.sample_id:
            raise ArticulatorAwareError("sample_id must match vectorized_pose source identity.")


@dataclass(frozen=True, slots=True)
class ArticulatorFrameTrainingSample:
    source: ArticulatorSourceSample
    frame_index: int
    target_values: np.ndarray
    channel_masks: Mapping[PoseChannel, np.ndarray]

    def __post_init__(self) -> None:
        if not isinstance(self.source, ArticulatorSourceSample):
            raise ArticulatorAwareError("source must be an ArticulatorSourceSample.")
        if (
            not isinstance(self.frame_index, int)
            or isinstance(self.frame_index, bool)
            or not 0 <= self.frame_index < self.source.frame_count
        ):
            raise ArticulatorAwareError("frame_index must identify one source frame.")
        feature_dim = self.source.vectorized_pose.layout.total_feature_dim
        values = np.asarray(self.target_values, dtype=np.float32).copy()
        if values.shape != (feature_dim,) or not np.all(np.isfinite(values)):
            raise ArticulatorAwareError(
                f"target_values must be finite with shape ({feature_dim},)."
            )
        values.setflags(write=False)
        masks: dict[PoseChannel, np.ndarray] = {}
        for channel in self.source.vectorized_pose.layout.channels:
            if channel not in self.channel_masks:
                raise ArticulatorAwareError(
                    f"channel_masks is missing primary channel {channel.value!r}."
                )
            mask = np.asarray(self.channel_masks[channel], dtype=np.bool_).copy()
            if mask.shape != (feature_dim,):
                raise ArticulatorAwareError(
                    f"channel mask for {channel.value!r} must have shape ({feature_dim},)."
                )
            mask.setflags(write=False)
            masks[channel] = mask
        object.__setattr__(self, "target_values", values)
        object.__setattr__(self, "channel_masks", MappingProxyType(masks))


def build_articulator_source_samples(
    *,
    topology: ArtifactTopology,
    manifest_family: ModelingManifestFamily,
    split: SampleSplit,
    max_samples: int | None = None,
    on_sample_loaded: Callable[[int, int, str], None] | None = None,
) -> tuple[ArticulatorSourceSample, ...]:
    """Load validated source samples from the selected manifest in manifest order."""

    if not isinstance(topology, ArtifactTopology):
        raise ArticulatorAwareError("topology must be an ArtifactTopology.")
    if not isinstance(manifest_family, ModelingManifestFamily):
        raise ArticulatorAwareError("manifest_family must be a ModelingManifestFamily.")
    if max_samples is not None and (
        not isinstance(max_samples, int) or isinstance(max_samples, bool) or max_samples <= 0
    ):
        raise ArticulatorAwareError("max_samples must be positive when provided.")
    resolved_split = SampleSplit(split)
    manifest = read_modeling_manifest(topology, manifest_family, resolved_split)
    entries = manifest.entries if max_samples is None else manifest.entries[:max_samples]
    samples: list[ArticulatorSourceSample] = []
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
        samples.append(
            ArticulatorSourceSample(
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
        )
    return tuple(samples)


def build_articulator_frame_training_samples(
    *,
    source_samples: Sequence[ArticulatorSourceSample],
    partition_policy: ArticulatorChannelPartitionPolicy,
    mask_config: ChannelMaskStrategyConfig,
    on_source_built: Callable[[int, int, str, int], None] | None = None,
) -> tuple[ArticulatorFrameTrainingSample, ...]:
    """Expand sequences to frame targets while retaining full-BFH channel masks."""

    sources = tuple(source_samples)
    if not sources:
        raise ArticulatorAwareError("source_samples must contain at least one sample.")
    frames: list[ArticulatorFrameTrainingSample] = []
    for source_index, source in enumerate(sources, start=1):
        if not isinstance(source, ArticulatorSourceSample):
            raise ArticulatorAwareError(
                "source_samples must contain ArticulatorSourceSample values."
            )
        summarize_channel_masks(
            vectorized=source.vectorized_pose,
            split=source.split,
            sample_id=source.sample_id,
            policy=partition_policy,
            config=mask_config,
        )
        masks = build_channel_loss_masks(
            vectorized=source.vectorized_pose,
            policy=partition_policy,
            config=mask_config,
        )
        target = flatten_bfh_vectorized_pose(source.vectorized_pose)
        for frame_index in range(source.frame_count):
            frames.append(
                ArticulatorFrameTrainingSample(
                    source=source,
                    frame_index=frame_index,
                    target_values=target[frame_index],
                    channel_masks={
                        channel: mask[frame_index] for channel, mask in masks.items()
                    },
                )
            )
        if on_source_built is not None:
            on_source_built(source_index, len(sources), source.sample_id, source.frame_count)
    return tuple(frames)


def build_articulator_source_surface(
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
    max_units_per_shard: int = 50_000,
    max_source_samples_per_shard: int = 512,
    on_sample_loaded: Callable[[int, int, str], None] | None = None,
) -> ModelDataSurface:
    """Build a tensor source sequence surface for articulator-aware training."""

    resolved_split = SampleSplit(split)
    manifest = read_modeling_manifest(topology, manifest_family, resolved_split)
    entries = manifest.entries if max_samples is None else manifest.entries[:max_samples]
    root = Path(surface_root)
    root.mkdir(parents=True, exist_ok=True)
    writer = ModelDataSurfaceWriter(
        root=root,
        provider_key="articulator_aware",
        surface_kind="articulator_source_sequences",
        split=resolved_split.value,
        manifest_family=manifest.manifest_family.family_id,
        source_manifest_path=manifest.manifest_path,
        source_manifest_sha256=source_manifest_sha256,
        provider_config_sha256=provider_config_sha256,
        cache_key=cache_key,
        manifest_entry_count=manifest_entry_count,
        run_mode=run_mode,
        data_version="bfh_source:v1",
        max_units_per_shard=max_units_per_shard,
        max_source_samples_per_shard=max_source_samples_per_shard,
    )
    sources_path = root / "sources.jsonl"
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
            target = flatten_bfh_vectorized_pose(vectorized)
            frame_index = np.arange(vectorized.frame_count, dtype=np.int64)
            writer.append_units(
                {
                    "target_values": target.astype(np.float32, copy=False),
                    "validity_mask": np.repeat(
                        vectorized.validity_mask,
                        vectorized.layout.coordinate_dimensions,
                        axis=1,
                    ).astype(np.bool_, copy=False),
                    "source_index": np.full((vectorized.frame_count,), source_index, dtype=np.int64),
                    "frame_index": frame_index,
                },
                sample_count=1,
                frame_count=vectorized.frame_count,
            )
            sources_handle.write(
                _source_json_line(
                    source_index=source_index,
                    sample_id=loaded.sample_id,
                    source_sentence_name=loaded.source_sentence_name,
                    text=loaded.text,
                    source_video_id=loaded.source_video_id,
                    source_sentence_id=loaded.source_sentence_id,
                    reference_payload_ref=loaded.payload_ref,
                    split=resolved_split.value,
                    frame_count=vectorized.frame_count,
                )
            )
    return writer.close()


def build_articulator_frame_surface_from_source_surface(
    *,
    source_surface: ModelDataSurface,
    partition_policy: ArticulatorChannelPartitionPolicy,
    mask_config: ChannelMaskStrategyConfig,
    surface_root: Path,
    source_manifest_sha256: str,
    provider_config_sha256: str,
    cache_key: str,
    run_mode: str,
    manifest_entry_count: int,
    max_units_per_shard: int = 50_000,
    max_source_samples_per_shard: int = 512,
) -> ModelDataSurface:
    """Write frame-unit masks from a source tensor surface one shard at a time."""

    from text_to_sign_production.modeling.data_surfaces import ModelDataSurfaceReader

    reader = ModelDataSurfaceReader(source_surface)
    writer = ModelDataSurfaceWriter(
        root=surface_root,
        provider_key="articulator_aware",
        surface_kind="articulator_frame_units",
        split=reader.metadata.split,
        manifest_family=reader.metadata.manifest_family,
        source_manifest_path=reader.metadata.source_manifest_path,
        source_manifest_sha256=source_manifest_sha256,
        provider_config_sha256=provider_config_sha256,
        cache_key=cache_key,
        manifest_entry_count=manifest_entry_count,
        run_mode=run_mode,
        data_version="bfh_frame_masks:v1",
        max_units_per_shard=max_units_per_shard,
        max_source_samples_per_shard=max_source_samples_per_shard,
    )
    del partition_policy, mask_config
    for shard in reader.iter_shards():
        values = shard["target_values"]
        validity = shard["validity_mask"]
        writer.append_units(
            {
                "target_values": values,
                "validity_mask": validity,
                "source_index": shard["source_index"],
                "frame_index": shard["frame_index"],
                "body_mask": validity,
                "left_hand_mask": validity,
                "right_hand_mask": validity,
                "face_mask": validity,
            },
            sample_count=int(shard["metadata"]["sample_count"]),
            frame_count=int(shard["metadata"]["unit_count"]),
        )
    return writer.close()


def _require_text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ArticulatorAwareError(f"{name} must be non-empty.")


def _source_json_line(
    *,
    source_index: int,
    sample_id: str,
    source_sentence_name: str,
    text: str,
    source_video_id: str,
    source_sentence_id: str,
    reference_payload_ref: str,
    split: str,
    frame_count: int,
) -> str:
    import json

    return (
        json.dumps(
            {
                "source_index": source_index,
                "sample_id": sample_id,
                "source_sentence_name": source_sentence_name,
                "text": text,
                "source_video_id": source_video_id,
                "source_sentence_id": source_sentence_id,
                "reference_payload_ref": reference_payload_ref,
                "split": split,
                "frame_count": frame_count,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    )


__all__ = [
    "ArticulatorFrameTrainingSample",
    "ArticulatorSourceSample",
    "build_articulator_frame_training_samples",
    "build_articulator_frame_surface_from_source_surface",
    "build_articulator_source_samples",
    "build_articulator_source_surface",
]
