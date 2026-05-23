"""Dataset and cache builders for latent_diffusion foundation targets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable
from pathlib import Path

import numpy as np

from text_to_sign_production.artifacts.store import ArtifactTopology
from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.backbones.bfh_standardization import (
    BfhStandardizationAccumulator,
    BfhStandardizationStats,
    apply_bfh_standardization,
    write_bfh_standardization_stats_json,
)
from text_to_sign_production.modeling.backbones.bfh_vectorization import (
    BfhTensorLayout,
    BfhVectorizedPose,
    flatten_bfh_vectorized_pose,
    vectorize_bfh_pose_arrays,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.config import (
    LatentTargetConfig,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.errors import (
    LatentDiffusionError,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.io import (
    write_latent_sequence_npz,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.latents import (
    LATENT_MANIFEST_SCHEMA_VERSION,
    LATENT_SEQUENCE_SCHEMA_VERSION,
    LATENT_TARGET_SPEC_SCHEMA_VERSION,
    LATENT_TARGET_TYPE_LEARNED_BFH_WINDOW_LATENT,
    LATENT_TARGET_TYPE_STANDARDIZED_BFH_FRAME,
    LatentManifestEntry,
    LatentSequence,
    LatentTargetSpec,
)
from text_to_sign_production.modeling.data import (
    ModelingManifestFamily,
    TemporalWindowSpec,
    extract_bfh_pose_windows,
    flatten_bfh_pose_window_validity,
    flatten_bfh_pose_windows,
    load_manifest_sample,
    read_modeling_manifest,
    temporal_window_starts,
)
from text_to_sign_production.modeling.data_surfaces import (
    ModelDataSurface,
    ModelDataSurfaceReader,
    ModelDataSurfaceWriter,
)


@dataclass(frozen=True, slots=True)
class LatentSourceSample:
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
        _require_positive_int(self.frame_count, "frame_count")
        if not isinstance(self.vectorized_pose, BfhVectorizedPose):
            raise LatentDiffusionError("vectorized_pose must be a BfhVectorizedPose.")
        if self.vectorized_pose.frame_count != self.frame_count:
            raise LatentDiffusionError("frame_count must match vectorized_pose.")
        if self.vectorized_pose.source_sample_id != self.sample_id:
            raise LatentDiffusionError("sample_id must match vectorized_pose source_sample_id.")


@dataclass(frozen=True, slots=True)
class LatentFrameTrainingSample:
    source: LatentSourceSample
    frame_index: int
    latent_value: np.ndarray
    validity_mask: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.source, LatentSourceSample):
            raise LatentDiffusionError("source must be a LatentSourceSample.")
        if (
            not isinstance(self.frame_index, int)
            or isinstance(self.frame_index, bool)
            or self.frame_index < 0
            or self.frame_index >= self.source.frame_count
        ):
            raise LatentDiffusionError("frame_index must identify a source frame.")
        value = np.asarray(self.latent_value, dtype=np.float32).copy()
        mask = np.asarray(self.validity_mask, dtype=np.bool_).copy()
        if value.ndim != 1 or value.shape[0] <= 0:
            raise LatentDiffusionError("latent_value must be a non-empty 1D array.")
        if mask.shape != value.shape:
            raise LatentDiffusionError("validity_mask must match latent_value shape.")
        if not np.all(np.isfinite(value[mask])):
            raise LatentDiffusionError("latent_value must be finite where validity_mask is true.")
        value.setflags(write=False)
        mask.setflags(write=False)
        object.__setattr__(self, "latent_value", value)
        object.__setattr__(self, "validity_mask", mask)


def build_latent_source_samples(
    *,
    topology: ArtifactTopology,
    manifest_family: ModelingManifestFamily,
    split: SampleSplit,
    max_samples: int | None = None,
    on_sample_loaded: Callable[[int, int, str], None] | None = None,
) -> tuple[LatentSourceSample, ...]:
    """Load source samples from a modeling manifest family in manifest order."""

    if not isinstance(topology, ArtifactTopology):
        raise LatentDiffusionError("topology must be an ArtifactTopology.")
    if not isinstance(manifest_family, ModelingManifestFamily):
        raise LatentDiffusionError("manifest_family must be a ModelingManifestFamily.")
    resolved_split = SampleSplit(split)
    if max_samples is not None and (
        not isinstance(max_samples, int) or isinstance(max_samples, bool) or max_samples <= 0
    ):
        raise LatentDiffusionError("max_samples must be positive when provided.")
    manifest = read_modeling_manifest(topology, manifest_family, resolved_split)
    entries = manifest.entries if max_samples is None else manifest.entries[:max_samples]
    samples: list[LatentSourceSample] = []
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
            LatentSourceSample(
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


def build_latent_source_surface(
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
    standardization_missing_observation_policy: str = "raise",
    on_sample_loaded: Callable[[int, int, str], None] | None = None,
) -> ModelDataSurface:
    """Build a sharded latent source sequence surface without retaining the split."""

    resolved_split = SampleSplit(split)
    manifest = read_modeling_manifest(topology, manifest_family, resolved_split)
    entries = manifest.entries if max_samples is None else manifest.entries[:max_samples]
    root = Path(surface_root)
    root.mkdir(parents=True, exist_ok=True)
    writer = ModelDataSurfaceWriter(
        root=root,
        provider_key="latent_diffusion",
        surface_kind="latent_source_sequences",
        split=resolved_split.value,
        manifest_family=manifest.manifest_family.family_id,
        source_manifest_path=manifest.manifest_path,
        source_manifest_sha256=source_manifest_sha256,
        provider_config_sha256=provider_config_sha256,
        cache_key=cache_key,
        manifest_entry_count=manifest_entry_count,
        run_mode=run_mode,
        data_version="standardized_bfh_source:v1",
        max_units_per_shard=max_units_per_shard,
        max_source_samples_per_shard=max_source_samples_per_shard,
    )
    accumulator = BfhStandardizationAccumulator()
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
            accumulator.update(vectorized)
            values = flatten_bfh_vectorized_pose(vectorized)
            mask = np.repeat(
                vectorized.validity_mask,
                vectorized.layout.coordinate_dimensions,
                axis=1,
            )
            frame_index = np.arange(vectorized.frame_count, dtype=np.int64)
            writer.append_units(
                {
                    "pose_values": values.astype(np.float32, copy=False),
                    "validity_mask": mask.astype(np.bool_, copy=False),
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
    stats = accumulator.finalize(
        epsilon=1e-6,
        missing_observation_policy=standardization_missing_observation_policy,
    )
    write_bfh_standardization_stats_json(root / "standardization_stats.json", stats)
    return writer.close()


def build_latent_sequence_surface_from_source_surface(
    *,
    source_surface: ModelDataSurface,
    stats: BfhStandardizationStats,
    surface_root: Path,
    source_manifest_sha256: str,
    provider_config_sha256: str,
    cache_key: str,
    run_mode: str,
    manifest_entry_count: int,
    target_spec: LatentTargetSpec,
    max_units_per_shard: int = 50_000,
    max_source_samples_per_shard: int = 512,
) -> ModelDataSurface:
    """Standardize latent source shards into a latent sequence tensor surface."""

    import torch

    from text_to_sign_production.modeling.data_surfaces import ModelDataSurfaceReader

    if target_spec.target_type != LATENT_TARGET_TYPE_STANDARDIZED_BFH_FRAME:
        raise LatentDiffusionError(
            "build_latent_sequence_surface_from_source_surface only supports standardized_bfh_frame."
        )
    reader = ModelDataSurfaceReader(source_surface)
    root = Path(surface_root)
    writer = ModelDataSurfaceWriter(
        root=root,
        provider_key="latent_diffusion",
        surface_kind="latent_sequences",
        split=reader.metadata.split,
        manifest_family=reader.metadata.manifest_family,
        source_manifest_path=reader.surface.metadata.source_manifest_path,
        source_manifest_sha256=source_manifest_sha256,
        provider_config_sha256=provider_config_sha256,
        cache_key=cache_key,
        manifest_entry_count=manifest_entry_count,
        run_mode=run_mode,
        data_version=f"{target_spec.target_type}:v1",
        max_units_per_shard=max_units_per_shard,
        max_source_samples_per_shard=max_source_samples_per_shard,
    )
    mean = torch.as_tensor(np.array(stats.mean, dtype=np.float32, copy=True).reshape(-1))
    std = torch.as_tensor(np.array(stats.std, dtype=np.float32, copy=True).reshape(-1))
    for shard in reader.iter_shards():
        values = shard["pose_values"].float()
        mask = shard["validity_mask"].bool()
        standardized = torch.where(mask, (values - mean.unsqueeze(0)) / std.unsqueeze(0), torch.zeros_like(values))
        writer.append_units(
            {
                "latent_values": standardized,
                "latent_mask": mask,
                "source_index": shard["source_index"].long(),
                "latent_index": shard["frame_index"].long(),
            },
            sample_count=int(shard["metadata"]["sample_count"]),
            frame_count=int(shard["metadata"]["unit_count"]),
        )
    return writer.close()


def build_latent_window_surface_from_source_surface(
    *,
    source_surface: ModelDataSurface,
    stats: BfhStandardizationStats,
    surface_root: Path,
    source_manifest_sha256: str,
    provider_config_sha256: str,
    cache_key: str,
    run_mode: str,
    manifest_entry_count: int,
    target_spec: LatentTargetSpec,
    max_units_per_shard: int = 50_000,
    max_source_samples_per_shard: int = 512,
    on_shard_built: Callable[[int, int, int], None] | None = None,
) -> ModelDataSurface:
    """Build temporal BFH window tensors from source shards one shard at a time."""

    import torch

    if target_spec.target_type != LATENT_TARGET_TYPE_LEARNED_BFH_WINDOW_LATENT:
        raise LatentDiffusionError(
            "build_latent_window_surface_from_source_surface requires learned_bfh_window_latent."
        )
    spec = target_spec.temporal_window_spec()
    if not isinstance(spec, TemporalWindowSpec):
        raise LatentDiffusionError("target temporal window spec is invalid.")
    reader = ModelDataSurfaceReader(source_surface)
    root = Path(surface_root)
    writer = ModelDataSurfaceWriter(
        root=root,
        provider_key="latent_diffusion",
        surface_kind="latent_windows",
        split=reader.metadata.split,
        manifest_family=reader.metadata.manifest_family,
        source_manifest_path=reader.surface.metadata.source_manifest_path,
        source_manifest_sha256=source_manifest_sha256,
        provider_config_sha256=provider_config_sha256,
        cache_key=cache_key,
        manifest_entry_count=manifest_entry_count,
        run_mode=run_mode,
        data_version=f"{target_spec.target_type}:windows:v1",
        max_units_per_shard=max_units_per_shard,
        max_source_samples_per_shard=max_source_samples_per_shard,
    )
    mean = torch.as_tensor(np.array(stats.mean, dtype=np.float32, copy=True).reshape(-1))
    std = torch.as_tensor(np.array(stats.std, dtype=np.float32, copy=True).reshape(-1))
    total_shards = source_surface.metadata.shard_count
    for shard_index, shard in enumerate(reader.iter_shards(), start=1):
        source_index = shard["source_index"].long()
        frame_index = shard["frame_index"].long()
        shard_window_count = 0
        for raw_source_index in torch.unique(source_index, sorted=True).tolist():
            mask = source_index == int(raw_source_index)
            order = torch.argsort(frame_index[mask])
            values = shard["pose_values"][mask][order].float()
            validity = shard["validity_mask"][mask][order].bool()
            standardized = torch.where(
                validity,
                (values - mean.unsqueeze(0)) / std.unsqueeze(0),
                torch.zeros_like(values),
            )
            frame_count = int(standardized.shape[0])
            starts = temporal_window_starts(frame_count=frame_count, spec=spec)
            windows = torch.zeros(
                (len(starts), spec.window_size, target_spec.base_feature_dim),
                dtype=torch.float32,
            )
            window_mask = torch.zeros_like(windows, dtype=torch.bool)
            for window_index, start in enumerate(starts):
                for offset in range(spec.window_size):
                    source_frame = int(start + offset)
                    if source_frame >= frame_count:
                        continue
                    windows[window_index, offset] = standardized[source_frame]
                    window_mask[window_index, offset] = validity[source_frame]
            valid_windows = torch.any(window_mask.reshape(window_mask.shape[0], -1), dim=1)
            if not torch.all(valid_windows):
                raise LatentDiffusionError(
                    "latent window surface contains a temporal window with no valid input features."
                )
            window_count = int(windows.shape[0])
            writer.append_units(
                {
                    "windows": windows,
                    "window_mask": window_mask,
                    "source_index": torch.full((window_count,), int(raw_source_index), dtype=torch.int64),
                    "window_index": torch.arange(window_count, dtype=torch.int64),
                },
                sample_count=1,
                frame_count=frame_count,
            )
            shard_window_count += window_count
        if on_shard_built is not None:
            on_shard_built(shard_index, total_shards, shard_window_count)
    return writer.close()


def build_latent_target_spec(
    *,
    config: LatentTargetConfig,
    layout: BfhTensorLayout,
    learned_latent_dim: int | None = None,
) -> LatentTargetSpec:
    """Build the versioned latent target spec for the configured target type."""

    if not isinstance(config, LatentTargetConfig):
        raise LatentDiffusionError("config must be a LatentTargetConfig.")
    if not isinstance(layout, BfhTensorLayout):
        raise LatentDiffusionError("layout must be a BfhTensorLayout.")
    return LatentTargetSpec(
        schema_version=LATENT_TARGET_SPEC_SCHEMA_VERSION,
        target_type=config.target_type,
        layout=layout,
        coordinate_mode=config.coordinate_mode,
        confidence_policy=config.confidence_policy,
        temporal_granularity=config.temporal_granularity,
        window_size=config.window_size,
        stride=config.stride,
        latent_dim=(
            layout.total_feature_dim
            if config.target_type == LATENT_TARGET_TYPE_STANDARDIZED_BFH_FRAME
            else _required_learned_latent_dim(learned_latent_dim)
        ),
        base_feature_dim=layout.total_feature_dim,
    )


def cache_latent_sequences(
    *,
    source_samples: Sequence[LatentSourceSample],
    stats: BfhStandardizationStats,
    output_root: Path,
    target_spec: LatentTargetSpec,
    on_sample_cached: Callable[[int, int, str, int], None] | None = None,
) -> tuple[LatentManifestEntry, ...]:
    """Standardize and cache one latent sequence per source sample."""

    sources = tuple(source_samples)
    if not sources:
        raise LatentDiffusionError("source_samples must be non-empty.")
    if any(not isinstance(source, LatentSourceSample) for source in sources):
        raise LatentDiffusionError("source_samples must contain LatentSourceSample values.")
    if not isinstance(stats, BfhStandardizationStats):
        raise LatentDiffusionError("stats must be BfhStandardizationStats.")
    if not isinstance(target_spec, LatentTargetSpec):
        raise LatentDiffusionError("target_spec must be a LatentTargetSpec.")
    root = Path(output_root)
    entries: list[LatentManifestEntry] = []
    if target_spec.target_type != LATENT_TARGET_TYPE_STANDARDIZED_BFH_FRAME:
        raise LatentDiffusionError("cache_latent_sequences only supports standardized_bfh_frame.")
    for index, source in enumerate(sources, start=1):
        standardized = apply_bfh_standardization(source.vectorized_pose, stats)
        values = flatten_bfh_vectorized_pose(standardized)
        mask = np.repeat(
            standardized.validity_mask,
            standardized.layout.coordinate_dimensions,
            axis=1,
        )
        if not np.any(mask):
            raise LatentDiffusionError(
                f"latent cache sample_id={source.sample_id!r} has no valid latent "
                "features after BFH standardization; fix the source pose validity before caching."
            )
        sequence = LatentSequence(
            schema_version=LATENT_SEQUENCE_SCHEMA_VERSION,
            sample_id=source.sample_id,
            source_sentence_name=source.source_sentence_name,
            split=source.split,
            values=values,
            validity_mask=mask,
            frame_count=source.frame_count,
            latent_count=source.frame_count,
            latent_dim=standardized.layout.total_feature_dim,
            target_spec=target_spec,
        )
        latent_path = root / f"{_safe_path_token(source.sample_id)}.npz"
        write_latent_sequence_npz(latent_path, sequence)
        entries.append(
            LatentManifestEntry(
                schema_version=LATENT_MANIFEST_SCHEMA_VERSION,
                sample_id=source.sample_id,
                source_sentence_name=source.source_sentence_name,
                split=source.split,
                latent_path=latent_path,
                frame_count=source.frame_count,
                latent_count=sequence.latent_count,
                latent_dim=sequence.latent_dim,
                target_type=target_spec.target_type,
                temporal_granularity=target_spec.temporal_granularity,
                window_size=target_spec.window_size,
                stride=target_spec.stride,
                issues=(),
            )
        )
        if on_sample_cached is not None:
            on_sample_cached(index, len(sources), source.sample_id, sequence.latent_count)
    return tuple(entries)


@dataclass(frozen=True, slots=True)
class WindowLatentSourceArrays:
    source: LatentSourceSample
    flattened_values: np.ndarray
    flattened_validity: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.source, LatentSourceSample):
            raise LatentDiffusionError("source must be a LatentSourceSample.")
        values = np.asarray(self.flattened_values, dtype=np.float32).copy()
        validity = np.asarray(self.flattened_validity, dtype=np.bool_).copy()
        if values.ndim != 2 or values.shape[0] <= 0:
            raise LatentDiffusionError("flattened_values must be a non-empty 2D array.")
        if validity.shape != values.shape:
            raise LatentDiffusionError("flattened_validity must match flattened_values shape.")
        if not np.any(validity):
            raise LatentDiffusionError(
                f"sample_id={self.source.sample_id!r} has no valid window features."
            )
        if not np.all(np.isfinite(values[validity])):
            raise LatentDiffusionError("window values must be finite where valid.")
        values.setflags(write=False)
        validity.setflags(write=False)
        object.__setattr__(self, "flattened_values", values)
        object.__setattr__(self, "flattened_validity", validity)


def build_standardized_window_arrays(
    *,
    source_samples: Sequence[LatentSourceSample],
    stats: BfhStandardizationStats,
    target_spec: LatentTargetSpec,
    on_sample_built: Callable[[int, int, str, int], None] | None = None,
) -> tuple[WindowLatentSourceArrays, ...]:
    sources = tuple(source_samples)
    if target_spec.target_type != LATENT_TARGET_TYPE_LEARNED_BFH_WINDOW_LATENT:
        raise LatentDiffusionError("window arrays require learned_bfh_window_latent target.")
    spec = target_spec.temporal_window_spec()
    if not isinstance(spec, TemporalWindowSpec):
        raise LatentDiffusionError("target temporal window spec is invalid.")
    arrays: list[WindowLatentSourceArrays] = []
    for index, source in enumerate(sources, start=1):
        standardized = apply_bfh_standardization(source.vectorized_pose, stats)
        windows = extract_bfh_pose_windows(standardized, spec=spec)
        flat_values = flatten_bfh_pose_windows(windows).reshape(
            windows.values.shape[0],
            spec.window_size * standardized.layout.total_feature_dim,
        )
        flat_validity = flatten_bfh_pose_window_validity(windows).reshape(flat_values.shape)
        window_validity = np.any(flat_validity, axis=1)
        if not np.all(window_validity):
            raise LatentDiffusionError(
                f"sample_id={source.sample_id!r} contains a temporal window with no valid "
                "input features; fix source pose validity before latent autoencoder caching."
            )
        arrays.append(
            WindowLatentSourceArrays(
                source=source,
                flattened_values=flat_values,
                flattened_validity=flat_validity,
            )
        )
        if on_sample_built is not None:
            on_sample_built(index, len(sources), source.sample_id, int(flat_values.shape[0]))
    return tuple(arrays)


def cache_encoded_window_latent_sequences(
    *,
    window_arrays: Sequence[WindowLatentSourceArrays],
    encoded_latents: Sequence[np.ndarray],
    output_root: Path,
    target_spec: LatentTargetSpec,
    on_sample_cached: Callable[[int, int, str, int], None] | None = None,
) -> tuple[LatentManifestEntry, ...]:
    arrays = tuple(window_arrays)
    latents = tuple(np.asarray(values, dtype=np.float32) for values in encoded_latents)
    if len(arrays) != len(latents):
        raise LatentDiffusionError("encoded_latents must align with window_arrays.")
    entries: list[LatentManifestEntry] = []
    root = Path(output_root)
    for index, (item, values) in enumerate(zip(arrays, latents, strict=True), start=1):
        source = item.source
        if values.shape != (item.flattened_values.shape[0], target_spec.latent_dim):
            raise LatentDiffusionError(
                "encoded window latent shape must be (latent_count, target_spec.latent_dim)."
            )
        if not np.all(np.isfinite(values)):
            raise LatentDiffusionError("encoded window latents must be finite.")
        latent_count = int(values.shape[0])
        sequence = LatentSequence(
            schema_version=LATENT_SEQUENCE_SCHEMA_VERSION,
            sample_id=source.sample_id,
            source_sentence_name=source.source_sentence_name,
            split=source.split,
            values=values,
            validity_mask=np.ones(values.shape, dtype=np.bool_),
            frame_count=source.frame_count,
            latent_count=latent_count,
            latent_dim=target_spec.latent_dim,
            target_spec=target_spec,
        )
        latent_path = root / f"{_safe_path_token(source.sample_id)}.npz"
        write_latent_sequence_npz(latent_path, sequence)
        entries.append(
            LatentManifestEntry(
                schema_version=LATENT_MANIFEST_SCHEMA_VERSION,
                sample_id=source.sample_id,
                source_sentence_name=source.source_sentence_name,
                split=source.split,
                latent_path=latent_path,
                frame_count=source.frame_count,
                latent_count=sequence.latent_count,
                latent_dim=sequence.latent_dim,
                target_type=target_spec.target_type,
                temporal_granularity=target_spec.temporal_granularity,
                window_size=target_spec.window_size,
                stride=target_spec.stride,
                issues=(),
            )
        )
        if on_sample_cached is not None:
            on_sample_cached(index, len(arrays), source.sample_id, sequence.latent_count)
    return tuple(entries)


def _safe_path_token(value: str) -> str:
    _require_text(value, "sample_id")
    if value in {".", ".."} or "/" in value or "\\" in value or "." in value:
        raise LatentDiffusionError("sample_id must be a safe path token for latent cache output.")
    return value


def _required_learned_latent_dim(value: int | None) -> int:
    _require_positive_int(value, "learned_latent_dim")
    return int(value)


def _require_text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise LatentDiffusionError(f"{name} must be non-empty.")


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


def _require_positive_int(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise LatentDiffusionError(f"{name} must be a positive integer.")


__all__ = [
    "LatentFrameTrainingSample",
    "LatentSourceSample",
    "WindowLatentSourceArrays",
    "build_latent_source_samples",
    "build_latent_source_surface",
    "build_latent_sequence_surface_from_source_surface",
    "build_latent_window_surface_from_source_surface",
    "build_latent_target_spec",
    "build_standardized_window_arrays",
    "cache_encoded_window_latent_sequences",
    "cache_latent_sequences",
]
