from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.backbones.bfh_standardization import (
    BfhStandardizationAccumulator,
)
from text_to_sign_production.modeling.backbones.bfh_vectorization import (
    BfhVectorizedPose,
    default_bfh_tensor_layout,
    flatten_bfh_vectorized_pose,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.config import (
    LatentTargetConfig,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.dataset import (
    build_latent_target_spec,
    build_latent_window_surface_from_source_surface,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.latents import (
    LATENT_TARGET_TYPE_LEARNED_BFH_WINDOW_LATENT,
)
from text_to_sign_production.modeling.data.temporal_windows import temporal_window_starts
from text_to_sign_production.modeling.data_surfaces import (
    ModelDataSurfaceReader,
    ModelDataSurfaceWriter,
)


def test_latent_window_surface_builds_from_source_shards_without_full_arrays(tmp_path: Path) -> None:
    source_surface, stats = _source_surface(tmp_path)
    layout = default_bfh_tensor_layout()
    target_spec = build_latent_target_spec(
        config=LatentTargetConfig(
            target_type=LATENT_TARGET_TYPE_LEARNED_BFH_WINDOW_LATENT,
            temporal_granularity="window",
            window_size=2,
            stride=1,
            coordinate_mode="xy",
            confidence_policy="mask_only",
            standardization="train_split",
        ),
        layout=layout,
        learned_latent_dim=4,
    )
    built_shards: list[tuple[int, int, int]] = []

    surface = build_latent_window_surface_from_source_surface(
        source_surface=source_surface,
        stats=stats,
        surface_root=tmp_path / "windows",
        source_manifest_sha256="a" * 64,
        provider_config_sha256="b" * 64,
        cache_key="window-test",
        run_mode="full",
        manifest_entry_count=1,
        target_spec=target_spec,
        on_shard_built=lambda index, total, windows: built_shards.append(
            (index, total, windows)
        ),
    )

    expected_windows = len(
        temporal_window_starts(frame_count=3, spec=target_spec.temporal_window_spec())
    )
    assert surface.metadata.surface_kind == "latent_windows"
    assert surface.metadata.unit_count == expected_windows
    assert surface.metadata.loaded_sample_count == 1
    assert built_shards == [(1, 1, expected_windows)]

    shard = next(ModelDataSurfaceReader(surface).iter_shards())
    assert shard["windows"].shape == (expected_windows, 2, layout.total_feature_dim)
    assert shard["window_mask"].shape == shard["windows"].shape
    assert torch.all(shard["window_mask"])
    assert shard["source_index"].tolist() == [0, 0]
    assert shard["window_index"].tolist() == [0, 1]


def _source_surface(tmp_path: Path):
    layout = default_bfh_tensor_layout()
    values = np.arange(
        3 * layout.total_joint_count * layout.coordinate_dimensions,
        dtype=np.float32,
    ).reshape(3, layout.total_joint_count, layout.coordinate_dimensions)
    pose = BfhVectorizedPose(
        layout=layout,
        values=values,
        validity_mask=np.ones((3, layout.total_joint_count), dtype=np.bool_),
        frame_validity_mask=np.ones((3,), dtype=np.bool_),
        confidence_values=np.ones((3, layout.total_joint_count), dtype=np.float32),
        frame_count=3,
        source_sample_id="sample-1",
    )
    accumulator = BfhStandardizationAccumulator()
    accumulator.update(pose)
    stats = accumulator.finalize(epsilon=1e-6, missing_observation_policy="raise")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    writer = ModelDataSurfaceWriter(
        root=tmp_path / "source",
        provider_key="latent_diffusion",
        surface_kind="latent_source_sequences",
        split=SampleSplit.TRAIN.value,
        manifest_family="tiered:clean:included",
        source_manifest_path=manifest,
        source_manifest_sha256="a" * 64,
        provider_config_sha256="b" * 64,
        cache_key="source-test",
        manifest_entry_count=1,
        run_mode="full",
        data_version="test",
    )
    writer.append_units(
        {
            "pose_values": flatten_bfh_vectorized_pose(pose),
            "validity_mask": np.repeat(
                pose.validity_mask,
                layout.coordinate_dimensions,
                axis=1,
            ),
            "source_index": np.zeros((3,), dtype=np.int64),
            "frame_index": np.arange(3, dtype=np.int64),
        },
        sample_count=1,
        frame_count=3,
    )
    return writer.close(), stats
