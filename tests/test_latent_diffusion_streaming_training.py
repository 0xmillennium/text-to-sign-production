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
from text_to_sign_production.modeling.candidates.latent_diffusion.autoencoder import (
    build_temporal_window_autoencoder,
    train_latent_autoencoder_from_surfaces,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.config import (
    LatentAutoencoderConfig,
    LatentTargetConfig,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.dataset import (
    build_latent_target_spec,
    build_latent_window_surface_from_source_surface,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.latents import (
    LATENT_TARGET_TYPE_LEARNED_BFH_WINDOW_LATENT,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.provider import (
    LatentDiffusionProvider,
)
from text_to_sign_production.modeling.data_surfaces import ModelDataSurfaceWriter


def test_latent_training_capability_is_verified_after_streaming_migration() -> None:
    capability = LatentDiffusionProvider().full_data_pipeline_capability

    assert capability.full_training_data_mode == "streaming_sharded"
    assert capability.verified is True
    assert capability.is_full_safe
    assert capability.verification_evidence


def test_latent_autoencoder_trains_from_window_surface_reader(tmp_path) -> None:
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
        learned_latent_dim=3,
    )
    surface = build_latent_window_surface_from_source_surface(
        source_surface=source_surface,
        stats=stats,
        surface_root=tmp_path / "windows",
        source_manifest_sha256="a" * 64,
        provider_config_sha256="b" * 64,
        cache_key="window-train-test",
        run_mode="full",
        manifest_entry_count=1,
        target_spec=target_spec,
    )
    config = LatentAutoencoderConfig(
        architecture="window_mlp_autoencoder",
        latent_dim=3,
        hidden_dim=8,
        max_epochs=1,
        batch_size=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        reconstruction_loss="masked_mse",
        velocity_loss_weight=0.0,
    )
    model = build_temporal_window_autoencoder(
        input_dim=target_spec.window_size * target_spec.base_feature_dim,
        config=config,
    )

    metrics = train_latent_autoencoder_from_surfaces(
        model=model,
        train_window_surface=surface,
        validation_window_surface=surface,
        config=config,
        window_size=target_spec.window_size,
        feature_dim=target_spec.base_feature_dim,
        device=torch.device("cpu"),
    )

    assert metrics["train_autoencoder_loss"] >= 0.0
    assert metrics["validation_autoencoder_loss"] >= 0.0


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
