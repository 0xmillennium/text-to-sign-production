from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.backbones.bfh_standardization import (
    BfhStandardizationAccumulator,
)
from text_to_sign_production.modeling.backbones.bfh_vectorization import (
    BfhVectorizedPose,
    flatten_bfh_vectorized_pose,
)
from text_to_sign_production.modeling.candidates.articulator_aware import trainer as articulator_trainer
from text_to_sign_production.modeling.candidates.latent_diffusion import dataset as latent_dataset
from text_to_sign_production.modeling.data_surfaces import ModelDataSurfaceWriter


class ForbiddenCall:
    def __init__(self, name: str):
        self.name = name
        self.called = False

    def __call__(self, *args, **kwargs):
        del args, kwargs
        self.called = True
        raise AssertionError(f"Forbidden eager full-path call: {self.name}")


def test_forbidden_call_harness_records_and_raises() -> None:
    forbidden = ForbiddenCall("legacy_builder")

    with pytest.raises(AssertionError, match="Forbidden eager full-path call: legacy_builder"):
        forbidden()

    assert forbidden.called is True


def test_articulator_surface_trainer_does_not_call_legacy_delegate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    forbidden = ForbiddenCall("_train_articulator_model_legacy")
    monkeypatch.setattr(articulator_trainer, "_train_articulator_model_legacy", forbidden)
    surface = _minimal_articulator_surface(tmp_path)

    with pytest.raises(Exception, match="config must be an ArticulatorAwareConfig"):
        articulator_trainer.train_articulator_model_from_surfaces(
            config=object(),
            train_surface=surface,
            validation_surface=surface,
            train_source_surface=surface,
            validation_source_surface=surface,
            partition_policy=object(),
            mask_config=object(),
            weighting=object(),
            output_root=tmp_path / "out",
            run_name="behavior",
            seed=0,
        )

    assert forbidden.called is False


def test_latent_temporal_window_surface_does_not_call_eager_array_builder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from text_to_sign_production.modeling.backbones.bfh_vectorization import (
        default_bfh_tensor_layout,
    )
    from text_to_sign_production.modeling.candidates.latent_diffusion.config import (
        LatentTargetConfig,
    )
    from text_to_sign_production.modeling.candidates.latent_diffusion.latents import (
        LATENT_TARGET_TYPE_LEARNED_BFH_WINDOW_LATENT,
    )

    forbidden = ForbiddenCall("build_standardized_window_arrays")
    monkeypatch.setattr(latent_dataset, "build_standardized_window_arrays", forbidden)
    source_surface, stats = _latent_source_surface(tmp_path)
    target_spec = latent_dataset.build_latent_target_spec(
        config=LatentTargetConfig(
            target_type=LATENT_TARGET_TYPE_LEARNED_BFH_WINDOW_LATENT,
            temporal_granularity="window",
            window_size=2,
            stride=1,
            coordinate_mode="xy",
            confidence_policy="mask_only",
            standardization="train_split",
        ),
        layout=default_bfh_tensor_layout(),
        learned_latent_dim=4,
    )

    surface = latent_dataset.build_latent_window_surface_from_source_surface(
        source_surface=source_surface,
        stats=stats,
        surface_root=tmp_path / "latent_windows",
        source_manifest_sha256="a" * 64,
        provider_config_sha256="b" * 64,
        cache_key="latent-window-behavior",
        run_mode="full",
        manifest_entry_count=1,
        target_spec=target_spec,
    )

    assert surface.metadata.surface_kind == "latent_windows"
    assert surface.metadata.unit_count > 0
    assert forbidden.called is False


def test_all_target_providers_now_report_behavior_evidence() -> None:
    from text_to_sign_production.modeling.candidates.articulator_aware.provider import (
        ArticulatorAwareProvider,
    )
    from text_to_sign_production.modeling.candidates.base_direct.provider import (
        BaseDirectProvider,
    )
    from text_to_sign_production.modeling.candidates.latent_diffusion.provider import (
        LatentDiffusionProvider,
    )
    from text_to_sign_production.modeling.candidates.learned_pose_token.provider import (
        LearnedPoseTokenProvider,
    )

    providers = (
        BaseDirectProvider(),
        LearnedPoseTokenProvider(),
        LatentDiffusionProvider(),
        ArticulatorAwareProvider(),
    )

    for provider in providers:
        capability = provider.full_data_pipeline_capability
        assert capability.verified is True
        assert capability.verification_evidence
        assert capability.is_full_safe is True


def _minimal_articulator_surface(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    root = tmp_path / "surface"
    writer = ModelDataSurfaceWriter(
        root=root,
        provider_key="articulator_aware",
        surface_kind="articulator_frame_units",
        split="train",
        manifest_family="tiered:clean:included",
        source_manifest_path=manifest,
        source_manifest_sha256="a" * 64,
        provider_config_sha256="b" * 64,
        cache_key="c" * 32,
        manifest_entry_count=1,
        run_mode="full",
        data_version="test",
    )
    writer.append_units(
        {
            "target_values": torch.ones((1, 4)),
            "validity_mask": torch.ones((1, 4), dtype=torch.bool),
            "source_index": torch.zeros((1,), dtype=torch.long),
            "frame_index": torch.zeros((1,), dtype=torch.long),
            "body_mask": torch.ones((1, 4), dtype=torch.bool),
        },
        sample_count=1,
        frame_count=1,
    )
    surface = writer.close()
    (root / "sources.jsonl").write_text(
        json.dumps(
            {
                "source_index": 0,
                "sample_id": "sample-1",
                "source_sentence_name": "sentence-1",
                "text": "hello",
                "source_video_id": "video-1",
                "source_sentence_id": "sentence-1",
                "reference_payload_ref": "payloads/sample-1.json",
                "split": "train",
                "frame_count": 1,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return surface


def _latent_source_surface(tmp_path: Path):
    from text_to_sign_production.modeling.backbones.bfh_vectorization import (
        default_bfh_tensor_layout,
    )

    layout = default_bfh_tensor_layout()
    values = np.ones(
        (3, layout.total_joint_count, layout.coordinate_dimensions),
        dtype=np.float32,
    )
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
        root=tmp_path / "latent_source",
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
