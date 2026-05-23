from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch

from text_to_sign_production.modeling.backbones.bfh_vectorization import (
    default_bfh_tensor_layout,
)
from text_to_sign_production.modeling.candidates.articulator_aware.config import (
    load_articulator_aware_config,
)
from text_to_sign_production.modeling.candidates.articulator_aware.losses import (
    build_channel_loss_weighting_policy,
)
from text_to_sign_production.modeling.candidates.articulator_aware.partitions import (
    build_articulator_partition_policy,
)
from text_to_sign_production.modeling.candidates.articulator_aware.provider import (
    ArticulatorAwareProvider,
)
from text_to_sign_production.modeling.candidates.articulator_aware.trainer import (
    train_articulator_model_from_surfaces,
)
from text_to_sign_production.modeling.data_surfaces import ModelDataSurfaceWriter


def test_articulator_training_capability_is_verified_after_surface_trainer_migration() -> None:
    capability = ArticulatorAwareProvider().full_data_pipeline_capability

    assert capability.full_training_data_mode == "streaming_sharded"
    assert capability.verified is True
    assert capability.verification_evidence
    assert capability.is_full_safe


def test_articulator_surface_trainer_reads_frame_surface_without_legacy_delegate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from text_to_sign_production.modeling.candidates.articulator_aware import trainer

    def fail_legacy(*args, **kwargs):
        raise AssertionError("legacy trainer must not be called")

    monkeypatch.setattr(trainer, "_train_articulator_model_legacy", fail_legacy)
    config = load_articulator_aware_config(Path("configs/modeling/articulator_aware.yaml"))
    config = replace(
        config,
        structure_variant=replace(config.structure_variant, hidden_dim=8),
        training=replace(
            config.training,
            max_epochs=1,
            batch_size=1,
            frame_batch_size=2,
            device="cpu",
        ),
    )
    layout = default_bfh_tensor_layout()
    policy = build_articulator_partition_policy(
        config=config.partition_policy,
        layout=layout,
    )
    result = train_articulator_model_from_surfaces(
        config=config,
        train_surface=_surface(tmp_path / "train", layout.total_feature_dim),
        validation_surface=_surface(tmp_path / "val", layout.total_feature_dim),
        train_source_surface=_surface(tmp_path / "train_source", layout.total_feature_dim),
        validation_source_surface=_surface(tmp_path / "val_source", layout.total_feature_dim),
        partition_policy=policy,
        mask_config=config.mask_strategy,
        weighting=build_channel_loss_weighting_policy(config.loss_weighting),
        output_root=tmp_path / "out",
        run_name="surface",
        seed=0,
        precision_policy_name="fp32",
    )

    assert result.best_checkpoint_path.is_file()
    assert result.training_metrics_path.is_file()


def _surface(root: Path, feature_dim: int):
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
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
    values = torch.zeros((2, feature_dim), dtype=torch.float32)
    mask = torch.ones((2, feature_dim), dtype=torch.bool)
    writer.append_units(
        {
            "target_values": values,
            "validity_mask": mask,
            "source_index": torch.zeros((2,), dtype=torch.long),
            "frame_index": torch.arange(2, dtype=torch.long),
            "body_mask": mask,
            "left_hand_mask": mask,
            "right_hand_mask": mask,
            "face_mask": mask,
        },
        sample_count=1,
        frame_count=2,
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
                "frame_count": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return surface
