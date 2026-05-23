from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from text_to_sign_production.modeling.candidates.articulator_aware.config import (
    load_articulator_aware_config,
)
from text_to_sign_production.modeling.candidates.articulator_aware.losses import (
    build_channel_loss_weighting_policy,
)
from text_to_sign_production.modeling.candidates.articulator_aware.partitions import (
    build_articulator_partition_policy,
)
from text_to_sign_production.modeling.candidates.articulator_aware.trainer import (
    train_articulator_model_from_surfaces,
)
from text_to_sign_production.modeling.candidates.base_direct.provider import BaseDirectProvider
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
    build_latent_sequence_surface_from_source_surface,
    build_latent_window_surface_from_source_surface,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.latents import (
    LATENT_TARGET_TYPE_LEARNED_BFH_WINDOW_LATENT,
    LATENT_TARGET_TYPE_STANDARDIZED_BFH_FRAME,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.provider import (
    LatentDenoiserBatch,
    iter_latent_sequence_batches_from_surface,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.exporter import (
    export_predicted_pose_samples_incremental,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.spec import (
    LEARNED_POSE_TOKEN_MODEL_KEY,
)
from text_to_sign_production.modeling.candidates.bootstrap import (
    ensure_model_provider_registered,
)
from text_to_sign_production.modeling.candidates import ModelRunRequest
from text_to_sign_production.modeling.candidates.registry import require_model_provider
from text_to_sign_production.modeling.data import dataset as dataset_module
from text_to_sign_production.modeling.data import parse_modeling_manifest_family
from text_to_sign_production.modeling.research import ModelKey
from text_to_sign_production.modeling.backbones.bfh_vectorization import default_bfh_tensor_layout
from text_to_sign_production.workflows.model.processing.calibration import (
    build_provider_real_calibration_context,
    run_provider_real_calibration,
    run_synthetic_device_smoke_benchmark,
)
from text_to_sign_production.workflows.model.processing.provider import load_model_provider_config
from helpers.real_modeling_topology import write_real_modeling_topology

from test_articulator_aware_streaming_training import _surface as _articulator_surface
from test_latent_diffusion_streaming_training import _source_surface as _latent_source_surface
from test_learned_pose_token_generated_pose_paths import _decoded_sample


class ForbiddenCall:
    def __init__(self, name: str):
        self.name = name
        self.called = False

    def __call__(self, *args, **kwargs):
        del args, kwargs
        self.called = True
        raise AssertionError(f"Forbidden full-path eager call: {self.name}")


def assert_provider_capability_verified_after_behavior(provider_key: str) -> None:
    ensure_model_provider_registered(provider_key)
    capability = require_model_provider(provider_key).full_data_pipeline_capability
    assert capability.verified is True
    assert capability.limitations == ()
    assert capability.is_full_safe is True
    assert all(item.startswith(("behavior:", "calibration:", "progress:")) for item in capability.verification_evidence)


def assert_no_authoritative_synthetic_calibration() -> None:
    rows = run_synthetic_device_smoke_benchmark({"batch_size": [1]}, max_batches_per_candidate=1)
    assert rows
    assert all(row["benchmark_type"] == "synthetic_device_smoke" for row in rows)


def assert_incremental_writer_used(path: Path) -> None:
    assert (path / "manifest.jsonl").is_file()
    assert (path / "samples").is_dir()


def test_final_capability_evidence_is_behavior_named() -> None:
    for provider_key in (
        "learned_pose_token",
        "latent_diffusion",
        "articulator_aware",
        "base_direct",
    ):
        assert_provider_capability_verified_after_behavior(provider_key)


def test_final_calibration_separates_synthetic_from_provider_real() -> None:
    assert_no_authoritative_synthetic_calibration()
    rows = run_provider_real_calibration(
        context=_calibration_context("latent_diffusion"),
        candidates={"denoiser_batch_size": [2]},
        max_batches_per_candidate=1,
        warmup_batches=0,
    )
    assert rows[0]["benchmark_type"] == "provider_real"
    assert rows[0]["surface_reader_used"] is True
    assert rows[0]["forward_executed"] is True
    assert rows[0]["backward_executed"] is True
    assert rows[0]["surface_derived_primary_input"] is True
    assert rows[0]["trace_validation_passed"] is True


def _calibration_context(provider_key: str):
    root = Path(tempfile.mkdtemp(prefix="t2sp-test-real-topology-"))
    topology = write_real_modeling_topology(root / "runtime")
    model_key = ModelKey(provider_key)
    request = ModelRunRequest(
        model_key=model_key,
        run_name="run",
        manifest_family=parse_modeling_manifest_family("tiered:clean:included"),
        config_path=Path(f"configs/modeling/{provider_key}.yaml"),
    )
    ensure_model_provider_registered(model_key)
    provider = require_model_provider(model_key)
    loaded = load_model_provider_config(provider, request).loaded_config
    return build_provider_real_calibration_context(
        loaded_config=loaded,
        stage_plan=provider.plan_stages(request, loaded),
        working_dir=root,
        topology=topology,
    )


def test_articulator_full_training_uses_surface_reader_valid_path(monkeypatch, tmp_path: Path) -> None:
    from dataclasses import replace
    from text_to_sign_production.modeling.candidates.articulator_aware import trainer

    forbidden = ForbiddenCall("_train_articulator_model_legacy")
    monkeypatch.setattr(trainer, "_train_articulator_model_legacy", forbidden)
    config = load_articulator_aware_config(Path("configs/modeling/articulator_aware.yaml"))
    config = replace(
        config,
        structure_variant=replace(config.structure_variant, hidden_dim=8),
        training=replace(config.training, max_epochs=1, batch_size=1, frame_batch_size=2, device="cpu"),
    )
    layout = default_bfh_tensor_layout()
    policy = build_articulator_partition_policy(config=config.partition_policy, layout=layout)
    result = train_articulator_model_from_surfaces(
        config=config,
        train_surface=_articulator_surface(tmp_path / "train", layout.total_feature_dim),
        validation_surface=_articulator_surface(tmp_path / "val", layout.total_feature_dim),
        train_source_surface=_articulator_surface(tmp_path / "train_source", layout.total_feature_dim),
        validation_source_surface=_articulator_surface(tmp_path / "val_source", layout.total_feature_dim),
        partition_policy=policy,
        mask_config=config.mask_strategy,
        weighting=build_channel_loss_weighting_policy(config.loss_weighting),
        output_root=tmp_path / "out",
        run_name="surface",
        seed=0,
        precision_policy_name="fp32",
    )
    assert result.best_checkpoint_path.is_file()
    assert forbidden.called is False


def test_latent_temporal_autoencoder_uses_window_surface_valid_path(tmp_path: Path) -> None:
    source_surface, stats = _latent_source_surface(tmp_path)
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
        layout=default_bfh_tensor_layout(),
        learned_latent_dim=3,
    )
    surface = build_latent_window_surface_from_source_surface(
        source_surface=source_surface,
        stats=stats,
        surface_root=tmp_path / "windows",
        source_manifest_sha256="a" * 64,
        provider_config_sha256="b" * 64,
        cache_key="window-final-acceptance",
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
    metrics = train_latent_autoencoder_from_surfaces(
        model=build_temporal_window_autoencoder(
            input_dim=target_spec.window_size * target_spec.base_feature_dim,
            config=config,
        ),
        train_window_surface=surface,
        validation_window_surface=surface,
        config=config,
        window_size=target_spec.window_size,
        feature_dim=target_spec.base_feature_dim,
        device=torch.device("cpu"),
    )
    assert metrics["train_autoencoder_loss"] >= 0.0


def test_learned_decode_writer_is_incremental_valid_path(tmp_path: Path) -> None:
    root = tmp_path / "generated_pose"
    written = export_predicted_pose_samples_incremental(
        decoded_samples=(_decoded_sample("sample1") for _ in range(1)),
        output_root=root,
        producer_key=LEARNED_POSE_TOKEN_MODEL_KEY,
        producer_stage="decode_to_pose",
        run_name="run001",
        split="val",
        generation_mode="deterministic",
        seed=1,
    )
    assert written.sample_count == 1
    assert written.payload_count == 1
    assert len(written.preview_payload_paths) == 1
    assert not hasattr(written, "payload_paths")
    assert not hasattr(written, "entries")
    assert_incremental_writer_used(root)


def test_final_acceptance_generated_pose_results_bounded(tmp_path: Path) -> None:
    root = tmp_path / "generated_pose"
    written = export_predicted_pose_samples_incremental(
        decoded_samples=(_decoded_sample(f"sample{index}") for index in range(12)),
        output_root=root,
        producer_key=LEARNED_POSE_TOKEN_MODEL_KEY,
        producer_stage="decode_to_pose",
        run_name="run001",
        split="val",
        generation_mode="deterministic",
        seed=1,
    )

    assert written.sample_count == 12
    assert written.payload_count == 12
    assert written.preview_limit == 10
    assert len(written.preview_entries) == 10
    assert len(written.preview_payload_paths) == 10
    assert not hasattr(written, "payload_paths")
    assert not hasattr(written, "entries")


def test_final_acceptance_latent_denoiser_batch_surface_reader(tmp_path: Path, monkeypatch) -> None:
    source_surface, stats = _latent_source_surface(tmp_path)
    target_spec = build_latent_target_spec(
        config=LatentTargetConfig(
            target_type=LATENT_TARGET_TYPE_STANDARDIZED_BFH_FRAME,
            temporal_granularity="frame",
            window_size=1,
            stride=1,
            coordinate_mode="xy",
            confidence_policy="mask_only",
            standardization="train_split",
        ),
        layout=default_bfh_tensor_layout(),
    )
    surface = build_latent_sequence_surface_from_source_surface(
        source_surface=source_surface,
        stats=stats,
        surface_root=tmp_path / "latent_sequences",
        source_manifest_sha256="a" * 64,
        provider_config_sha256="b" * 64,
        cache_key="latent-sequence-batch-final-acceptance",
        run_mode="full",
        manifest_entry_count=1,
        target_spec=target_spec,
    )
    forbidden = ForbiddenCall("_iter_latent_sequences_from_surface")
    monkeypatch.setattr(
        "text_to_sign_production.modeling.candidates.latent_diffusion.provider._iter_latent_sequences_from_surface",
        forbidden,
    )

    batches = tuple(iter_latent_sequence_batches_from_surface(surface, batch_size=2))

    assert batches
    assert all(isinstance(batch, LatentDenoiserBatch) for batch in batches)
    assert sum(int(batch.latent_values.shape[0]) for batch in batches) == surface.metadata.unit_count
    assert forbidden.called is False


def test_base_direct_lazy_loader_does_not_load_all_payloads_before_first_batch(monkeypatch) -> None:
    BaseDirectProvider()
    records = tuple(
        SimpleNamespace(sample_id=f"sample-{index}", sample_path=Path(f"sample-{index}.npz"))
        for index in range(8)
    )
    calls: list[str] = []
    monkeypatch.setattr(dataset_module, "read_processed_modeling_manifest", lambda *args, **kwargs: records)
    monkeypatch.setattr(
        dataset_module,
        "load_processed_pose_sample",
        lambda record: calls.append(record.sample_id) or object(),
    )
    monkeypatch.setattr(
        dataset_module.ProcessedPoseItem,
        "from_manifest_and_sample",
        staticmethod(lambda record, sample: {"sample_id": record.sample_id, "sample": sample}),
    )
    loader = DataLoader(
        dataset_module.ProcessedPoseDataset(Path("manifest.jsonl"), split="train"),
        batch_size=2,
        collate_fn=lambda batch: batch,
    )
    assert calls == []
    first_batch = next(iter(loader))
    assert len(first_batch) == 2
    assert calls == ["sample-0", "sample-1"]
