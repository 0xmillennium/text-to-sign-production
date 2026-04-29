"""Reusable Sprint 3 modeling fixtures for workflow tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

import text_to_sign_production.core.paths as paths_mod
from tests.support.builders.manifests import processed_record, write_jsonl_records
from tests.support.builders.samples import write_processed_sample_npz
from text_to_sign_production.modeling.contracts import (
    BASELINE_ID,
    BASELINE_NAME,
    BASELINE_ROLE,
    CONFIDENCE_POLICY,
    DEFAULT_CHANNEL_WEIGHTS,
    LENGTH_POLICY,
    PREDICTION_MANIFEST_SCHEMA_VERSION,
    PREDICTION_SCHEMA_VERSION,
)
from text_to_sign_production.modeling.data import M0_CHANNEL_POLICY, M0_TARGET_CHANNELS
from text_to_sign_production.modeling.training.config import (
    BaselineBackboneConfig,
    BaselineCheckpointConfig,
    BaselineDataConfig,
    BaselineIdentityConfig,
    BaselineLoopConfig,
    BaselineLossConfig,
    BaselineModelConfig,
    BaselineOptimizerConfig,
    BaselineSchedulerConfig,
    BaselineTargetStandardizationConfig,
    BaselineTrainingConfig,
)


@dataclass(frozen=True, slots=True)
class ModelingWorkspace:
    """Tiny processed Dataset Build workspace used by Sprint 3 tests."""

    root: Path
    config_path: Path
    train_manifest_path: Path
    val_manifest_path: Path
    artifact_runs_root: Path
    run_name: str = "baseline-test"


def patch_modeling_repo_root(monkeypatch: Any, root: Path) -> None:
    """Patch the repo-root boundary used by repo-relative Sprint 3 test paths."""

    monkeypatch.setattr(paths_mod, "DEFAULT_REPO_ROOT", root.resolve())
    monkeypatch.setattr(paths_mod, "DEFAULT_DATA_ROOT", (root / "data").resolve())


def processed_sample_relative_path(split: str = "train", sample_id: str = "sample") -> str:
    """Return the processed Dataset Build sample path stored in manifests."""

    return f"data/processed/v1/samples/{split}/{sample_id}.npz"


def baseline_config_payload(
    *,
    checkpoint_output_dir: str = ("runs/base/m0-direct-text-to-full-bfh/baseline-test/checkpoints"),
    backbone_name: str = "dummy",
) -> dict[str, Any]:
    """Return the default tiny Sprint 3 baseline config payload for tests."""

    return {
        "baseline": {
            "id": BASELINE_ID,
            "name": BASELINE_NAME,
            "role": BASELINE_ROLE,
            "channels": list(M0_TARGET_CHANNELS),
            "channel_policy": M0_CHANNEL_POLICY,
            "length_policy": LENGTH_POLICY,
            "confidence_policy": CONFIDENCE_POLICY,
            "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
            "prediction_manifest_schema_version": PREDICTION_MANIFEST_SCHEMA_VERSION,
        },
        "data": {
            "train_manifest": "data/processed/v1/manifests/train.jsonl",
            "val_manifest": "data/processed/v1/manifests/val.jsonl",
            "train_split": "train",
            "val_split": "val",
            "prediction_splits": ["val"],
        },
        "backbone": {
            "name": backbone_name,
            "revision": "main",
            "max_length": 8,
            "local_files_only": True,
            "trainable": False,
            "freeze_strategy": "none",
            "encoder_learning_rate": 0.0,
        },
        "model": {
            "decoder_hidden_dim": 8,
            "decoder_layers": 1,
            "decoder_dropout": 0.0,
            "frame_position_encoding_dim": 8,
        },
        "loss": {
            "channel_weights": dict(DEFAULT_CHANNEL_WEIGHTS),
        },
        "training": {
            "epochs": 1,
            "min_epochs": 1,
            "early_stopping_patience": 5,
            "early_stopping_metric": "val_loss",
            "early_stopping_mode": "min",
            "validate_every_epochs": 1,
            "batch_size": 1,
            "shuffle_train": False,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": None,
            "non_blocking_transfers": False,
            "seed": 5,
            "device": "cpu",
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "mixed_precision": "no",
            "progress_interval_batches": 1,
            "length_bucketed_batching": False,
        },
        "optimizer": {
            "name": "adamw",
            "decoder_learning_rate": 0.001,
            "weight_decay": 0.0,
        },
        "scheduler": {
            "name": "constant",
            "warmup_ratio": 0.0,
        },
        "target_standardization": {
            "enabled": False,
            "epsilon": 1e-6,
        },
        "checkpoint": {
            "output_dir": checkpoint_output_dir,
        },
    }


def write_processed_modeling_split(
    root: Path,
    *,
    split: str,
    sample_ids: tuple[str, ...] = ("sample",),
) -> None:
    """Write a tiny processed Dataset Build split for Sprint 3 workflow tests."""

    records: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        sample_path = f"data/processed/v1/samples/{split}/{sample_id}.npz"
        write_processed_sample_npz(root / sample_path)
        records.append(
            processed_record(
                sample_path,
                split=split,
                sample_id=sample_id,
                text=f"text for {sample_id}",
            )
        )
    write_jsonl_records(root / f"data/processed/v1/manifests/{split}.jsonl", records)


def write_processed_modeling_manifest(
    root: Path,
    records: list[dict[str, Any]],
    *,
    split: str = "train",
) -> Path:
    """Write a processed modeling manifest and return its path."""

    manifest_path = root / f"data/processed/v1/manifests/{split}.jsonl"
    write_jsonl_records(manifest_path, records)
    return manifest_path


def write_processed_modeling_sample(
    root: Path,
    *,
    split: str = "train",
    sample_id: str = "sample",
    num_frames: int = 2,
    frame_valid_mask: npt.NDArray[np.bool_] | None = None,
    overrides: dict[str, Any] | None = None,
    drop_keys: tuple[str, ...] = (),
) -> str:
    """Write a processed sample payload and return the repo-relative sample path."""

    sample_relative_path = processed_sample_relative_path(split=split, sample_id=sample_id)
    resolved_overrides = {} if overrides is None else dict(overrides)
    if frame_valid_mask is not None:
        resolved_overrides["frame_valid_mask"] = frame_valid_mask
    write_processed_sample_npz(
        root / sample_relative_path,
        num_frames=num_frames,
        overrides=resolved_overrides,
        drop_keys=drop_keys,
    )
    return sample_relative_path


def write_baseline_modeling_config(
    root: Path,
    *,
    checkpoint_output_dir: str = ("runs/base/m0-direct-text-to-full-bfh/baseline-test/checkpoints"),
    backbone_name: str = "dummy",
) -> Path:
    """Write a tiny baseline config that points at processed Dataset Build manifests."""

    config_path = root / "baseline.yaml"
    config_path.write_text(
        yaml.safe_dump(
            baseline_config_payload(
                checkpoint_output_dir=checkpoint_output_dir,
                backbone_name=backbone_name,
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path


def write_baseline_config_payload(path: Path, payload: dict[str, Any]) -> Path:
    """Write an explicit baseline config payload for config-contract tests."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def baseline_training_config(
    root: Path,
    *,
    checkpoint_dir: Path | None = None,
) -> BaselineTrainingConfig:
    """Return a typed tiny baseline training config for unit tests."""

    resolved_checkpoint_dir = root if checkpoint_dir is None else checkpoint_dir
    return BaselineTrainingConfig(
        source_path=root / "baseline.yaml",
        raw_config={},
        baseline=BaselineIdentityConfig(
            baseline_id=BASELINE_ID,
            name=BASELINE_NAME,
            role=BASELINE_ROLE,
            channels=M0_TARGET_CHANNELS,
            channel_policy=M0_CHANNEL_POLICY,
            length_policy=LENGTH_POLICY,
            confidence_policy=CONFIDENCE_POLICY,
            prediction_schema_version=PREDICTION_SCHEMA_VERSION,
            prediction_manifest_schema_version=PREDICTION_MANIFEST_SCHEMA_VERSION,
        ),
        data=BaselineDataConfig(
            train_manifest=root / "train.jsonl",
            val_manifest=root / "val.jsonl",
            train_split="train",
            val_split="val",
            prediction_splits=("val",),
        ),
        backbone=BaselineBackboneConfig(
            name="dummy",
            revision="main",
            max_length=8,
            local_files_only=True,
            trainable=False,
            freeze_strategy="none",
            encoder_learning_rate=0.0,
        ),
        model=BaselineModelConfig(
            decoder_hidden_dim=8,
            decoder_layers=1,
            decoder_dropout=0.0,
            frame_position_encoding_dim=8,
        ),
        loss=BaselineLossConfig(
            channel_weights=dict(DEFAULT_CHANNEL_WEIGHTS),
        ),
        training=BaselineLoopConfig(
            epochs=1,
            min_epochs=1,
            early_stopping_patience=5,
            early_stopping_metric="val_loss",
            early_stopping_mode="min",
            validate_every_epochs=1,
            batch_size=1,
            shuffle_train=False,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=None,
            non_blocking_transfers=False,
            seed=5,
            device="cpu",
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            mixed_precision="no",
            progress_interval_batches=1,
            length_bucketed_batching=False,
        ),
        optimizer=BaselineOptimizerConfig(
            name="adamw",
            decoder_learning_rate=0.001,
            weight_decay=0.0,
        ),
        scheduler=BaselineSchedulerConfig(
            name="constant",
            warmup_ratio=0.0,
        ),
        target_standardization=BaselineTargetStandardizationConfig(
            enabled=False,
            epsilon=1e-6,
        ),
        checkpoint=BaselineCheckpointConfig(output_dir=resolved_checkpoint_dir),
    )


def write_tiny_baseline_modeling_workspace(
    root: Path,
    *,
    train_sample_ids: tuple[str, ...] = ("train-sample",),
    val_sample_ids: tuple[str, ...] = ("val-sample",),
    run_name: str = "baseline-test",
    checkpoint_output_dir: str = ("runs/base/m0-direct-text-to-full-bfh/baseline-test/checkpoints"),
    backbone_name: str = "dummy",
) -> ModelingWorkspace:
    """Write tiny processed train/val inputs plus a baseline config."""

    write_processed_modeling_split(root, split="train", sample_ids=train_sample_ids)
    write_processed_modeling_split(root, split="val", sample_ids=val_sample_ids)
    config_path = write_baseline_modeling_config(
        root,
        checkpoint_output_dir=checkpoint_output_dir,
        backbone_name=backbone_name,
    )
    return ModelingWorkspace(
        root=root,
        config_path=config_path,
        train_manifest_path=root / "data/processed/v1/manifests/train.jsonl",
        val_manifest_path=root / "data/processed/v1/manifests/val.jsonl",
        artifact_runs_root=root / "runs",
        run_name=run_name,
    )


def write_fake_training_outputs(checkpoint_dir: Path) -> None:
    """Write the minimal checkpoint/summary surface used by workflow resume tests."""

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "last.pt").write_bytes(b"last checkpoint")
    (checkpoint_dir / "best.pt").write_bytes(b"best checkpoint")
    (checkpoint_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "backbone_name": "dummy",
                "checkpoint_output_path": checkpoint_dir.as_posix(),
                "last_checkpoint_path": (checkpoint_dir / "last.pt").as_posix(),
                "best_checkpoint_path": (checkpoint_dir / "best.pt").as_posix(),
                "target_channels": ["body", "left_hand", "right_hand"],
            }
        ),
        encoding="utf-8",
    )


def write_fake_qualitative_outputs(output_dir: Path) -> None:
    """Write the minimal qualitative panel surface used by workflow resume tests."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "panel_definition.json").write_text(
        json.dumps(
            {
                "schema_version": "t2sp-qualitative-panel-v1",
                "split": "val",
                "sample_ids": ["val-sample"],
                "selection_rule": "test",
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "records.jsonl").write_text('{"sample_id": "val-sample"}\n', encoding="utf-8")
    (output_dir / "panel_summary.json").write_text(
        json.dumps({"schema_version": "t2sp-qualitative-export-v1", "sample_count": 1}),
        encoding="utf-8",
    )
    (output_dir / "baseline_evidence_bundle.json").write_text(
        json.dumps({"schema_version": "t2sp-baseline-evidence-v1"}),
        encoding="utf-8",
    )


def touch_file(path: Path, content: str = "x") -> Path:
    """Create a small text file and return its path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
