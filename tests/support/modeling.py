"""Reusable Sprint 3 modeling fixtures for workflow tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

import text_to_sign_production.data.utils as utils_mod
from tests.support.builders.manifests import processed_record, write_jsonl_records
from tests.support.builders.samples import write_processed_sample_npz
from text_to_sign_production.modeling.training.config import (
    BaselineBackboneConfig,
    BaselineCheckpointConfig,
    BaselineDataConfig,
    BaselineLoopConfig,
    BaselineModelConfig,
    BaselineOptimizerConfig,
    BaselineTrainingConfig,
)
from text_to_sign_production.workflows.baseline_modeling import (
    BaselineRunLayout,
    resolve_baseline_run_layout,
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

    @property
    def layout(self) -> BaselineRunLayout:
        return resolve_baseline_run_layout(
            run_name=self.run_name,
            artifact_runs_root=self.artifact_runs_root,
        )


def patch_modeling_repo_root(monkeypatch: Any, root: Path) -> None:
    """Patch the repo-root boundary used by repo-relative Sprint 3 test paths."""

    monkeypatch.setattr(utils_mod, "REPO_ROOT", root.resolve())


def processed_sample_relative_path(split: str = "train", sample_id: str = "sample") -> str:
    """Return the processed Dataset Build sample path stored in manifests."""

    return f"data/processed/v1/samples/{split}/{sample_id}.npz"


def baseline_config_payload(
    *,
    checkpoint_output_dir: str = "outputs/modeling/baseline-test",
    backbone_name: str = "dummy",
) -> dict[str, Any]:
    """Return the default tiny Sprint 3 baseline config payload for tests."""

    return {
        "data": {
            "train_manifest": "data/processed/v1/manifests/train.jsonl",
            "val_manifest": "data/processed/v1/manifests/val.jsonl",
            "train_split": "train",
            "val_split": "val",
        },
        "backbone": {
            "name": backbone_name,
            "max_length": 8,
            "local_files_only": True,
            "freeze": True,
        },
        "model": {
            "decoder_hidden_dim": 8,
            "latent_dim": 6,
        },
        "training": {
            "epochs": 1,
            "batch_size": 1,
            "shuffle_train": False,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": None,
            "non_blocking_transfers": False,
            "seed": 5,
            "device": "cpu",
        },
        "optimizer": {
            "name": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
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
    checkpoint_output_dir: str = "outputs/modeling/baseline-test",
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
        data=BaselineDataConfig(
            train_manifest=root / "train.jsonl",
            val_manifest=root / "val.jsonl",
            train_split="train",
            val_split="val",
        ),
        backbone=BaselineBackboneConfig(
            name="dummy",
            max_length=8,
            local_files_only=True,
            freeze=True,
        ),
        model=BaselineModelConfig(
            decoder_hidden_dim=8,
            latent_dim=6,
        ),
        training=BaselineLoopConfig(
            epochs=1,
            batch_size=1,
            shuffle_train=False,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=None,
            non_blocking_transfers=False,
            seed=5,
            device="cpu",
        ),
        optimizer=BaselineOptimizerConfig(
            name="adamw",
            learning_rate=0.001,
            weight_decay=0.0,
        ),
        checkpoint=BaselineCheckpointConfig(output_dir=resolved_checkpoint_dir),
    )


def write_tiny_baseline_modeling_workspace(
    root: Path,
    *,
    train_sample_ids: tuple[str, ...] = ("train-sample",),
    val_sample_ids: tuple[str, ...] = ("val-sample",),
    run_name: str = "baseline-test",
    checkpoint_output_dir: str = "outputs/modeling/baseline-test",
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


def write_fake_record_outputs(layout: BaselineRunLayout) -> None:
    """Write the minimal runtime package surface used by resume tests."""

    layout.record_dir.mkdir(parents=True, exist_ok=True)
    (layout.package_manifest_path).write_text(
        json.dumps({"schema_version": "t2sp-baseline-modeling-package-v1"}),
        encoding="utf-8",
    )
    (layout.record_evidence_bundle_path).write_text(
        json.dumps({"schema_version": "t2sp-baseline-evidence-v1"}),
        encoding="utf-8",
    )
    (layout.record_run_summary_path).write_text(
        json.dumps({"backbone_name": "dummy"}),
        encoding="utf-8",
    )


def touch_file(path: Path, content: str = "x") -> Path:
    """Create a small text file and return its path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def fake_create_archive(
    *,
    archive_path: Path,
    members: tuple[Path, ...],
    cwd: Path,
    label: str,
    snapshot_parent: Path | None = None,
    artifact_description: str | None = None,
) -> Path:
    """Boundary fake for archive creation in CI-safe workflow tests."""

    del members, cwd, label, snapshot_parent, artifact_description
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_bytes(b"archive")
    return archive_path


def raise_dataset_build_archive_error(
    *,
    archive_path: Path,
    members: tuple[Path, ...],
    cwd: Path,
    label: str,
    snapshot_parent: Path | None = None,
) -> Path:
    """Archive fake that preserves the historical Dataset Build wording regression."""

    del archive_path, members, cwd, label, snapshot_parent
    raise FileNotFoundError("Missing required Dataset Build outputs:\n- stale")


def fake_extract_archive_with(
    build_extracted_tree: Callable[[Path], None],
) -> Callable[..., None]:
    """Return an extraction fake that writes a tiny extracted member tree."""

    def fake_extract(archive: Path, destination: Path, *, label: str) -> None:
        del archive, label
        build_extracted_tree(destination)

    return fake_extract
