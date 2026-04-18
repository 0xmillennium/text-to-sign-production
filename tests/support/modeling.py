"""Reusable Sprint 3 modeling fixtures for workflow tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from tests.support.builders.manifests import processed_record, write_jsonl_records
from tests.support.builders.samples import write_processed_sample_npz


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
            {
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
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path


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
