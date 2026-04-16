"""Isolated tests for Dataset Build Colab packaging helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

import text_to_sign_production.ops.colab_workflow as colab_workflow_mod
from tests.support.builders.manifests import processed_record, write_jsonl_records

pytestmark = pytest.mark.unit


def test_sample_archive_members_for_split_reads_manifest_sample_paths(tmp_path: Path) -> None:
    records = [
        processed_record(
            "data/processed/v1/samples/train/sample_a.npz",
            split="train",
            sample_id="sample_a",
        ),
        processed_record(
            "data/processed/v1/samples/train/sample_b.npz",
            split="train",
            sample_id="sample_b",
        ),
    ]
    _write_processed_manifest(tmp_path, "train", records)

    members = colab_workflow_mod._sample_archive_members_for_split(
        project_root=tmp_path,
        split="train",
    )

    assert members == (
        Path("data/processed/v1/samples/train/sample_a.npz"),
        Path("data/processed/v1/samples/train/sample_b.npz"),
    )


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (
            processed_record(
                "data/processed/v1/samples/train/sample.npz",
                split="val",
                sample_id="sample",
            ),
            "record split does not match",
        ),
        (
            processed_record("/tmp/sample.npz", split="train", sample_id="sample"),
            "must be repo-relative",
        ),
        (
            processed_record(
                "data/processed/v1/samples/train/sample.txt",
                split="train",
                sample_id="sample",
            ),
            r"must end with \.npz",
        ),
        (
            processed_record(
                "data/processed/v1/samples/val/sample.npz",
                split="train",
                sample_id="sample",
            ),
            "must stay under data/processed/v1/samples/train",
        ),
        (
            processed_record(
                "data/processed/v1/samples/train/../val/sample.npz",
                split="train",
                sample_id="sample",
            ),
            "must stay under data/processed/v1/samples/train",
        ),
        (
            processed_record(
                "data/processed/v1/samples/train/nested/sample.npz",
                split="train",
                sample_id="sample",
            ),
            "must follow data/processed/v1/samples/<split>/<sample_id>.npz",
        ),
        (
            processed_record(
                "data/processed/v1/samples/train/other_sample.npz",
                split="train",
                sample_id="sample",
            ),
            "filename does not match sample_id",
        ),
    ],
)
def test_sample_archive_members_for_split_rejects_invalid_manifest_paths(
    tmp_path: Path,
    record: dict[str, object],
    message: str,
) -> None:
    _write_processed_manifest(tmp_path, "train", [record])

    with pytest.raises(ValueError, match=message):
        colab_workflow_mod._sample_archive_members_for_split(
            project_root=tmp_path,
            split="train",
        )


def test_sample_archive_members_for_split_rejects_duplicate_manifest_sample_paths(
    tmp_path: Path,
) -> None:
    sample_path = "data/processed/v1/samples/train/sample.npz"
    _write_processed_manifest(
        tmp_path,
        "train",
        [
            processed_record(sample_path, split="train", sample_id="sample"),
            processed_record(sample_path, split="train", sample_id="sample"),
        ],
    )

    with pytest.raises(ValueError, match="Duplicate processed manifest sample_path"):
        colab_workflow_mod._sample_archive_members_for_split(
            project_root=tmp_path,
            split="train",
        )


def _write_processed_manifest(
    root: Path,
    split: str,
    records: list[dict[str, object]],
) -> None:
    write_jsonl_records(root / f"data/processed/v1/manifests/{split}.jsonl", records)
