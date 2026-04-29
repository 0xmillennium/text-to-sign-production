"""Integration tests for processed sample indexing and timing joins."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.builders.manifests import (
    normalized_manifest_entry,
    processed_record,
    raw_manifest_entry,
    write_jsonl_records,
)
from tests.support.builders.samples import write_processed_sample_npz
from tests.support.builders.visual_debug import write_visual_debug_workspace
from text_to_sign_production.data.processed_samples import (
    DuplicateSampleIdError,
    ProcessedSampleDataError,
    SampleSelectionError,
    build_sample_index,
    select_sample,
)

pytestmark = pytest.mark.integration


def test_loads_all_available_processed_manifests_and_selects_exact_sample_id(
    tmp_path: Path,
) -> None:
    data_root = write_visual_debug_workspace(tmp_path, splits=("train", "val"), samples_per_split=1)

    index = build_sample_index(data_root)
    selected = select_sample(index, sample_id="val-sample-0")

    assert index.splits == ("train", "val")
    assert selected.sample_id == "val-sample-0"
    assert selected.split == "val"


def test_missing_sample_id_reports_loaded_splits(tmp_path: Path) -> None:
    data_root = write_visual_debug_workspace(tmp_path, splits=("val",), samples_per_split=1)
    index = build_sample_index(data_root)

    with pytest.raises(SampleSelectionError, match="loaded processed manifest splits: val"):
        select_sample(index, sample_id="missing")


def test_duplicate_sample_id_across_splits_errors_for_exact_and_random_selection(
    tmp_path: Path,
) -> None:
    data_root = write_visual_debug_workspace(
        tmp_path,
        splits=("train", "val"),
        samples_per_split=1,
        duplicate_sample_id="duplicate",
    )
    index = build_sample_index(data_root)

    with pytest.raises(DuplicateSampleIdError, match="appears in 2 loaded"):
        select_sample(index, sample_id="duplicate")
    with pytest.raises(DuplicateSampleIdError, match="appears in 2 loaded"):
        select_sample(index, random_selection=True, seed=7)


def test_deterministic_random_selection_is_seeded(tmp_path: Path) -> None:
    data_root = write_visual_debug_workspace(tmp_path, splits=("val",), samples_per_split=3)
    index = build_sample_index(data_root)

    first = select_sample(index, random_selection=True, seed=7)
    second = select_sample(index, random_selection=True, seed=7)

    assert first.sample_id == second.sample_id


def test_npz_path_resolution_uses_manifest_sample_path_not_arbitrary_discovery(
    tmp_path: Path,
) -> None:
    data_root = write_visual_debug_workspace(tmp_path, splits=("val",), samples_per_split=1)
    write_processed_sample_npz(data_root / "processed/v1/samples/val/unlisted.npz")

    index = build_sample_index(data_root)

    assert [record.sample_id for record in index.records] == ["val-sample-0"]
    assert (
        index.records[0].sample_path
        == (data_root / "processed/v1/samples/val/val-sample-0.npz").resolve()
    )


def test_strict_index_rejects_missing_manifest_sample_file(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    write_jsonl_records(
        data_root / "processed/v1/manifests/val.jsonl",
        [
            processed_record(
                "data/processed/v1/samples/val/sample-a.npz",
                split="val",
                sample_id="sample-a",
            )
        ],
    )

    with pytest.raises(ProcessedSampleDataError, match="does not exist"):
        build_sample_index(data_root, splits=("val",))


def test_manifest_only_index_accepts_missing_canonical_sample_file(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    write_jsonl_records(
        data_root / "processed/v1/manifests/val.jsonl",
        [
            processed_record(
                "data/processed/v1/samples/val/sample-a.npz",
                split="val",
                sample_id="sample-a",
            )
        ],
    )

    index = build_sample_index(data_root, splits=("val",), require_sample_files=False)

    assert index.splits == ("val",)
    assert index.records[0].sample_id == "sample-a"
    assert (
        index.records[0].sample_path
        == (data_root / "processed/v1/samples/val/sample-a.npz").resolve()
    )
    assert not index.records[0].sample_path.exists()


@pytest.mark.parametrize(
    ("sample_path", "sample_id", "match"),
    [
        ("/tmp/sample-a.npz", "sample-a", "repo-relative"),
        ("data/processed/v1/samples/val/sample-a.npy", "sample-a", "must end with .npz"),
        (
            "data/processed/v1/samples/val/../sample-a.npz",
            "sample-a",
            "parent traversal",
        ),
        ("data/processed/v1/samples/sample-a.npz", "sample-a", "must follow"),
        ("data/processed/v1/samples/train/sample-a.npz", "sample-a", "split does not match"),
        ("data/processed/v1/samples/val/sample-a.npz", "sample-b", "filename does not match"),
    ],
)
def test_manifest_only_index_still_rejects_invalid_sample_paths(
    tmp_path: Path,
    sample_path: str,
    sample_id: str,
    match: str,
) -> None:
    data_root = tmp_path / "data"
    write_jsonl_records(
        data_root / "processed/v1/manifests/val.jsonl",
        [
            processed_record(
                sample_path,
                split="val",
                sample_id=sample_id,
            )
        ],
    )

    with pytest.raises(ProcessedSampleDataError, match=match):
        build_sample_index(data_root, splits=("val",), require_sample_files=False)


def test_rejects_absolute_processed_manifest_sample_path(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    sample_path = data_root / "processed/v1/samples/val/sample-a.npz"
    write_processed_sample_npz(sample_path)
    write_jsonl_records(
        data_root / "processed/v1/manifests/val.jsonl",
        [
            processed_record(
                sample_path.as_posix(),
                split="val",
                sample_id="sample-a",
            )
        ],
    )

    with pytest.raises(ProcessedSampleDataError, match="repo-relative"):
        build_sample_index(data_root, splits=("val",))


def test_rejects_noncanonical_processed_manifest_sample_path(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    sample_path = data_root / "processed/v1/samples/val/nested/sample-a.npz"
    write_processed_sample_npz(sample_path)
    write_jsonl_records(
        data_root / "processed/v1/manifests/val.jsonl",
        [
            processed_record(
                "data/processed/v1/samples/val/nested/sample-a.npz",
                split="val",
                sample_id="sample-a",
            )
        ],
    )

    with pytest.raises(ProcessedSampleDataError, match="must follow"):
        build_sample_index(data_root, splits=("val",))


def test_processed_manifest_records_are_schema_backed(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    sample_path = data_root / "processed/v1/samples/val/sample-a.npz"
    write_processed_sample_npz(sample_path)
    record = processed_record(
        "data/processed/v1/samples/val/sample-a.npz",
        split="val",
        sample_id="sample-a",
    )
    record.pop("processed_schema_version")
    write_jsonl_records(data_root / "processed/v1/manifests/val.jsonl", [record])

    with pytest.raises(ProcessedSampleDataError, match="Could not parse processed manifest"):
        build_sample_index(data_root, splits=("val",))


def test_default_index_skips_manifest_splits_without_present_samples(tmp_path: Path) -> None:
    data_root = write_visual_debug_workspace(tmp_path, splits=("val",), samples_per_split=1)
    write_jsonl_records(
        data_root / "processed/v1/manifests/train.jsonl",
        [
            processed_record(
                "data/processed/v1/samples/train/train-sample-0.npz",
                split="train",
                sample_id="train-sample-0",
            )
        ],
    )

    index = build_sample_index(data_root)

    assert index.splits == ("val",)


def test_filtered_timing_join_uses_sample_id(tmp_path: Path) -> None:
    data_root = write_visual_debug_workspace(tmp_path, splits=("val",), samples_per_split=1)
    index = build_sample_index(data_root)
    record = select_sample(index, sample_id="val-sample-0")

    timing = index.timing_for(record)

    assert timing.start_time == 1.25
    assert timing.end_time == 2.5
    assert timing.fps == 30.0
    assert timing.num_frames == 3
    assert timing.join_key == "filtered:sample_id"


def test_timing_join_falls_back_to_normalized_sentence_name(tmp_path: Path) -> None:
    data_root = _write_timing_fallback_workspace(tmp_path, source="normalized")

    index = build_sample_index(data_root)
    timing = index.timing_for(select_sample(index, sample_id="sample-a"))

    assert timing.start_time == 3.0
    assert timing.join_key == "normalized:source_sentence_name"


def test_timing_join_falls_back_to_raw_sentence_id(tmp_path: Path) -> None:
    data_root = _write_timing_fallback_workspace(tmp_path, source="raw")

    index = build_sample_index(data_root)
    timing = index.timing_for(select_sample(index, sample_id="sample-a"))

    assert timing.start_time == 5.0
    assert timing.join_key == "raw:source_sentence_id"


def test_timing_join_is_keyed_by_split_and_sample_id(tmp_path: Path) -> None:
    data_root = write_visual_debug_workspace(
        tmp_path,
        splits=("train", "val"),
        samples_per_split=1,
        duplicate_sample_id="shared-sample",
    )
    val_timing = normalized_manifest_entry(
        split="val",
        sample_id="shared-sample",
        sample_path="data/processed/v1/samples/val/shared-sample.npz",
    )
    val_timing.start_time = 7.0
    val_timing.end_time = 8.0
    val_timing.num_frames = 3
    val_timing.source_sentence_id = "val-sentence-0"
    val_timing.source_sentence_name = "val-sentence-name-0"
    write_jsonl_records(
        data_root / "interim/filtered_manifests/filtered_val.jsonl",
        [val_timing.to_record()],
    )

    index = build_sample_index(data_root)
    timing_by_split = {record.split: index.timing_for(record) for record in index.records}

    assert timing_by_split["train"].start_time == 1.25
    assert timing_by_split["val"].start_time == 7.0


def _write_timing_fallback_workspace(tmp_path: Path, *, source: str) -> Path:
    data_root = tmp_path / "data"
    sample_path = data_root / "processed/v1/samples/val/sample-a.npz"
    write_processed_sample_npz(sample_path, num_frames=3)
    record = processed_record(
        "data/processed/v1/samples/val/sample-a.npz",
        split="val",
        sample_id="sample-a",
        num_frames=3,
    )
    record.update(
        {
            "source_sentence_id": "sentence-a",
            "source_sentence_name": "sentence-name-a",
        }
    )
    write_jsonl_records(data_root / "processed/v1/manifests/val.jsonl", [record])
    write_jsonl_records(data_root / "interim/filtered_manifests/filtered_val.jsonl", [])
    write_jsonl_records(data_root / "interim/normalized_manifests/normalized_val.jsonl", [])
    write_jsonl_records(data_root / "interim/raw_manifests/raw_val.jsonl", [])

    if source == "normalized":
        normalized_entry = normalized_manifest_entry(
            split="val",
            sample_id="different-id",
            sample_path="data/processed/v1/samples/val/different-id.npz",
        )
        normalized_entry.source_sentence_name = "sentence-name-a"
        normalized_entry.start_time = 3.0
        normalized_entry.end_time = 4.0
        normalized_entry.fps = 25.0
        normalized_entry.num_frames = 4
        normalized_entry.source_video_path = "data/raw/how2sign/bfh_keypoints/val/fallback.mp4"
        normalized_entry.video_width = 640
        normalized_entry.video_height = 360
        write_jsonl_records(
            data_root / "interim/normalized_manifests/normalized_val.jsonl",
            [normalized_entry.to_record()],
        )
    elif source == "raw":
        raw_entry = raw_manifest_entry(
            split="val",
            sample_id="different-id",
            keypoints_dir="data/raw/how2sign/bfh_keypoints/val/sample",
            num_frames=5,
        )
        raw_entry.sentence_id = "sentence-a"
        raw_entry.start_time = 5.0
        raw_entry.end_time = 6.0
        raw_entry.video_fps = 24.0
        raw_entry.source_video_path = "data/raw/how2sign/bfh_keypoints/val/raw.mp4"
        raw_entry.video_width = 640
        raw_entry.video_height = 360
        write_jsonl_records(
            data_root / "interim/raw_manifests/raw_val.jsonl",
            [raw_entry.to_record()],
        )
    else:
        raise ValueError(source)
    return data_root
