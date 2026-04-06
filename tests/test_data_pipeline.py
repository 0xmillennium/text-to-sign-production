"""Tests for the Sprint 2 How2Sign data pipeline."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import scripts.export_training_manifest as export_training_manifest_script
import scripts.filter_samples as filter_samples_script
import scripts.normalize_keypoints as normalize_keypoints_script
import scripts.prepare_raw as prepare_raw_script
import scripts.validate_manifest as validate_manifest_script
import scripts.view_sample as view_sample_script
import text_to_sign_production.data.filtering as filtering_mod
import text_to_sign_production.data.normalize as normalize_mod
import text_to_sign_production.data.raw as raw_mod
import text_to_sign_production.data.reports as reports_mod
import text_to_sign_production.data.utils as utils_mod
from text_to_sign_production.data.constants import PROCESSED_SCHEMA_VERSION
from text_to_sign_production.data.filtering import (
    FilterConfig,
    determine_drop_reasons,
    load_filter_config,
)
from text_to_sign_production.data.openpose import parse_frame
from text_to_sign_production.data.schemas import NormalizedManifestEntry, RawManifestEntry
from text_to_sign_production.data.validate import (
    validate_normalized_records,
    validate_processed_records,
    validate_raw_records,
)


def _make_box(box_type: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", len(payload) + 8, box_type) + payload


def _write_minimal_mp4(
    path: Path, *, width: int = 1280, height: int = 720, fps: float = 24.0
) -> None:
    timescale = 12288
    sample_count = 12
    sample_delta = int(round(timescale / fps))
    duration = sample_count * sample_delta
    tkhd_payload = (
        b"\x00\x00\x00\x00" + (b"\x00" * 72) + struct.pack(">II", width << 16, height << 16)
    )
    mdhd_payload = (
        b"\x00\x00\x00\x00"
        + (b"\x00" * 8)
        + struct.pack(">II", timescale, duration)
        + (b"\x00" * 4)
    )
    hdlr_payload = b"\x00\x00\x00\x00" + (b"\x00" * 4) + b"vide" + (b"\x00" * 12)
    stts_payload = (
        b"\x00\x00\x00\x00" + struct.pack(">I", 1) + struct.pack(">II", sample_count, sample_delta)
    )
    stbl = _make_box(b"stbl", _make_box(b"stts", stts_payload))
    minf = _make_box(b"minf", stbl)
    mdia = _make_box(
        b"mdia",
        _make_box(b"mdhd", mdhd_payload) + _make_box(b"hdlr", hdlr_payload) + minf,
    )
    trak = _make_box(b"trak", _make_box(b"tkhd", tkhd_payload) + mdia)
    moov = _make_box(b"moov", trak)
    path.write_bytes(moov)


def _build_channel(
    num_points: int, *, x_offset: float = 100.0, y_offset: float = 200.0, zero_last: bool = False
) -> list[float]:
    values: list[float] = []
    for index in range(num_points):
        if zero_last and index == num_points - 1:
            values.extend([0.0, 0.0, 0.0])
        else:
            values.extend([x_offset + float(index), y_offset + float(index), 0.9])
    return values


def _person_payload(*, zero_body_joint: bool = False) -> dict[str, Any]:
    return {
        "person_id": [-1],
        "pose_keypoints_2d": _build_channel(25, zero_last=zero_body_joint),
        "face_keypoints_2d": _build_channel(70, x_offset=300.0, y_offset=150.0),
        "hand_left_keypoints_2d": _build_channel(21, x_offset=400.0, y_offset=350.0),
        "hand_right_keypoints_2d": _build_channel(21, x_offset=500.0, y_offset=320.0),
        "pose_keypoints_3d": [],
        "face_keypoints_3d": [],
        "hand_left_keypoints_3d": [],
        "hand_right_keypoints_3d": [],
    }


def _write_openpose_frame(path: Path, *, people: list[dict[str, Any]]) -> None:
    payload = {"version": 1.3, "people": people}
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_translation_file(path: Path, rows: list[dict[str, str]]) -> None:
    header = "\t".join(
        [
            "VIDEO_ID",
            "VIDEO_NAME",
            "SENTENCE_ID",
            "SENTENCE_NAME",
            "START_REALIGNED",
            "END_REALIGNED",
            "SENTENCE",
        ]
    )
    lines = [header]
    for row in rows:
        lines.append("\t".join(row[column] for column in header.split("\t")))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _patch_pipeline_paths(monkeypatch: Any, root: Path) -> None:
    monkeypatch.setattr(utils_mod, "REPO_ROOT", root)
    monkeypatch.setattr(raw_mod, "TRANSLATIONS_DIR", root / "data/raw/how2sign/translations")
    monkeypatch.setattr(raw_mod, "BFH_KEYPOINTS_ROOT", root / "data/raw/how2sign/bfh_keypoints")
    monkeypatch.setattr(raw_mod, "RAW_MANIFESTS_ROOT", root / "data/interim/raw_manifests")
    monkeypatch.setattr(raw_mod, "INTERIM_REPORTS_ROOT", root / "data/interim/reports")
    monkeypatch.setattr(normalize_mod, "RAW_MANIFESTS_ROOT", root / "data/interim/raw_manifests")
    monkeypatch.setattr(
        normalize_mod, "NORMALIZED_MANIFESTS_ROOT", root / "data/interim/normalized_manifests"
    )
    monkeypatch.setattr(normalize_mod, "PROCESSED_SAMPLES_ROOT", root / "data/processed/v1/samples")
    monkeypatch.setattr(
        filtering_mod, "NORMALIZED_MANIFESTS_ROOT", root / "data/interim/normalized_manifests"
    )
    monkeypatch.setattr(
        filtering_mod, "FILTERED_MANIFESTS_ROOT", root / "data/interim/filtered_manifests"
    )
    monkeypatch.setattr(filtering_mod, "INTERIM_REPORTS_ROOT", root / "data/interim/reports")
    monkeypatch.setattr(reports_mod, "RAW_MANIFESTS_ROOT", root / "data/interim/raw_manifests")
    monkeypatch.setattr(
        reports_mod, "FILTERED_MANIFESTS_ROOT", root / "data/interim/filtered_manifests"
    )
    monkeypatch.setattr(
        reports_mod, "PROCESSED_MANIFESTS_ROOT", root / "data/processed/v1/manifests"
    )
    monkeypatch.setattr(reports_mod, "PROCESSED_REPORTS_ROOT", root / "data/processed/v1/reports")
    monkeypatch.setattr(
        view_sample_script, "PROCESSED_MANIFESTS_ROOT", root / "data/processed/v1/manifests"
    )
    monkeypatch.setattr(filter_samples_script, "PROJECT_ROOT", root)
    monkeypatch.setattr(export_training_manifest_script, "PROJECT_ROOT", root)


def _create_fixture_dataset(root: Path) -> None:
    translations_dir = root / "data/raw/how2sign/translations"
    bfh_root = root / "data/raw/how2sign/bfh_keypoints"
    filter_config_path = root / "configs/data/filter-v1.yaml"
    translations_dir.mkdir(parents=True, exist_ok=True)
    filter_config_path.parent.mkdir(parents=True, exist_ok=True)
    filter_config_path.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "require_nonempty_text: true",
                "require_positive_duration: true",
                "require_keypoints_dir: true",
                "require_frames: true",
                "drop_on_sample_parse_error: true",
                "require_at_least_one_valid_frame: true",
                "minimum_nonzero_frames_per_core_channel: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    split_to_dir = {
        "train": "train_2D_keypoints",
        "val": "val_2D_keypoints",
        "test": "test_2D_keypoints",
    }
    split_rows: dict[str, list[dict[str, str]]] = {}
    for split, split_dir in split_to_dir.items():
        sentence_name = f"{split}_sample_0-1-rgb_front"
        rows = [
            {
                "VIDEO_ID": f"{split}_video",
                "VIDEO_NAME": f"{split}_video-1-rgb_front",
                "SENTENCE_ID": f"{split}_video_0",
                "SENTENCE_NAME": sentence_name,
                "START_REALIGNED": "0.0",
                "END_REALIGNED": "0.5",
                "SENTENCE": f"{split} text",
            }
        ]
        if split == "train":
            rows.append(
                {
                    "VIDEO_ID": "train_video",
                    "VIDEO_NAME": "train_video-1-rgb_front",
                    "SENTENCE_ID": "train_video_1",
                    "SENTENCE_NAME": "train_unmatched_1-1-rgb_front",
                    "START_REALIGNED": "0.5",
                    "END_REALIGNED": "1.0",
                    "SENTENCE": "train unmatched text",
                }
            )
        split_rows[split] = rows
        _write_translation_file(translations_dir / f"how2sign_realigned_{split}.csv", rows)

        json_dir = bfh_root / split_dir / "openpose_output/json" / sentence_name
        video_dir = bfh_root / split_dir / "openpose_output/video"
        json_dir.mkdir(parents=True, exist_ok=True)
        video_dir.mkdir(parents=True, exist_ok=True)
        _write_openpose_frame(
            json_dir / f"{sentence_name}_000000000000_keypoints.json", people=[_person_payload()]
        )
        _write_openpose_frame(
            json_dir / f"{sentence_name}_000000000001_keypoints.json", people=[_person_payload()]
        )
        _write_minimal_mp4(video_dir / f"{sentence_name}.mp4")


def test_parse_frame_tracks_multi_person_and_any_zeroed_required_joint(tmp_path: Path) -> None:
    frame_path = tmp_path / "frame.json"
    _write_openpose_frame(
        frame_path,
        people=[_person_payload(zero_body_joint=True), _person_payload()],
    )

    parsed = parse_frame(frame_path)

    assert parsed.people_count == 2
    assert parsed.frame_valid is True
    assert parsed.has_any_zeroed_required_joint is True
    assert parsed.coords["body"].shape == (25, 2)
    assert np.isclose(parsed.coords["body"][0, 0], 100.0 / 1280.0)
    assert parsed.confidences["left_hand"].shape == (21,)


def test_parse_frame_counts_canvas_edge_coordinates_as_out_of_bounds(tmp_path: Path) -> None:
    frame_path = tmp_path / "frame.json"
    payload = _person_payload()
    payload["pose_keypoints_2d"][:3] = [1280.0, 720.0, 0.9]
    _write_openpose_frame(frame_path, people=[payload])

    parsed = parse_frame(frame_path)

    assert parsed.out_of_bounds_coordinate_count == 2


@pytest.mark.parametrize("schema_version", [None, 2])
def test_load_filter_config_requires_supported_schema_version(
    tmp_path: Path, schema_version: int | None
) -> None:
    config_path = tmp_path / "filter.yaml"
    lines = [
        "require_nonempty_text: true",
        "require_positive_duration: true",
        "require_keypoints_dir: true",
        "require_frames: true",
        "drop_on_sample_parse_error: true",
        "require_at_least_one_valid_frame: true",
        "minimum_nonzero_frames_per_core_channel: 1",
    ]
    if schema_version is not None:
        lines.insert(0, f"schema_version: {schema_version}")
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported filter config schema_version"):
        load_filter_config(config_path)


def test_determine_drop_reasons_respects_core_channel_policy() -> None:
    entry = NormalizedManifestEntry(
        sample_id="sample",
        processed_schema_version=PROCESSED_SCHEMA_VERSION,
        text="text",
        split="train",
        start_time=0.0,
        end_time=1.0,
        num_frames=2,
        sample_path="data/processed/v1/samples/train/sample.npz",
        source_video_id="video",
        source_sentence_id="sentence",
        source_sentence_name="sample",
        source_metadata_path="data/raw/how2sign/translations/how2sign_realigned_train.csv",
        source_keypoints_dir="data/raw/how2sign/bfh_keypoints/train_2D_keypoints/openpose_output/json/sample",
        source_video_path="data/raw/how2sign/bfh_keypoints/train_2D_keypoints/openpose_output/video/sample.mp4",
        fps=24.0,
        video_width=1280,
        video_height=720,
        video_metadata_error=None,
        selected_person_index=0,
        multi_person_frame_count=0,
        max_people_per_frame=1,
        frame_valid_count=2,
        frame_invalid_count=0,
        face_missing_frame_count=0,
        out_of_bounds_coordinate_count=0,
        frames_with_any_zeroed_required_joint=0,
        frame_issue_counts={},
        core_channel_nonzero_frames={"body": 2, "left_hand": 0, "right_hand": 2},
        sample_parse_error=None,
    )
    config = FilterConfig(
        require_nonempty_text=True,
        require_positive_duration=True,
        require_keypoints_dir=True,
        require_frames=True,
        drop_on_sample_parse_error=True,
        require_at_least_one_valid_frame=True,
        minimum_nonzero_frames_per_core_channel=1,
    )

    drop_reasons = determine_drop_reasons(entry, config)

    assert "unusable_core_channel:left_hand" in drop_reasons
    assert all(
        reason.startswith("unusable_core_channel") or reason == "missing_sample_file"
        for reason in drop_reasons
    )


def test_validate_normalized_records_reports_parse_errors() -> None:
    issues = validate_normalized_records(Path("normalized.jsonl"), [{"sample_id": "broken"}])

    assert len(issues) == 1
    assert issues[0].code == "normalized_record_parse_error"


@pytest.mark.parametrize(
    ("validator", "path"),
    [
        (validate_raw_records, Path("raw.jsonl")),
        (validate_normalized_records, Path("normalized.jsonl")),
        (validate_processed_records, Path("processed.jsonl")),
    ],
)
def test_validators_do_not_report_duplicate_sample_id_for_blank_values(
    validator: Any, path: Path
) -> None:
    issues = validator(path, [{"sample_id": " "}, {"sample_id": ""}])

    assert all(issue.code != "duplicate_sample_id" for issue in issues)


def test_validate_processed_records_resolves_repo_relative_sample_paths(
    tmp_path: Path, monkeypatch: Any
) -> None:
    root = tmp_path / "repo"
    sample_path = root / "data/processed/v1/samples/train/sample.npz"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(utils_mod, "REPO_ROOT", root)

    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    monkeypatch.chdir(outside_dir)

    issues = validate_processed_records(
        root / "data/processed/v1/manifests/train.jsonl",
        [
            {
                "sample_id": "sample",
                "processed_schema_version": PROCESSED_SCHEMA_VERSION,
                "text": "text",
                "split": "train",
                "fps": 24.0,
                "num_frames": 2,
                "sample_path": "data/processed/v1/samples/train/sample.npz",
                "source_video_id": "video",
                "source_sentence_id": "sentence",
                "source_sentence_name": "sample",
                "selected_person_index": 0,
                "multi_person_frame_count": 0,
                "max_people_per_frame": 1,
                "source_metadata_path": "data/raw/how2sign/translations/train.tsv",
                "source_keypoints_dir": "data/raw/how2sign/bfh_keypoints/train/sample",
                "source_video_path": "data/raw/how2sign/bfh_keypoints/train/sample.mp4",
                "video_width": 1280,
                "video_height": 720,
                "video_metadata_error": None,
                "frame_valid_count": 2,
                "frame_invalid_count": 0,
                "face_missing_frame_count": 0,
                "out_of_bounds_coordinate_count": 0,
                "frames_with_any_zeroed_required_joint": 0,
                "frame_issue_counts": {},
                "core_channel_nonzero_frames": {"body": 2, "left_hand": 2, "right_hand": 2},
            }
        ],
    )

    assert [issue.code for issue in issues] == []


def test_validate_processed_records_reports_missing_required_fields() -> None:
    issues = validate_processed_records(
        Path("processed.jsonl"),
        [
            {
                "sample_id": "sample",
                "processed_schema_version": PROCESSED_SCHEMA_VERSION,
                "text": "text",
                "split": "train",
                "fps": 24.0,
                "num_frames": 2,
                "sample_path": "data/processed/v1/samples/train/sample.npz",
                "source_video_id": "video",
                "source_sentence_id": "sentence",
                "source_sentence_name": "sample",
                "selected_person_index": 0,
                "multi_person_frame_count": 0,
                "max_people_per_frame": 1,
                "source_video_path": "data/raw/how2sign/bfh_keypoints/train/sample.mp4",
                "video_width": 1280,
                "video_height": 720,
                "video_metadata_error": None,
                "frame_valid_count": 2,
                "frame_invalid_count": 0,
                "face_missing_frame_count": 0,
                "out_of_bounds_coordinate_count": 0,
                "frames_with_any_zeroed_required_joint": 0,
                "frame_issue_counts": {},
                "core_channel_nonzero_frames": {"body": 2, "left_hand": 2, "right_hand": 2},
            }
        ],
    )

    missing_fields_issue = next(
        issue for issue in issues if issue.code == "missing_required_fields"
    )
    assert "source_metadata_path" in missing_fields_issue.message
    assert "source_keypoints_dir" in missing_fields_issue.message


def test_validate_processed_records_rejects_absolute_sample_paths() -> None:
    issues = validate_processed_records(
        Path("processed.jsonl"),
        [
            {
                "sample_id": "sample",
                "processed_schema_version": PROCESSED_SCHEMA_VERSION,
                "text": "text",
                "split": "train",
                "fps": 24.0,
                "num_frames": 2,
                "sample_path": "/tmp/sample.npz",
                "source_video_id": "video",
                "source_sentence_id": "sentence",
                "source_sentence_name": "sample",
                "selected_person_index": 0,
                "multi_person_frame_count": 0,
                "max_people_per_frame": 1,
                "source_metadata_path": "data/raw/how2sign/translations/train.tsv",
                "source_keypoints_dir": "data/raw/how2sign/bfh_keypoints/train/sample",
                "source_video_path": "data/raw/how2sign/bfh_keypoints/train/sample.mp4",
                "video_width": 1280,
                "video_height": 720,
                "video_metadata_error": None,
                "frame_valid_count": 2,
                "frame_invalid_count": 0,
                "face_missing_frame_count": 0,
                "out_of_bounds_coordinate_count": 0,
                "frames_with_any_zeroed_required_joint": 0,
                "frame_issue_counts": {},
                "core_channel_nonzero_frames": {"body": 2, "left_hand": 2, "right_hand": 2},
            }
        ],
    )

    assert any(issue.code == "absolute_sample_path" for issue in issues)


def test_validate_processed_records_rejects_repo_escaping_sample_paths() -> None:
    issues = validate_processed_records(
        Path("processed.jsonl"),
        [
            {
                "sample_id": "sample",
                "processed_schema_version": PROCESSED_SCHEMA_VERSION,
                "text": "text",
                "split": "train",
                "fps": 24.0,
                "num_frames": 2,
                "sample_path": "../../outside/sample.npz",
                "source_video_id": "video",
                "source_sentence_id": "sentence",
                "source_sentence_name": "sample",
                "selected_person_index": 0,
                "multi_person_frame_count": 0,
                "max_people_per_frame": 1,
                "source_metadata_path": "data/raw/how2sign/translations/train.tsv",
                "source_keypoints_dir": "data/raw/how2sign/bfh_keypoints/train/sample",
                "source_video_path": "data/raw/how2sign/bfh_keypoints/train/sample.mp4",
                "video_width": 1280,
                "video_height": 720,
                "video_metadata_error": None,
                "frame_valid_count": 2,
                "frame_invalid_count": 0,
                "face_missing_frame_count": 0,
                "out_of_bounds_coordinate_count": 0,
                "frames_with_any_zeroed_required_joint": 0,
                "frame_issue_counts": {},
                "core_channel_nonzero_frames": {"body": 2, "left_hand": 2, "right_hand": 2},
            }
        ],
    )

    assert any(issue.code == "invalid_sample_path" for issue in issues)


def test_export_final_manifests_rejects_missing_source_keypoints_dir(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(
        reports_mod, "PROCESSED_MANIFESTS_ROOT", tmp_path / "data/processed/manifests"
    )
    monkeypatch.setattr(reports_mod, "PROCESSED_REPORTS_ROOT", tmp_path / "data/processed/reports")
    monkeypatch.setattr(reports_mod, "_load_raw_records", lambda split: [])
    monkeypatch.setattr(
        reports_mod,
        "_load_filtered_records",
        lambda split: (
            [
                NormalizedManifestEntry(
                    sample_id="sample",
                    processed_schema_version=PROCESSED_SCHEMA_VERSION,
                    text="text",
                    split=split,
                    start_time=0.0,
                    end_time=1.0,
                    num_frames=2,
                    sample_path=f"data/processed/v1/samples/{split}/sample.npz",
                    source_video_id="video",
                    source_sentence_id="sentence",
                    source_sentence_name="sample",
                    source_metadata_path="data/raw/how2sign/translations/train.tsv",
                    source_keypoints_dir=None,
                    source_video_path="data/raw/how2sign/bfh_keypoints/train/sample.mp4",
                    fps=24.0,
                    video_width=1280,
                    video_height=720,
                    video_metadata_error=None,
                    selected_person_index=0,
                    multi_person_frame_count=0,
                    max_people_per_frame=1,
                    frame_valid_count=2,
                    frame_invalid_count=0,
                    face_missing_frame_count=0,
                    out_of_bounds_coordinate_count=0,
                    frames_with_any_zeroed_required_joint=0,
                    frame_issue_counts={},
                    core_channel_nonzero_frames={"body": 2, "left_hand": 2, "right_hand": 2},
                    sample_parse_error=None,
                )
            ]
            if split == "train"
            else []
        ),
    )

    with pytest.raises(ValueError, match="missing source_keypoints_dir"):
        reports_mod.export_final_manifests(
            assumption_report={"generated_at": "2026-04-07T00:00:00+00:00", "splits": {}},
            filter_report={"generated_at": "2026-04-07T00:00:00+00:00", "splits": {}},
        )


def test_export_final_manifests_supports_subset_splits(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(
        reports_mod, "PROCESSED_MANIFESTS_ROOT", tmp_path / "data/processed/manifests"
    )
    monkeypatch.setattr(reports_mod, "PROCESSED_REPORTS_ROOT", tmp_path / "data/processed/reports")
    monkeypatch.setattr(
        reports_mod,
        "_load_raw_records",
        lambda split: (
            [
                RawManifestEntry(
                    sample_id="sample",
                    source_split=split,
                    video_id="video",
                    video_name="video-rgb_front",
                    sentence_id="sentence",
                    sentence_name="sample",
                    text="text",
                    start_time=0.0,
                    end_time=1.0,
                    keypoints_dir=f"data/raw/how2sign/bfh_keypoints/{split}/sample",
                    source_metadata_path="data/raw/how2sign/translations/train.tsv",
                    has_face=True,
                    num_frames=2,
                    source_video_path=f"data/raw/how2sign/bfh_keypoints/{split}/sample.mp4",
                    video_width=1280,
                    video_height=720,
                    video_fps=24.0,
                    video_metadata_error=None,
                )
            ]
            if split == "train"
            else pytest.fail(f"Unexpected raw split request: {split}")
        ),
    )
    monkeypatch.setattr(
        reports_mod,
        "_load_filtered_records",
        lambda split: (
            [
                NormalizedManifestEntry(
                    sample_id="sample",
                    processed_schema_version=PROCESSED_SCHEMA_VERSION,
                    text="text",
                    split=split,
                    start_time=0.0,
                    end_time=1.0,
                    num_frames=2,
                    sample_path=f"data/processed/v1/samples/{split}/sample.npz",
                    source_video_id="video",
                    source_sentence_id="sentence",
                    source_sentence_name="sample",
                    source_metadata_path="data/raw/how2sign/translations/train.tsv",
                    source_keypoints_dir=f"data/raw/how2sign/bfh_keypoints/{split}/sample",
                    source_video_path=f"data/raw/how2sign/bfh_keypoints/{split}/sample.mp4",
                    fps=24.0,
                    video_width=1280,
                    video_height=720,
                    video_metadata_error=None,
                    selected_person_index=0,
                    multi_person_frame_count=0,
                    max_people_per_frame=1,
                    frame_valid_count=2,
                    frame_invalid_count=0,
                    face_missing_frame_count=0,
                    out_of_bounds_coordinate_count=0,
                    frames_with_any_zeroed_required_joint=0,
                    frame_issue_counts={},
                    core_channel_nonzero_frames={"body": 2, "left_hand": 2, "right_hand": 2},
                    sample_parse_error=None,
                )
            ]
            if split == "train"
            else pytest.fail(f"Unexpected filtered split request: {split}")
        ),
    )

    result = reports_mod.export_final_manifests(
        assumption_report={
            "generated_at": "2026-04-07T00:00:00+00:00",
            "splits": {
                "train": {
                    "translation_row_count": 1,
                    "matched_sample_count": 1,
                    "unmatched_sample_count": 0,
                    "video_metadata": {"readable_count": 1, "unreadable_count": 0},
                    "first_frame_people_counter": {"1": 1},
                    "openpose_schema": {"deviation_counts": {}},
                }
            },
        },
        filter_report={
            "generated_at": "2026-04-07T00:00:00+00:00",
            "splits": {
                "train": {
                    "dropped_samples": 0,
                    "drop_reason_counts": {},
                }
            },
        },
        splits=("train",),
    )

    assert result["split_report"]["official_split_names"] == ["train"]
    assert result["split_report"]["video_id_overlap"] == {}
    assert set(result["split_report"]["splits"]) == {"train"}
    assert (tmp_path / "data/processed/manifests/train.jsonl").exists()
    assert not (tmp_path / "data/processed/manifests/val.jsonl").exists()
    assert not (tmp_path / "data/processed/manifests/test.jsonl").exists()


def test_cli_pipeline_end_to_end(tmp_path: Path, monkeypatch: Any) -> None:
    _create_fixture_dataset(tmp_path)
    _patch_pipeline_paths(monkeypatch, tmp_path)
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    monkeypatch.chdir(outside_dir)

    monkeypatch.setattr(sys, "argv", ["prepare_raw.py", "--splits", "train", "val", "test"])
    assert prepare_raw_script.main() == 0

    raw_manifest = tmp_path / "data/interim/raw_manifests/raw_train.jsonl"
    raw_records = [
        json.loads(line) for line in raw_manifest.read_text(encoding="utf-8").splitlines() if line
    ]
    assert len(raw_records) == 2
    assert raw_records[0]["sample_id"] == "train_sample_0-1-rgb_front"
    assert raw_records[0]["num_frames"] == 2
    assert raw_records[0]["video_fps"] == 24.0

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_manifest.py",
            "--manifest",
            str(raw_manifest),
            "--kind",
            "raw",
        ],
    )
    assert validate_manifest_script.main() == 0

    monkeypatch.setattr(sys, "argv", ["normalize_keypoints.py", "--splits", "train", "val", "test"])
    assert normalize_keypoints_script.main() == 0

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "filter_samples.py",
            "--splits",
            "train",
            "val",
            "test",
        ],
    )
    assert filter_samples_script.main() == 0

    monkeypatch.setattr(sys, "argv", ["export_training_manifest.py"])
    assert export_training_manifest_script.main() == 0

    train_manifest = tmp_path / "data/processed/v1/manifests/train.jsonl"
    processed_records = [
        json.loads(line) for line in train_manifest.read_text(encoding="utf-8").splitlines() if line
    ]
    assert len(processed_records) == 1
    record = processed_records[0]
    assert record["processed_schema_version"] == PROCESSED_SCHEMA_VERSION
    assert record["selected_person_index"] == 0
    assert record["multi_person_frame_count"] == 0

    sample_path = tmp_path / record["sample_path"]
    assert sample_path.exists()
    with np.load(sample_path) as sample:
        assert set(sample.files) >= {
            "processed_schema_version",
            "body",
            "body_confidence",
            "left_hand",
            "left_hand_confidence",
            "right_hand",
            "right_hand_confidence",
            "face",
            "face_confidence",
            "people_per_frame",
            "selected_person_index",
            "frame_valid_mask",
        }
        assert str(sample["processed_schema_version"]) == PROCESSED_SCHEMA_VERSION
        assert sample["body"].shape == (2, 25, 2)

    filter_report = json.loads(
        (tmp_path / "data/interim/reports/filter-report.json").read_text(encoding="utf-8")
    )
    assert filter_report["config_path"] == "configs/data/filter-v1.yaml"
    assert filter_report["splits"]["train"]["dropped_samples"] == 1
    assert filter_report["splits"]["train"]["drop_reason_counts"]["missing_keypoints_dir"] == 1

    assumption_report = json.loads(
        (tmp_path / "data/interim/reports/assumption-report.json").read_text(encoding="utf-8")
    )
    assert assumption_report["splits"]["train"]["matched_sample_count"] == 1
    assert assumption_report["splits"]["train"]["unmatched_sample_count"] == 1

    monkeypatch.setattr(
        sys, "argv", ["view_sample.py", "--split", "train", "--sample-id", record["sample_id"]]
    )
    assert view_sample_script.main() == 0

    assert (tmp_path / "data/processed/v1/reports/assumption-report.md").exists()
    assert (tmp_path / "data/processed/v1/reports/data-quality-report.md").exists()
    assert (tmp_path / "data/processed/v1/reports/split-report.md").exists()
