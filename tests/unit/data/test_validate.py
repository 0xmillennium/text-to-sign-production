"""Manifest and processed sample validation tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import text_to_sign_production.core.paths as paths_mod
from tests.support.builders.manifests import processed_record
from tests.support.builders.samples import write_processed_sample_npz
from text_to_sign_production.data.validate import (
    validate_manifest,
    validate_normalized_records,
    validate_processed_records,
    validate_raw_records,
)

pytestmark = pytest.mark.unit


def test_validate_normalized_records_reports_parse_errors() -> None:
    issues = validate_normalized_records(Path("normalized.jsonl"), [{"sample_id": "broken"}])

    assert len(issues) == 1
    assert issues[0].code == "normalized_record_parse_error"


@pytest.mark.parametrize(
    ("kind", "record_payload", "expected_followup_code"),
    [
        ("raw", {"sample_id": "sample"}, "raw_record_parse_error"),
        ("normalized", {"sample_id": "sample"}, "normalized_record_parse_error"),
        ("processed", {"sample_id": "sample"}, "processed_record_parse_error"),
    ],
)
def test_validate_manifest_reports_invalid_jsonl_lines_and_continues(
    tmp_path: Path,
    kind: str,
    record_payload: dict[str, Any],
    expected_followup_code: str,
) -> None:
    manifest_path = tmp_path / f"{kind}.jsonl"
    manifest_path.write_text(
        "{broken\n" + json.dumps(record_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    issues = validate_manifest(manifest_path, kind=kind)

    assert any(issue.code == "invalid_jsonl_line" and "Line 1" in issue.message for issue in issues)
    assert any(issue.code == expected_followup_code for issue in issues)


def test_validate_manifest_reports_non_object_records_and_continues_for_raw(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "raw.jsonl"
    manifest_path.write_text("[]\n" + json.dumps({"sample_id": "sample"}) + "\n", encoding="utf-8")

    issues = validate_manifest(manifest_path, kind="raw")

    assert any(issue.code == "non_object_record" and "Line 1" in issue.message for issue in issues)
    assert any(issue.code == "raw_record_parse_error" for issue in issues)


def test_validate_manifest_reports_non_object_records_and_continues_for_processed(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "processed.jsonl"
    manifest_path.write_text(
        "[]\n" + json.dumps({"sample_id": "sample"}) + "\n",
        encoding="utf-8",
    )

    issues = validate_manifest(manifest_path, kind="processed")

    assert any(issue.code == "non_object_record" and "Line 1" in issue.message for issue in issues)
    assert any(issue.code == "processed_record_parse_error" for issue in issues)


@pytest.mark.parametrize(
    ("validator", "path"),
    [
        (validate_raw_records, Path("raw.jsonl")),
        (validate_normalized_records, Path("normalized.jsonl")),
        (validate_processed_records, Path("processed.jsonl")),
    ],
)
def test_validators_do_not_report_duplicate_sample_id_for_blank_values(
    validator: Any,
    path: Path,
) -> None:
    issues = validator(path, [{"sample_id": " "}, {"sample_id": ""}])

    assert all(issue.code != "duplicate_sample_id" for issue in issues)


def test_validate_processed_records_resolves_repo_relative_sample_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    sample_path = root / "data/processed/v1/samples/train/sample.npz"
    write_processed_sample_npz(sample_path)
    monkeypatch.setattr(paths_mod, "DEFAULT_REPO_ROOT", root)

    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    monkeypatch.chdir(outside_dir)

    issues = validate_processed_records(
        root / "data/processed/v1/manifests/train.jsonl",
        [processed_record("data/processed/v1/samples/train/sample.npz")],
    )

    assert [issue.code for issue in issues] == []


def test_validate_processed_records_reports_missing_required_fields() -> None:
    record = processed_record("data/processed/v1/samples/train/sample.npz")
    record.pop("source_metadata_path")
    record.pop("source_keypoints_dir")

    issues = validate_processed_records(Path("processed.jsonl"), [record])

    missing_fields_issue = next(
        issue for issue in issues if issue.code == "missing_required_fields"
    )
    assert "source_metadata_path" in missing_fields_issue.message
    assert "source_keypoints_dir" in missing_fields_issue.message


def test_validate_processed_records_rejects_absolute_sample_paths() -> None:
    issues = validate_processed_records(
        Path("processed.jsonl"),
        [processed_record("/tmp/sample.npz")],
    )

    assert any(issue.code == "absolute_sample_path" for issue in issues)


def test_validate_processed_records_rejects_repo_escaping_sample_paths() -> None:
    issues = validate_processed_records(
        Path("processed.jsonl"),
        [processed_record("../../outside/sample.npz")],
    )

    assert any(issue.code == "invalid_sample_path" for issue in issues)


@pytest.mark.parametrize(
    ("sample_path", "message_fragment"),
    [
        ("", "non-empty"),
        ("   ", "non-empty"),
        ("data/processed/v1/samples/train/sample.txt", "must end with .npz"),
    ],
)
def test_validate_processed_records_rejects_blank_or_non_npz_sample_paths(
    sample_path: str,
    message_fragment: str,
) -> None:
    issues = validate_processed_records(Path("processed.jsonl"), [processed_record(sample_path)])

    issue = next(issue for issue in issues if issue.code == "invalid_sample_path")
    assert message_fragment in issue.message


def test_validate_processed_records_rejects_directory_sample_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    sample_dir = root / "data/processed/v1/samples/train/sample_dir.npz"
    sample_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(paths_mod, "DEFAULT_REPO_ROOT", root)

    issues = validate_processed_records(
        Path("processed.jsonl"),
        [
            processed_record(
                "data/processed/v1/samples/train/sample_dir.npz",
                sample_id="sample_dir",
            )
        ],
    )

    issue = next(issue for issue in issues if issue.code == "missing_sample_file")
    assert "is a directory" in issue.message


def test_validate_processed_records_rejects_in_repo_npz_outside_processed_samples_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    sample_path = root / "data/interim/sample.npz"
    write_processed_sample_npz(sample_path)
    monkeypatch.setattr(paths_mod, "DEFAULT_REPO_ROOT", root)

    issues = validate_processed_records(
        Path("processed.jsonl"),
        [processed_record("data/interim/sample.npz")],
    )

    issue = next(issue for issue in issues if issue.code == "invalid_sample_path")
    assert "data/processed/v1/samples" in issue.message


def test_validate_processed_records_rejects_processed_sample_path_with_wrong_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    sample_path = root / "data/processed/v1/samples/val/sample.npz"
    write_processed_sample_npz(sample_path)
    monkeypatch.setattr(paths_mod, "DEFAULT_REPO_ROOT", root)

    issues = validate_processed_records(
        Path("processed.jsonl"),
        [processed_record("data/processed/v1/samples/val/sample.npz")],
    )

    issue = next(issue for issue in issues if issue.code == "invalid_sample_path")
    assert "split does not match manifest split" in issue.message


def test_validate_processed_records_rejects_processed_sample_path_with_wrong_filename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    sample_path = root / "data/processed/v1/samples/train/other_sample.npz"
    write_processed_sample_npz(sample_path)
    monkeypatch.setattr(paths_mod, "DEFAULT_REPO_ROOT", root)

    issues = validate_processed_records(
        Path("processed.jsonl"),
        [processed_record("data/processed/v1/samples/train/other_sample.npz")],
    )

    issue = next(issue for issue in issues if issue.code == "invalid_sample_path")
    assert "filename does not match manifest sample_id" in issue.message


def test_validate_processed_records_rejects_unreadable_npz_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    sample_path = root / "data/processed/v1/samples/train/sample.npz"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_text("not-an-npz", encoding="utf-8")
    monkeypatch.setattr(paths_mod, "DEFAULT_REPO_ROOT", root)

    issues = validate_processed_records(
        Path("processed.jsonl"),
        [processed_record("data/processed/v1/samples/train/sample.npz")],
    )

    issue = next(issue for issue in issues if issue.code == "invalid_sample_file")
    assert ".npz" in issue.message
    assert issue.path == sample_path.as_posix()


def test_validate_processed_records_rejects_missing_required_sample_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    sample_path = root / "data/processed/v1/samples/train/sample.npz"
    write_processed_sample_npz(sample_path, drop_keys=("face", "frame_valid_mask"))
    monkeypatch.setattr(paths_mod, "DEFAULT_REPO_ROOT", root)

    issues = validate_processed_records(
        Path("processed.jsonl"),
        [processed_record("data/processed/v1/samples/train/sample.npz")],
    )

    issue = next(issue for issue in issues if issue.code == "missing_sample_arrays")
    assert "face" in issue.message
    assert "frame_valid_mask" in issue.message
    assert issue.path == sample_path.as_posix()


def test_validate_processed_records_rejects_unexpected_sample_payload_schema_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    sample_path = root / "data/processed/v1/samples/train/sample.npz"
    write_processed_sample_npz(
        sample_path,
        overrides={"processed_schema_version": np.asarray("t2sp-processed-v999")},
    )
    monkeypatch.setattr(paths_mod, "DEFAULT_REPO_ROOT", root)

    issues = validate_processed_records(
        Path("processed.jsonl"),
        [processed_record("data/processed/v1/samples/train/sample.npz")],
    )

    issue = next(issue for issue in issues if issue.code == "unexpected_sample_schema_version")
    assert "t2sp-processed-v999" in issue.message
    assert issue.path == sample_path.as_posix()


def test_validate_processed_records_rejects_invalid_sample_array_shapes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    sample_path = root / "data/processed/v1/samples/train/sample.npz"
    write_processed_sample_npz(
        sample_path,
        overrides={"body": np.zeros((3, 25, 2), dtype=np.float32)},
    )
    monkeypatch.setattr(paths_mod, "DEFAULT_REPO_ROOT", root)

    issues = validate_processed_records(
        Path("processed.jsonl"),
        [processed_record("data/processed/v1/samples/train/sample.npz")],
    )

    issue = next(issue for issue in issues if issue.code == "invalid_sample_array_shape")
    assert "'body'" in issue.message
    assert "(3, 25, 2)" in issue.message
    assert issue.path == sample_path.as_posix()


def test_validate_processed_records_rejects_unexpected_sample_selected_person_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    sample_path = root / "data/processed/v1/samples/train/sample.npz"
    write_processed_sample_npz(
        sample_path,
        overrides={"selected_person_index": np.asarray(1, dtype=np.int16)},
    )
    monkeypatch.setattr(paths_mod, "DEFAULT_REPO_ROOT", root)

    issues = validate_processed_records(
        Path("processed.jsonl"),
        [processed_record("data/processed/v1/samples/train/sample.npz")],
    )

    issue = next(
        issue for issue in issues if issue.code == "unexpected_sample_selected_person_index"
    )
    assert "payload=1" in issue.message
    assert issue.path == sample_path.as_posix()


def test_validate_processed_records_allows_matching_nonzero_selected_person_index_with_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    sample_path = root / "data/processed/v1/samples/train/sample.npz"
    write_processed_sample_npz(
        sample_path,
        overrides={"selected_person_index": np.asarray(1, dtype=np.int16)},
    )
    monkeypatch.setattr(paths_mod, "DEFAULT_REPO_ROOT", root)

    issues = validate_processed_records(
        Path("processed.jsonl"),
        [
            processed_record(
                "data/processed/v1/samples/train/sample.npz",
                selected_person_index=1,
            )
        ],
    )

    assert all(issue.code != "unexpected_sample_selected_person_index" for issue in issues)
    warning = next(issue for issue in issues if issue.code == "unexpected_selected_person_index")
    assert warning.severity == "warning"


def test_validate_processed_records_rejects_frame_valid_count_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    sample_path = root / "data/processed/v1/samples/train/sample.npz"
    write_processed_sample_npz(
        sample_path,
        overrides={"frame_valid_mask": np.asarray([True, False], dtype=np.bool_)},
    )
    monkeypatch.setattr(paths_mod, "DEFAULT_REPO_ROOT", root)

    issues = validate_processed_records(
        Path("processed.jsonl"),
        [processed_record("data/processed/v1/samples/train/sample.npz")],
    )

    valid_issue = next(issue for issue in issues if issue.code == "frame_valid_count_mismatch")
    invalid_issue = next(issue for issue in issues if issue.code == "frame_invalid_count_mismatch")
    assert "payload=1" in valid_issue.message
    assert "payload=1" in invalid_issue.message
    assert valid_issue.path == sample_path.as_posix()
    assert invalid_issue.path == sample_path.as_posix()


def test_expected_processed_sample_keys_fixture_matches_builder(
    tmp_path: Path,
    fixtures_dir: Path,
) -> None:
    sample_path = tmp_path / "sample.npz"
    expected_keys = set(
        (fixtures_dir / "expected/processed_sample_keys.txt").read_text(encoding="utf-8").split()
    )
    write_processed_sample_npz(sample_path)

    with np.load(sample_path) as sample:
        assert set(sample.files) == expected_keys
