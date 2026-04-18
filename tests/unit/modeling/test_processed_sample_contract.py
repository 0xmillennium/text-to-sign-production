"""Sprint 3 processed sample payload contract tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from tests.support.builders.manifests import processed_record
from tests.support.modeling import (
    patch_modeling_repo_root,
    write_processed_modeling_manifest,
    write_processed_modeling_sample,
)
from text_to_sign_production.data.constants import PROCESSED_SCHEMA_VERSION
from text_to_sign_production.modeling.data import (
    SPRINT3_TARGET_CHANNEL_SHAPES,
    SPRINT3_TARGET_CHANNELS,
    ProcessedModelingDataError,
    load_processed_pose_sample,
    read_processed_modeling_manifest,
)

pytestmark = pytest.mark.unit


def test_sprint3_target_channel_contract_uses_expected_processed_shapes() -> None:
    assert SPRINT3_TARGET_CHANNELS == ("body", "left_hand", "right_hand")
    assert SPRINT3_TARGET_CHANNEL_SHAPES == {
        "body": (25, 2),
        "left_hand": (21, 2),
        "right_hand": (21, 2),
    }


@pytest.mark.parametrize(
    ("overrides", "drop_keys", "frame_valid_mask", "match"),
    [
        (
            {"processed_schema_version": np.asarray("t2sp-processed-v999")},
            (),
            None,
            PROCESSED_SCHEMA_VERSION,
        ),
        (
            {"selected_person_index": np.asarray(1, dtype=np.int16)},
            (),
            None,
            "selected_person_index",
        ),
        ({}, ("left_hand",), None, "left_hand"),
        (
            {"body": np.zeros((3, 25, 2), dtype=np.float32)},
            (),
            None,
            r"expected \(2, 25, 2\)",
        ),
        ({}, (), np.asarray([True, False, True], dtype=np.bool_), "frame_valid_mask"),
        ({}, (), np.asarray([True, False], dtype=np.bool_), "valid count"),
    ],
)
def test_load_processed_pose_sample_rejects_payload_contract_violations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, Any],
    drop_keys: tuple[str, ...],
    frame_valid_mask: npt.NDArray[np.bool_] | None,
    match: str,
) -> None:
    patch_modeling_repo_root(monkeypatch, tmp_path)
    sample_path = write_processed_modeling_sample(
        tmp_path,
        frame_valid_mask=frame_valid_mask,
        overrides=overrides,
        drop_keys=drop_keys,
    )
    manifest_path = write_processed_modeling_manifest(tmp_path, [processed_record(sample_path)])
    record = read_processed_modeling_manifest(manifest_path, split="train")[0]

    with pytest.raises(ProcessedModelingDataError, match=match):
        load_processed_pose_sample(record)
