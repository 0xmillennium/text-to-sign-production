"""Filtering policy validation and drop-reason tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.support.builders.manifests import normalized_manifest_entry
from text_to_sign_production.data.filtering import (
    FilterConfig,
    determine_drop_reasons,
    load_filter_config,
)

pytestmark = pytest.mark.unit


def _valid_filter_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "require_nonempty_text": True,
        "require_positive_duration": True,
        "require_keypoints_dir": True,
        "require_frames": True,
        "drop_on_sample_parse_error": True,
        "require_at_least_one_valid_frame": True,
        "minimum_nonzero_frames_per_core_channel": 1,
    }


def test_load_filter_config_requires_schema_version_field(tmp_path: Path) -> None:
    config_path = tmp_path / "filter.yaml"
    payload = _valid_filter_payload()
    payload.pop("schema_version")
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required field 'schema_version'"):
        load_filter_config(config_path)


def test_load_filter_config_requires_supported_schema_version(tmp_path: Path) -> None:
    config_path = tmp_path / "filter.yaml"
    payload = _valid_filter_payload()
    payload["schema_version"] = 2
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported filter config schema_version"):
        load_filter_config(config_path)


@pytest.mark.parametrize("schema_version", [True, 1.0])
def test_load_filter_config_rejects_non_integer_schema_version(
    tmp_path: Path,
    schema_version: Any,
) -> None:
    config_path = tmp_path / "filter.yaml"
    payload = _valid_filter_payload()
    payload["schema_version"] = schema_version
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version.*integer"):
        load_filter_config(config_path)


@pytest.mark.parametrize(
    "field_name",
    [
        "require_nonempty_text",
        "require_positive_duration",
        "require_keypoints_dir",
        "require_frames",
        "drop_on_sample_parse_error",
        "require_at_least_one_valid_frame",
    ],
)
def test_load_filter_config_rejects_non_boolean_flags(tmp_path: Path, field_name: str) -> None:
    config_path = tmp_path / "filter.yaml"
    payload = _valid_filter_payload()
    payload[field_name] = "false"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"{field_name}.*boolean"):
        load_filter_config(config_path)


@pytest.mark.parametrize("invalid_value", [True, 1.5, "1", -1])
def test_load_filter_config_rejects_invalid_nonnegative_integer_threshold(
    tmp_path: Path,
    invalid_value: Any,
) -> None:
    config_path = tmp_path / "filter.yaml"
    payload = _valid_filter_payload()
    payload["minimum_nonzero_frames_per_core_channel"] = invalid_value
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="non-negative integer"):
        load_filter_config(config_path)


@pytest.mark.parametrize(
    "field_name",
    ["require_positive_duration", "minimum_nonzero_frames_per_core_channel"],
)
def test_load_filter_config_reports_missing_required_fields(
    tmp_path: Path,
    field_name: str,
) -> None:
    config_path = tmp_path / "filter.yaml"
    payload = _valid_filter_payload()
    payload.pop(field_name)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=field_name):
        load_filter_config(config_path)


def test_determine_drop_reasons_respects_core_channel_policy() -> None:
    entry = normalized_manifest_entry()
    entry.core_channel_nonzero_frames["left_hand"] = 0
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
