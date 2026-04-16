"""Filtering policy validation and drop-reason tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.support.builders.manifests import normalized_manifest_entry
from text_to_sign_production.data.constants import DEFAULT_FILTER_CONFIG_RELATIVE_PATH
from text_to_sign_production.data.filtering import (
    FilterConfig,
    determine_drop_reasons,
    load_filter_config,
)
from text_to_sign_production.data.schemas import NormalizedManifestEntry

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LEGACY_FILTER_CONFIG_PATH = PROJECT_ROOT / "configs/data/filter-v1.yaml"
ACTIVE_FILTER_CONFIG_PATH = PROJECT_ROOT / DEFAULT_FILTER_CONFIG_RELATIVE_PATH


def _valid_filter_payload(*, schema_version: int = 2) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": schema_version,
        "require_nonempty_text": True,
        "require_positive_duration": True,
        "require_keypoints_dir": True,
        "require_frames": True,
        "drop_on_sample_parse_error": True,
        "require_at_least_one_valid_frame": True,
        "minimum_nonzero_frames_per_core_channel": 1,
    }
    if schema_version == 2:
        payload["required_all_core_channels"] = ["body"]
        payload["required_any_core_channel_groups"] = [["left_hand", "right_hand"]]
    return payload


def _v2_filter_config() -> FilterConfig:
    return FilterConfig(
        schema_version=2,
        require_nonempty_text=True,
        require_positive_duration=True,
        require_keypoints_dir=True,
        require_frames=True,
        drop_on_sample_parse_error=True,
        require_at_least_one_valid_frame=True,
        minimum_nonzero_frames_per_core_channel=1,
        required_all_core_channels=("body",),
        required_any_core_channel_groups=(("left_hand", "right_hand"),),
    )


def _entry_with_core_counts(
    *,
    body: int = 2,
    left_hand: int = 2,
    right_hand: int = 2,
) -> NormalizedManifestEntry:
    entry = normalized_manifest_entry()
    entry.core_channel_nonzero_frames = {
        "body": body,
        "left_hand": left_hand,
        "right_hand": right_hand,
    }
    return entry


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
    payload["schema_version"] = 999
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
    [
        "require_positive_duration",
        "minimum_nonzero_frames_per_core_channel",
        "required_all_core_channels",
        "required_any_core_channel_groups",
    ],
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


def test_load_filter_config_reads_active_v2_policy() -> None:
    config = load_filter_config(ACTIVE_FILTER_CONFIG_PATH)

    assert config.schema_version == 2
    assert config.required_all_core_channels == ("body",)
    assert config.required_any_core_channel_groups == (("left_hand", "right_hand"),)


def test_load_filter_config_keeps_legacy_v1_policy_strict() -> None:
    config = load_filter_config(LEGACY_FILTER_CONFIG_PATH)

    assert config.schema_version == 1
    assert config.required_all_core_channels == ("body", "left_hand", "right_hand")
    assert config.required_any_core_channel_groups == ()


def test_load_filter_config_canonicalizes_group_drop_reason_order(tmp_path: Path) -> None:
    config_path = tmp_path / "filter.yaml"
    payload = _valid_filter_payload()
    payload["required_any_core_channel_groups"] = [["right_hand", "left_hand"]]
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    config = load_filter_config(config_path)

    assert config.required_any_core_channel_groups == (("left_hand", "right_hand"),)


@pytest.mark.parametrize(
    "payload_update,match",
    [
        ({"required_all_core_channels": ["face"]}, "unsupported core channel"),
        ({"required_any_core_channel_groups": [["left_hand", "left_hand"]]}, "duplicate"),
        ({"required_any_core_channel_groups": []}, "must not be empty"),
    ],
)
def test_load_filter_config_rejects_invalid_v2_channel_policy(
    tmp_path: Path,
    payload_update: dict[str, Any],
    match: str,
) -> None:
    config_path = tmp_path / "filter.yaml"
    payload = _valid_filter_payload()
    payload.update(payload_update)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_filter_config(config_path)


@pytest.mark.parametrize(
    "left_hand,right_hand",
    [
        (0, 2),
        (2, 0),
    ],
)
def test_determine_drop_reasons_keeps_one_hand_samples(
    left_hand: int,
    right_hand: int,
) -> None:
    entry = _entry_with_core_counts(left_hand=left_hand, right_hand=right_hand)

    drop_reasons = determine_drop_reasons(entry, _v2_filter_config())

    assert drop_reasons == []


def test_determine_drop_reasons_drops_when_both_hands_are_unusable() -> None:
    entry = _entry_with_core_counts(left_hand=0, right_hand=0)

    drop_reasons = determine_drop_reasons(entry, _v2_filter_config())

    assert drop_reasons == ["unusable_core_channel_group:left_hand|right_hand"]


def test_determine_drop_reasons_drops_when_body_is_unusable_with_one_hand_present() -> None:
    entry = _entry_with_core_counts(body=0, left_hand=0, right_hand=2)

    drop_reasons = determine_drop_reasons(entry, _v2_filter_config())

    assert drop_reasons == ["unusable_core_channel:body"]


def test_same_one_hand_sample_is_dropped_by_v1_and_kept_by_v2() -> None:
    entry = _entry_with_core_counts(left_hand=0, right_hand=2)
    legacy_config = load_filter_config(LEGACY_FILTER_CONFIG_PATH)
    active_config = load_filter_config(ACTIVE_FILTER_CONFIG_PATH)

    assert determine_drop_reasons(entry, legacy_config) == ["unusable_core_channel:left_hand"]
    assert determine_drop_reasons(entry, active_config) == []
