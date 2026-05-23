"""Strict deterministic IO for articulator-aware foundation artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.data.gate.pose.types import PoseChannel
from text_to_sign_production.modeling.backbones.bfh_masks import BfhChannelPartition
from text_to_sign_production.modeling.backbones.bfh_vectorization import (
    bfh_tensor_layout_from_dict,
)
from text_to_sign_production.modeling.candidates.articulator_aware.diagnostics import (
    CHANNEL_DIAGNOSTIC_RECORD_SCHEMA_VERSION,
    CHANNEL_DIAGNOSTIC_SUMMARY_SCHEMA_VERSION,
    ChannelDiagnosticRecord,
    ChannelDiagnosticSummary,
)
from text_to_sign_production.modeling.candidates.articulator_aware.config import (
    ChannelMaskStrategyConfig,
)
from text_to_sign_production.modeling.candidates.articulator_aware.errors import (
    ArticulatorAwareError,
)
from text_to_sign_production.modeling.candidates.articulator_aware.losses import (
    CHANNEL_LOSS_RECORD_SCHEMA_VERSION,
    CHANNEL_LOSS_WEIGHTING_SCHEMA_VERSION,
    ChannelLossRecord,
    ChannelLossWeightingPolicy,
)
from text_to_sign_production.modeling.candidates.articulator_aware.masks import (
    CHANNEL_MASK_SUMMARY_SCHEMA_VERSION,
    ChannelMaskSummary,
)
from text_to_sign_production.modeling.candidates.articulator_aware.partitions import (
    ARTICULATOR_PARTITION_POLICY_SCHEMA_VERSION,
    ArticulatorChannelPartitionPolicy,
)


def write_partition_policy_json(path: Path, policy: ArticulatorChannelPartitionPolicy) -> None:
    if not isinstance(policy, ArticulatorChannelPartitionPolicy):
        raise ArticulatorAwareError("policy must be ArticulatorChannelPartitionPolicy.")
    _write_json(path, policy.to_dict())


def read_partition_policy_json(path: Path) -> ArticulatorChannelPartitionPolicy:
    record = _read_json_object(path, "partition policy")
    _require_keys(
        record,
        {
            "schema_version",
            "source",
            "layout",
            "primary_channels",
            "partitions",
            "composite_groups",
        },
        "partition policy",
    )
    if record["schema_version"] != ARTICULATOR_PARTITION_POLICY_SCHEMA_VERSION:
        raise ArticulatorAwareError("partition policy schema_version is unsupported.")
    raw_partitions = _list(record["partitions"], "partitions")
    return ArticulatorChannelPartitionPolicy(
        schema_version=_text(record["schema_version"], "schema_version"),
        source=_text(record["source"], "source"),
        layout=bfh_tensor_layout_from_dict(_mapping(record["layout"], "layout")),
        primary_channels=tuple(
            PoseChannel(_text(channel, "primary channel"))
            for channel in _list(record["primary_channels"], "primary_channels")
        ),
        partitions=tuple(_partition_from_record(item) for item in raw_partitions),
        composite_groups=_channel_groups(record["composite_groups"]),
    )


def write_mask_strategy_json(path: Path, strategy: ChannelMaskStrategyConfig) -> None:
    if not isinstance(strategy, ChannelMaskStrategyConfig):
        raise ArticulatorAwareError("strategy must be ChannelMaskStrategyConfig.")
    _write_json(
        path,
        {"schema_version": "t2sp-articulator-mask-strategy-v1", **strategy.to_dict()},
    )


def read_mask_strategy_json(path: Path) -> ChannelMaskStrategyConfig:
    record = _read_json_object(path, "mask strategy")
    _require_keys(
        record,
        {
            "schema_version",
            "validity_source",
            "confidence_policy",
            "missing_channel_policy",
            "all_invalid_sample_policy",
            "channel_min_valid_fraction",
        },
        "mask strategy",
    )
    if record["schema_version"] != "t2sp-articulator-mask-strategy-v1":
        raise ArticulatorAwareError("mask strategy schema_version is unsupported.")
    return ChannelMaskStrategyConfig(
        validity_source=_text(record["validity_source"], "validity_source"),
        confidence_policy=_text(record["confidence_policy"], "confidence_policy"),
        missing_channel_policy=_text(
            record["missing_channel_policy"], "missing_channel_policy"
        ),
        all_invalid_sample_policy=_text(
            record["all_invalid_sample_policy"], "all_invalid_sample_policy"
        ),
        channel_min_valid_fraction=_channel_float_mapping(
            record["channel_min_valid_fraction"], "channel_min_valid_fraction"
        ),
    )


def write_mask_summaries_jsonl(path: Path, summaries: Iterable[ChannelMaskSummary]) -> None:
    _write_jsonl(path, tuple(_require_type(item, ChannelMaskSummary) for item in summaries))


def read_mask_summaries_jsonl(path: Path) -> tuple[ChannelMaskSummary, ...]:
    return tuple(_mask_summary_from_record(record, line_number=line) for line, record in _read_jsonl(path))


def write_loss_weighting_policy_json(path: Path, policy: ChannelLossWeightingPolicy) -> None:
    if not isinstance(policy, ChannelLossWeightingPolicy):
        raise ArticulatorAwareError("policy must be ChannelLossWeightingPolicy.")
    _write_json(path, policy.to_dict())


def read_loss_weighting_policy_json(path: Path) -> ChannelLossWeightingPolicy:
    record = _read_json_object(path, "loss weighting policy")
    _require_keys(
        record,
        {
            "schema_version",
            "policy",
            "normalize_weights",
            "channel_weights",
            "normalized_channel_weights",
            "velocity_weight",
            "symmetry_weight",
            "cross_channel_consistency_weight",
        },
        "loss weighting policy",
    )
    if record["schema_version"] != CHANNEL_LOSS_WEIGHTING_SCHEMA_VERSION:
        raise ArticulatorAwareError("loss weighting policy schema_version is unsupported.")
    return ChannelLossWeightingPolicy(
        schema_version=_text(record["schema_version"], "schema_version"),
        policy=_text(record["policy"], "policy"),
        normalize_weights=_bool(record["normalize_weights"], "normalize_weights"),
        channel_weights=_channel_float_mapping(record["channel_weights"], "channel_weights"),
        normalized_channel_weights=_channel_float_mapping(
            record["normalized_channel_weights"],
            "normalized_channel_weights",
        ),
        velocity_weight=_float(record["velocity_weight"], "velocity_weight"),
        symmetry_weight=_float(record["symmetry_weight"], "symmetry_weight"),
        cross_channel_consistency_weight=_float(
            record["cross_channel_consistency_weight"],
            "cross_channel_consistency_weight",
        ),
    )


def write_channel_loss_records_jsonl(path: Path, records: Iterable[ChannelLossRecord]) -> None:
    _write_jsonl(path, tuple(_require_type(item, ChannelLossRecord) for item in records))


def read_channel_loss_records_jsonl(path: Path) -> tuple[ChannelLossRecord, ...]:
    return tuple(_loss_record_from_record(record, line_number=line) for line, record in _read_jsonl(path))


def write_channel_diagnostic_records_jsonl(path: Path, records: Iterable[ChannelDiagnosticRecord]) -> None:
    _write_jsonl(path, tuple(_require_type(item, ChannelDiagnosticRecord) for item in records))


def read_channel_diagnostic_records_jsonl(path: Path) -> tuple[ChannelDiagnosticRecord, ...]:
    return tuple(
        _diagnostic_record_from_record(record, line_number=line)
        for line, record in _read_jsonl(path)
    )


def write_channel_diagnostic_summary_json(path: Path, summary: ChannelDiagnosticSummary) -> None:
    if not isinstance(summary, ChannelDiagnosticSummary):
        raise ArticulatorAwareError("summary must be ChannelDiagnosticSummary.")
    _write_json(path, summary.to_dict())


def read_channel_diagnostic_summary_json(path: Path) -> ChannelDiagnosticSummary:
    record = _read_json_object(path, "channel diagnostic summary")
    _require_keys(
        record,
        {
            "schema_version",
            "split",
            "channel_count",
            "sample_count",
            "records_count",
            "skipped_records_count",
            "per_channel_valid_fraction_mean",
            "per_channel_loss_mean",
            "caveats",
        },
        "channel diagnostic summary",
    )
    if record["schema_version"] != CHANNEL_DIAGNOSTIC_SUMMARY_SCHEMA_VERSION:
        raise ArticulatorAwareError("channel diagnostic summary schema_version is unsupported.")
    return ChannelDiagnosticSummary(
        schema_version=_text(record["schema_version"], "schema_version"),
        split=SampleSplit(_text(record["split"], "split")),
        channel_count=_int(record["channel_count"], "channel_count"),
        sample_count=_int(record["sample_count"], "sample_count"),
        records_count=_int(record["records_count"], "records_count"),
        skipped_records_count=_int(record["skipped_records_count"], "skipped_records_count"),
        per_channel_valid_fraction_mean=_optional_float_mapping(
            record["per_channel_valid_fraction_mean"],
            "per_channel_valid_fraction_mean",
        ),
        per_channel_loss_mean=_optional_float_mapping(
            record["per_channel_loss_mean"],
            "per_channel_loss_mean",
        ),
        caveats=tuple(_text(item, "caveat") for item in _list(record["caveats"], "caveats")),
    )


def _mask_summary_from_record(record: Mapping[str, object], *, line_number: int) -> ChannelMaskSummary:
    _require_keys(
        record,
        {
            "schema_version",
            "sample_id",
            "split",
            "frame_count",
            "channel_valid_counts",
            "channel_total_counts",
            "channel_valid_fractions",
            "all_invalid_channels",
            "issues",
        },
        f"mask summary line {line_number}",
    )
    if record["schema_version"] != CHANNEL_MASK_SUMMARY_SCHEMA_VERSION:
        raise ArticulatorAwareError(f"mask summary schema_version is unsupported at line {line_number}.")
    return ChannelMaskSummary(
        schema_version=_text(record["schema_version"], "schema_version"),
        sample_id=_text(record["sample_id"], "sample_id"),
        split=SampleSplit(_text(record["split"], "split")),
        frame_count=_int(record["frame_count"], "frame_count"),
        channel_valid_counts=_string_int_mapping(record["channel_valid_counts"], "channel_valid_counts"),
        channel_total_counts=_string_int_mapping(record["channel_total_counts"], "channel_total_counts"),
        channel_valid_fractions=_string_float_mapping(
            record["channel_valid_fractions"],
            "channel_valid_fractions",
        ),
        all_invalid_channels=tuple(
            _text(item, "all_invalid_channel")
            for item in _list(record["all_invalid_channels"], "all_invalid_channels")
        ),
        issues=tuple(_text(item, "issue") for item in _list(record["issues"], "issues")),
    )


def _loss_record_from_record(record: Mapping[str, object], *, line_number: int) -> ChannelLossRecord:
    _require_keys(
        record,
        {
            "schema_version",
            "sample_id",
            "split",
            "channel",
            "loss_name",
            "value",
            "valid_observation_count",
            "skipped",
            "reason",
        },
        f"loss record line {line_number}",
    )
    if record["schema_version"] != CHANNEL_LOSS_RECORD_SCHEMA_VERSION:
        raise ArticulatorAwareError(f"loss record schema_version is unsupported at line {line_number}.")
    return ChannelLossRecord(
        schema_version=_text(record["schema_version"], "schema_version"),
        sample_id=_text(record["sample_id"], "sample_id"),
        split=SampleSplit(_text(record["split"], "split")),
        channel=PoseChannel(_text(record["channel"], "channel")),
        loss_name=_text(record["loss_name"], "loss_name"),
        value=_optional_float(record["value"], "value"),
        valid_observation_count=_int(record["valid_observation_count"], "valid_observation_count"),
        skipped=_bool(record["skipped"], "skipped"),
        reason=_optional_text(record["reason"], "reason"),
    )


def _diagnostic_record_from_record(record: Mapping[str, object], *, line_number: int) -> ChannelDiagnosticRecord:
    _require_keys(
        record,
        {
            "schema_version",
            "sample_id",
            "split",
            "channel",
            "valid_fraction",
            "weighted_loss",
            "masked_l1_hint",
            "skipped",
            "issues",
        },
        f"diagnostic record line {line_number}",
    )
    if record["schema_version"] != CHANNEL_DIAGNOSTIC_RECORD_SCHEMA_VERSION:
        raise ArticulatorAwareError(f"diagnostic record schema_version is unsupported at line {line_number}.")
    return ChannelDiagnosticRecord(
        schema_version=_text(record["schema_version"], "schema_version"),
        sample_id=_text(record["sample_id"], "sample_id"),
        split=SampleSplit(_text(record["split"], "split")),
        channel=PoseChannel(_text(record["channel"], "channel")),
        valid_fraction=_float(record["valid_fraction"], "valid_fraction"),
        weighted_loss=_optional_float(record["weighted_loss"], "weighted_loss"),
        masked_l1_hint=_optional_float(record["masked_l1_hint"], "masked_l1_hint"),
        skipped=_bool(record["skipped"], "skipped"),
        issues=tuple(_text(item, "issue") for item in _list(record["issues"], "issues")),
    )


def _partition_from_record(record: object) -> BfhChannelPartition:
    item = _mapping(record, "partition")
    _require_keys(
        item,
        {"channel", "joint_start", "joint_stop", "joint_count", "feature_start", "feature_stop"},
        "partition",
    )
    return BfhChannelPartition(
        channel=PoseChannel(_text(item["channel"], "partition.channel")),
        joint_start=_int(item["joint_start"], "partition.joint_start"),
        joint_stop=_int(item["joint_stop"], "partition.joint_stop"),
        joint_count=_int(item["joint_count"], "partition.joint_count"),
        feature_start=_int(item["feature_start"], "partition.feature_start"),
        feature_stop=_int(item["feature_stop"], "partition.feature_stop"),
    )


def _write_json(path: Path, value: object) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Iterable[object]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True))
            handle.write("\n")


def _read_json_object(path: Path, label: str) -> Mapping[str, object]:
    try:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArticulatorAwareError(f"malformed {label} JSON: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise ArticulatorAwareError(f"{label} JSON must contain an object.")
    if any(not isinstance(key, str) for key in loaded):
        raise ArticulatorAwareError(f"{label} JSON keys must be strings.")
    return loaded


def _read_jsonl(path: Path) -> tuple[tuple[int, Mapping[str, object]], ...]:
    records: list[tuple[int, Mapping[str, object]]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                loaded = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ArticulatorAwareError(f"malformed JSONL at line {line_number}: {exc}") from exc
            if not isinstance(loaded, Mapping):
                raise ArticulatorAwareError(f"JSONL record at line {line_number} must be an object.")
            if any(not isinstance(key, str) for key in loaded):
                raise ArticulatorAwareError(f"JSONL record keys at line {line_number} must be strings.")
            records.append((line_number, loaded))
    return tuple(records)


def _require_type(value, expected_type):
    if not isinstance(value, expected_type):
        raise ArticulatorAwareError(f"expected {expected_type.__name__} records.")
    return value


def _require_keys(record: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(record) != expected:
        raise ArticulatorAwareError(
            f"{label} keys mismatch: expected={sorted(expected)}, observed={sorted(record)}."
        )


def _channel_groups(value: object) -> Mapping[str, tuple[PoseChannel, ...]]:
    raw = _mapping(value, "composite_groups")
    return {
        name: tuple(PoseChannel(_text(channel, f"composite_groups.{name}")) for channel in _list(channels, name))
        for name, channels in raw.items()
    }


def _channel_float_mapping(value: object, name: str) -> Mapping[PoseChannel, float]:
    raw = _mapping(value, name)
    return {PoseChannel(_text(channel, f"{name} channel")): _float(amount, name) for channel, amount in raw.items()}


def _string_int_mapping(value: object, name: str) -> Mapping[str, int]:
    raw = _mapping(value, name)
    return {key: _int(amount, f"{name}.{key}") for key, amount in raw.items()}


def _string_float_mapping(value: object, name: str) -> Mapping[str, float]:
    raw = _mapping(value, name)
    return {key: _float(amount, f"{name}.{key}") for key, amount in raw.items()}


def _optional_float_mapping(value: object, name: str) -> Mapping[str, float | None]:
    raw = _mapping(value, name)
    return {key: _optional_float(amount, f"{name}.{key}") for key, amount in raw.items()}


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ArticulatorAwareError(f"{name} must be a JSON object with string keys.")
    return value


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ArticulatorAwareError(f"{name} must be a list.")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArticulatorAwareError(f"{name} must be non-empty text.")
    return value


def _optional_text(value: object, name: str) -> str | None:
    return None if value is None else _text(value, name)


def _int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ArticulatorAwareError(f"{name} must be an integer.")
    return value


def _float(value: object, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ArticulatorAwareError(f"{name} must be a number.")
    amount = float(value)
    if amount != amount or amount in {float("inf"), float("-inf")}:
        raise ArticulatorAwareError(f"{name} must be finite.")
    return amount


def _optional_float(value: object, name: str) -> float | None:
    return None if value is None else _float(value, name)


def _bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ArticulatorAwareError(f"{name} must be a boolean.")
    return value


__all__ = [
    "read_channel_diagnostic_records_jsonl",
    "read_channel_diagnostic_summary_json",
    "read_channel_loss_records_jsonl",
    "read_mask_strategy_json",
    "read_loss_weighting_policy_json",
    "read_mask_summaries_jsonl",
    "read_partition_policy_json",
    "write_channel_diagnostic_records_jsonl",
    "write_channel_diagnostic_summary_json",
    "write_channel_loss_records_jsonl",
    "write_mask_strategy_json",
    "write_loss_weighting_policy_json",
    "write_mask_summaries_jsonl",
    "write_partition_policy_json",
]
