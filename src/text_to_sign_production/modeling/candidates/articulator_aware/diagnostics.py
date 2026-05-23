"""Channel diagnostics records and summaries."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from statistics import fmean
from types import MappingProxyType

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.data.gate.pose.types import PoseChannel
from text_to_sign_production.modeling.candidates.articulator_aware.errors import (
    ArticulatorAwareError,
)
from text_to_sign_production.modeling.candidates.articulator_aware.losses import (
    ChannelLossRecord,
)
from text_to_sign_production.modeling.candidates.articulator_aware.masks import (
    ChannelMaskSummary,
)

CHANNEL_DIAGNOSTIC_RECORD_SCHEMA_VERSION = "t2sp-channel-diagnostic-record-v1"
CHANNEL_DIAGNOSTIC_SUMMARY_SCHEMA_VERSION = "t2sp-channel-diagnostic-summary-v1"

_CAVEATS = (
    "Channel-level diagnostics do not prove sign-level adequacy.",
    "Face channel proxy is not full non-manual linguistic annotation.",
)


@dataclass(frozen=True, slots=True)
class ChannelDiagnosticRecord:
    schema_version: str
    sample_id: str
    split: SampleSplit
    channel: PoseChannel
    valid_fraction: float
    weighted_loss: float | None
    masked_l1_hint: float | None
    skipped: bool
    issues: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != CHANNEL_DIAGNOSTIC_RECORD_SCHEMA_VERSION:
            raise ArticulatorAwareError("channel diagnostic record schema_version is unsupported.")
        _require_text(self.sample_id, "sample_id")
        object.__setattr__(self, "split", SampleSplit(self.split))
        object.__setattr__(self, "channel", PoseChannel(self.channel))
        if not 0.0 <= float(self.valid_fraction) <= 1.0:
            raise ArticulatorAwareError("valid_fraction must be in [0, 1].")
        object.__setattr__(self, "valid_fraction", float(self.valid_fraction))
        if self.weighted_loss is not None:
            object.__setattr__(self, "weighted_loss", _finite(self.weighted_loss, "weighted_loss"))
        if self.masked_l1_hint is not None:
            object.__setattr__(self, "masked_l1_hint", _finite(self.masked_l1_hint, "masked_l1_hint"))
        if not isinstance(self.skipped, bool):
            raise ArticulatorAwareError("skipped must be a boolean.")
        if self.skipped and (
            self.weighted_loss is not None or self.masked_l1_hint is not None
        ):
            raise ArticulatorAwareError(
                "skipped diagnostic records must not contain weighted_loss or masked_l1_hint."
            )
        if not self.skipped and self.weighted_loss is None:
            raise ArticulatorAwareError(
                "computed diagnostic records must contain finite weighted_loss."
            )
        object.__setattr__(self, "issues", tuple(self.issues))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "split": self.split.value,
            "channel": self.channel.value,
            "valid_fraction": self.valid_fraction,
            "weighted_loss": self.weighted_loss,
            "masked_l1_hint": self.masked_l1_hint,
            "skipped": self.skipped,
            "issues": list(self.issues),
        }


@dataclass(frozen=True, slots=True)
class ChannelDiagnosticSummary:
    schema_version: str
    split: SampleSplit
    channel_count: int
    sample_count: int
    records_count: int
    skipped_records_count: int
    per_channel_valid_fraction_mean: Mapping[str, float | None] = field(
        default_factory=lambda: MappingProxyType({})
    )
    per_channel_loss_mean: Mapping[str, float | None] = field(
        default_factory=lambda: MappingProxyType({})
    )
    caveats: tuple[str, ...] = _CAVEATS

    def __post_init__(self) -> None:
        if self.schema_version != CHANNEL_DIAGNOSTIC_SUMMARY_SCHEMA_VERSION:
            raise ArticulatorAwareError("channel diagnostic summary schema_version is unsupported.")
        object.__setattr__(self, "split", SampleSplit(self.split))
        for name in ("channel_count", "sample_count", "records_count", "skipped_records_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ArticulatorAwareError(f"{name} must be a non-negative integer.")
        if self.channel_count <= 0:
            raise ArticulatorAwareError("channel_count must be positive.")
        if self.skipped_records_count > self.records_count:
            raise ArticulatorAwareError(
                "skipped_records_count must not exceed records_count."
            )
        valid_means = _optional_float_mapping(self.per_channel_valid_fraction_mean)
        loss_means = _optional_float_mapping(self.per_channel_loss_mean)
        if len(valid_means) != self.channel_count or len(loss_means) != self.channel_count:
            raise ArticulatorAwareError(
                "channel_count must equal the number of per-channel summary means."
            )
        for channel, value in valid_means.items():
            if value is not None and not 0.0 <= value <= 1.0:
                raise ArticulatorAwareError(
                    f"per_channel_valid_fraction_mean.{channel} must be null or in [0, 1]."
                )
        for channel, value in loss_means.items():
            if value is not None and value < 0.0:
                raise ArticulatorAwareError(
                    f"per_channel_loss_mean.{channel} must be null or non-negative."
                )
        object.__setattr__(
            self,
            "per_channel_valid_fraction_mean",
            MappingProxyType(valid_means),
        )
        object.__setattr__(
            self,
            "per_channel_loss_mean",
            MappingProxyType(loss_means),
        )
        caveats = tuple(self.caveats)
        for required in _CAVEATS:
            if required not in caveats:
                raise ArticulatorAwareError(f"channel diagnostic summary caveats must include: {required}")
        object.__setattr__(self, "caveats", caveats)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "split": self.split.value,
            "channel_count": self.channel_count,
            "sample_count": self.sample_count,
            "records_count": self.records_count,
            "skipped_records_count": self.skipped_records_count,
            "per_channel_valid_fraction_mean": {
                key: self.per_channel_valid_fraction_mean[key]
                for key in sorted(self.per_channel_valid_fraction_mean)
            },
            "per_channel_loss_mean": {
                key: self.per_channel_loss_mean[key]
                for key in sorted(self.per_channel_loss_mean)
            },
            "caveats": list(self.caveats),
        }


def build_channel_diagnostic_records(
    *,
    mask_summaries: Iterable[ChannelMaskSummary],
    loss_records: Iterable[ChannelLossRecord],
) -> tuple[ChannelDiagnosticRecord, ...]:
    """Join mask summaries and loss records while preserving skipped status."""

    summaries = tuple(mask_summaries)
    losses = tuple(loss_records)
    loss_by_key = {
        (loss.sample_id, loss.split, loss.channel): loss
        for loss in losses
    }
    records: list[ChannelDiagnosticRecord] = []
    for summary in summaries:
        if not isinstance(summary, ChannelMaskSummary):
            raise ArticulatorAwareError("mask_summaries must contain ChannelMaskSummary values.")
        for channel_value, valid_fraction in summary.channel_valid_fractions.items():
            channel = PoseChannel(channel_value)
            issues = [
                issue
                for issue in summary.issues
                if issue.startswith(f"{channel.value}:")
            ]
            loss = loss_by_key.get((summary.sample_id, summary.split, channel))
            if loss is None:
                issues.append(f"{channel.value}: missing channel loss record.")
                records.append(
                    ChannelDiagnosticRecord(
                        schema_version=CHANNEL_DIAGNOSTIC_RECORD_SCHEMA_VERSION,
                        sample_id=summary.sample_id,
                        split=summary.split,
                        channel=channel,
                        valid_fraction=valid_fraction,
                        weighted_loss=None,
                        masked_l1_hint=None,
                        skipped=True,
                        issues=tuple(issues),
                    )
                )
                continue
            if loss.reason is not None:
                issues.append(loss.reason)
            records.append(
                ChannelDiagnosticRecord(
                    schema_version=CHANNEL_DIAGNOSTIC_RECORD_SCHEMA_VERSION,
                    sample_id=summary.sample_id,
                    split=summary.split,
                    channel=channel,
                    valid_fraction=valid_fraction,
                    weighted_loss=loss.value,
                    masked_l1_hint=None,
                    skipped=loss.skipped,
                    issues=tuple(issues),
                )
            )
    return tuple(records)


def aggregate_channel_diagnostics(
    records: Iterable[ChannelDiagnosticRecord],
    *,
    split: SampleSplit,
) -> ChannelDiagnosticSummary:
    """Aggregate channel diagnostics without assigning fake zero to skipped records."""

    resolved_split = SampleSplit(split)
    materialized = tuple(records)
    for record in materialized:
        if not isinstance(record, ChannelDiagnosticRecord):
            raise ArticulatorAwareError("records must contain ChannelDiagnosticRecord values.")
        if record.split is not resolved_split:
            raise ArticulatorAwareError("all diagnostic records must match the requested split.")
    channels = tuple(sorted({record.channel.value for record in materialized}))
    samples = tuple(sorted({record.sample_id for record in materialized}))
    valid_means = {
        channel: _mean_or_none(
            record.valid_fraction
            for record in materialized
            if record.channel.value == channel
        )
        for channel in channels
    }
    loss_means = {
        channel: _mean_or_none(
            record.weighted_loss
            for record in materialized
            if record.channel.value == channel and record.weighted_loss is not None
        )
        for channel in channels
    }
    return ChannelDiagnosticSummary(
        schema_version=CHANNEL_DIAGNOSTIC_SUMMARY_SCHEMA_VERSION,
        split=resolved_split,
        channel_count=len(channels),
        sample_count=len(samples),
        records_count=len(materialized),
        skipped_records_count=sum(1 for record in materialized if record.skipped),
        per_channel_valid_fraction_mean=valid_means,
        per_channel_loss_mean=loss_means,
        caveats=_CAVEATS,
    )


def _mean_or_none(values: Iterable[float | None]) -> float | None:
    materialized = tuple(value for value in values if value is not None)
    return None if not materialized else float(fmean(materialized))


def _finite(value: object, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ArticulatorAwareError(f"{name} must be a number.")
    amount = float(value)
    if amount != amount or amount in {float("inf"), float("-inf")}:
        raise ArticulatorAwareError(f"{name} must be finite.")
    return amount


def _optional_float_mapping(value: Mapping[str, float | None]) -> dict[str, float | None]:
    if not isinstance(value, Mapping):
        raise ArticulatorAwareError("summary channel means must be mappings.")
    result: dict[str, float | None] = {}
    for key, amount in value.items():
        _require_text(key, "summary channel key")
        result[key] = None if amount is None else _finite(amount, key)
    return result


def _require_text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ArticulatorAwareError(f"{name} must be non-empty.")


__all__ = [
    "CHANNEL_DIAGNOSTIC_RECORD_SCHEMA_VERSION",
    "CHANNEL_DIAGNOSTIC_SUMMARY_SCHEMA_VERSION",
    "ChannelDiagnosticRecord",
    "ChannelDiagnosticSummary",
    "aggregate_channel_diagnostics",
    "build_channel_diagnostic_records",
]
