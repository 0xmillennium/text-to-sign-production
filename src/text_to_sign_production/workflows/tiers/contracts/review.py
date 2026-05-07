from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.workflows.tiers.contracts.config import (
    TiersWorkflowInputError,
)


@dataclass(frozen=True, slots=True)
class TiersRuntimeAssetRow:
    label: str
    path: Path
    exists: bool

    def __post_init__(self) -> None:
        _validate_non_empty_text("label", self.label)
        object.__setattr__(self, "label", self.label.strip())


@dataclass(frozen=True, slots=True)
class TiersMembershipCountRow:
    tier: str
    split: str
    included_count: int
    excluded_count: int

    def __post_init__(self) -> None:
        _validate_non_empty_text("tier", self.tier)
        _validate_non_empty_text("split", self.split)
        _ensure_non_negative("included_count", self.included_count)
        _ensure_non_negative("excluded_count", self.excluded_count)
        object.__setattr__(self, "tier", self.tier.strip())
        object.__setattr__(self, "split", self.split.strip())


@dataclass(frozen=True, slots=True)
class TiersReportArtifactRow:
    label: str
    path: Path

    def __post_init__(self) -> None:
        _validate_non_empty_text("label", self.label)
        object.__setattr__(self, "label", self.label.strip())


@dataclass(frozen=True, slots=True)
class TiersTieredManifestRow:
    tier: str
    membership: str
    split: str
    path: Path

    def __post_init__(self) -> None:
        _validate_non_empty_text("tier", self.tier)
        _validate_non_empty_text("membership", self.membership)
        _validate_non_empty_text("split", self.split)
        object.__setattr__(self, "tier", self.tier.strip())
        object.__setattr__(self, "membership", self.membership.strip())
        object.__setattr__(self, "split", self.split.strip())


@dataclass(frozen=True, slots=True)
class TiersPublishTargetRow:
    label: str
    kind: str
    source_path: Path
    target_path: Path
    tier: str | None = None
    membership: str | None = None
    split: str | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_text("label", self.label)
        _validate_non_empty_text("kind", self.kind)
        object.__setattr__(self, "label", self.label.strip())
        object.__setattr__(self, "kind", self.kind.strip())


def _validate_non_empty_text(field_name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise TiersWorkflowInputError(f"{field_name} must be a non-empty string")


def _ensure_non_negative(field_name: str, value: int) -> None:
    if value < 0:
        raise TiersWorkflowInputError(f"{field_name} must be >= 0")
