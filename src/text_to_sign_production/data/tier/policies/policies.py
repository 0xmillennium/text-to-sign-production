"""Typed tier-policy parsing from ``configs/data/tiers.yaml``."""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TypeVar

import yaml

from text_to_sign_production.core.ids import TierName
from text_to_sign_production.data.tier.families.types import BindingQualityFamily

DEFAULT_TIER_POLICIES_CONFIG_PATH = Path("configs/data/tiers.yaml")
EnumT = TypeVar("EnumT", bound=enum.StrEnum)


class TierLeakageSeverity(enum.StrEnum):
    """Parsed leakage-severity policy value carried by existing tier config."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class TierPolicy:
    """Typed policy for one named tier."""

    tier_name: TierName
    family_filter_levels: Mapping[BindingQualityFamily, TierName]
    max_allowed_leakage_severity: TierLeakageSeverity = TierLeakageSeverity.NONE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "family_filter_levels",
            MappingProxyType(dict(self.family_filter_levels)),
        )


@dataclass(frozen=True, slots=True)
class TierPoliciesConfig:
    """Typed policy authority for tier ordering and selected-tier semantics."""

    policies_by_tier: Mapping[TierName, TierPolicy]
    tier_order: tuple[TierName, ...] = tuple(TierName)

    def __post_init__(self) -> None:
        object.__setattr__(self, "policies_by_tier", MappingProxyType(dict(self.policies_by_tier)))
        object.__setattr__(self, "tier_order", tuple(self.tier_order))

    @property
    def strongest_first(self) -> tuple[TierName, ...]:
        """Return tiers ordered from strictest to weakest."""
        return tuple(reversed(self.tier_order))

    def policy_for(self, tier: TierName) -> TierPolicy:
        """Return the policy for a named tier."""
        return self.policies_by_tier[tier]

    def filter_level_for(self, tier: TierName, family: BindingQualityFamily) -> TierName:
        """Return the filter level that applies to one family under one tier."""
        return self.policy_for(tier).family_filter_levels[family]

    def select_best_supported_tier(
        self,
        supported_by_family: Mapping[BindingQualityFamily, tuple[TierName, ...]],
        binding_families: tuple[BindingQualityFamily, ...],
    ) -> TierName | None:
        """Choose the strongest tier supported by every binding family."""
        for tier in self.strongest_first:
            if all(tier in supported_by_family.get(family, ()) for family in binding_families):
                return tier
        return None


def load_tier_policies_config(
    path: Path = DEFAULT_TIER_POLICIES_CONFIG_PATH,
) -> TierPoliciesConfig:
    """Load and parse tier policies from YAML."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid tier policies YAML: {exc}") from exc

    config = parse_tier_policies_config_mapping(loaded)
    from text_to_sign_production.data.tier.policies.validate import (
        validate_tier_policies_config,
    )

    issues = validate_tier_policies_config(config)
    if issues:
        raise ValueError(f"Invalid tier policies config: {issues}")
    return config


def parse_tier_policies_config_mapping(value: object) -> TierPoliciesConfig:
    """Parse a loaded YAML mapping into typed tier policies."""
    root = _require_mapping(value, "tier policies root")
    _require_exact_keys(root, ("tiers",), "tier policies root")
    tiers = _require_mapping(root["tiers"], "tiers")
    _require_exact_keys(tiers, tuple(tier.value for tier in TierName), "tiers")

    policies: dict[TierName, TierPolicy] = {}
    for tier in TierName:
        tier_payload = _require_mapping(tiers[tier.value], f"tiers.{tier.value}")
        _require_exact_keys(
            tier_payload,
            ("families", "max_allowed_leakage_severity"),
            f"tiers.{tier.value}",
        )
        family_payload = _require_mapping(
            tier_payload["families"],
            f"tiers.{tier.value}.families",
        )
        _require_exact_keys(
            family_payload,
            tuple(family.value for family in BindingQualityFamily),
            f"tiers.{tier.value}.families",
        )
        family_levels: dict[BindingQualityFamily, TierName] = {}
        for family in BindingQualityFamily:
            family_levels[family] = _parse_enum(
                TierName,
                family_payload[family.value],
                f"tiers.{tier.value}.families.{family.value}",
            )
        policies[tier] = TierPolicy(
            tier_name=tier,
            family_filter_levels=family_levels,
            max_allowed_leakage_severity=_parse_enum(
                TierLeakageSeverity,
                tier_payload["max_allowed_leakage_severity"],
                f"tiers.{tier.value}.max_allowed_leakage_severity",
            ),
        )
    return TierPoliciesConfig(policies_by_tier=policies)


def _parse_enum(enum_type: type[EnumT], value: object, label: str) -> EnumT:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ValueError(
            f"{label} must be one of {[member.value for member in enum_type]}, got {value!r}."
        ) from exc


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return value


def _require_exact_keys(
    value: Mapping[str, object],
    expected_keys: tuple[str, ...],
    label: str,
) -> None:
    expected = set(expected_keys)
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} must contain exactly {sorted(expected)} "
            f"(missing={sorted(expected - actual)}, unknown={sorted(actual - expected)})."
        )


__all__ = [
    "DEFAULT_TIER_POLICIES_CONFIG_PATH",
    "TierLeakageSeverity",
    "TierPoliciesConfig",
    "TierPolicy",
    "load_tier_policies_config",
    "parse_tier_policies_config_mapping",
]
