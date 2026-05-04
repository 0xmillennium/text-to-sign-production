"""Strict loading and validation for configs/data/tiers.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml

from text_to_sign_production.data.leakages.types import LeakageSeverity
from text_to_sign_production.data.tiers._shared.parsing import (
    parse_enum_value,
    require_exact_keys,
    require_mapping,
)
from text_to_sign_production.data.tiers.types import (
    BindingTierFamily,
    FilterLevel,
    TierName,
    TierPolicy,
)

_TIER_KEYS = tuple(tier.value for tier in TierName)
_FAMILY_KEYS = tuple(family.value for family in BindingTierFamily)


def load_tier_policies(path: str | Path) -> tuple[TierPolicy, ...]:
    """Load tiers.yaml into strict typed tier policies."""
    config_path = Path(path)
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid tiers YAML: {exc}") from exc
    return parse_tier_policies(loaded)


def parse_tier_policies(payload: object) -> tuple[TierPolicy, ...]:
    """Parse a loaded YAML object into strict typed tier policies."""
    root = require_mapping(payload, "tiers root")
    require_exact_keys(root, ("tiers",), "tiers root")

    tiers = require_mapping(root["tiers"], "tiers")
    require_exact_keys(tiers, _TIER_KEYS, "tiers")

    parsed: list[TierPolicy] = []
    for tier_name in _TIER_KEYS:
        tier_payload = require_mapping(tiers[tier_name], f"tiers.{tier_name}")
        require_exact_keys(
            tier_payload,
            ("families", "max_allowed_leakage_severity"),
            f"tiers.{tier_name}",
        )

        family_payload = require_mapping(tier_payload["families"], f"tiers.{tier_name}.families")
        require_exact_keys(family_payload, _FAMILY_KEYS, f"tiers.{tier_name}.families")

        family_levels: dict[BindingTierFamily, FilterLevel] = {}
        for family in BindingTierFamily:
            raw_level = family_payload[family.value]
            if not isinstance(raw_level, str):
                raise ValueError(f"tiers.{tier_name}.families.{family.value} must be a string")
            try:
                family_levels[family] = parse_enum_value(
                    FilterLevel,
                    raw_level,
                    f"tiers.{tier_name}.families.{family.value}",
                )
            except ValueError as exc:
                raise ValueError(
                    f"tiers.{tier_name}.families.{family.value} must be one of "
                    f"{[level.value for level in FilterLevel]}, got {raw_level!r}"
                ) from exc

        severity = parse_enum_value(
            LeakageSeverity,
            tier_payload["max_allowed_leakage_severity"],
            f"tiers.{tier_name}.max_allowed_leakage_severity",
        )
        parsed.append(
            TierPolicy(
                tier_name=TierName(tier_name),
                family_levels=family_levels,
                max_allowed_leakage_severity=severity,
            )
        )

    return tuple(parsed)
