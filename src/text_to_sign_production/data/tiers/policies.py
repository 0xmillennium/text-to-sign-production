"""Strict loading and validation for configs/data/tiers.yaml."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import yaml

from text_to_sign_production.data.leakages.types import LeakageSeverity
from text_to_sign_production.data.tiers.types import FilterLevel, MetricFamily, TierPolicy

_TIER_KEYS = ("loose", "clean", "tight")
_FAMILY_KEYS = tuple(family.value for family in MetricFamily)


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
    root = _require_mapping(payload, "tiers root")
    _require_exact_keys(root, ("tiers",), "tiers root")

    tiers = _require_mapping(root["tiers"], "tiers")
    _require_exact_keys(tiers, _TIER_KEYS, "tiers")

    parsed: list[TierPolicy] = []
    for tier_name in _TIER_KEYS:
        tier_payload = _require_mapping(tiers[tier_name], f"tiers.{tier_name}")
        _require_exact_keys(
            tier_payload,
            ("families", "max_allowed_leakage_severity"),
            f"tiers.{tier_name}",
        )

        family_payload = _require_mapping(tier_payload["families"], f"tiers.{tier_name}.families")
        _require_exact_keys(family_payload, _FAMILY_KEYS, f"tiers.{tier_name}.families")

        family_levels: dict[str, FilterLevel] = {}
        for family in _FAMILY_KEYS:
            raw_level = family_payload[family]
            if not isinstance(raw_level, str):
                raise ValueError(f"tiers.{tier_name}.families.{family} must be a string")
            try:
                family_levels[family] = FilterLevel(raw_level)
            except ValueError as exc:
                raise ValueError(
                    f"tiers.{tier_name}.families.{family} must be one of "
                    f"{[level.value for level in FilterLevel]}, got {raw_level!r}"
                ) from exc

        severity = _parse_leakage_severity(
            tier_payload["max_allowed_leakage_severity"],
            f"tiers.{tier_name}.max_allowed_leakage_severity",
        )
        parsed.append(
            TierPolicy(
                tier_name=tier_name,
                family_levels=family_levels,
                max_allowed_leakage_severity=severity,
            )
        )

    return tuple(parsed)


def _parse_leakage_severity(value: object, name: str) -> LeakageSeverity:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    try:
        return LeakageSeverity[value]
    except KeyError as exc:
        raise ValueError(
            f"{name} must be one of {list(LeakageSeverity.__members__)}, got {value!r}"
        ) from exc


def _require_mapping(payload: object, name: str) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must be a mapping")
    for key in payload:
        if not isinstance(key, str):
            raise ValueError(f"{name} contains non-string key {key!r}")
    return cast(Mapping[str, object], payload)


def _require_exact_keys(
    payload: Mapping[str, object],
    expected_keys: tuple[str, ...],
    name: str,
) -> None:
    actual = set(payload)
    expected = set(expected_keys)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if unknown:
            details.append(f"unknown={unknown}")
        raise ValueError(
            f"{name} keys must be exactly {list(expected_keys)} ({', '.join(details)})"
        )
