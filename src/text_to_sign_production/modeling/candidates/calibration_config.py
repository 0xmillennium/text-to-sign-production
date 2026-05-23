"""Strict config lookup helpers for provider-owned calibration specs."""

from __future__ import annotations

from collections.abc import Mapping

from text_to_sign_production.workflows.model.contracts.config import (
    ModelWorkflowInvariantError,
)


def require_config_path(config: Mapping[str, object], path: str) -> object:
    """Return a dot-path config value or fail without fallback."""

    current: object = config
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ModelWorkflowInvariantError(
                f"provider-real calibration config missing required path: {path}"
            )
        current = current[part]
    if current is None:
        raise ModelWorkflowInvariantError(
            f"provider-real calibration config required path is null: {path}"
        )
    return current


def optional_config_path(config: Mapping[str, object], path: str) -> object | None:
    """Return a dot-path config value when present, otherwise None."""

    current: object = config
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def require_positive_int_config_path(config: Mapping[str, object], path: str) -> int:
    """Return a positive integer config value or fail without fallback."""

    value = require_config_path(config, path)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ModelWorkflowInvariantError(
            f"provider-real calibration config path must be a positive integer: {path}"
        )
    return value


def require_text_config_path(config: Mapping[str, object], path: str) -> str:
    """Return a non-empty text config value or fail without fallback."""

    value = require_config_path(config, path)
    if not isinstance(value, str) or not value.strip():
        raise ModelWorkflowInvariantError(
            f"provider-real calibration config path must be non-empty text: {path}"
        )
    return value


__all__ = [
    "optional_config_path",
    "require_config_path",
    "require_positive_int_config_path",
    "require_text_config_path",
]
