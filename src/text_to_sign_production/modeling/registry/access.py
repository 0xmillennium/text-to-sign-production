"""Registry key coercion helpers for modeling research specifications."""

from __future__ import annotations

import enum

from text_to_sign_production.modeling.research import (
    ComparatorKey,
    EvaluationProtocolKey,
    ModelKey,
    ObjectiveKey,
)


class ModelingRegistryError(KeyError):
    """Raised when a modeling registry key cannot be resolved."""


def coerce_model_key(value: ModelKey | str) -> ModelKey:
    """Return a model key from an enum instance or string value."""

    return _coerce_key(value, ModelKey, "model")


def coerce_objective_key(value: ObjectiveKey | str) -> ObjectiveKey:
    """Return an objective key from an enum instance or string value."""

    return _coerce_key(value, ObjectiveKey, "objective")


def coerce_comparator_key(value: ComparatorKey | str) -> ComparatorKey:
    """Return a comparator key from an enum instance or string value."""

    return _coerce_key(value, ComparatorKey, "comparator")


def coerce_evaluation_protocol_key(
    value: EvaluationProtocolKey | str,
) -> EvaluationProtocolKey:
    """Return an evaluation protocol key from an enum instance or string value."""

    return _coerce_key(value, EvaluationProtocolKey, "evaluation protocol")


def _coerce_key(
    value: enum.StrEnum | str,
    key_type: type[enum.StrEnum],
    label: str,
) -> enum.StrEnum:
    if isinstance(value, key_type):
        return value
    if isinstance(value, str):
        try:
            return key_type(value)
        except ValueError as exc:
            allowed = ", ".join(item.value for item in key_type)
            raise ModelingRegistryError(
                f"unknown {label} key {value!r}; expected one of: {allowed}"
            ) from exc
    raise ModelingRegistryError(f"{label} key must be a string or {key_type.__name__}.")


__all__ = [
    "ModelingRegistryError",
    "coerce_comparator_key",
    "coerce_evaluation_protocol_key",
    "coerce_model_key",
    "coerce_objective_key",
]
