"""Honest interpretation limits for automatic validation metrics."""

from __future__ import annotations

from collections.abc import Iterable

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.validation.errors import ModelValidationError
from text_to_sign_production.modeling.validation.records import (
    VALIDATION_SCHEMA_VERSION,
    ValidationLimitations,
    ValidationPairingEntry,
)

_NOTES = (
    "Automatic pose/keypoint metrics do not prove sign intelligibility.",
    "Validation split results are for model development/checkpoint selection.",
    "These results must not be interpreted as final test performance.",
    "Metrics use simple prefix alignment by minimum sequence length; temporal misalignment is not fully corrected.",
)


def build_validation_limitations(
    *,
    split: SampleSplit,
    pairing_entries: Iterable[ValidationPairingEntry],
) -> ValidationLimitations:
    """Build required claims and pairing availability counts."""

    resolved_split = SampleSplit(split)
    if resolved_split is not SampleSplit.VAL:
        raise ModelValidationError("validation limitations must use split='val'.")
    entries = tuple(pairing_entries)
    if any(entry.key.split is not resolved_split for entry in entries):
        raise ModelValidationError("validation pairing entry split does not match limitations split.")
    return ValidationLimitations(
        VALIDATION_SCHEMA_VERSION,
        resolved_split,
        True,
        True,
        True,
        sum(entry.status == "missing_generated" for entry in entries),
        sum(entry.status == "missing_reference" for entry in entries),
        sum(entry.status == "paired" for entry in entries),
        sum(entry.status == "generated_failed" for entry in entries),
        sum(entry.status == "identity_mismatch" for entry in entries),
        _NOTES,
    )


__all__ = ["build_validation_limitations"]
