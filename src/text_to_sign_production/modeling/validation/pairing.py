"""Deterministic pairing of validation reference and generated manifests."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.core.models import PassedManifestEntry
from text_to_sign_production.modeling.artifacts import GeneratedPoseManifestEntry
from text_to_sign_production.modeling.validation.errors import ModelValidationError
from text_to_sign_production.modeling.validation.inputs import (
    ValidationGeneratedInput,
    ValidationReferenceInput,
)
from text_to_sign_production.modeling.validation.records import (
    VALIDATION_SCHEMA_VERSION,
    ValidationPairingEntry,
    ValidationPairingKey,
)


def build_validation_pairing(
    *,
    split: SampleSplit,
    reference_entries: Iterable[PassedManifestEntry | ValidationReferenceInput],
    generated_entries: Iterable[GeneratedPoseManifestEntry | ValidationGeneratedInput],
) -> tuple[ValidationPairingEntry, ...]:
    """Pair validation entries without discarding missing or failed generated rows."""

    resolved_split = SampleSplit(split)
    if resolved_split is not SampleSplit.VAL:
        raise ModelValidationError("validation pairing must use split='val'.")
    references: dict[ValidationPairingKey, ValidationReferenceInput] = {}
    generated: dict[ValidationPairingKey, ValidationGeneratedInput] = {}
    for value in reference_entries:
        entry = _reference_input(value)
        if entry.entry.split is not resolved_split:
            raise ModelValidationError(
                f"reference manifest contains non-validation split row: {entry.entry.sample_id!r}."
            )
        key = ValidationPairingKey(resolved_split, entry.entry.sample_id, 0)
        if key in references:
            raise ModelValidationError(f"duplicate validation reference pairing key: {key.to_dict()}.")
        references[key] = entry
    for value in generated_entries:
        entry = _generated_input(value)
        if entry.entry.split is not resolved_split:
            raise ModelValidationError(
                f"generated manifest contains non-validation split row: {entry.entry.sample_id!r}."
            )
        key = ValidationPairingKey(
            resolved_split,
            entry.entry.sample_id,
            entry.entry.generation_index,
        )
        if key in generated:
            raise ModelValidationError(f"duplicate validation generated pairing key: {key.to_dict()}.")
        generated[key] = entry
    pairs: list[ValidationPairingEntry] = []
    for key in sorted(set(references) | set(generated), key=lambda item: (item.sample_id, item.generation_index)):
        reference = references.get(key)
        candidate = generated.get(key)
        pairs.append(_pair_entry(key, reference, candidate))
    return tuple(pairs)


def _reference_input(
    value: PassedManifestEntry | ValidationReferenceInput,
) -> ValidationReferenceInput:
    if isinstance(value, ValidationReferenceInput):
        return value
    if not isinstance(value, PassedManifestEntry):
        raise ModelValidationError("reference entries must contain passed-manifest rows.")
    return ValidationReferenceInput(value, Path(value.payload_ref))


def _generated_input(
    value: GeneratedPoseManifestEntry | ValidationGeneratedInput,
) -> ValidationGeneratedInput:
    if isinstance(value, ValidationGeneratedInput):
        return value
    if not isinstance(value, GeneratedPoseManifestEntry):
        raise ModelValidationError("generated entries must contain generated-pose manifest rows.")
    path = None if value.generated_payload_ref is None else Path(value.generated_payload_ref)
    return ValidationGeneratedInput(value, path)


def _pair_entry(
    key: ValidationPairingKey,
    reference: ValidationReferenceInput | None,
    generated: ValidationGeneratedInput | None,
) -> ValidationPairingEntry:
    if reference is None and generated is not None:
        return ValidationPairingEntry(
            VALIDATION_SCHEMA_VERSION,
            key,
            "missing_reference",
            None,
            generated.entry.sample_id,
            None,
            generated.payload_path,
            ("generated validation output has no matching validation reference row.",),
        )
    if reference is not None and generated is None:
        return ValidationPairingEntry(
            VALIDATION_SCHEMA_VERSION,
            key,
            "missing_generated",
            reference.entry.sample_id,
            None,
            reference.payload_path,
            None,
            ("validation reference row has no matching generated output.",),
        )
    if reference is None or generated is None:
        raise ModelValidationError("validation pairing encountered an impossible empty key.")
    if not _identity_matches(reference.entry, generated.entry):
        return ValidationPairingEntry(
            VALIDATION_SCHEMA_VERSION,
            key,
            "identity_mismatch",
            reference.entry.sample_id,
            generated.entry.sample_id,
            reference.payload_path,
            generated.payload_path,
            ("generated and reference source identity fields do not match.",),
        )
    if generated.entry.failure_reason is not None or generated.payload_path is None:
        issue = generated.entry.failure_reason or "generated validation output has no payload."
        return ValidationPairingEntry(
            VALIDATION_SCHEMA_VERSION,
            key,
            "generated_failed",
            reference.entry.sample_id,
            generated.entry.sample_id,
            reference.payload_path,
            generated.payload_path,
            (issue,),
        )
    return ValidationPairingEntry(
        VALIDATION_SCHEMA_VERSION,
        key,
        "paired",
        reference.entry.sample_id,
        generated.entry.sample_id,
        reference.payload_path,
        generated.payload_path,
        (),
    )


def _identity_matches(reference: PassedManifestEntry, generated: GeneratedPoseManifestEntry) -> bool:
    return all(
        (
            reference.sample_id == generated.sample_id,
            reference.source_video_id == generated.source_video_id,
            reference.source_sentence_id == generated.source_sentence_id,
            reference.source_sentence_name == generated.source_sentence_name,
            reference.text == generated.text,
            reference.payload_ref == generated.reference_payload_ref,
        )
    )


__all__ = ["build_validation_pairing"]
