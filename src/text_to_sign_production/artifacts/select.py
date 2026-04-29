"""Processed sample selection policy for generated artifacts."""

from __future__ import annotations

import random

from text_to_sign_production.artifacts.index import SampleIndex
from text_to_sign_production.artifacts.refs import (
    DuplicateSampleIdError,
    ManifestSampleRecord,
    SampleSelectionError,
)


def select_sample(
    index: SampleIndex,
    *,
    sample_id: str | None = None,
    random_selection: bool = False,
    seed: int | None = None,
) -> ManifestSampleRecord:
    """Select one sample by exact id or random choice with optional deterministic seeding."""

    modes = [sample_id is not None, random_selection]
    if sum(modes) != 1:
        raise SampleSelectionError("Exactly one selection mode is required: sample_id or random.")

    if sample_id is not None:
        normalized_sample_id = sample_id.strip()
        if not normalized_sample_id:
            raise SampleSelectionError("Sample id must not be blank.")
        matches = index.records_for_sample_id(normalized_sample_id)
        if not matches:
            splits = ", ".join(index.splits) or "<none>"
            raise SampleSelectionError(
                f"Sample id {normalized_sample_id!r} was not found in loaded processed "
                f"manifest splits: {splits}."
            )
        if len(matches) > 1:
            raise DuplicateSampleIdError(normalized_sample_id, matches)
        return matches[0]

    if index.duplicate_sample_ids:
        duplicate_id, candidates = next(iter(index.duplicate_sample_ids.items()))
        raise DuplicateSampleIdError(duplicate_id, candidates)
    return random.Random(seed).choice(list(index.records))


__all__ = ["select_sample"]
