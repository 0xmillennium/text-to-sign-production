"""Codebook metadata and usage-stability reports."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.candidates.learned_pose_token.errors import (
    LearnedPoseTokenError,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.tokens import (
    PoseTokenSequence,
)

CODEBOOK_METADATA_SCHEMA_VERSION = "t2sp-pose-codebook-v1"
CODEBOOK_STABILITY_SCHEMA_VERSION = "t2sp-pose-codebook-stability-v1"


@dataclass(frozen=True, slots=True)
class PoseCodebookMetadata:
    schema_version: str
    codebook_size: int
    embedding_dim: int
    initialization: str
    checkpoint_path: Path | None
    created_at_utc: str
    training_sample_count: int
    training_frame_count: int
    notes: tuple[str, ...]
    training_token_unit_count: int | None = None
    temporal_granularity: str | None = None
    window_size: int | None = None
    stride: int | None = None

    def __post_init__(self) -> None:
        if self.schema_version != CODEBOOK_METADATA_SCHEMA_VERSION:
            raise LearnedPoseTokenError("codebook metadata schema_version is unsupported.")
        if not isinstance(self.codebook_size, int) or isinstance(self.codebook_size, bool) or self.codebook_size <= 1:
            raise LearnedPoseTokenError("codebook_size must be greater than 1.")
        _require_positive_int(self.embedding_dim, "embedding_dim")
        _require_text(self.initialization, "initialization")
        if self.checkpoint_path is not None:
            object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))
        _require_text(self.created_at_utc, "created_at_utc")
        _require_non_negative_int(self.training_sample_count, "training_sample_count")
        _require_non_negative_int(self.training_frame_count, "training_frame_count")
        if self.training_token_unit_count is not None:
            _require_non_negative_int(
                self.training_token_unit_count,
                "training_token_unit_count",
            )
        if self.temporal_granularity is not None:
            if self.temporal_granularity not in {"frame", "window"}:
                raise LearnedPoseTokenError(
                    "temporal_granularity must be one of {'frame', 'window'}."
                )
        if self.window_size is not None:
            _require_positive_int(self.window_size, "window_size")
        if self.stride is not None:
            _require_positive_int(self.stride, "stride")
        object.__setattr__(self, "notes", _notes(self.notes))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "codebook_size": self.codebook_size,
            "embedding_dim": self.embedding_dim,
            "initialization": self.initialization,
            "checkpoint_path": None if self.checkpoint_path is None else str(self.checkpoint_path),
            "created_at_utc": self.created_at_utc,
            "training_sample_count": self.training_sample_count,
            "training_frame_count": self.training_frame_count,
            "training_token_unit_count": self.training_token_unit_count,
            "temporal_granularity": self.temporal_granularity,
            "window_size": self.window_size,
            "stride": self.stride,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class PoseCodebookStabilityReport:
    schema_version: str
    split: SampleSplit
    codebook_size: int
    used_code_count: int
    dead_code_count: int
    usage_entropy: float
    perplexity: float
    max_usage_fraction: float
    collapse_detected: bool
    dead_code_indices: tuple[int, ...]
    notes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != CODEBOOK_STABILITY_SCHEMA_VERSION:
            raise LearnedPoseTokenError("codebook stability schema_version is unsupported.")
        object.__setattr__(self, "split", SampleSplit(self.split))
        if not isinstance(self.codebook_size, int) or isinstance(self.codebook_size, bool) or self.codebook_size <= 1:
            raise LearnedPoseTokenError("codebook_size must be greater than 1.")
        _require_non_negative_int(self.used_code_count, "used_code_count")
        _require_non_negative_int(self.dead_code_count, "dead_code_count")
        for value, name in (
            (self.usage_entropy, "usage_entropy"),
            (self.perplexity, "perplexity"),
            (self.max_usage_fraction, "max_usage_fraction"),
        ):
            if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(float(value)):
                raise LearnedPoseTokenError(f"{name} must be finite.")
        if not isinstance(self.collapse_detected, bool):
            raise LearnedPoseTokenError("collapse_detected must be a boolean.")
        dead = tuple(self.dead_code_indices)
        if any(not isinstance(index, int) or isinstance(index, bool) for index in dead):
            raise LearnedPoseTokenError("dead_code_indices must be integers.")
        if any(index < 0 or index >= self.codebook_size for index in dead):
            raise LearnedPoseTokenError("dead_code_indices must be within the codebook range.")
        object.__setattr__(self, "dead_code_indices", dead)
        object.__setattr__(self, "notes", _notes(self.notes))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "split": self.split.value,
            "codebook_size": self.codebook_size,
            "used_code_count": self.used_code_count,
            "dead_code_count": self.dead_code_count,
            "usage_entropy": self.usage_entropy,
            "perplexity": self.perplexity,
            "max_usage_fraction": self.max_usage_fraction,
            "collapse_detected": self.collapse_detected,
            "dead_code_indices": list(self.dead_code_indices),
            "notes": list(self.notes),
        }


def compute_codebook_stability(
    token_sequences: Iterable[PoseTokenSequence],
    *,
    codebook_size: int,
    dead_code_threshold: int,
    collapse_perplexity_threshold: float,
    split: SampleSplit,
) -> PoseCodebookStabilityReport:
    """Compute deterministic token usage stability from materialized token sequences."""

    if not isinstance(codebook_size, int) or isinstance(codebook_size, bool) or codebook_size <= 1:
        raise LearnedPoseTokenError("codebook_size must be greater than 1.")
    _require_non_negative_int(dead_code_threshold, "dead_code_threshold")
    if not isinstance(collapse_perplexity_threshold, int | float) or isinstance(
        collapse_perplexity_threshold,
        bool,
    ) or float(collapse_perplexity_threshold) <= 0.0:
        raise LearnedPoseTokenError("collapse_perplexity_threshold must be positive.")
    sequences = tuple(token_sequences)
    if not sequences:
        raise LearnedPoseTokenError("codebook stability requires at least one token sequence.")
    counts = np.zeros((codebook_size,), dtype=np.int64)
    for sequence in sequences:
        if not isinstance(sequence, PoseTokenSequence):
            raise LearnedPoseTokenError("token_sequences must contain PoseTokenSequence values.")
        if sequence.codebook_size != codebook_size:
            raise LearnedPoseTokenError("token sequence codebook_size does not match report.")
        if sequence.split is not SampleSplit(split):
            raise LearnedPoseTokenError("token sequence split does not match stability split.")
        ids = np.asarray(sequence.token_ids, dtype=np.int64)
        if np.any((ids < 0) | (ids >= codebook_size)):
            raise LearnedPoseTokenError("token sequence contains token ids outside the codebook.")
        counts += np.bincount(ids, minlength=codebook_size)
    total = int(counts.sum())
    if total <= 0:
        raise LearnedPoseTokenError("codebook stability cannot be computed from empty tokens.")
    used = counts > 0
    probabilities = counts[used].astype(np.float64) / float(total)
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    perplexity = float(math.exp(entropy))
    max_usage_fraction = float(counts.max() / total)
    dead_indices = tuple(int(index) for index in np.flatnonzero(counts <= dead_code_threshold))
    collapse_detected = (
        perplexity < float(collapse_perplexity_threshold)
        or max_usage_fraction >= 0.90
    )
    notes = (
        "Dead codes are counted with usage <= dead_code_threshold.",
        "Codebook collapse is flagged by low perplexity or one code using at least 90 percent of tokens.",
        "Codebook usage stability does not prove sign intelligibility.",
    )
    if collapse_detected:
        notes = notes + ("Potential token collapse detected; inspect training data and codebook usage.",)
    return PoseCodebookStabilityReport(
        schema_version=CODEBOOK_STABILITY_SCHEMA_VERSION,
        split=SampleSplit(split),
        codebook_size=codebook_size,
        used_code_count=int(np.count_nonzero(used)),
        dead_code_count=len(dead_indices),
        usage_entropy=entropy,
        perplexity=perplexity,
        max_usage_fraction=max_usage_fraction,
        collapse_detected=collapse_detected,
        dead_code_indices=dead_indices,
        notes=notes,
    )


def write_codebook_json(path: Path, payload: PoseCodebookMetadata | PoseCodebookStabilityReport) -> None:
    """Write a deterministic UTF-8 JSON codebook metadata/report artifact."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n",
        encoding="utf-8",
    )


def _notes(values: tuple[str, ...]) -> tuple[str, ...]:
    notes = tuple(values)
    for note in notes:
        _require_text(note, "note")
    return notes


def _require_text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise LearnedPoseTokenError(f"{name} must be non-empty.")


def _require_positive_int(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise LearnedPoseTokenError(f"{name} must be a positive integer.")


def _require_non_negative_int(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise LearnedPoseTokenError(f"{name} must be a non-negative integer.")


__all__ = [
    "CODEBOOK_METADATA_SCHEMA_VERSION",
    "CODEBOOK_STABILITY_SCHEMA_VERSION",
    "PoseCodebookMetadata",
    "PoseCodebookStabilityReport",
    "compute_codebook_stability",
    "write_codebook_json",
]
