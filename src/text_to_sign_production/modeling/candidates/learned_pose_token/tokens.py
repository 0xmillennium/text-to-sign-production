"""Pose token sequence and manifest contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.candidates.learned_pose_token.errors import (
    LearnedPoseTokenError,
)
from text_to_sign_production.modeling.data import TemporalWindowSpec, temporal_window_starts

POSE_TOKEN_SCHEMA_VERSION = "t2sp-learned-pose-token-v1"
POSE_TOKEN_MANIFEST_SCHEMA_VERSION = "t2sp-learned-pose-token-manifest-v1"


@dataclass(frozen=True, slots=True)
class PoseTokenSequence:
    schema_version: str
    sample_id: str
    source_sentence_name: str
    split: SampleSplit
    token_ids: np.ndarray
    frame_count: int
    token_count: int
    codebook_size: int
    temporal_granularity: str
    window_size: int
    stride: int

    def __post_init__(self) -> None:
        if self.schema_version != POSE_TOKEN_SCHEMA_VERSION:
            raise LearnedPoseTokenError("pose token sequence schema_version is unsupported.")
        _require_text(self.sample_id, "sample_id")
        _require_text(self.source_sentence_name, "source_sentence_name")
        object.__setattr__(self, "split", SampleSplit(self.split))
        _require_positive_int(self.frame_count, "frame_count")
        _require_positive_int(self.token_count, "token_count")
        if not isinstance(self.codebook_size, int) or isinstance(self.codebook_size, bool) or self.codebook_size <= 1:
            raise LearnedPoseTokenError("codebook_size must be greater than 1.")
        _validate_temporal_contract(
            frame_count=self.frame_count,
            token_count=self.token_count,
            temporal_granularity=self.temporal_granularity,
            window_size=self.window_size,
            stride=self.stride,
        )
        raw = np.asarray(self.token_ids)
        if raw.shape != (self.token_count,):
            raise LearnedPoseTokenError(
                f"token_ids must have shape ({self.token_count},); got {raw.shape}."
            )
        if raw.dtype.kind not in {"i", "u"}:
            raise LearnedPoseTokenError("token_ids must be an integer array.")
        ids = np.asarray(raw, dtype=np.int64).copy()
        if ids.size == 0:
            raise LearnedPoseTokenError("token_ids must be non-empty.")
        if np.any((ids < 0) | (ids >= self.codebook_size)):
            raise LearnedPoseTokenError(
                f"token_ids must be in range [0, {self.codebook_size})."
            )
        ids.setflags(write=False)
        object.__setattr__(self, "token_ids", ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "source_sentence_name": self.source_sentence_name,
            "split": self.split.value,
            "token_ids": self.token_ids.tolist(),
            "frame_count": self.frame_count,
            "token_count": self.token_count,
            "codebook_size": self.codebook_size,
            "temporal_granularity": self.temporal_granularity,
            "window_size": self.window_size,
            "stride": self.stride,
        }


@dataclass(frozen=True, slots=True)
class PoseTokenManifestEntry:
    schema_version: str
    sample_id: str
    source_sentence_name: str
    split: SampleSplit
    token_path: Path
    token_count: int
    frame_count: int
    codebook_size: int
    temporal_granularity: str
    window_size: int
    stride: int
    issues: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != POSE_TOKEN_MANIFEST_SCHEMA_VERSION:
            raise LearnedPoseTokenError("pose token manifest schema_version is unsupported.")
        _require_text(self.sample_id, "sample_id")
        _require_text(self.source_sentence_name, "source_sentence_name")
        object.__setattr__(self, "split", SampleSplit(self.split))
        object.__setattr__(self, "token_path", Path(self.token_path))
        if not str(self.token_path).strip():
            raise LearnedPoseTokenError("token_path must be non-empty.")
        _require_positive_int(self.token_count, "token_count")
        _require_positive_int(self.frame_count, "frame_count")
        if not isinstance(self.codebook_size, int) or isinstance(self.codebook_size, bool) or self.codebook_size <= 1:
            raise LearnedPoseTokenError("codebook_size must be greater than 1.")
        _validate_temporal_contract(
            frame_count=self.frame_count,
            token_count=self.token_count,
            temporal_granularity=self.temporal_granularity,
            window_size=self.window_size,
            stride=self.stride,
        )
        issues = tuple(self.issues)
        for issue in issues:
            _require_text(issue, "issue")
        object.__setattr__(self, "issues", issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "source_sentence_name": self.source_sentence_name,
            "split": self.split.value,
            "token_path": str(self.token_path),
            "token_count": self.token_count,
            "frame_count": self.frame_count,
            "codebook_size": self.codebook_size,
            "temporal_granularity": self.temporal_granularity,
            "window_size": self.window_size,
            "stride": self.stride,
            "issues": list(self.issues),
        }


def pose_token_temporal_spec(
    temporal_granularity: str,
    window_size: int,
    stride: int,
) -> TemporalWindowSpec:
    if temporal_granularity not in {"frame", "window"}:
        raise LearnedPoseTokenError(
            "tokenizer.temporal_granularity must be one of {'frame', 'window'}."
        )
    _require_positive_int(window_size, "window_size")
    _require_positive_int(stride, "stride")
    if temporal_granularity == "frame":
        if window_size != 1 or stride != 1:
            raise LearnedPoseTokenError(
                "frame tokenizer requires window_size=1 and stride=1."
            )
        return TemporalWindowSpec.frame()
    if window_size <= 1 or stride < 1:
        raise LearnedPoseTokenError(
            "window tokenizer requires window_size > 1 and stride >= 1."
        )
    try:
        return TemporalWindowSpec.window(window_size=window_size, stride=stride)
    except Exception as exc:
        raise LearnedPoseTokenError(str(exc)) from exc


def expected_token_count_for_frame_count(
    frame_count: int,
    temporal_granularity: str,
    window_size: int,
    stride: int,
) -> int:
    _require_positive_int(frame_count, "frame_count")
    spec = pose_token_temporal_spec(temporal_granularity, window_size, stride)
    return len(temporal_window_starts(frame_count=frame_count, spec=spec))


def generated_frame_count_from_token_count(
    token_count: int,
    temporal_granularity: str,
    window_size: int,
    stride: int,
) -> int:
    _require_positive_int(token_count, "token_count")
    pose_token_temporal_spec(temporal_granularity, window_size, stride)
    if temporal_granularity == "frame":
        return token_count
    if token_count == 1:
        return window_size
    return (token_count - 1) * stride + window_size


def _validate_temporal_contract(
    *,
    frame_count: int,
    token_count: int,
    temporal_granularity: str,
    window_size: int,
    stride: int,
) -> None:
    expected = expected_token_count_for_frame_count(
        frame_count,
        temporal_granularity,
        window_size,
        stride,
    )
    if temporal_granularity == "frame" and token_count != frame_count:
        raise LearnedPoseTokenError("frame token_count must equal frame_count.")
    if temporal_granularity == "window" and token_count != expected:
        raise LearnedPoseTokenError(
            "window token_count must equal len(temporal_window_starts(frame_count, spec))."
        )
    if token_count != expected:
        raise LearnedPoseTokenError("token_count does not match temporal window spec.")


def _require_text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise LearnedPoseTokenError(f"{name} must be non-empty.")


def _require_positive_int(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise LearnedPoseTokenError(f"{name} must be a positive integer.")


__all__ = [
    "POSE_TOKEN_MANIFEST_SCHEMA_VERSION",
    "POSE_TOKEN_SCHEMA_VERSION",
    "PoseTokenManifestEntry",
    "PoseTokenSequence",
    "expected_token_count_for_frame_count",
    "generated_frame_count_from_token_count",
    "pose_token_temporal_spec",
]
