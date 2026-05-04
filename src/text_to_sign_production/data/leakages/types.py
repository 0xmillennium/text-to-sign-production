"""Typed models for leakage detection."""

from __future__ import annotations

import enum
from dataclasses import dataclass


class LeakageRelation(enum.StrEnum):
    """Deterministic relations between samples."""

    SAME_SOURCE_SENTENCE = "same_source_sentence"
    EXACT_NORMALIZED_TEXT = "exact_normalized_text"
    SAME_SOURCE_VIDEO = "same_source_video"


class LeakageSeverity(enum.StrEnum):
    """Deterministic severity of leakage."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class LeakageInput:
    """The only input surface for leakage detection."""

    sample_id: str
    split: str
    source_video_id: str
    source_sentence_id: str
    source_sentence_name: str
    normalized_text: str


@dataclass(frozen=True, slots=True)
class LeakagePairFact:
    """A single leakage relationship between two cross-split samples."""

    left_sample_id: str
    right_sample_id: str
    left_split: str
    right_split: str
    relations: tuple[LeakageRelation, ...]
    severity: LeakageSeverity


@dataclass(frozen=True, slots=True)
class LeakageSampleSummary:
    """Summary of all cross-split leakages involving a specific sample."""

    sample_id: str
    split: str
    has_leakage: bool
    max_severity: LeakageSeverity
    same_source_sentence_match_count: int
    exact_normalized_text_match_count: int
    same_source_video_match_count: int
    matched_sample_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LeakageBundle:
    """The fully composed deterministic leakage facts for a set of samples."""

    pair_facts: tuple[LeakagePairFact, ...]
    sample_summaries: tuple[LeakageSampleSummary, ...]
