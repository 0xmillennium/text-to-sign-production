"""Typed models for leakage detection."""

from __future__ import annotations

import enum
from dataclasses import dataclass

from text_to_sign_production.data._shared.identities import SampleSplit


class LeakageRelation(enum.StrEnum):
    """Deterministic relations between samples."""

    SAME_SOURCE_SENTENCE = "same_source_sentence"
    EXACT_NORMALIZED_TEXT = "exact_normalized_text"
    SAME_SOURCE_VIDEO = "same_source_video"


@dataclass(frozen=True, slots=True)
class LeakageRelationSpec:
    """Central exact-match definition for one leakage relation."""

    relation: LeakageRelation
    input_field: str


LEAKAGE_RELATION_SPECS: tuple[LeakageRelationSpec, ...] = (
    LeakageRelationSpec(
        relation=LeakageRelation.SAME_SOURCE_SENTENCE,
        input_field="source_sentence_id",
    ),
    LeakageRelationSpec(
        relation=LeakageRelation.EXACT_NORMALIZED_TEXT,
        input_field="normalized_text",
    ),
    LeakageRelationSpec(
        relation=LeakageRelation.SAME_SOURCE_VIDEO,
        input_field="source_video_id",
    ),
)
LEAKAGE_RELATION_ORDER: tuple[LeakageRelation, ...] = tuple(
    spec.relation for spec in LEAKAGE_RELATION_SPECS
)


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
    split: SampleSplit
    source_video_id: str
    source_sentence_id: str
    source_sentence_name: str
    normalized_text: str


@dataclass(frozen=True, slots=True)
class LeakagePairFact:
    """A single leakage relationship between two cross-split samples."""

    left_sample_id: str
    right_sample_id: str
    left_split: SampleSplit
    right_split: SampleSplit
    relations: tuple[LeakageRelation, ...]
    severity: LeakageSeverity


@dataclass(frozen=True, slots=True)
class LeakageSampleRef:
    """A split-aware reference to a matched sample."""

    split: SampleSplit
    sample_id: str


@dataclass(frozen=True, slots=True)
class LeakageSampleSummary:
    """Summary of all cross-split leakages involving a specific sample."""

    sample_id: str
    split: SampleSplit
    has_leakage: bool
    max_severity: LeakageSeverity
    same_source_sentence_match_count: int
    exact_normalized_text_match_count: int
    same_source_video_match_count: int
    matched_samples: tuple[LeakageSampleRef, ...]


@dataclass(frozen=True, slots=True)
class LeakageBundle:
    """The fully composed deterministic leakage facts for a set of samples."""

    pair_facts: tuple[LeakagePairFact, ...]
    sample_summaries: tuple[LeakageSampleSummary, ...]


@dataclass(frozen=True, slots=True)
class LeakageValidationIssue:
    """A specific issue found during leakage bundle validation."""

    code: str
    message: str
