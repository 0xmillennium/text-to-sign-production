"""Typed models for leakage detection."""

from __future__ import annotations

import enum
from dataclasses import dataclass

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.data._shared.types import ValidationIssue


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


LeakageValidationIssue = ValidationIssue


@dataclass(frozen=True, slots=True)
class LeakageRelationFrequencyRecord:
    """Frequency of a leakage relation across pair facts."""

    relation: LeakageRelation
    pair_count: int


@dataclass(frozen=True, slots=True)
class LeakageSeverityDistributionRecord:
    """Frequency of leakage severities across pair facts."""

    severity: LeakageSeverity
    pair_count: int


@dataclass(frozen=True, slots=True)
class LeakageSampleSeverityDistributionRecord:
    """Split-aware sample count by maximum leakage severity."""

    split: SampleSplit
    severity: LeakageSeverity
    sample_count: int


@dataclass(frozen=True, slots=True)
class LeakageRelationCooccurrenceRecord:
    """Ordered relation co-occurrence count across pair facts."""

    left_relation: LeakageRelation
    right_relation: LeakageRelation
    pair_count: int


@dataclass(frozen=True, slots=True)
class LeakageSplitPairCountRecord:
    """Cross-split pair count for leakage pair facts."""

    left_split: SampleSplit
    right_split: SampleSplit
    pair_count: int


@dataclass(frozen=True, slots=True)
class LeakageAffectedSampleCoverageRecord:
    """Affected-sample coverage by split."""

    split: SampleSplit
    sample_count: int
    affected_sample_count: int
    affected_sample_ratio: float
