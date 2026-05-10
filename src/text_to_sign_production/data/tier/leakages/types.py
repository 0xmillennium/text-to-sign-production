"""Type system for quality-domain cross-split leakage semantics."""

from __future__ import annotations

import enum
from dataclasses import dataclass

from text_to_sign_production.core.ids import SampleSplit


class LeakageRelation(enum.StrEnum):
    """Deterministic exact-match relations between accepted samples."""

    SAME_SOURCE_SENTENCE = "same_source_sentence"
    EXACT_NORMALIZED_TEXT = "exact_normalized_text"
    SAME_SOURCE_VIDEO = "same_source_video"


LEAKAGE_RELATION_ORDER: tuple[LeakageRelation, ...] = (
    LeakageRelation.SAME_SOURCE_SENTENCE,
    LeakageRelation.EXACT_NORMALIZED_TEXT,
    LeakageRelation.SAME_SOURCE_VIDEO,
)


class LeakageSeverity(enum.StrEnum):
    """Deterministic severity of a leakage relation set."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class LeakageInput:
    """Accepted checkpoint/sample authority for deterministic leakage detection."""

    sample_id: str
    split: SampleSplit
    source_video_id: str
    source_sentence_id: str
    source_sentence_name: str
    canonical_normalized_text: str


@dataclass(frozen=True, slots=True)
class LeakagePairFact:
    """One exact cross-split leakage relationship between two samples."""

    left_sample_id: str
    right_sample_id: str
    left_split: SampleSplit
    right_split: SampleSplit
    relations: tuple[LeakageRelation, ...]
    severity: LeakageSeverity

    def __post_init__(self) -> None:
        object.__setattr__(self, "relations", tuple(LeakageRelation(rel) for rel in self.relations))
        object.__setattr__(self, "severity", LeakageSeverity(self.severity))


@dataclass(frozen=True, slots=True)
class LeakageSampleRef:
    """A split-aware reference to another leakage-matched sample."""

    split: SampleSplit
    sample_id: str


@dataclass(frozen=True, slots=True)
class LeakageSampleSummary:
    """Per-sample summary of cross-split leakage involvement."""

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
    """Fully composed deterministic leakage facts for accepted samples."""

    pair_facts: tuple[LeakagePairFact, ...]
    sample_summaries: tuple[LeakageSampleSummary, ...]


class LeakageValidationIssueCode(enum.StrEnum):
    """Stable leakage-domain validation issue codes."""

    DUPLICATE_PAIR = "duplicate_pair"
    SELF_PAIR = "self_pair"
    SAME_SPLIT_PAIR = "same_split_pair"
    EMPTY_RELATIONS = "empty_relations"
    INVALID_RELATION_ORDER = "invalid_relation_order"
    NORMALIZED_TEXT_AUTHORITY_MISSING = "normalized_text_authority_missing"
    DUPLICATE_SAMPLE_SUMMARY = "duplicate_sample_summary"
    NEGATIVE_COUNT = "negative_count"
    MISSING_SAMPLE_SUMMARY_FOR_PAIR = "missing_sample_summary_for_pair"


@dataclass(frozen=True, slots=True)
class LeakageValidationIssue:
    """Structured leakage-domain validation issue."""

    code: LeakageValidationIssueCode
    message: str
    field_path: str | None = None


@dataclass(frozen=True, slots=True)
class LeakageRelationFrequencyRecord:
    """Frequency of one leakage relation across pair facts."""

    relation: LeakageRelation
    pair_count: int


@dataclass(frozen=True, slots=True)
class LeakageSeverityDistributionRecord:
    """Frequency of leakage severities across pair facts."""

    severity: LeakageSeverity
    pair_count: int


__all__ = [
    "LEAKAGE_RELATION_ORDER",
    "LeakageBundle",
    "LeakageInput",
    "LeakagePairFact",
    "LeakageRelation",
    "LeakageRelationFrequencyRecord",
    "LeakageSampleRef",
    "LeakageSampleSummary",
    "LeakageSeverity",
    "LeakageSeverityDistributionRecord",
    "LeakageValidationIssue",
    "LeakageValidationIssueCode",
]
