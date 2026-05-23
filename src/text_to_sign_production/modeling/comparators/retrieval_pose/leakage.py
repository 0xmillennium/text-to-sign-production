"""Leakage policy enforcement for retrieval-pose comparison."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.comparators.retrieval_pose.config import (
    RetrievalLeakagePolicyConfig,
)
from text_to_sign_production.modeling.comparators.retrieval_pose.errors import (
    RetrievalPoseComparatorError,
)


@dataclass(frozen=True, slots=True)
class RetrievalLeakageSummary:
    blocked_candidates: int
    blocked_by_reason: dict[str, int]
    queries_without_safe_candidate: int
    leakage_checked: bool = True


def assert_bank_query_split_policy(
    *,
    bank_split: SampleSplit | str,
    query_split: SampleSplit | str,
    policy: RetrievalLeakagePolicyConfig,
) -> None:
    resolved_bank = SampleSplit(bank_split)
    resolved_query = SampleSplit(query_split)
    if resolved_bank is resolved_query and not policy.allow_same_split:
        raise RetrievalPoseComparatorError(
            "retrieval leakage policy forbids using query split "
            f"{resolved_query.value!r} as retrieval bank split."
        )


def detect_retrieval_leakage_reasons(
    query: object,
    bank_item: object,
    policy: RetrievalLeakagePolicyConfig,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if SampleSplit(getattr(query, "split")) is SampleSplit(getattr(bank_item, "split")):
        if not policy.allow_same_split:
            reasons.append("same_split_forbidden")
    if policy.exclude_same_sample_id and getattr(query, "sample_id") == getattr(
        bank_item, "sample_id"
    ):
        reasons.append("same_sample_id")
    if policy.exclude_same_source_sentence_id and getattr(query, "source_sentence_id") == getattr(
        bank_item, "source_sentence_id"
    ):
        reasons.append("same_source_sentence_id")
    if policy.exclude_same_source_video_id and getattr(query, "source_video_id") == getattr(
        bank_item, "source_video_id"
    ):
        reasons.append("same_source_video_id")
    if policy.exclude_identical_text and _normalized_text(getattr(query, "text")) == _normalized_text(
        getattr(bank_item, "text")
    ):
        reasons.append("identical_text")
    return tuple(reasons)


def summarize_leakage(results: Iterable[object]) -> RetrievalLeakageSummary:
    blocked = 0
    reasons: Counter[str] = Counter()
    no_safe = 0
    for result in results:
        blocked += int(getattr(result, "blocked_count", 0))
        reasons.update(getattr(result, "blocked_reasons", ()))
        if not getattr(result, "success", False):
            no_safe += 1
    return RetrievalLeakageSummary(
        blocked_candidates=blocked,
        blocked_by_reason=dict(sorted(reasons.items())),
        queries_without_safe_candidate=no_safe,
    )


def _normalized_text(value: str) -> str:
    return " ".join(value.lower().split())


__all__ = [
    "RetrievalLeakageSummary",
    "assert_bank_query_split_policy",
    "detect_retrieval_leakage_reasons",
    "summarize_leakage",
]
