"""Cosine scoring and leakage-aware selection for retrieval comparator."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.comparators.retrieval_pose.bank import (
    RetrievalBank,
    RetrievalBankItem,
)
from text_to_sign_production.modeling.comparators.retrieval_pose.config import (
    RetrievalLeakagePolicyConfig,
    RetrievalScoringConfig,
)
from text_to_sign_production.modeling.comparators.retrieval_pose.errors import (
    RetrievalPoseComparatorError,
)
from text_to_sign_production.modeling.comparators.retrieval_pose.leakage import (
    detect_retrieval_leakage_reasons,
)


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    sample_id: str
    split: SampleSplit
    text: str
    source_video_id: str
    source_sentence_id: str
    source_sentence_name: str
    reference_payload_ref: str
    embedding: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "split", SampleSplit(self.split))
        embedding = np.asarray(self.embedding, dtype=np.float32).copy()
        if embedding.ndim != 1 or embedding.shape[0] < 1:
            raise RetrievalPoseComparatorError("retrieval query embedding must be one-dimensional.")
        if not np.all(np.isfinite(embedding)):
            raise RetrievalPoseComparatorError("retrieval text embeddings contain non-finite values.")
        embedding.setflags(write=False)
        object.__setattr__(self, "embedding", embedding)


@dataclass(frozen=True, slots=True)
class RetrievalCandidateScore:
    rank: int
    retrieved_sample_id: str
    retrieval_source_split: SampleSplit
    score: float
    blocked_by_leakage: bool
    leakage_reasons: tuple[str, ...]
    source_sentence_name: str
    frame_count: int
    valid_frame_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "retrieval_source_split", SampleSplit(self.retrieval_source_split))


@dataclass(frozen=True, slots=True)
class RetrievalQueryResult:
    query_sample_id: str
    query_split: SampleSplit
    query_text: str
    source_video_id: str
    source_sentence_id: str
    source_sentence_name: str
    reference_payload_ref: str
    success: bool
    selected_candidate: RetrievalCandidateScore | None
    selected_bank_item: RetrievalBankItem | None
    top_k_safe_candidates: tuple[RetrievalCandidateScore, ...]
    blocked_count: int
    blocked_reasons: tuple[str, ...]
    failure_reason: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_split", SampleSplit(self.query_split))
        if self.success and (self.selected_candidate is None or self.selected_bank_item is None):
            raise RetrievalPoseComparatorError("successful retrieval result requires a selected candidate.")
        if not self.success and not self.failure_reason:
            raise RetrievalPoseComparatorError("failed retrieval result requires a failure_reason.")


def score_retrieval_query(
    *,
    query: RetrievalQuery,
    bank: RetrievalBank,
    scoring: RetrievalScoringConfig,
    leakage_policy: RetrievalLeakagePolicyConfig,
) -> RetrievalQueryResult:
    if query.embedding.shape[0] != bank.embedding_matrix.shape[1]:
        raise RetrievalPoseComparatorError("retrieval query and bank embedding dimensions differ.")
    scores = cosine_similarity_scores(query.embedding, bank.embedding_matrix, scoring=scoring)
    order = np.argsort(-scores, kind="stable")
    safe_pairs: list[tuple[RetrievalCandidateScore, RetrievalBankItem]] = []
    blocked_reasons: list[str] = []
    blocked_count = 0
    for ordered_rank, bank_index in enumerate(order, start=1):
        item = bank.items[int(bank_index)]
        reasons = detect_retrieval_leakage_reasons(query, item, leakage_policy)
        blocked = bool(reasons)
        if blocked:
            blocked_count += 1
            blocked_reasons.extend(reasons)
        candidate = RetrievalCandidateScore(
            rank=ordered_rank,
            retrieved_sample_id=item.sample_id,
            retrieval_source_split=item.split,
            score=float(scores[int(bank_index)]),
            blocked_by_leakage=blocked,
            leakage_reasons=reasons,
            source_sentence_name=item.source_sentence_name,
            frame_count=item.frame_count,
            valid_frame_count=item.valid_frame_count,
        )
        if not blocked:
            safe_pairs.append((candidate, item))
    top_k = tuple(candidate for candidate, _ in safe_pairs[: scoring.top_k])
    if not safe_pairs:
        reason = f"no leakage-safe retrieval candidate remained for query sample_id={query.sample_id!r}."
        return RetrievalQueryResult(
            query_sample_id=query.sample_id,
            query_split=query.split,
            query_text=query.text,
            source_video_id=query.source_video_id,
            source_sentence_id=query.source_sentence_id,
            source_sentence_name=query.source_sentence_name,
            reference_payload_ref=query.reference_payload_ref,
            success=False,
            selected_candidate=None,
            selected_bank_item=None,
            top_k_safe_candidates=(),
            blocked_count=blocked_count,
            blocked_reasons=tuple(sorted(Counter(blocked_reasons).elements())),
            failure_reason=reason,
        )
    selected, item = safe_pairs[0]
    return RetrievalQueryResult(
        query_sample_id=query.sample_id,
        query_split=query.split,
        query_text=query.text,
        source_video_id=query.source_video_id,
        source_sentence_id=query.source_sentence_id,
        source_sentence_name=query.source_sentence_name,
        reference_payload_ref=query.reference_payload_ref,
        success=True,
        selected_candidate=selected,
        selected_bank_item=item,
        top_k_safe_candidates=top_k,
        blocked_count=blocked_count,
        blocked_reasons=tuple(sorted(Counter(blocked_reasons).elements())),
        failure_reason=None,
    )


def cosine_similarity_scores(
    query_embedding: np.ndarray,
    bank_matrix: np.ndarray,
    *,
    scoring: RetrievalScoringConfig,
) -> np.ndarray:
    if scoring.score_metric != "cosine_similarity":
        raise RetrievalPoseComparatorError("retrieval.score_metric must be 'cosine_similarity'.")
    query = np.asarray(query_embedding, dtype=np.float32)
    bank = np.asarray(bank_matrix, dtype=np.float32)
    if not np.all(np.isfinite(query)) or not np.all(np.isfinite(bank)):
        raise RetrievalPoseComparatorError("retrieval text embeddings contain non-finite values.")
    query_norm = float(np.linalg.norm(query))
    bank_norms = np.linalg.norm(bank, axis=1)
    if query_norm <= 0.0 or np.any(bank_norms <= 0.0):
        raise RetrievalPoseComparatorError(
            "retrieval cosine similarity cannot use zero-norm embeddings."
        )
    if scoring.normalize_embeddings:
        return (bank / bank_norms[:, None]) @ (query / query_norm)
    return bank @ query / (bank_norms * query_norm)


__all__ = [
    "RetrievalCandidateScore",
    "RetrievalQuery",
    "RetrievalQueryResult",
    "cosine_similarity_scores",
    "score_retrieval_query",
]
