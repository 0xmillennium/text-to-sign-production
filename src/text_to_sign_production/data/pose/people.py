"""Selected-person logic for multi-person scenarios."""

from __future__ import annotations

import numpy as np

from text_to_sign_production.data.pose.schema import CANONICAL_POSE_CHANNELS
from text_to_sign_production.data.pose.types import (
    ParsedFrameResult,
    ParsedPerson,
    PersonSelectionCandidateScore,
    PersonSelectionPolicy,
    PersonSelectionResult,
)
from text_to_sign_production.data.samples.types import SelectedPersonMetadata

DEFAULT_PERSON_SELECTION_POLICY = PersonSelectionPolicy.HIGHEST_CANONICAL_SIGNAL


def resolve_target_person_index(
    frames: list[ParsedFrameResult],
    policy: PersonSelectionPolicy = DEFAULT_PERSON_SELECTION_POLICY,
) -> int:
    """Resolve which person index should be parsed from frames."""
    return resolve_person_selection(frames, policy).target_index


def resolve_person_selection(
    frames: list[ParsedFrameResult],
    policy: PersonSelectionPolicy = DEFAULT_PERSON_SELECTION_POLICY,
) -> PersonSelectionResult:
    """Resolve person selection with inspectable deterministic diagnostics."""
    if policy is PersonSelectionPolicy.OPENPOSE_PRIMARY:
        return PersonSelectionResult(
            target_index=0,
            policy=policy,
            fallback_used=False,
            fallback_reason=None,
            candidate_scores=_score_person_indices(frames),
        )
    if policy is PersonSelectionPolicy.HIGHEST_CANONICAL_SIGNAL:
        candidate_scores = _score_person_indices(frames)
        if not candidate_scores:
            return PersonSelectionResult(
                target_index=0,
                policy=policy,
                fallback_used=True,
                fallback_reason="no_people_detected",
                candidate_scores=(),
            )
        selected = min(
            candidate_scores,
            key=lambda score: (
                -score.aggregate_canonical_signal,
                -score.aggregate_body_signal,
                -score.valid_frame_presence_count,
                score.index,
            ),
        )
        return PersonSelectionResult(
            target_index=selected.index,
            policy=policy,
            fallback_used=False,
            fallback_reason=None,
            candidate_scores=candidate_scores,
        )
    raise ValueError(f"Unsupported person selection policy: {policy!r}")


def person_selection_policy_name(policy: PersonSelectionPolicy) -> str:
    """Return the stable serialized name for a person selection policy."""
    return policy.value


def build_person_metadata(
    *, target_index: int, multi_person_frame_count: int, max_people_per_frame: int
) -> SelectedPersonMetadata:
    """Build the final person selection facts."""
    return SelectedPersonMetadata(
        index=target_index,
        multi_person_frame_count=multi_person_frame_count,
        max_people_per_frame=max_people_per_frame,
    )


def _score_person_indices(
    frames: list[ParsedFrameResult],
) -> tuple[PersonSelectionCandidateScore, ...]:
    aggregate: dict[int, tuple[float, float, int]] = {}
    for frame in frames:
        for index, person in enumerate(frame.people):
            canonical_signal = _canonical_signal(person)
            body_signal = _body_signal(person)
            valid_presence = int(person.person_valid)
            prev_canonical, prev_body, prev_presence = aggregate.get(index, (0.0, 0.0, 0))
            aggregate[index] = (
                prev_canonical + canonical_signal,
                prev_body + body_signal,
                prev_presence + valid_presence,
            )

    return tuple(
        PersonSelectionCandidateScore(
            index=index,
            aggregate_canonical_signal=canonical_signal,
            aggregate_body_signal=body_signal,
            valid_frame_presence_count=valid_presence,
        )
        for index, (canonical_signal, body_signal, valid_presence) in sorted(aggregate.items())
    )


def _canonical_signal(person: ParsedPerson) -> float:
    return sum(
        _positive_confidence_signal(person.confidences[channel])
        for channel in CANONICAL_POSE_CHANNELS
    )


def _body_signal(person: ParsedPerson) -> float:
    return _positive_confidence_signal(person.confidences["body"])


def _positive_confidence_signal(confidences: np.ndarray) -> float:
    positive = confidences[confidences > 0.0]
    if positive.size == 0:
        return 0.0
    return float(positive.sum())
