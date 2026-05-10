"""Selected-person logic for multi-person scenarios."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np

from text_to_sign_production.legacy_data.pose.schema import CANONICAL_POSE_CHANNELS
from text_to_sign_production.legacy_data.pose.types import (
    ParsedFrameResult,
    ParsedPerson,
    PersonSelectionCandidateScore,
    PersonSelectionPolicy,
    PersonSelectionResult,
    PersonTrackingResult,
)
from text_to_sign_production.legacy_data.samples.types import SelectedPersonMetadata

DEFAULT_PERSON_SELECTION_POLICY = PersonSelectionPolicy.HIGHEST_CANONICAL_SIGNAL
_UPPER_BODY_TRACKING_LANDMARK_INDICES = (1, 2, 3, 4, 5, 6, 7)
_UPPER_BODY_TRACKING_LANDMARK_INDEXER = list(_UPPER_BODY_TRACKING_LANDMARK_INDICES)


@dataclass(frozen=True, slots=True)
class _ContinuityScore:
    index: int
    shared_upper_body_landmarks: int
    upper_body_centroid_distance: float
    canonical_signal: float
    body_signal: float
    person_valid: bool


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


def resolve_person_tracking(
    frames: list[ParsedFrameResult],
    policy: PersonSelectionPolicy = DEFAULT_PERSON_SELECTION_POLICY,
) -> PersonTrackingResult:
    """Resolve an anchor person, then track frame-wise raw person indices.

    OpenPose person ordering is frame-local, so the globally selected anchor
    index is a starting fact rather than a safe tensor index for every frame.
    """
    anchor_selection = resolve_person_selection(frames, policy)
    selected_indices: list[int] = []
    previous_person: ParsedPerson | None = None
    continuity_break_count = 0
    reanchor_count = 0
    tracked_target_missing_frame_count = 0
    had_tracked_person = False

    for frame in frames:
        if not frame.people:
            selected_indices.append(-1)
            if previous_person is not None:
                continuity_break_count += 1
            previous_person = None
            tracked_target_missing_frame_count += 1
            continue

        if previous_person is None:
            selected_index = _reanchor_person_index(frame, anchor_selection.target_index)
            if had_tracked_person:
                reanchor_count += 1
        else:
            selected_index = _tracked_person_index(frame, previous_person)
            if selected_index is None:
                continuity_break_count += 1
                reanchor_count += 1
                selected_index = _reanchor_person_index(frame, anchor_selection.target_index)

        if selected_index is None:
            selected_indices.append(-1)
            previous_person = None
            tracked_target_missing_frame_count += 1
            continue

        selected_indices.append(selected_index)
        previous_person = frame.people[selected_index]
        had_tracked_person = True

    frame_count = len(frames)
    missing_ratio = tracked_target_missing_frame_count / frame_count if frame_count > 0 else 0.0
    continuity_break_ratio = continuity_break_count / frame_count if frame_count > 0 else 0.0
    reanchor_ratio = reanchor_count / frame_count if frame_count > 0 else 0.0
    return PersonTrackingResult(
        anchor_selection=anchor_selection,
        selected_person_indices=tuple(selected_indices),
        continuity_break_count=continuity_break_count,
        continuity_break_ratio=continuity_break_ratio,
        reanchor_count=reanchor_count,
        reanchor_ratio=reanchor_ratio,
        tracked_target_missing_frame_count=tracked_target_missing_frame_count,
        tracked_target_missing_frame_ratio=missing_ratio,
    )


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


def _tracked_person_index(
    frame: ParsedFrameResult,
    previous_person: ParsedPerson,
) -> int | None:
    scores = tuple(
        _continuity_score(index, person, previous_person)
        for index, person in enumerate(frame.people)
    )
    continuity_scores = tuple(score for score in scores if _has_continuity_signal(score))
    if not continuity_scores:
        return None

    selected = min(
        continuity_scores,
        key=lambda score: (
            -score.shared_upper_body_landmarks,
            score.upper_body_centroid_distance,
            -score.canonical_signal,
            -score.body_signal,
            not score.person_valid,
            score.index,
        ),
    )
    return selected.index


def _reanchor_person_index(frame: ParsedFrameResult, anchor_index: int) -> int | None:
    if anchor_index < len(frame.people):
        return anchor_index
    if not frame.people:
        return None

    selected = min(
        enumerate(frame.people),
        key=lambda item: (
            -_canonical_signal(item[1]),
            -_body_signal(item[1]),
            not item[1].person_valid,
            item[0],
        ),
    )
    return selected[0]


def _continuity_score(
    index: int,
    person: ParsedPerson,
    previous_person: ParsedPerson,
) -> _ContinuityScore:
    current_mask = _upper_body_available_mask(person)
    previous_mask = _upper_body_available_mask(previous_person)
    shared = int(np.count_nonzero(current_mask & previous_mask))
    distance = _upper_body_centroid_distance(person, previous_person)
    return _ContinuityScore(
        index=index,
        shared_upper_body_landmarks=shared,
        upper_body_centroid_distance=distance,
        canonical_signal=_canonical_signal(person),
        body_signal=_body_signal(person),
        person_valid=person.person_valid,
    )


def _has_continuity_signal(score: _ContinuityScore) -> bool:
    return score.shared_upper_body_landmarks > 0 or math.isfinite(
        score.upper_body_centroid_distance
    )


def _upper_body_available_mask(person: ParsedPerson) -> np.ndarray:
    return cast(
        np.ndarray,
        person.confidences["body"][_UPPER_BODY_TRACKING_LANDMARK_INDEXER] > 0.0,
    )


def _upper_body_centroid_distance(
    person: ParsedPerson,
    previous_person: ParsedPerson,
) -> float:
    current = _upper_body_centroid(person)
    previous = _upper_body_centroid(previous_person)
    if current is None or previous is None:
        return math.inf
    return float(np.linalg.norm(current - previous))


def _upper_body_centroid(person: ParsedPerson) -> np.ndarray | None:
    mask = _upper_body_available_mask(person)
    if not np.any(mask):
        return None
    coordinates = person.coords["body"][_UPPER_BODY_TRACKING_LANDMARK_INDEXER]
    return cast(np.ndarray, np.mean(coordinates[mask], axis=0))


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
