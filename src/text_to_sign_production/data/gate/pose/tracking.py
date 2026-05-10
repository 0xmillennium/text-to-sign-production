"""Frame-wise person tracking truth."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np

from text_to_sign_production.data.gate.pose.anatomy import UPPER_BODY_TRACKING_LANDMARK_INDICES
from text_to_sign_production.data.gate.pose.schema import CANONICAL_POSE_CHANNELS
from text_to_sign_production.data.gate.pose.types import (
    AnchorSelection,
    FrameTrackingSelection,
    ParsedFrame,
    ParsedPerson,
    PersonSelectionPolicy,
    PersonSelectionScore,
    PersonTrackingResult,
    PoseChannel,
    PoseDiagnostic,
    PoseDiagnosticCode,
    PoseDiagnosticSeverity,
    TrackingSelectionReason,
)

DEFAULT_PERSON_SELECTION_POLICY = PersonSelectionPolicy.HIGHEST_CANONICAL_SIGNAL


@dataclass(frozen=True, slots=True)
class _ContinuityScore:
    index: int
    shared_upper_body_landmarks: int
    upper_body_centroid_distance: float
    canonical_signal: float
    body_signal: float
    person_valid: bool


def _diagnostic(
    code: PoseDiagnosticCode,
    message: str,
    *,
    frame_index: int | None = None,
    person_index: int | None = None,
    severity: PoseDiagnosticSeverity = PoseDiagnosticSeverity.WARNING,
) -> PoseDiagnostic:
    return PoseDiagnostic(
        code=code,
        severity=severity,
        message=message,
        frame_index=frame_index,
        person_index=person_index,
    )


def select_anchor_person(
    frames: tuple[ParsedFrame, ...],
    policy: PersonSelectionPolicy = DEFAULT_PERSON_SELECTION_POLICY,
) -> AnchorSelection:
    """Select the global anchor person used to initialize frame-wise tracking."""
    scores = _score_person_indices(frames)
    if policy is PersonSelectionPolicy.OPENPOSE_PRIMARY:
        return AnchorSelection(
            anchor_person_index=0,
            policy=policy,
            fallback_used=False,
            fallback_reason=None,
            candidate_scores=scores,
        )
    if policy is PersonSelectionPolicy.HIGHEST_CANONICAL_SIGNAL:
        if not scores:
            return AnchorSelection(
                anchor_person_index=0,
                policy=policy,
                fallback_used=True,
                fallback_reason=PoseDiagnosticCode.NO_PEOPLE_DETECTED,
                candidate_scores=(),
            )
        selected = min(
            scores,
            key=lambda score: (
                -score.aggregate_canonical_signal,
                -score.aggregate_body_signal,
                -score.valid_frame_presence_count,
                score.person_index,
            ),
        )
        return AnchorSelection(
            anchor_person_index=selected.person_index,
            policy=policy,
            fallback_used=False,
            fallback_reason=None,
            candidate_scores=scores,
        )
    raise ValueError(f"Unsupported person selection policy: {policy!r}.")


def track_person_frames(
    frames: tuple[ParsedFrame, ...],
    anchor: AnchorSelection,
) -> PersonTrackingResult:
    """Resolve selected person indices per frame from an anchor selection."""
    selections: list[FrameTrackingSelection] = []
    diagnostics: list[PoseDiagnostic] = []
    previous_person: ParsedPerson | None = None
    continuity_break_count = 0
    reanchor_count = 0
    missing_count = 0
    had_tracked_person = False

    for frame in frames:
        if not frame.people:
            diagnostic = _diagnostic(
                PoseDiagnosticCode.TARGET_PERSON_MISSING,
                "No people available for tracking in frame.",
                frame_index=frame.frame_index,
            )
            selections.append(
                FrameTrackingSelection(
                    frame_index=frame.frame_index,
                    selected_person_index=None,
                    reason=TrackingSelectionReason.MISSING,
                    diagnostics=(diagnostic,),
                )
            )
            diagnostics.append(diagnostic)
            if previous_person is not None:
                continuity_break_count += 1
            previous_person = None
            missing_count += 1
            continue

        reason = TrackingSelectionReason.ANCHOR
        selection_diagnostics: list[PoseDiagnostic] = []
        if previous_person is None:
            selected_index = _reanchor_person_index(frame, anchor.anchor_person_index)
            if had_tracked_person:
                reason = TrackingSelectionReason.REANCHOR
                reanchor_count += 1
                diagnostic = _diagnostic(
                    PoseDiagnosticCode.TRACKING_REANCHOR,
                    "Tracking re-anchored after missing continuity.",
                    frame_index=frame.frame_index,
                    person_index=selected_index,
                )
                selection_diagnostics.append(diagnostic)
                diagnostics.append(diagnostic)
        else:
            tracked_index = _tracked_person_index(frame, previous_person)
            if tracked_index is None:
                continuity_break_count += 1
                reanchor_count += 1
                selected_index = _reanchor_person_index(frame, anchor.anchor_person_index)
                reason = TrackingSelectionReason.REANCHOR
                for code, message in (
                    (
                        PoseDiagnosticCode.TRACKING_CONTINUITY_BREAK,
                        "Tracking continuity could not be resolved from previous frame.",
                    ),
                    (
                        PoseDiagnosticCode.TRACKING_REANCHOR,
                        "Tracking re-anchored deterministically.",
                    ),
                ):
                    diagnostic = _diagnostic(
                        code,
                        message,
                        frame_index=frame.frame_index,
                        person_index=selected_index,
                    )
                    selection_diagnostics.append(diagnostic)
                    diagnostics.append(diagnostic)
            else:
                selected_index = tracked_index
                reason = TrackingSelectionReason.CONTINUITY

        if selected_index is None:
            diagnostic = _diagnostic(
                PoseDiagnosticCode.TARGET_PERSON_MISSING,
                "No selected person could be resolved.",
                frame_index=frame.frame_index,
            )
            selection_diagnostics.append(diagnostic)
            diagnostics.append(diagnostic)
            selections.append(
                FrameTrackingSelection(
                    frame_index=frame.frame_index,
                    selected_person_index=None,
                    reason=TrackingSelectionReason.MISSING,
                    diagnostics=tuple(selection_diagnostics),
                )
            )
            previous_person = None
            missing_count += 1
            continue

        selections.append(
            FrameTrackingSelection(
                frame_index=frame.frame_index,
                selected_person_index=selected_index,
                reason=reason,
                diagnostics=tuple(selection_diagnostics),
            )
        )
        previous_person = frame.people[selected_index]
        had_tracked_person = True

    return PersonTrackingResult(
        anchor=anchor,
        frame_selections=tuple(selections),
        continuity_break_count=continuity_break_count,
        reanchor_count=reanchor_count,
        target_missing_frame_count=missing_count,
        diagnostics=tuple(diagnostics),
    )


def build_tracking_result(
    frames: tuple[ParsedFrame, ...],
    policy: PersonSelectionPolicy = DEFAULT_PERSON_SELECTION_POLICY,
) -> PersonTrackingResult:
    """Select an anchor person and resolve frame-wise tracking."""
    return track_person_frames(frames, select_anchor_person(frames, policy))


def _score_person_indices(frames: tuple[ParsedFrame, ...]) -> tuple[PersonSelectionScore, ...]:
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
        PersonSelectionScore(
            person_index=index,
            aggregate_canonical_signal=canonical_signal,
            aggregate_body_signal=body_signal,
            valid_frame_presence_count=valid_presence,
        )
        for index, (canonical_signal, body_signal, valid_presence) in sorted(aggregate.items())
    )


def _tracked_person_index(frame: ParsedFrame, previous_person: ParsedPerson) -> int | None:
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


def _reanchor_person_index(frame: ParsedFrame, anchor_index: int) -> int | None:
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
    indexer = list(UPPER_BODY_TRACKING_LANDMARK_INDICES)
    return cast(np.ndarray, person.channels[PoseChannel.BODY].confidences[indexer] > 0.0)


def _upper_body_centroid_distance(person: ParsedPerson, previous_person: ParsedPerson) -> float:
    current = _upper_body_centroid(person)
    previous = _upper_body_centroid(previous_person)
    if current is None or previous is None:
        return math.inf
    return float(np.linalg.norm(current - previous))


def _upper_body_centroid(person: ParsedPerson) -> np.ndarray | None:
    mask = _upper_body_available_mask(person)
    if not np.any(mask):
        return None
    coordinates = person.channels[PoseChannel.BODY].coordinates[
        list(UPPER_BODY_TRACKING_LANDMARK_INDICES)
    ]
    return cast(np.ndarray, np.mean(coordinates[mask], axis=0))


def _canonical_signal(person: ParsedPerson) -> float:
    return sum(
        _positive_confidence_signal(person.channels[channel].confidences)
        for channel in CANONICAL_POSE_CHANNELS
    )


def _body_signal(person: ParsedPerson) -> float:
    return _positive_confidence_signal(person.channels[PoseChannel.BODY].confidences)


def _positive_confidence_signal(confidences: np.ndarray) -> float:
    positive = confidences[confidences > 0.0]
    if positive.size == 0:
        return 0.0
    return float(positive.sum())


__all__ = [
    "DEFAULT_PERSON_SELECTION_POLICY",
    "build_tracking_result",
    "select_anchor_person",
    "track_person_frames",
]
