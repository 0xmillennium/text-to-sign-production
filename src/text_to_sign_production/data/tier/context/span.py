"""Active-span and representative-articulator context builders."""

from __future__ import annotations

import numpy as np

from text_to_sign_production.core.models import PreparedSample
from text_to_sign_production.data.tier.context.types import (
    ActiveSpanContext,
    ArticulatorSource,
    RepresentativeArticulatorContext,
    RepresentativeArticulatorFrame,
    TransitionKind,
)
from text_to_sign_production.data.tier.facts import QualityFacts

UPPER_BODY_TRACKING_LANDMARK_INDICES: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
_ARTICULATOR_ORDER: tuple[ArticulatorSource, ...] = (
    ArticulatorSource.LEFT_HAND,
    ArticulatorSource.RIGHT_HAND,
    ArticulatorSource.BODY,
)


def build_active_span_context(
    sample: PreparedSample,
    quality_facts: QualityFacts,
) -> ActiveSpanContext:
    """Derive active-span masks from PreparedSample pose truth and facts."""
    frame_count = quality_facts.frame.frame_count
    frame_valid = np.asarray(sample.pose.valid_frame_mask, dtype=np.bool_)[:frame_count]
    left_available = _channel_available(sample.pose.left_hand_xyc)
    right_available = _channel_available(sample.pose.right_hand_xyc)
    body_available = _upper_body_available(sample.pose.body_xyc)
    raw = frame_valid & (left_available | right_available | body_available)
    if not np.any(raw):
        raw = frame_valid
    active = np.asarray(raw, dtype=np.bool_)
    indices = np.flatnonzero(active)
    start = int(indices[0]) if indices.size else 0
    end = int(indices[-1]) + 1 if indices.size else 0
    transitions = active[:-1] & active[1:] if frame_count > 1 else np.zeros((0,), dtype=np.bool_)
    mask = tuple(bool(value) for value in active.tolist())
    return ActiveSpanContext(
        frame_count=frame_count,
        start_frame_index=start,
        end_frame_index_exclusive=end,
        raw_evidence_mask=mask,
        stabilized_evidence_mask=mask,
        bridged_evidence_mask=mask,
        padded_active_mask=mask,
        active_frame_mask=mask,
        active_transition_mask=tuple(bool(value) for value in transitions.tolist()),
        active_frame_count=int(np.count_nonzero(active)),
    )


def build_representative_articulator_context(
    sample: PreparedSample,
    active_span: ActiveSpanContext,
) -> RepresentativeArticulatorContext:
    """Build a stable representative articulator sequence from PreparedSample."""
    candidates = {
        ArticulatorSource.LEFT_HAND: _points_and_quality(sample.pose.left_hand_xyc),
        ArticulatorSource.RIGHT_HAND: _points_and_quality(sample.pose.right_hand_xyc),
        ArticulatorSource.BODY: _points_and_quality(
            sample.pose.body_xyc[:, list(UPPER_BODY_TRACKING_LANDMARK_INDICES), :]
        ),
    }
    frames: list[RepresentativeArticulatorFrame] = []
    comparable: list[bool] = []
    source_switch: list[bool] = []
    unavailable: list[bool] = []
    previous: ArticulatorSource | None = None
    for frame_index in range(active_span.frame_count):
        selected = _select_source(candidates, frame_index)
        if selected is None or not active_span.active_frame_mask[frame_index]:
            transition = _transition(previous, None)
            frames.append(
                RepresentativeArticulatorFrame(frame_index, None, False, None, transition)
            )
            previous = None
        else:
            points, _, _ = candidates[selected]
            transition = _transition(previous, selected)
            frames.append(
                RepresentativeArticulatorFrame(
                    frame_index,
                    selected,
                    True,
                    (float(points[frame_index, 0]), float(points[frame_index, 1])),
                    transition,
                )
            )
            previous = selected
        if frame_index > 0:
            comparable.append(transition is TransitionKind.COMPARABLE)
            source_switch.append(transition is TransitionKind.SOURCE_SWITCH)
            unavailable.append(transition is TransitionKind.UNAVAILABLE)
    return RepresentativeArticulatorContext(
        frames=tuple(frames),
        comparable_transition_mask=tuple(comparable),
        source_switch_transition_mask=tuple(source_switch),
        unavailable_transition_mask=tuple(unavailable),
    )


def _channel_available(xyc: np.ndarray) -> np.ndarray:
    return np.asarray(np.any(xyc[..., 2] > 0.0, axis=1), dtype=np.bool_)


def _upper_body_available(xyc: np.ndarray) -> np.ndarray:
    return _channel_available(xyc[:, list(UPPER_BODY_TRACKING_LANDMARK_INDICES), :])


def _points_and_quality(xyc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    conf = xyc[..., 2]
    coords = xyc[..., :2]
    available = np.any(conf > 0.0, axis=1)
    points = np.zeros((xyc.shape[0], 2), dtype=float)
    quality = np.zeros((xyc.shape[0],), dtype=float)
    for index in np.flatnonzero(available):
        mask = conf[index] > 0.0
        points[index] = np.mean(coords[index, mask], axis=0)
        quality[index] = float(np.mean(conf[index, mask]))
    return points, available, quality


def _select_source(
    candidates: dict[ArticulatorSource, tuple[np.ndarray, np.ndarray, np.ndarray]],
    frame_index: int,
) -> ArticulatorSource | None:
    available = tuple(
        source for source in _ARTICULATOR_ORDER if bool(candidates[source][1][frame_index])
    )
    if not available:
        return None
    return min(
        available,
        key=lambda source: (
            -float(candidates[source][2][frame_index]),
            _ARTICULATOR_ORDER.index(source),
        ),
    )


def _transition(
    previous: ArticulatorSource | None,
    current: ArticulatorSource | None,
) -> TransitionKind | None:
    if previous is None and current is None:
        return TransitionKind.UNAVAILABLE
    if previous is None:
        return None
    if current is None:
        return TransitionKind.UNAVAILABLE
    if previous != current:
        return TransitionKind.SOURCE_SWITCH
    return TransitionKind.COMPARABLE


__all__ = ["build_active_span_context", "build_representative_articulator_context"]
