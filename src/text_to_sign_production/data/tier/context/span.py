"""Active-span and representative-articulator context builders."""

from __future__ import annotations

import numpy as np

from text_to_sign_production.core.models import PreparedSample
from text_to_sign_production.data.tier.context.masks import (
    bridge_short_false_gaps,
    mask_bounds,
    pad_true_runs,
    remove_short_true_runs,
)
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
_MIN_ACTIVE_EVIDENCE_RUN = 2
_MAX_ACTIVE_GAP_BRIDGE = 2
_ACTIVE_EDGE_PAD = 1
_SOURCE_STICKINESS_MARGIN = 0.05


def build_active_span_context(
    sample: PreparedSample,
    quality_facts: QualityFacts,
) -> ActiveSpanContext:
    """Derive active-span masks from PreparedSample pose truth and facts."""
    frame_count = quality_facts.frame.frame_count
    frame_valid = np.asarray(sample.pose.valid_frame_mask, dtype=np.bool_)[:frame_count]
    left_available = _channel_available(sample.pose.left_hand_xyc)[:frame_count]
    right_available = _channel_available(sample.pose.right_hand_xyc)[:frame_count]
    body_available = _upper_body_available(sample.pose.body_xyc)[:frame_count]
    raw = frame_valid & (left_available | right_available | body_available)
    stabilized = remove_short_true_runs(raw, min_run_length=_MIN_ACTIVE_EVIDENCE_RUN)
    bridged = bridge_short_false_gaps(stabilized, max_gap_length=_MAX_ACTIVE_GAP_BRIDGE)
    padded = pad_true_runs(bridged, pad=_ACTIVE_EDGE_PAD, support_mask=frame_valid)
    fallback_used = False
    if not np.any(padded):
        fallback_used = True
        padded = np.array(raw if np.any(raw) else frame_valid, copy=True)
    active = np.asarray(padded, dtype=np.bool_)
    start, end = mask_bounds(active)
    transitions = active[:-1] & active[1:] if frame_count > 1 else np.zeros((0,), dtype=np.bool_)
    return ActiveSpanContext(
        frame_count=frame_count,
        start_frame_index=start,
        end_frame_index_exclusive=end,
        frame_valid_mask=_to_bool_tuple(frame_valid),
        raw_evidence_mask=_to_bool_tuple(raw),
        stabilized_evidence_mask=_to_bool_tuple(stabilized),
        bridged_evidence_mask=_to_bool_tuple(bridged),
        padded_active_mask=_to_bool_tuple(padded),
        active_frame_mask=_to_bool_tuple(active),
        active_transition_mask=tuple(bool(value) for value in transitions.tolist()),
        active_frame_count=int(np.count_nonzero(active)),
        fallback_used=fallback_used,
        raw_evidence_frame_count=int(np.count_nonzero(raw)),
        stabilized_evidence_frame_count=int(np.count_nonzero(stabilized)),
        bridged_evidence_frame_count=int(np.count_nonzero(bridged)),
        padded_active_frame_count=int(np.count_nonzero(padded)),
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
        selected = _select_source(candidates, frame_index, previous)
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
    previous: ArticulatorSource | None = None,
) -> ArticulatorSource | None:
    available = tuple(
        source for source in _ARTICULATOR_ORDER if bool(candidates[source][1][frame_index])
    )
    if not available:
        return None
    best = min(
        available,
        key=lambda source: (
            -float(candidates[source][2][frame_index]),
            _ARTICULATOR_ORDER.index(source),
        ),
    )
    if previous in available:
        previous_quality = float(candidates[previous][2][frame_index])
        best_quality = float(candidates[best][2][frame_index])
        if best_quality - previous_quality <= _SOURCE_STICKINESS_MARGIN:
            return previous
    return best


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


def _to_bool_tuple(mask: np.ndarray) -> tuple[bool, ...]:
    return tuple(bool(value) for value in np.asarray(mask, dtype=np.bool_).tolist())


__all__ = ["build_active_span_context", "build_representative_articulator_context"]
