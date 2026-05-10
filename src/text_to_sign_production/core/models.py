"""Repository-wide root dataclass contracts."""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np

from text_to_sign_production.core.ids import SampleSplit, SampleStatus, TierName
from text_to_sign_production.data.gate.pose import CoordinateSpace
from text_to_sign_production.data.gate.sources import SourceIssueCode


class SamplesDropStage(enum.StrEnum):
    """Samples-stage location where a sample can be dropped."""

    SOURCE = "source"
    POSE = "pose"
    GATES = "gates"
    PAYLOAD = "payload"
    MANIFEST = "manifest"


class SamplesIssueCode(enum.StrEnum):
    """Stable samples-stage issue codes."""

    SOURCE_TRUTH_INVALID = "source_truth_invalid"
    POSE_TRUTH_INVALID = "pose_truth_invalid"
    PAYLOAD_INVALID = "payload_invalid"
    MANIFEST_INVALID = "manifest_invalid"
    GATE_FAILED = "gate_failed"
    MATERIALIZATION_FAILED = "materialization_failed"


class GateName(enum.StrEnum):
    """Samples admission gate names in evaluation order."""

    SOURCE = "source"
    FRAMES = "frames"
    BODY = "body"
    HAND = "hand"
    FACE = "face"


class GateStatus(enum.StrEnum):
    """Controlled samples admission gate result status."""

    PASS = "pass"
    FAIL = "fail"
    SKIPPED = "skipped"


class GateIssueCode(enum.StrEnum):
    """Stable machine-readable samples admission gate issue codes."""

    STRUCTURAL_CORRUPTION = "structural_corruption"
    PREREQUISITE_MISSING = "prerequisite_missing"
    SOURCE_TEXT_MISSING = "source_text_missing"
    SOURCE_TEXT_TOO_SHORT = "source_text_too_short"
    SOURCE_ISSUE_PRESENT = "source_issue_present"
    POSE_OUTPUT_INCOMPLETE = "pose_output_incomplete"
    FRAME_COUNT_TOO_LOW = "frame_count_too_low"
    VALID_FRAME_COUNT_TOO_LOW = "valid_frame_count_too_low"
    DURATION_OUT_OF_RANGE = "duration_out_of_range"
    TRACKING_MISSING_TOO_HIGH = "tracking_missing_too_high"
    TRACKING_CONTINUITY_TOO_HIGH = "tracking_continuity_too_high"
    TRACKING_REANCHOR_TOO_HIGH = "tracking_reanchor_too_high"
    CHANNEL_EVIDENCE_TOO_LOW = "channel_evidence_too_low"
    UPPER_BODY_SUPPORT_TOO_LOW = "upper_body_support_too_low"
    MANUAL_VISIBILITY_TOO_LOW = "manual_visibility_too_low"
    FACE_VISIBILITY_TOO_LOW = "face_visibility_too_low"


class TierStatus(enum.StrEnum):
    """Controlled downstream tier-decision status."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class SourceTruth:
    """Source-side truth carried by a prepared sample."""

    sample_id: str
    split: SampleSplit

    text: str
    canonical_normalized_text: str
    fps: float

    source_video_id: str
    source_sentence_id: str
    source_sentence_name: str

    source_issue_codes: tuple[SourceIssueCode, ...]

    def __post_init__(self) -> None:
        _require_text(self.sample_id, "sample_id")
        _require_text(self.text, "text")
        _require_text(self.canonical_normalized_text, "canonical_normalized_text")
        if self.fps <= 0:
            raise ValueError("fps must be positive.")
        _require_text(self.source_video_id, "source_video_id")
        _require_text(self.source_sentence_id, "source_sentence_id")
        _require_text(self.source_sentence_name, "source_sentence_name")
        codes = tuple(SourceIssueCode(code) for code in self.source_issue_codes)
        _require_unique(codes, "source_issue_codes")
        object.__setattr__(self, "source_issue_codes", codes)


@dataclass(frozen=True, slots=True)
class PoseTruth:
    """Pose-side truth carried by a prepared sample."""

    coordinate_space: CoordinateSpace
    frame_count: int

    valid_frame_mask: np.ndarray
    selected_person_indices: tuple[int | None, ...]

    tracked_target_missing_frame_count: int
    continuity_break_count: int
    reanchor_count: int

    body_xyc: np.ndarray
    face_xyc: np.ndarray
    left_hand_xyc: np.ndarray
    right_hand_xyc: np.ndarray

    body_nonzero_frame_count: int
    face_nonzero_frame_count: int
    left_hand_nonzero_frame_count: int
    right_hand_nonzero_frame_count: int

    def __post_init__(self) -> None:
        if self.coordinate_space is not CoordinateSpace.NORMALIZED_IMAGE:
            raise ValueError("coordinate_space must be normalized image.")
        if self.frame_count <= 0:
            raise ValueError("frame_count must be positive.")
        if len(self.valid_frame_mask) != self.frame_count:
            raise ValueError("valid_frame_mask length must match frame_count.")
        if len(self.selected_person_indices) != self.frame_count:
            raise ValueError("selected_person_indices length must match frame_count.")
        for name, tensor in (
            ("body_xyc", self.body_xyc),
            ("face_xyc", self.face_xyc),
            ("left_hand_xyc", self.left_hand_xyc),
            ("right_hand_xyc", self.right_hand_xyc),
        ):
            if tensor.shape[0] != self.frame_count:
                raise ValueError(f"{name} time dimension must match frame_count.")
        for name, count in (
            ("tracked_target_missing_frame_count", self.tracked_target_missing_frame_count),
            ("continuity_break_count", self.continuity_break_count),
            ("reanchor_count", self.reanchor_count),
            ("body_nonzero_frame_count", self.body_nonzero_frame_count),
            ("face_nonzero_frame_count", self.face_nonzero_frame_count),
            ("left_hand_nonzero_frame_count", self.left_hand_nonzero_frame_count),
            ("right_hand_nonzero_frame_count", self.right_hand_nonzero_frame_count),
        ):
            if count < 0:
                raise ValueError(f"{name} cannot be negative.")
            if count > self.frame_count:
                raise ValueError(f"{name} cannot exceed frame_count.")


@dataclass(frozen=True, slots=True)
class PreparedSample:
    """Canonical samples-stage object consumed by downstream stages."""

    schema_version: str
    source: SourceTruth
    pose: PoseTruth

    def __post_init__(self) -> None:
        _require_text(self.schema_version, "schema_version")


@dataclass(frozen=True, slots=True)
class PassedManifestEntry:
    """Indexing and handoff record for a passed prepared sample payload."""

    schema_version: str

    sample_id: str
    split: SampleSplit
    payload_ref: str

    text: str
    canonical_normalized_text: str
    fps: float
    frame_count: int

    source_video_id: str
    source_sentence_id: str
    source_sentence_name: str

    valid_frame_count: int
    body_nonzero_frame_count: int
    face_nonzero_frame_count: int
    left_hand_nonzero_frame_count: int
    right_hand_nonzero_frame_count: int

    def __post_init__(self) -> None:
        for value, name in (
            (self.schema_version, "schema_version"),
            (self.sample_id, "sample_id"),
            (self.payload_ref, "payload_ref"),
            (self.text, "text"),
            (self.canonical_normalized_text, "canonical_normalized_text"),
            (self.source_video_id, "source_video_id"),
            (self.source_sentence_id, "source_sentence_id"),
            (self.source_sentence_name, "source_sentence_name"),
        ):
            _require_text(value, name)
        if self.fps <= 0:
            raise ValueError("fps must be positive.")
        if self.frame_count <= 0:
            raise ValueError("frame_count must be positive.")
        for name, count in (
            ("valid_frame_count", self.valid_frame_count),
            ("body_nonzero_frame_count", self.body_nonzero_frame_count),
            ("face_nonzero_frame_count", self.face_nonzero_frame_count),
            ("left_hand_nonzero_frame_count", self.left_hand_nonzero_frame_count),
            ("right_hand_nonzero_frame_count", self.right_hand_nonzero_frame_count),
        ):
            if count < 0 or count > self.frame_count:
                raise ValueError(f"{name} must be between 0 and frame_count.")


@dataclass(frozen=True, slots=True)
class DroppedManifestEntry:
    """Indexing and review record for a sample dropped before passed handoff."""

    schema_version: str

    sample_id: str
    split: SampleSplit

    text: str | None
    canonical_normalized_text: str | None

    source_video_id: str | None
    source_sentence_id: str | None
    source_sentence_name: str | None

    drop_stage: SamplesDropStage
    issue_codes: tuple[SamplesIssueCode, ...]
    debug_ref: str | None

    def __post_init__(self) -> None:
        _require_text(self.schema_version, "schema_version")
        _require_text(self.sample_id, "sample_id")
        object.__setattr__(self, "drop_stage", SamplesDropStage(self.drop_stage))
        codes = tuple(SamplesIssueCode(code) for code in self.issue_codes)
        if not codes:
            raise ValueError("issue_codes must be non-empty.")
        _require_unique(codes, "issue_codes")
        object.__setattr__(self, "issue_codes", codes)


@dataclass(frozen=True, slots=True)
class GateDecision:
    """Decision for one samples admission gate."""

    gate: GateName
    status: GateStatus
    issue_codes: tuple[GateIssueCode, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "gate", GateName(self.gate))
        object.__setattr__(self, "status", GateStatus(self.status))
        codes = tuple(GateIssueCode(code) for code in self.issue_codes)
        _require_unique(codes, "issue_codes")
        object.__setattr__(self, "issue_codes", codes)


@dataclass(frozen=True, slots=True)
class GateDecisionBundle:
    """Samples admission gate decisions for one prepared sample."""

    sample_id: str
    split: SampleSplit

    final_status: SampleStatus
    terminal_gate: GateName | None

    decisions: tuple[GateDecision, ...]
    failed_gates: tuple[GateName, ...]

    def __post_init__(self) -> None:
        _require_text(self.sample_id, "sample_id")
        decisions = tuple(self.decisions)
        if not decisions:
            raise ValueError("decisions must be non-empty.")
        failed_gates = tuple(GateName(gate) for gate in self.failed_gates)
        _require_unique(failed_gates, "failed_gates")
        terminal_gate = None if self.terminal_gate is None else GateName(self.terminal_gate)
        if self.final_status is SampleStatus.PASSED:
            if failed_gates:
                raise ValueError("passed gate bundle cannot carry failed_gates.")
            if terminal_gate is not None:
                raise ValueError("passed gate bundle cannot carry terminal_gate.")
        if self.final_status is SampleStatus.DROPPED:
            if not failed_gates:
                raise ValueError("dropped gate bundle requires failed_gates.")
            if terminal_gate is None:
                raise ValueError("dropped gate bundle requires terminal_gate.")
        object.__setattr__(self, "terminal_gate", terminal_gate)
        object.__setattr__(self, "decisions", decisions)
        object.__setattr__(self, "failed_gates", failed_gates)


@dataclass(frozen=True, slots=True)
class TierFamilyDecision:
    """Root downstream tier decision for one metric family."""

    family: str
    status: TierStatus
    supported_tiers: tuple[TierName, ...]
    best_supported_tier: TierName | None
    issue_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.family, "family")
        object.__setattr__(self, "status", TierStatus(self.status))
        object.__setattr__(
            self, "supported_tiers", tuple(TierName(t) for t in self.supported_tiers)
        )
        best = None if self.best_supported_tier is None else TierName(self.best_supported_tier)
        object.__setattr__(self, "best_supported_tier", best)
        object.__setattr__(self, "issue_codes", tuple(self.issue_codes))


@dataclass(frozen=True, slots=True)
class TierDecisionBundle:
    """Root downstream tier-band decision for one sample."""

    sample_id: str
    split: SampleSplit
    status: TierStatus
    selected_tier: TierName | None
    family_decisions: tuple[TierFamilyDecision, ...]
    issue_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.sample_id, "sample_id")
        object.__setattr__(self, "status", TierStatus(self.status))
        selected = None if self.selected_tier is None else TierName(self.selected_tier)
        object.__setattr__(self, "selected_tier", selected)
        object.__setattr__(self, "family_decisions", tuple(self.family_decisions))
        object.__setattr__(self, "issue_codes", tuple(self.issue_codes))


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-blank string.")


def _require_unique(values: tuple[object, ...], name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{name} cannot contain duplicates.")


__all__ = [
    "DroppedManifestEntry",
    "GateDecision",
    "GateDecisionBundle",
    "GateIssueCode",
    "GateName",
    "GateStatus",
    "PassedManifestEntry",
    "PoseTruth",
    "PreparedSample",
    "SamplesDropStage",
    "SamplesIssueCode",
    "SourceTruth",
    "TierDecisionBundle",
    "TierFamilyDecision",
    "TierStatus",
]
