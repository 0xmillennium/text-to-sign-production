"""Repository-wide root dataclass contracts."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from text_to_sign_production.core.ids import (
    CoordinateSpace,
    SampleSplit,
    SampleStatus,
    SourceIssueCode,
    TierName,
)


class GateDropStage(enum.StrEnum):
    """Gate-stage location where a sample can be dropped."""

    SOURCE = "source"
    POSE = "pose"
    GATES = "gates"


class GateDropIssueCode(enum.StrEnum):
    """Stable gate-stage dropped-manifest issue codes."""

    SOURCE_TRUTH_INVALID = "source_truth_invalid"
    POSE_TRUTH_INVALID = "pose_truth_invalid"
    GATE_FAILED = "gate_failed"


class GateName(enum.StrEnum):
    """Gate-stage admission gate names in evaluation order."""

    SOURCE = "source"
    FRAMES = "frames"
    BODY = "body"
    HAND = "hand"
    FACE = "face"


_SAMPLE_DROP_STAGES = frozenset({GateDropStage.SOURCE, GateDropStage.POSE, GateDropStage.GATES})


class GateStatus(enum.StrEnum):
    """Controlled admission-gate result status."""

    PASS = "pass"
    FAIL = "fail"
    SKIPPED = "skipped"


class GateIssueCode(enum.StrEnum):
    """Stable machine-readable admission-gate issue codes."""

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


class TierIssueCode(enum.StrEnum):
    """Stable machine-readable tier issue codes."""

    GATE_FAILED = "gate_failed"
    FAMILY_THRESHOLD_NOT_MET = "family_threshold_not_met"
    FAMILY_UNSUPPORTED = "family_unsupported"
    NO_SUPPORTED_TIER = "no_supported_tier"
    LEAKAGE_SEVERITY_EXCEEDS_TIER_POLICY = "leakage_severity_exceeds_tier_policy"
    CONFIG_INVALID = "config_invalid"
    POLICY_INVALID = "policy_invalid"
    DECISION_INVALID = "decision_invalid"


@dataclass(frozen=True, slots=True)
class SourceTruth:
    """Source-side truth carried by a prepared sample."""

    sample_id: str
    split: SampleSplit

    text: str
    fps: float

    source_video_id: str
    source_sentence_id: str
    source_sentence_name: str

    source_issue_codes: tuple[SourceIssueCode, ...]

    def __post_init__(self) -> None:
        _require_text(self.sample_id, "sample_id")
        _require_text(self.text, "text")
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
    """Canonical prepared-sample object consumed by downstream stages."""

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

    source_video_id: str | None
    source_sentence_id: str | None
    source_sentence_name: str | None

    drop_stage: GateDropStage
    issue_codes: tuple[GateDropIssueCode, ...]
    dropped_sample_ref: str

    def __post_init__(self) -> None:
        _require_text(self.schema_version, "schema_version")
        _require_text(self.sample_id, "sample_id")
        object.__setattr__(self, "drop_stage", GateDropStage(self.drop_stage))
        if self.drop_stage not in _SAMPLE_DROP_STAGES:
            raise ValueError("drop_stage must be source, pose, or gates.")
        codes = tuple(GateDropIssueCode(code) for code in self.issue_codes)
        if not codes:
            raise ValueError("issue_codes must be non-empty.")
        _require_unique(codes, "issue_codes")
        object.__setattr__(self, "issue_codes", codes)
        _require_text(self.dropped_sample_ref, "dropped_sample_ref")


@dataclass(frozen=True, slots=True)
class DroppedSampleSourceSnapshot:
    """Source/match evidence carried by a dropped sample payload."""

    text: str | None
    source_video_id: str | None
    source_sentence_id: str | None
    source_sentence_name: str | None
    match_status: str
    unmatched_reason: str | None
    ambiguity_reasons: tuple[str, ...]
    source_issue_codes: tuple[str, ...]
    video_match_count: int
    keypoint_match_count: int

    def __post_init__(self) -> None:
        _require_text(self.match_status, "match_status")
        if self.video_match_count < 0:
            raise ValueError("video_match_count cannot be negative.")
        if self.keypoint_match_count < 0:
            raise ValueError("keypoint_match_count cannot be negative.")
        object.__setattr__(self, "ambiguity_reasons", tuple(self.ambiguity_reasons))
        object.__setattr__(self, "source_issue_codes", tuple(self.source_issue_codes))


@dataclass(frozen=True, slots=True)
class DroppedSamplePoseSnapshot:
    """Candidate/pose viability evidence carried by a dropped sample payload."""

    candidate_available: bool
    video_path: str | None
    keypoints_dir: str | None
    candidate_frame_count: int | None
    observed_frame_count: int | None
    video_metadata_readable: bool | None
    video_metadata_error: str | None
    missing_frame_files: bool | None
    viability_status: str | None
    viability_issue_codes: tuple[str, ...]
    viability_issue_messages: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("candidate_frame_count", self.candidate_frame_count),
            ("observed_frame_count", self.observed_frame_count),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} cannot be negative.")
        object.__setattr__(self, "viability_issue_codes", tuple(self.viability_issue_codes))
        object.__setattr__(
            self,
            "viability_issue_messages",
            tuple(self.viability_issue_messages),
        )


@dataclass(frozen=True, slots=True)
class DroppedSampleGateSnapshot:
    """Prepared-sample and gate decision evidence carried by a dropped sample payload."""

    prepared_sample_available: bool
    frame_count: int
    valid_frame_count: int
    body_nonzero_frame_count: int
    face_nonzero_frame_count: int
    left_hand_nonzero_frame_count: int
    right_hand_nonzero_frame_count: int
    final_status: SampleStatus
    terminal_gate: GateName | None
    failed_gates: tuple[GateName, ...]
    decision_issue_codes: tuple[GateIssueCode, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("frame_count", self.frame_count),
            ("valid_frame_count", self.valid_frame_count),
            ("body_nonzero_frame_count", self.body_nonzero_frame_count),
            ("face_nonzero_frame_count", self.face_nonzero_frame_count),
            ("left_hand_nonzero_frame_count", self.left_hand_nonzero_frame_count),
            ("right_hand_nonzero_frame_count", self.right_hand_nonzero_frame_count),
        ):
            if value < 0:
                raise ValueError(f"{name} cannot be negative.")
        object.__setattr__(self, "final_status", SampleStatus(self.final_status))
        object.__setattr__(
            self,
            "terminal_gate",
            None if self.terminal_gate is None else GateName(self.terminal_gate),
        )
        object.__setattr__(
            self,
            "failed_gates",
            tuple(GateName(gate) for gate in self.failed_gates),
        )
        object.__setattr__(
            self,
            "decision_issue_codes",
            tuple(GateIssueCode(code) for code in self.decision_issue_codes),
        )


@dataclass(frozen=True, slots=True)
class DroppedSample:
    """Canonical in-memory dropped sample evidence payload."""

    schema_version: str
    sample_id: str
    split: SampleSplit
    drop_stage: GateDropStage
    issue_codes: tuple[GateDropIssueCode, ...]
    source: DroppedSampleSourceSnapshot
    pose: DroppedSamplePoseSnapshot | None
    gate: DroppedSampleGateSnapshot | None

    def __post_init__(self) -> None:
        _require_text(self.schema_version, "schema_version")
        _require_text(self.sample_id, "sample_id")
        object.__setattr__(self, "split", SampleSplit(self.split))
        object.__setattr__(self, "drop_stage", GateDropStage(self.drop_stage))
        if self.drop_stage not in _SAMPLE_DROP_STAGES:
            raise ValueError("drop_stage must be source, pose, or gates.")
        codes = tuple(GateDropIssueCode(code) for code in self.issue_codes)
        if not codes:
            raise ValueError("issue_codes must be non-empty.")
        _require_unique(codes, "issue_codes")
        object.__setattr__(self, "issue_codes", codes)
        if self.drop_stage is GateDropStage.SOURCE:
            if self.pose is not None or self.gate is not None:
                raise ValueError("source-stage dropped samples cannot carry pose or gate evidence.")
        elif self.drop_stage is GateDropStage.POSE:
            if self.pose is None or self.gate is not None:
                raise ValueError("pose-stage dropped samples require pose evidence only.")
        elif self.drop_stage is GateDropStage.GATES:
            if self.pose is None or self.gate is None:
                raise ValueError("gate-stage dropped samples require pose and gate evidence.")


@dataclass(frozen=True, slots=True)
class GateDecision:
    """Decision for one admission gate."""

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
    """Admission-gate decisions for one prepared sample."""

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
class CheckpointAdmission:
    """Checkpoint manifest provenance for downstream tier processing.

    This is not gate predicate detail. It records the exact passed-checkpoint
    manifest surface and row identity consumed by tier processing.
    """

    sample_id: str
    split: SampleSplit
    checkpoint_surface: str = "passed_manifest"
    source_manifest_path: Path | None = None
    source_manifest_sha256: str | None = None
    payload_ref: str | None = None
    gate_detail_available: bool = False
    gate_detail_status: str = "unavailable_in_checkpoint_tier_flow"

    def __post_init__(self) -> None:
        _require_text(self.sample_id, "sample_id")
        _require_text(self.checkpoint_surface, "checkpoint_surface")
        if self.source_manifest_path is not None and not isinstance(
            self.source_manifest_path,
            Path,
        ):
            object.__setattr__(self, "source_manifest_path", Path(self.source_manifest_path))
        if self.source_manifest_sha256 is not None:
            _require_text(self.source_manifest_sha256, "source_manifest_sha256")
        if self.payload_ref is not None:
            _require_text(self.payload_ref, "payload_ref")
        if self.gate_detail_available:
            raise ValueError("checkpoint-only tier flow must not claim gate detail availability.")
        _require_text(self.gate_detail_status, "gate_detail_status")
        object.__setattr__(self, "split", SampleSplit(self.split))


@dataclass(frozen=True, slots=True)
class TierIssue:
    """One structured tier policy issue."""

    code: TierIssueCode
    message: str
    family: enum.StrEnum | str | None = None
    tier: TierName | None = None
    metric_name: str | None = None
    observed_value: int | float | str | bool | None = None
    threshold_value: int | float | str | bool | None = None
    gate_name: GateName | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", TierIssueCode(self.code))
        _require_text(self.message, "message")
        tier = None if self.tier is None else TierName(self.tier)
        gate_name = None if self.gate_name is None else GateName(self.gate_name)
        object.__setattr__(self, "tier", tier)
        object.__setattr__(self, "gate_name", gate_name)


@dataclass(frozen=True, slots=True)
class TierFamilyDecision:
    """Root downstream tier decision for one metric family."""

    family: enum.StrEnum | str
    status: TierStatus
    supported_tiers: tuple[TierName, ...]
    best_supported_tier: TierName | None
    issues: tuple[TierIssue, ...] = ()

    def __post_init__(self) -> None:
        _require_text(str(self.family), "family")
        object.__setattr__(self, "status", TierStatus(self.status))
        object.__setattr__(
            self, "supported_tiers", tuple(TierName(t) for t in self.supported_tiers)
        )
        best = None if self.best_supported_tier is None else TierName(self.best_supported_tier)
        object.__setattr__(self, "best_supported_tier", best)
        object.__setattr__(self, "issues", tuple(self.issues))


@dataclass(frozen=True, slots=True)
class TierLeakageDecision:
    """Sample-local leakage admissibility applied to tier policy selection."""

    observed_max_severity: str
    admissible_tiers: tuple[TierName, ...]
    rejected_tiers: tuple[TierName, ...]

    def __post_init__(self) -> None:
        _require_text(self.observed_max_severity, "observed_max_severity")
        object.__setattr__(
            self,
            "admissible_tiers",
            tuple(TierName(tier) for tier in self.admissible_tiers),
        )
        object.__setattr__(
            self,
            "rejected_tiers",
            tuple(TierName(tier) for tier in self.rejected_tiers),
        )


@dataclass(frozen=True, slots=True)
class TierDecisionBundle:
    """Root downstream tier-band decision for one sample."""

    status: TierStatus
    selected_tier: TierName | None
    family_decisions: tuple[TierFamilyDecision, ...]
    leakage_decision: TierLeakageDecision | None = None
    issues: tuple[TierIssue, ...] = ()
    sample_id: str | None = None
    split: SampleSplit | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", TierStatus(self.status))
        selected = None if self.selected_tier is None else TierName(self.selected_tier)
        split = None if self.split is None else SampleSplit(self.split)
        object.__setattr__(self, "selected_tier", selected)
        object.__setattr__(self, "family_decisions", tuple(self.family_decisions))
        object.__setattr__(self, "leakage_decision", self.leakage_decision)
        object.__setattr__(self, "issues", tuple(self.issues))
        object.__setattr__(self, "split", split)


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-blank string.")


def _require_unique(values: tuple[object, ...], name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{name} cannot contain duplicates.")


__all__ = [
    "CheckpointAdmission",
    "DroppedManifestEntry",
    "DroppedSample",
    "DroppedSampleGateSnapshot",
    "DroppedSamplePoseSnapshot",
    "DroppedSampleSourceSnapshot",
    "GateDecision",
    "GateDecisionBundle",
    "GateDropIssueCode",
    "GateIssueCode",
    "GateName",
    "GateStatus",
    "PassedManifestEntry",
    "PoseTruth",
    "PreparedSample",
    "GateDropStage",
    "SourceTruth",
    "TierDecisionBundle",
    "TierFamilyDecision",
    "TierIssue",
    "TierIssueCode",
    "TierLeakageDecision",
    "TierStatus",
]
