"""Public surface for repository-wide core primitives."""

from __future__ import annotations

from text_to_sign_production.core.ids import VALID_SAMPLE_SPLITS, SampleSplit
from text_to_sign_production.core.models import (
    CheckpointAdmission,
    DroppedManifestEntry,
    GateDecision,
    GateDecisionBundle,
    GateDropIssueCode,
    GateDropStage,
    GateIssueCode,
    GateName,
    GateStatus,
    PassedManifestEntry,
    PoseTruth,
    PreparedSample,
    SourceTruth,
    TierDecisionBundle,
    TierFamilyDecision,
    TierIssue,
    TierIssueCode,
    TierStatus,
)
from text_to_sign_production.core.paths import (
    RepoRoots,
    build_repo_roots,
    discover_repo_root,
    load_repo_roots,
)

__all__ = [
    "RepoRoots",
    "CheckpointAdmission",
    "DroppedManifestEntry",
    "GateDropIssueCode",
    "GateDecision",
    "GateDecisionBundle",
    "GateIssueCode",
    "GateName",
    "GateStatus",
    "PassedManifestEntry",
    "PoseTruth",
    "PreparedSample",
    "SampleSplit",
    "GateDropStage",
    "SourceTruth",
    "TierDecisionBundle",
    "TierFamilyDecision",
    "TierIssue",
    "TierIssueCode",
    "TierStatus",
    "VALID_SAMPLE_SPLITS",
    "build_repo_roots",
    "discover_repo_root",
    "load_repo_roots",
]
