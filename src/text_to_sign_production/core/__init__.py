"""Public surface for repository-wide core primitives."""

from __future__ import annotations

from text_to_sign_production.core.ids import VALID_SAMPLE_SPLITS, SampleSplit
from text_to_sign_production.core.models import (
    DroppedManifestEntry,
    GateDecision,
    GateDecisionBundle,
    GateIssueCode,
    GateName,
    GateStatus,
    PassedManifestEntry,
    PoseTruth,
    PreparedSample,
    SamplesDropStage,
    SamplesIssueCode,
    SourceTruth,
    TierDecisionBundle,
    TierFamilyDecision,
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
    "DroppedManifestEntry",
    "GateDecision",
    "GateDecisionBundle",
    "GateIssueCode",
    "GateName",
    "GateStatus",
    "PassedManifestEntry",
    "PoseTruth",
    "PreparedSample",
    "SampleSplit",
    "SamplesDropStage",
    "SamplesIssueCode",
    "SourceTruth",
    "TierDecisionBundle",
    "TierFamilyDecision",
    "TierStatus",
    "VALID_SAMPLE_SPLITS",
    "build_repo_roots",
    "discover_repo_root",
    "load_repo_roots",
]
