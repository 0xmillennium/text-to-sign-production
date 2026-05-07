"""Public surface for repository-wide core primitives."""

from __future__ import annotations

from text_to_sign_production.core.ids import SampleSplit, VALID_SAMPLE_SPLITS
from text_to_sign_production.core.paths import (
    RepoRoots,
    build_repo_roots,
    discover_repo_root,
    load_repo_roots,
)

__all__ = [
    "RepoRoots",
    "SampleSplit",
    "VALID_SAMPLE_SPLITS",
    "build_repo_roots",
    "discover_repo_root",
    "load_repo_roots",
]
