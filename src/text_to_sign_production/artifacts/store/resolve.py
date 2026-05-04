"""Resolution helpers for physical artifact store paths."""

from __future__ import annotations

from pathlib import Path

from text_to_sign_production.artifacts.store.topology import ArtifactTopology
from text_to_sign_production.artifacts.store.types import (
    ArtifactPathRef,
    ManifestPathRef,
    SamplePathRef,
)


def resolve_repo_relative(
    topology: ArtifactTopology,
    relative_path: str | Path,
) -> ArtifactPathRef:
    """Resolve a path relative to the repository root."""

    path = _require_relative_path(relative_path)
    return ArtifactPathRef(topology.repo_root / path)


def resolve_samples_relative(
    topology: ArtifactTopology,
    relative_path: str | Path,
) -> SamplePathRef:
    """Resolve a path relative to the physical samples root."""

    path = _require_relative_path(relative_path)
    return SamplePathRef(topology.samples_root / path)


def resolve_manifests_relative(
    topology: ArtifactTopology,
    relative_path: str | Path,
) -> ManifestPathRef:
    """Resolve a path relative to the physical manifests root."""

    path = _require_relative_path(relative_path)
    return ManifestPathRef(topology.manifests_root / path)


def _require_relative_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        raise ValueError(f"Expected a relative path, got absolute path: {resolved}")
    if not resolved.parts:
        raise ValueError("Expected a non-empty relative path.")
    if ".." in resolved.parts:
        raise ValueError(f"Relative path must not contain parent directory references: {resolved}")
    return resolved


__all__ = [
    "resolve_manifests_relative",
    "resolve_repo_relative",
    "resolve_samples_relative",
]
