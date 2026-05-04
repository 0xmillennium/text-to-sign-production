"""Repository root discovery and top-level root derivation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RepoRoots:
    """Top-level repository roots."""

    repo_root: Path
    assets_root: Path
    manifests_root: Path
    samples_root: Path
    models_root: Path
    evaluations_root: Path
    reports_root: Path


def discover_repo_root(start: Path | None = None) -> Path:
    """Discover the repository root by walking upward to a ``.git`` directory."""

    current = Path.cwd() if start is None else start.expanduser()
    if not current.exists():
        raise FileNotFoundError(f"Repository discovery start path does not exist: {current}")

    current = current.resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if (candidate / ".git").is_dir():
            return candidate

    raise FileNotFoundError(f"Could not discover repository root from: {current}")


def build_repo_roots(repo_root: Path) -> RepoRoots:
    """Build top-level roots from a repository root."""

    root = repo_root.expanduser().resolve()
    return RepoRoots(
        repo_root=root,
        assets_root=root / "assets",
        manifests_root=root / "manifests",
        samples_root=root / "samples",
        models_root=root / "models",
        evaluations_root=root / "evaluations",
        reports_root=root / "reports",
    )


def load_repo_roots(start: Path | None = None) -> RepoRoots:
    """Discover the repository root and build its top-level roots."""

    return build_repo_roots(discover_repo_root(start))
