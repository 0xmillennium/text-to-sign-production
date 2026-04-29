"""Project layout plus neutral repository/data path resolution helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from text_to_sign_production.data.constants import (
    SPLIT_TO_KEYPOINT_DIR,
    SPLIT_TO_TRANSLATION_FILE,
)


def _resolve_repo_root() -> Path:
    override = os.environ.get("T2SP_REPO_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class ProjectLayout:
    """Canonical mirrored project layout rooted at a repository checkout."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())

    @property
    def data_root(self) -> Path:
        return self.root / "data"

    @property
    def raw_root(self) -> Path:
        return self.data_root / "raw"

    @property
    def interim_root(self) -> Path:
        return self.data_root / "interim"

    @property
    def processed_root(self) -> Path:
        return self.data_root / "processed"

    @property
    def artifacts_root(self) -> Path:
        return self.data_root / "artifacts"

    @property
    def how2sign_root(self) -> Path:
        return self.raw_root / "how2sign"

    @property
    def how2sign_translations_root(self) -> Path:
        return self.how2sign_root / "translations"

    @property
    def how2sign_bfh_keypoints_root(self) -> Path:
        return self.how2sign_root / "bfh_keypoints"

    @property
    def raw_manifests_root(self) -> Path:
        return self.interim_root / "raw_manifests"

    @property
    def normalized_manifests_root(self) -> Path:
        return self.interim_root / "normalized_manifests"

    @property
    def filtered_manifests_root(self) -> Path:
        return self.interim_root / "filtered_manifests"

    @property
    def interim_reports_root(self) -> Path:
        return self.interim_root / "reports"

    @property
    def processed_v1_root(self) -> Path:
        return self.processed_root / "v1"

    @property
    def processed_v1_manifests_root(self) -> Path:
        return self.processed_v1_root / "manifests"

    @property
    def processed_v1_samples_root(self) -> Path:
        return self.processed_v1_root / "samples"

    @property
    def processed_v1_reports_root(self) -> Path:
        return self.processed_v1_root / "reports"

    @property
    def dataset_artifacts_root(self) -> Path:
        return self.artifacts_root / "dataset"

    @property
    def visualization_artifacts_root(self) -> Path:
        return self.artifacts_root / "visualization"

    @property
    def base_artifacts_root(self) -> Path:
        return self.artifacts_root / "base"

    def how2sign_translation_file(self, split: str) -> Path:
        return self.how2sign_translations_root / SPLIT_TO_TRANSLATION_FILE[_require_split(split)]

    def raw_bfh_keypoints_split_root(self, split: str) -> Path:
        split_dir = SPLIT_TO_KEYPOINT_DIR[_require_split(split)]
        return self.how2sign_bfh_keypoints_root / split_dir / "openpose_output"

    def raw_bfh_keypoints_archive(self, split: str) -> Path:
        return self.how2sign_bfh_keypoints_root / f"{_require_split(split)}_2D_keypoints.tar.zst"

    def processed_manifest_path(self, split: str) -> Path:
        return self.processed_v1_manifests_root / f"{_require_split(split)}.jsonl"

    def processed_samples_split_root(self, split: str) -> Path:
        return self.processed_v1_samples_root / _require_split(split)

    def processed_sample_archive(self, split: str) -> Path:
        return (
            self.processed_v1_samples_root
            / f"dataset_build_samples_{_require_split(split)}.tar.zst"
        )

    def processed_manifests_reports_archive(self) -> Path:
        return self.processed_v1_root / "dataset_build_manifests_reports.tar.zst"

    def base_run_root(self, run_name: str) -> Path:
        resolved_run_name = _require_filename_token(run_name, label="run_name")
        return self.base_artifacts_root / resolved_run_name

    def base_run_archive(self, run_name: str) -> Path:
        resolved_run_name = _require_filename_token(run_name, label="run_name")
        return self.base_artifacts_root / f"{resolved_run_name}.tar.zst"


def _require_split(split: str) -> str:
    from text_to_sign_production.data.constants import SPLITS

    if split not in SPLITS:
        expected = ", ".join(SPLITS)
        raise ValueError(f"Unsupported split {split!r}; expected one of: {expected}.")
    return split


def _require_filename_token(value: str, *, label: str) -> str:
    resolved = value.strip()
    if not resolved:
        raise ValueError(f"{label} must not be blank.")
    if Path(resolved).name != resolved:
        raise ValueError(f"{label} must be a single path component: {value!r}")
    return resolved


DEFAULT_REPO_ROOT: Final[Path] = _resolve_repo_root()
DEFAULT_DATA_ROOT: Final[Path] = ProjectLayout(DEFAULT_REPO_ROOT).data_root
_CANONICAL_REPO_ROOT: Final[Path] = DEFAULT_REPO_ROOT
_CANONICAL_DATA_ROOT: Final[Path] = DEFAULT_DATA_ROOT


def _resolved_root(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _effective_repo_root(repo_root: str | Path | None) -> Path:
    return _resolved_root(DEFAULT_REPO_ROOT if repo_root is None else repo_root)


def _effective_data_root(data_root: str | Path | None) -> Path:
    if data_root is not None:
        return _resolved_root(data_root)
    if DEFAULT_DATA_ROOT == _CANONICAL_DATA_ROOT and DEFAULT_REPO_ROOT != _CANONICAL_REPO_ROOT:
        return _resolved_root(DEFAULT_REPO_ROOT / "data")
    return _resolved_root(DEFAULT_DATA_ROOT)


def _contains_parent_traversal(path: Path) -> bool:
    return ".." in path.parts


def _without_data_prefix(path: Path, *, allow_repo_prefix: bool) -> Path:
    if not path.parts or path.parts[0] != "data":
        return path
    if not allow_repo_prefix:
        raise ValueError(f"Repo-prefixed data path is not allowed: {path}")
    if len(path.parts) == 1:
        return Path(".")
    return Path(*path.parts[1:])


def _ensure_under_root(path: Path, root: Path, *, label: str) -> Path:
    if not path.is_relative_to(root):
        raise ValueError(f"{label} must stay under {root}: {path}")
    return path


def resolve_repo_path(
    path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> Path:
    """Resolve an absolute path or repo-relative path on the local filesystem."""

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    root = _effective_repo_root(repo_root)
    resolved = (root / candidate).resolve()
    return _ensure_under_root(resolved, root, label="Repo-relative path")


def repo_relative_path(
    path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> str:
    """Return a stable repo-relative POSIX path string."""

    root = _effective_repo_root(repo_root)
    resolved = Path(path).expanduser().resolve()
    return _ensure_under_root(resolved, root, label="Path").relative_to(root).as_posix()


def resolve_data_root_path(
    path: str | Path,
    *,
    data_root: str | Path | None = None,
    allow_repo_prefix: bool = True,
) -> Path:
    """Resolve a path under the runtime data root.

    Relative values may be data-root-relative (``processed/v1/...``) or,
    when ``allow_repo_prefix`` is true, repo-style values (``data/processed/v1/...``).
    Absolute canonical data-layout constants are remapped under ``data_root``.
    """

    root = _effective_data_root(data_root)
    candidate = Path(path).expanduser()
    if _contains_parent_traversal(candidate):
        raise ValueError(f"Data path must not contain parent traversal: {path}")

    if candidate.is_absolute():
        resolved = candidate.resolve()
        if resolved.is_relative_to(root):
            return resolved

        canonical_root = _CANONICAL_DATA_ROOT.resolve()
        if resolved.is_relative_to(canonical_root):
            remapped = (root / resolved.relative_to(canonical_root)).resolve()
            return _ensure_under_root(remapped, root, label="Data path")

        raise ValueError(f"Data path must stay under {root}: {path}")

    relative = _without_data_prefix(candidate, allow_repo_prefix=allow_repo_prefix)
    resolved = (root / relative).resolve()
    return _ensure_under_root(resolved, root, label="Data path")


def data_root_relative_path(
    path: str | Path,
    *,
    data_root: str | Path | None = None,
    allow_repo_prefix: bool = True,
) -> Path:
    """Return a path relative to the runtime data root."""

    root = _effective_data_root(data_root)
    resolved = resolve_data_root_path(
        path,
        data_root=root,
        allow_repo_prefix=allow_repo_prefix,
    )
    return resolved.relative_to(root)


def _resolve_allowed_root(allowed_root: str | Path, *, data_root: Path) -> Path:
    boundary = Path(allowed_root).expanduser()
    if boundary.is_absolute():
        try:
            return resolve_data_root_path(boundary, data_root=data_root)
        except ValueError:
            return boundary.resolve()
    return resolve_data_root_path(boundary, data_root=data_root)


def resolve_manifest_path(
    value: str | Path,
    *,
    data_root: str | Path | None = None,
    allowed_root: str | Path | None = None,
    allow_absolute: bool = False,
    allow_repo_prefix: bool = True,
) -> Path:
    """Resolve a path value read from a manifest into a runtime absolute path."""

    raw_value = str(value).strip()
    if not raw_value:
        raise ValueError("Manifest path must be a non-empty relative path.")

    root = _effective_data_root(data_root)
    candidate = Path(raw_value).expanduser()
    if _contains_parent_traversal(candidate):
        raise ValueError(f"Manifest path must not contain parent traversal: {raw_value}")

    if candidate.is_absolute():
        if not allow_absolute:
            raise ValueError(f"Manifest path must be relative: {raw_value}")
        resolved = candidate.resolve()
    else:
        resolved = resolve_data_root_path(
            candidate,
            data_root=root,
            allow_repo_prefix=allow_repo_prefix,
        )

    if allowed_root is not None:
        boundary = _resolve_allowed_root(allowed_root, data_root=root)
        _ensure_under_root(resolved, boundary, label="Manifest path")
    return resolved


__all__ = [
    "DEFAULT_DATA_ROOT",
    "DEFAULT_REPO_ROOT",
    "ProjectLayout",
    "data_root_relative_path",
    "repo_relative_path",
    "resolve_data_root_path",
    "resolve_manifest_path",
    "resolve_repo_path",
]
