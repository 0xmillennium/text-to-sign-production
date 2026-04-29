"""Canonical Dataset artifact reference paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.core.paths import ProjectLayout
from text_to_sign_production.data.constants import (
    SPLIT_TO_KEYPOINT_DIR,
    SPLIT_TO_TRANSLATION_FILE,
    SPLITS,
)


@dataclass(frozen=True, slots=True)
class DatasetArtifactLayout:
    """Canonical data-lifecycle artifact paths for Dataset Build outputs and inputs."""

    project_layout: ProjectLayout

    @classmethod
    def from_project_root(cls, project_root: Path | str) -> DatasetArtifactLayout:
        return cls(ProjectLayout(Path(project_root)))

    @property
    def data_root(self) -> Path:
        return self.project_layout.data_root

    @property
    def raw_root(self) -> Path:
        return self.project_layout.raw_root

    @property
    def interim_root(self) -> Path:
        return self.project_layout.interim_root

    @property
    def processed_root(self) -> Path:
        return self.project_layout.processed_root

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
    def processed_v1_root(self) -> Path:
        return self.processed_root / "v1"

    @property
    def processed_v1_manifests_root(self) -> Path:
        return self.processed_v1_root / "manifests"

    @property
    def processed_v1_samples_root(self) -> Path:
        return self.processed_v1_root / "samples"

    def how2sign_translation_file(self, split: str) -> Path:
        return self.how2sign_translations_root / SPLIT_TO_TRANSLATION_FILE[_require_split(split)]

    def raw_bfh_keypoints_split_root(self, split: str) -> Path:
        split_dir = SPLIT_TO_KEYPOINT_DIR[_require_split(split)]
        return self.how2sign_bfh_keypoints_root / split_dir

    def raw_bfh_keypoints_openpose_root(self, split: str) -> Path:
        return self.raw_bfh_keypoints_split_root(split) / "openpose_output"

    def raw_bfh_keypoints_archive(self, split: str) -> Path:
        return self.how2sign_bfh_keypoints_root / f"{_require_split(split)}_2D_keypoints.tar.zst"

    def raw_manifest_path(self, split: str) -> Path:
        return self.raw_manifests_root / f"raw_{_require_split(split)}.jsonl"

    def normalized_manifest_path(self, split: str) -> Path:
        return self.normalized_manifests_root / f"normalized_{_require_split(split)}.jsonl"

    def filtered_manifest_path(self, split: str) -> Path:
        return self.filtered_manifests_root / f"filtered_{_require_split(split)}.jsonl"

    def processed_manifest_path(self, split: str) -> Path:
        return self.processed_v1_manifests_root / f"{_require_split(split)}.jsonl"

    def processed_samples_split_root(self, split: str) -> Path:
        return self.processed_v1_samples_root / _require_split(split)

    def processed_sample_archive(self, split: str) -> Path:
        return (
            self.processed_v1_samples_root
            / f"dataset_build_samples_{_require_split(split)}.tar.zst"
        )

    def processed_tier_manifest_path(self, tier: str, split: str) -> Path:
        resolved_tier = tier.strip()
        if not resolved_tier:
            raise ValueError("Quality tier must not be blank.")
        if "/" in resolved_tier or "\\" in resolved_tier or resolved_tier in {".", ".."}:
            raise ValueError(f"Quality tier must be a single path component: {tier!r}")
        return self.processed_v1_root / "tiers" / resolved_tier / f"{_require_split(split)}.jsonl"


def _require_split(split: str) -> str:
    if split not in SPLITS:
        expected = ", ".join(SPLITS)
        raise ValueError(f"Unsupported split {split!r}; expected one of: {expected}.")
    return split


__all__ = ["DatasetArtifactLayout"]
