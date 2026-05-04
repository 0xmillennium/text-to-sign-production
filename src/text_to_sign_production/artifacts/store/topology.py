"""Physical artifact store topology."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from text_to_sign_production.artifacts.store.types import (
    ArchiveMemberPathRef,
    ArchivePathRef,
    ArtifactPathRef,
    ManifestPathRef,
    SamplePathRef,
    SampleStatus,
    SplitName,
    TierMembership,
    TierName,
    split_name_from_sample_split,
)
from text_to_sign_production.foundation.roots import RepoRoots


@dataclass(frozen=True, slots=True)
class AssetsTopology:
    """Physical assets topology."""

    how2sign_root: Path
    how2sign_bfh_keypoints_root: Path
    how2sign_translations_root: Path

    def translation_csv(self, split: SplitName | str) -> ArtifactPathRef:
        split = split_name_from_sample_split(split)
        path = self.how2sign_translations_root / f"how2sign_realigned_{split.value}.csv"
        return ArtifactPathRef(path)

    def keypoint_split_root(self, split: SplitName | str) -> ArtifactPathRef:
        split = split_name_from_sample_split(split)
        path = self.how2sign_bfh_keypoints_root / f"{split.value}_2D_keypoints"
        return ArtifactPathRef(path)

    def keypoint_archive(self, split: SplitName | str) -> ArchivePathRef:
        split = split_name_from_sample_split(split)
        path = self.how2sign_bfh_keypoints_root / f"{split.value}_2D_keypoints.tar.zst"
        return ArchivePathRef(path)

    def keypoint_json_dir(self, split: SplitName | str) -> ArtifactPathRef:
        path = self.keypoint_split_root(split).path / "openpose_output" / "json"
        return ArtifactPathRef(path)

    def keypoint_video_dir(self, split: SplitName | str) -> ArtifactPathRef:
        path = self.keypoint_split_root(split).path / "openpose_output" / "video"
        return ArtifactPathRef(path)


@dataclass(frozen=True, slots=True)
class ManifestsTopology:
    """Physical manifests topology."""

    untiered_root: Path
    tiered_root: Path

    def untiered_passed_manifest(self, split: SplitName | str) -> ManifestPathRef:
        split = split_name_from_sample_split(split)
        return ManifestPathRef(self.untiered_root / "passed" / f"{split.value}.jsonl")

    def untiered_dropped_manifest(self, split: SplitName | str) -> ManifestPathRef:
        split = split_name_from_sample_split(split)
        return ManifestPathRef(self.untiered_root / "dropped" / f"{split.value}.jsonl")

    def tiered_manifest(
        self,
        tier: TierName | str,
        membership: TierMembership | str,
        split: SplitName | str,
    ) -> ManifestPathRef:
        split = split_name_from_sample_split(split)
        path = self.tiered_root / str(tier) / str(membership) / f"{split.value}.jsonl"
        return ManifestPathRef(path)


@dataclass(frozen=True, slots=True)
class SamplesTopology:
    """Physical samples topology."""

    passed_root: Path
    dropped_root: Path

    def passed_split_dir(self, split: SplitName | str) -> ArtifactPathRef:
        split = split_name_from_sample_split(split)
        return ArtifactPathRef(self.passed_root / split.value)

    def dropped_split_dir(self, split: SplitName | str) -> ArtifactPathRef:
        split = split_name_from_sample_split(split)
        return ArtifactPathRef(self.dropped_root / split.value)

    def sample_dir(self, status: SampleStatus, split: SplitName | str) -> ArtifactPathRef:
        if status is SampleStatus.PASSED:
            return self.passed_split_dir(split)
        return self.dropped_split_dir(split)

    def sample_file(
        self,
        status: SampleStatus,
        split: SplitName | str,
        sample_id: str,
    ) -> SamplePathRef:
        return SamplePathRef(self.sample_dir(status, split).path / _sample_filename(sample_id))

    def split_archive(self, status: SampleStatus, split: SplitName | str) -> ArchivePathRef:
        split = split_name_from_sample_split(split)
        return ArchivePathRef(self._status_root(status) / f"{split.value}.tar.zst")

    def split_extract_root(self, status: SampleStatus) -> ArtifactPathRef:
        return ArtifactPathRef(self._status_root(status))

    def archive_member(self, split: SplitName | str, sample_id: str) -> ArchiveMemberPathRef:
        split = split_name_from_sample_split(split)
        path = PurePosixPath(split.value) / _sample_filename(sample_id)
        return ArchiveMemberPathRef(path)

    def _status_root(self, status: SampleStatus) -> Path:
        if status is SampleStatus.PASSED:
            return self.passed_root
        return self.dropped_root


@dataclass(frozen=True, slots=True)
class ModelsTopology:
    """Physical models topology."""

    root: Path

    def checkpoint_file(self, relative_path: str | Path) -> ArtifactPathRef:
        path = _checkpoint_relative_path(relative_path)
        return ArtifactPathRef(self.root / path)

    def checkpoint_archive(self, relative_path: str | Path) -> ArchivePathRef:
        path = _checkpoint_relative_path(relative_path)
        return ArchivePathRef(self.root / path.with_suffix(f"{path.suffix}.zst"))


@dataclass(frozen=True, slots=True)
class EvaluationsTopology:
    """Physical evaluations topology."""

    root: Path


@dataclass(frozen=True, slots=True)
class ReportsTopology:
    """Physical reports topology."""

    root: Path
    samples_root: Path
    tiers_root: Path
    modeling_root: Path
    visualization_root: Path
    evaluation_root: Path


@dataclass(frozen=True, slots=True)
class ArtifactTopology:
    """Physical artifact store topology."""

    repo_root: Path
    assets_root: Path
    manifests_root: Path
    samples_root: Path
    models_root: Path
    evaluations_root: Path
    reports_root: Path
    assets: AssetsTopology
    manifests: ManifestsTopology
    samples: SamplesTopology
    models: ModelsTopology
    evaluations: EvaluationsTopology
    reports: ReportsTopology


@dataclass(frozen=True, slots=True)
class ArtifactStores:
    """Runtime and Drive physical projections for the same artifact topology."""

    runtime: ArtifactTopology
    drive: ArtifactTopology


def build_artifact_topology(roots: RepoRoots) -> ArtifactTopology:
    """Build the physical artifact topology from repository roots."""

    assets = AssetsTopology(
        how2sign_root=roots.assets_root / "how2sign",
        how2sign_bfh_keypoints_root=roots.assets_root / "how2sign" / "bfh_keypoints",
        how2sign_translations_root=roots.assets_root / "how2sign" / "translations",
    )
    manifests = ManifestsTopology(
        untiered_root=roots.manifests_root / "untiered",
        tiered_root=roots.manifests_root / "tiered",
    )
    samples = SamplesTopology(
        passed_root=roots.samples_root / "passed",
        dropped_root=roots.samples_root / "dropped",
    )
    models = ModelsTopology(root=roots.models_root)
    evaluations = EvaluationsTopology(root=roots.evaluations_root)
    reports = ReportsTopology(
        root=roots.reports_root,
        samples_root=roots.reports_root / "samples",
        tiers_root=roots.reports_root / "tiers",
        modeling_root=roots.reports_root / "modeling",
        visualization_root=roots.reports_root / "visualization",
        evaluation_root=roots.reports_root / "evaluation",
    )
    return ArtifactTopology(
        repo_root=roots.repo_root,
        assets_root=roots.assets_root,
        manifests_root=roots.manifests_root,
        samples_root=roots.samples_root,
        models_root=roots.models_root,
        evaluations_root=roots.evaluations_root,
        reports_root=roots.reports_root,
        assets=assets,
        manifests=manifests,
        samples=samples,
        models=models,
        evaluations=evaluations,
        reports=reports,
    )


def build_artifact_stores(runtime_roots: RepoRoots, drive_roots: RepoRoots) -> ArtifactStores:
    """Build runtime and Drive artifact topology projections."""

    return ArtifactStores(
        runtime=build_artifact_topology(runtime_roots),
        drive=build_artifact_topology(drive_roots),
    )


def sample_manifest_relative_path(
    status: SampleStatus,
    split: SplitName | str,
    sample_id: str,
) -> Path:
    """Return the canonical samples-root-relative manifest sample path."""
    split = split_name_from_sample_split(split)
    return Path(status.value) / split.value / _sample_filename(sample_id)


def _sample_filename(sample_id: str) -> str:
    if not sample_id or Path(sample_id).name != sample_id or sample_id in {".", ".."}:
        raise ValueError(f"Sample id must be a concrete file stem: {sample_id!r}")
    return f"{sample_id}.npz"


def _checkpoint_relative_path(relative_path: str | Path) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        raise ValueError(f"Checkpoint path must be models-relative, got absolute path: {path}")
    if not path.parts:
        raise ValueError("Checkpoint path must be a non-empty models-relative path.")
    if ".." in path.parts:
        raise ValueError(f"Checkpoint path must not contain parent references: {path}")
    if path.name in {"", ".", ".."} or path.suffix != ".pt" or not path.stem:
        raise ValueError(f"Checkpoint path must end in a concrete .pt file name: {path}")
    return path


__all__ = [
    "ArtifactStores",
    "ArtifactTopology",
    "AssetsTopology",
    "EvaluationsTopology",
    "ManifestsTopology",
    "ModelsTopology",
    "ReportsTopology",
    "SamplesTopology",
    "build_artifact_stores",
    "build_artifact_topology",
    "sample_manifest_relative_path",
]
