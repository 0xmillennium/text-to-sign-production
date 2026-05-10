"""Physical artifact store topology."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from text_to_sign_production.artifacts.store.types import (
    ArchiveMemberPathRef,
    ArchivePathRef,
    ArtifactPathRef,
    ManifestPathRef,
    ReportPathRef,
    SamplePathRef,
    sample_split_from_value,
    sample_status_from_value,
    tier_membership_from_value,
    tier_name_from_value,
)
from text_to_sign_production.core.ids import (
    SampleSplit,
    SampleStatus,
    TierMembership,
    TierName,
)
from text_to_sign_production.core.paths import RepoRoots


@dataclass(frozen=True, slots=True)
class AssetsTopology:
    """Physical assets topology."""

    how2sign_root: Path
    how2sign_bfh_keypoints_root: Path
    how2sign_translations_root: Path

    def translation_csv(self, split: SampleSplit | str) -> ArtifactPathRef:
        split = sample_split_from_value(split)
        path = self.how2sign_translations_root / f"how2sign_realigned_{split.value}.csv"
        return ArtifactPathRef(path)

    def keypoint_split_root(self, split: SampleSplit | str) -> ArtifactPathRef:
        split = sample_split_from_value(split)
        path = self.how2sign_bfh_keypoints_root / f"{split.value}_2D_keypoints"
        return ArtifactPathRef(path)

    def keypoint_archive(self, split: SampleSplit | str) -> ArchivePathRef:
        split = sample_split_from_value(split)
        path = self.how2sign_bfh_keypoints_root / f"{split.value}_2D_keypoints.tar.zst"
        return ArchivePathRef(path)

    def keypoint_extract_root(self) -> ArtifactPathRef:
        """Root for extracting keypoint archives; archive members are relative to this path."""
        return ArtifactPathRef(self.how2sign_bfh_keypoints_root)

    def keypoint_json_dir(self, split: SampleSplit | str) -> ArtifactPathRef:
        path = self.keypoint_split_root(split).path / "openpose_output" / "json"
        return ArtifactPathRef(path)

    def keypoint_video_dir(self, split: SampleSplit | str) -> ArtifactPathRef:
        path = self.keypoint_split_root(split).path / "openpose_output" / "video"
        return ArtifactPathRef(path)


@dataclass(frozen=True, slots=True)
class ManifestsTopology:
    """Physical manifests topology for checkpoint inputs and tier-stage outputs."""

    untiered_root: Path
    tiered_root: Path

    def untiered_passed_manifest(self, split: SampleSplit | str) -> ManifestPathRef:
        split = sample_split_from_value(split)
        return ManifestPathRef(self.untiered_root / "passed" / f"{split.value}.jsonl")

    def untiered_dropped_manifest(self, split: SampleSplit | str) -> ManifestPathRef:
        split = sample_split_from_value(split)
        return ManifestPathRef(self.untiered_root / "dropped" / f"{split.value}.jsonl")

    def tiered_manifest(
        self,
        tier: TierName | str,
        membership: TierMembership | str,
        split: SampleSplit | str,
    ) -> ManifestPathRef:
        tier = tier_name_from_value(tier)
        membership = tier_membership_from_value(membership)
        split = sample_split_from_value(split)
        path = self.tiered_root / tier.value / membership.value / f"{split.value}.jsonl"
        return ManifestPathRef(path)


@dataclass(frozen=True, slots=True)
class SamplesTopology:
    """Physical PreparedSample checkpoint payload topology."""

    passed_root: Path
    dropped_root: Path

    def passed_split_dir(self, split: SampleSplit | str) -> ArtifactPathRef:
        split = sample_split_from_value(split)
        return ArtifactPathRef(self.passed_root / split.value)

    def dropped_split_dir(self, split: SampleSplit | str) -> ArtifactPathRef:
        split = sample_split_from_value(split)
        return ArtifactPathRef(self.dropped_root / split.value)

    def sample_dir(self, status: SampleStatus | str, split: SampleSplit | str) -> ArtifactPathRef:
        status = sample_status_from_value(status)
        if status is SampleStatus.PASSED:
            return self.passed_split_dir(split)
        return self.dropped_split_dir(split)

    def sample_file(
        self,
        status: SampleStatus | str,
        split: SampleSplit | str,
        sample_id: str,
    ) -> SamplePathRef:
        """Return the PreparedSample checkpoint payload path for one sample."""
        return SamplePathRef(self.sample_dir(status, split).path / _sample_filename(sample_id))

    def split_archive(self, status: SampleStatus | str, split: SampleSplit | str) -> ArchivePathRef:
        split = sample_split_from_value(split)
        return ArchivePathRef(self._status_root(status) / f"{split.value}.tar.zst")

    def split_extract_root(self, status: SampleStatus | str) -> ArtifactPathRef:
        return ArtifactPathRef(self._status_root(status))

    def archive_member(self, split: SampleSplit | str, sample_id: str) -> ArchiveMemberPathRef:
        split = sample_split_from_value(split)
        path = PurePosixPath(split.value) / _sample_filename(sample_id)
        return ArchiveMemberPathRef(path)

    def _status_root(self, status: SampleStatus | str) -> Path:
        status = sample_status_from_value(status)
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
    """Physical report topology; samples paths are samples-stage projections only."""

    root: Path
    samples_root: Path
    tiers_root: Path
    modeling_root: Path
    visualization_root: Path
    evaluation_root: Path

    def samples_summary(self) -> ReportPathRef:
        return ReportPathRef(self.samples_root / "summary.md")

    def samples_processing_summary(self) -> ReportPathRef:
        return ReportPathRef(self.samples_root / "processing" / "summary.jsonl")

    def samples_processing_detail(self) -> ReportPathRef:
        return ReportPathRef(self.samples_root / "processing" / "detail.jsonl")

    def samples_gate_summary(self) -> ReportPathRef:
        return ReportPathRef(self.samples_root / "gates" / "summary.jsonl")

    def samples_gate_detail(self) -> ReportPathRef:
        return ReportPathRef(self.samples_root / "gates" / "detail.jsonl")

    def samples_source_issue_summary(self) -> ReportPathRef:
        return ReportPathRef(self.samples_root / "source_issues" / "summary.jsonl")

    def samples_source_issue_detail(self) -> ReportPathRef:
        return ReportPathRef(self.samples_root / "source_issues" / "detail.jsonl")

    def samples_index(self) -> ReportPathRef:
        return ReportPathRef(self.samples_root / "index.json")

    def tiers_summary(self) -> ReportPathRef:
        """Tier-stage quality summary report."""
        return ReportPathRef(self.tiers_root / "summary.md")

    def tiers_calibration(self) -> ReportPathRef:
        """Tier-stage calibration and policy report."""
        return ReportPathRef(self.tiers_root / "calibration.md")

    def tiers_decision_detail(self) -> ReportPathRef:
        """Tier-stage per-sample quality and decision detail report."""
        return ReportPathRef(self.tiers_root / "decisions" / "detail.jsonl")

    def tiers_calibration_surfaces(self) -> ReportPathRef:
        """Tier-stage aggregate calibration surface payload."""
        return ReportPathRef(self.tiers_root / "calibration" / "surfaces.json")

    def tiers_calibration_detail(self) -> ReportPathRef:
        """Tier-stage detailed calibration payload."""
        return ReportPathRef(self.tiers_root / "calibration" / "detail.json")

    def tiers_index(self) -> ReportPathRef:
        """Tier-stage report index."""
        return ReportPathRef(self.tiers_root / "index.json")


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
    status: SampleStatus | str,
    split: SampleSplit | str,
    sample_id: str,
) -> Path:
    """Return the canonical samples-root-relative manifest sample path."""
    status = sample_status_from_value(status)
    split = sample_split_from_value(split)
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
