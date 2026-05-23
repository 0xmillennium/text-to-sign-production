"""Physical artifact store topology."""

from __future__ import annotations

import re
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
        return ManifestPathRef(self.untiered_root / "passed" / f"{split.value}.json")

    def untiered_dropped_manifest(self, split: SampleSplit | str) -> ManifestPathRef:
        split = sample_split_from_value(split)
        return ManifestPathRef(self.untiered_root / "dropped" / f"{split.value}.json")

    def tiered_manifest(
        self,
        tier: TierName | str,
        membership: TierMembership | str,
        split: SampleSplit | str,
    ) -> ManifestPathRef:
        tier_token = _tier_manifest_token(tier)
        membership = tier_membership_from_value(membership)
        split = sample_split_from_value(split)
        path = self.tiered_root / tier_token / membership.value / f"{split.value}.json"
        return ManifestPathRef(path)


@dataclass(frozen=True, slots=True)
class SamplesTopology:
    """Physical sample payload topology."""

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
        """Return the status-specific sample payload path for one sample."""
        status = sample_status_from_value(status)
        suffix = ".npz" if status is SampleStatus.PASSED else ".json"
        return SamplePathRef(
            self.sample_dir(status, split).path / _sample_filename(sample_id, suffix)
        )

    def passed_sample_file(self, split: SampleSplit | str, sample_id: str) -> SamplePathRef:
        return SamplePathRef(
            self.passed_split_dir(split).path / _sample_filename(sample_id, ".npz")
        )

    def dropped_sample_file(self, split: SampleSplit | str, sample_id: str) -> SamplePathRef:
        return SamplePathRef(
            self.dropped_split_dir(split).path / _sample_filename(sample_id, ".json")
        )

    def split_archive(self, status: SampleStatus | str, split: SampleSplit | str) -> ArchivePathRef:
        split = sample_split_from_value(split)
        return ArchivePathRef(self._status_root(status) / f"{split.value}.tar.zst")

    def split_extract_root(self, status: SampleStatus | str) -> ArtifactPathRef:
        return ArtifactPathRef(self._status_root(status))

    def archive_member(self, split: SampleSplit | str, sample_id: str) -> ArchiveMemberPathRef:
        """Legacy passed PreparedSample archive member helper.

        Always returns the ``.npz`` (PreparedSample) member path.
        Do NOT use for DroppedSample JSON payloads, and do NOT use in any
        code path where passed/dropped status must be distinguished.
        Prefer ``passed_archive_member(...)`` or ``dropped_archive_member(...)``."""
        return self.passed_archive_member(split, sample_id)

    def passed_archive_member(
        self,
        split: SampleSplit | str,
        sample_id: str,
    ) -> ArchiveMemberPathRef:
        split = sample_split_from_value(split)
        path = PurePosixPath(split.value) / _sample_filename(sample_id, ".npz")
        return ArchiveMemberPathRef(path)

    def dropped_archive_member(
        self,
        split: SampleSplit | str,
        sample_id: str,
    ) -> ArchiveMemberPathRef:
        split = sample_split_from_value(split)
        path = PurePosixPath(split.value) / _sample_filename(sample_id, ".json")
        return ArchiveMemberPathRef(path)

    def _status_root(self, status: SampleStatus | str) -> Path:
        status = sample_status_from_value(status)
        if status is SampleStatus.PASSED:
            return self.passed_root
        return self.dropped_root


def _tier_manifest_token(tier: TierName | str) -> str:
    token = tier.value if isinstance(tier, TierName) else str(tier).strip()
    if not re.fullmatch(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", token):
        raise ValueError(f"tier must be a lowercase safe path token: {tier!r}")
    return token


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

    def model_run_root(self, model_key: str, run_name: str) -> ArtifactPathRef:
        model_key = _safe_path_token(model_key, field_name="model_key")
        run_name = _safe_path_token(run_name, field_name="run_name")
        return ArtifactPathRef(self.root / model_key / run_name)

    def model_effective_config_file(self, model_key: str, run_name: str) -> ArtifactPathRef:
        return ArtifactPathRef(
            self.model_run_root(model_key, run_name).path / "effective_config.json"
        )

    def model_research_spec_file(self, model_key: str, run_name: str) -> ArtifactPathRef:
        return ArtifactPathRef(self.model_run_root(model_key, run_name).path / "research_spec.json")

    def model_run_metadata_file(self, model_key: str, run_name: str) -> ArtifactPathRef:
        return ArtifactPathRef(self.model_run_root(model_key, run_name).path / "run_metadata.json")

    def model_runtime_support_manifest_file(
        self,
        model_key: str,
        run_name: str,
    ) -> ArtifactPathRef:
        return ArtifactPathRef(
            self.model_run_root(model_key, run_name).path / "runtime_support_manifest.json"
        )

    def model_stage_artifacts_index_file(self, model_key: str, run_name: str) -> ArtifactPathRef:
        return ArtifactPathRef(
            self.model_run_root(model_key, run_name).path / "stage_artifacts_index.json"
        )

    def model_checkpoints_root(self, model_key: str, run_name: str) -> ArtifactPathRef:
        return ArtifactPathRef(self.model_run_root(model_key, run_name).path / "checkpoints")

    def model_checkpoint_file(
        self,
        model_key: str,
        run_name: str,
        checkpoint_name: str = "last.pt",
    ) -> ArtifactPathRef:
        filename = _safe_filename(
            checkpoint_name,
            field_name="checkpoint_name",
            allowed_suffixes=(".pt",),
        )
        return ArtifactPathRef(self.model_checkpoints_root(model_key, run_name).path / filename)

    def model_checkpoint_archive(
        self,
        model_key: str,
        run_name: str,
        checkpoint_name: str = "last.pt",
    ) -> ArchivePathRef:
        checkpoint = self.model_checkpoint_file(model_key, run_name, checkpoint_name).path
        return ArchivePathRef(checkpoint.with_suffix(f"{checkpoint.suffix}.zst"))

    def model_training_root(self, model_key: str, run_name: str) -> ArtifactPathRef:
        return ArtifactPathRef(self.model_run_root(model_key, run_name).path / "training")

    def model_training_metrics_file(self, model_key: str, run_name: str) -> ArtifactPathRef:
        return ArtifactPathRef(self.model_training_root(model_key, run_name).path / "metrics.jsonl")

    def model_training_summary_file(self, model_key: str, run_name: str) -> ArtifactPathRef:
        return ArtifactPathRef(self.model_training_root(model_key, run_name).path / "summary.json")

    def model_training_log_file(self, model_key: str, run_name: str) -> ArtifactPathRef:
        return ArtifactPathRef(self.model_training_root(model_key, run_name).path / "live.log")

    def model_intermediates_root(self, model_key: str, run_name: str) -> ArtifactPathRef:
        return ArtifactPathRef(self.model_run_root(model_key, run_name).path / "intermediates")

    def model_intermediate_root(
        self,
        model_key: str,
        run_name: str,
        intermediate_name: str,
    ) -> ArtifactPathRef:
        intermediate_name = _safe_path_token(intermediate_name, field_name="intermediate_name")
        return ArtifactPathRef(
            self.model_intermediates_root(model_key, run_name).path / intermediate_name
        )


@dataclass(frozen=True, slots=True)
class EvaluationsTopology:
    """Physical evaluations topology."""

    root: Path

    def generated_pose_root(self) -> ArtifactPathRef:
        return ArtifactPathRef(self.root / "generated_pose")

    def generated_pose_run_root(
        self,
        producer_key: str,
        run_name: str,
    ) -> ArtifactPathRef:
        producer_key = _safe_path_token(producer_key, field_name="producer_key")
        run_name = _safe_path_token(run_name, field_name="run_name")
        return ArtifactPathRef(self.generated_pose_root().path / producer_key / run_name)

    def generated_pose_split_root(
        self,
        producer_key: str,
        run_name: str,
        split: SampleSplit | str,
    ) -> ArtifactPathRef:
        split = sample_split_from_value(split)
        return ArtifactPathRef(
            self.generated_pose_run_root(producer_key, run_name).path / split.value
        )

    def generated_pose_manifest(
        self,
        producer_key: str,
        run_name: str,
        split: SampleSplit | str,
    ) -> ManifestPathRef:
        return ManifestPathRef(
            self.generated_pose_split_root(producer_key, run_name, split).path / "manifest.jsonl"
        )

    def generated_pose_samples_root(
        self,
        producer_key: str,
        run_name: str,
        split: SampleSplit | str,
    ) -> ArtifactPathRef:
        return ArtifactPathRef(
            self.generated_pose_split_root(producer_key, run_name, split).path / "samples"
        )

    def generated_pose_sample_file(
        self,
        producer_key: str,
        run_name: str,
        split: SampleSplit | str,
        sample_id: str,
        generation_index: int = 0,
    ) -> ArtifactPathRef:
        filename = _generated_pose_sample_filename(sample_id, generation_index)
        return ArtifactPathRef(
            self.generated_pose_samples_root(producer_key, run_name, split).path / filename
        )

@dataclass(frozen=True, slots=True)
class ReportsTopology:
    """Physical report topology; samples paths are samples-stage projections only."""

    root: Path
    samples_root: Path
    tiers_root: Path
    modeling_root: Path
    visualization_root: Path

    def samples_summary(self) -> ReportPathRef:
        return ReportPathRef(self.samples_root / "summary.md")

    def samples_processing_summary(self) -> ReportPathRef:
        return ReportPathRef(self.samples_root / "processing" / "summary.md")

    def samples_processing_detail(self) -> ReportPathRef:
        return ReportPathRef(self.samples_root / "processing" / "detail.json")

    def samples_gate_summary(self) -> ReportPathRef:
        return ReportPathRef(self.samples_root / "gates" / "summary.md")

    def samples_gate_detail(self) -> ReportPathRef:
        return ReportPathRef(self.samples_root / "gates" / "detail.json")

    def samples_source_issue_summary(self) -> ReportPathRef:
        return ReportPathRef(self.samples_root / "source_issues" / "summary.md")

    def samples_source_issue_detail(self) -> ReportPathRef:
        return ReportPathRef(self.samples_root / "source_issues" / "detail.json")

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
        return ReportPathRef(self.tiers_root / "decisions" / "detail.json")

    def tiers_calibration_surfaces(self) -> ReportPathRef:
        """Tier-stage aggregate calibration surface payload."""
        return ReportPathRef(self.tiers_root / "calibration" / "surfaces.json")

    def tiers_calibration_detail(self) -> ReportPathRef:
        """Tier-stage detailed calibration payload."""
        return ReportPathRef(self.tiers_root / "calibration" / "detail.json")

    def tiers_index(self) -> ReportPathRef:
        """Tier-stage report index."""
        return ReportPathRef(self.tiers_root / "index.json")

    def model_run_root(self, model_key: str, run_name: str) -> ReportPathRef:
        model_key = _safe_path_token(model_key, field_name="model_key")
        run_name = _safe_path_token(run_name, field_name="run_name")
        return ReportPathRef(self.modeling_root / model_key / run_name)

    def model_spec_report(self, model_key: str, run_name: str) -> ReportPathRef:
        return self.model_report_file(model_key, run_name, "model_spec.md")

    def model_architecture_report(self, model_key: str, run_name: str) -> ReportPathRef:
        return self.model_report_file(model_key, run_name, "architecture.md")

    def model_training_summary_report(self, model_key: str, run_name: str) -> ReportPathRef:
        return self.model_report_file(model_key, run_name, "training_summary.md")

    def model_generated_pose_summary_report(
        self,
        model_key: str,
        run_name: str,
    ) -> ReportPathRef:
        return self.model_report_file(model_key, run_name, "generated_pose_summary.md")

    def model_risk_controls_report(self, model_key: str, run_name: str) -> ReportPathRef:
        return self.model_report_file(model_key, run_name, "risk_controls.md")

    def model_index(self, model_key: str, run_name: str) -> ReportPathRef:
        return self.model_report_file(model_key, run_name, "index.json")

    def model_validation_root(self, model_key: str, run_name: str) -> ReportPathRef:
        return ReportPathRef(self.model_run_root(model_key, run_name).path / "validation")

    def model_validation_report_file(
        self,
        model_key: str,
        run_name: str,
        filename: str,
    ) -> ReportPathRef:
        filename = _safe_filename(
            filename,
            field_name="filename",
            allowed_suffixes=(".md", ".json", ".jsonl"),
        )
        return ReportPathRef(self.model_validation_root(model_key, run_name).path / filename)

    def model_report_file(
        self,
        model_key: str,
        run_name: str,
        filename: str,
    ) -> ReportPathRef:
        filename = _safe_filename(
            filename,
            field_name="filename",
            allowed_suffixes=(".md", ".json"),
        )
        return ReportPathRef(self.model_run_root(model_key, run_name).path / filename)

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
    suffix = ".npz" if status is SampleStatus.PASSED else ".json"
    return Path(status.value) / split.value / _sample_filename(sample_id, suffix)


def _sample_filename(sample_id: str, suffix: str = ".npz") -> str:
    if not sample_id or Path(sample_id).name != sample_id or sample_id in {".", ".."}:
        raise ValueError(f"Sample id must be a concrete file stem: {sample_id!r}")
    return f"{sample_id}{suffix}"


def _safe_path_token(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string path token.")
    token = value.strip()
    if not token or token in {".", ".."}:
        raise ValueError(f"{field_name} must be a non-empty concrete path token.")
    if "." in token:
        raise ValueError(f"{field_name} must not contain dot path tokens.")
    if "/" in token or "\\" in token:
        raise ValueError(f"{field_name} must not contain path separators.")
    if Path(token).parts != (token,) or ".." in Path(token).parts:
        raise ValueError(f"{field_name} must be a single safe path token.")
    return token


def _safe_filename(
    value: str,
    *,
    field_name: str,
    allowed_suffixes: tuple[str, ...],
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string filename.")
    filename = value.strip()
    if not filename or filename in {".", ".."}:
        raise ValueError(f"{field_name} must be a non-empty concrete filename.")
    if "/" in filename or "\\" in filename:
        raise ValueError(f"{field_name} must not contain path separators.")
    if Path(filename).parts != (filename,) or ".." in Path(filename).parts:
        raise ValueError(f"{field_name} must be a single safe filename.")
    if not any(filename.endswith(suffix) for suffix in allowed_suffixes):
        expected = ", ".join(allowed_suffixes)
        raise ValueError(f"{field_name} must end with one of: {expected}.")
    if filename in allowed_suffixes:
        raise ValueError(f"{field_name} must include a concrete file stem.")
    return filename


def _safe_generation_index(value: int) -> int:
    if type(value) is not int:
        raise ValueError("generation_index must be an integer.")
    if value < 0:
        raise ValueError("generation_index must be non-negative.")
    return value


def _generated_pose_sample_filename(sample_id: str, generation_index: int) -> str:
    sample_id = _safe_path_token(sample_id, field_name="sample_id")
    generation_index = _safe_generation_index(generation_index)
    return f"{sample_id}__g{generation_index}.npz"


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
