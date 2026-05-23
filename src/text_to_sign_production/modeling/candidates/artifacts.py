"""Shared provider artifact-reference constructors."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.candidates.results import ModelStageArtifactRef


class GeneratedPoseManifestArtifactSubtype(enum.StrEnum):
    """Semantic use of a generated-pose manifest reference."""

    FINAL_VALIDATION = "final_validation"
    RECONSTRUCTION = "reconstruction"
    DECODED_INTERMEDIATE = "decoded_intermediate"
    TEST_MODEL = "test_model"


class GeneratedPosePublishPolicy(enum.StrEnum):
    """Persistence policy for a generated-pose surface."""

    PERSISTENT = "persistent"
    RUNTIME_ONLY = "runtime_only"


@dataclass(frozen=True, slots=True)
class GeneratedPoseSurface:
    """Provider-neutral generated-pose output surface.

    The surface, not individual ``samples/*.npz`` refs, is the publishable unit.
    Persistent surfaces are archived by the model workflow before publication.
    """

    manifest_path: Path
    samples_dir: Path
    split: SampleSplit
    model_key: str
    model_run_name: str
    manifest_family: str
    producer_stage_id: str
    artifact_subtype: GeneratedPoseManifestArtifactSubtype | str
    publish_policy: GeneratedPosePublishPolicy | str
    archive_required: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest_path", Path(self.manifest_path))
        object.__setattr__(self, "samples_dir", Path(self.samples_dir))
        object.__setattr__(self, "split", SampleSplit(self.split))
        object.__setattr__(
            self,
            "artifact_subtype",
            GeneratedPoseManifestArtifactSubtype(self.artifact_subtype),
        )
        object.__setattr__(
            self,
            "publish_policy",
            GeneratedPosePublishPolicy(self.publish_policy),
        )
        for value, name in (
            (self.model_key, "model_key"),
            (self.model_run_name, "model_run_name"),
            (self.manifest_family, "manifest_family"),
            (self.producer_stage_id, "producer_stage_id"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"generated pose surface {name} must be non-empty.")
        expected = _SURFACE_POLICY_RULES[self.artifact_subtype]
        if self.publish_policy is not expected[0] or self.archive_required is not expected[1]:
            raise ValueError(
                "generated pose surface policy mismatch for "
                f"{self.artifact_subtype.value}: expected "
                f"publish_policy={expected[0].value!r}, archive_required={expected[1]!r}."
            )


_SURFACE_POLICY_RULES = {
    GeneratedPoseManifestArtifactSubtype.FINAL_VALIDATION: (
        GeneratedPosePublishPolicy.PERSISTENT,
        True,
    ),
    GeneratedPoseManifestArtifactSubtype.RECONSTRUCTION: (
        GeneratedPosePublishPolicy.RUNTIME_ONLY,
        False,
    ),
    GeneratedPoseManifestArtifactSubtype.DECODED_INTERMEDIATE: (
        GeneratedPosePublishPolicy.RUNTIME_ONLY,
        False,
    ),
    GeneratedPoseManifestArtifactSubtype.TEST_MODEL: (
        GeneratedPosePublishPolicy.PERSISTENT,
        False,
    ),
}


def generated_pose_surface(
    *,
    manifest_path: Path,
    samples_dir: Path,
    split: SampleSplit,
    model_key: str,
    model_run_name: str,
    manifest_family: str,
    producer_stage_id: str,
    artifact_subtype: GeneratedPoseManifestArtifactSubtype | str,
) -> GeneratedPoseSurface:
    """Build a generated-pose surface with canonical subtype policy."""

    subtype = GeneratedPoseManifestArtifactSubtype(artifact_subtype)
    publish_policy, archive_required = _SURFACE_POLICY_RULES[subtype]
    return GeneratedPoseSurface(
        manifest_path=manifest_path,
        samples_dir=samples_dir,
        split=split,
        model_key=model_key,
        model_run_name=model_run_name,
        manifest_family=manifest_family,
        producer_stage_id=producer_stage_id,
        artifact_subtype=subtype,
        publish_policy=publish_policy,
        archive_required=archive_required,
    )


def generated_pose_manifest_artifact_ref(
    *,
    path: Path,
    split: SampleSplit,
    producer_stage: str,
    generation_mode: str,
    model_key: str,
    artifact_subtype: GeneratedPoseManifestArtifactSubtype | str,
    kind: str = "generated_pose_manifest",
    description: str | None = None,
) -> ModelStageArtifactRef:
    """Build the canonical reference for a generated-pose manifest file."""

    if not isinstance(split, SampleSplit):
        split = SampleSplit(split)
    try:
        subtype = GeneratedPoseManifestArtifactSubtype(artifact_subtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown generated-pose manifest artifact_subtype: {artifact_subtype!r}.") from exc
    for value, name in (
        (producer_stage, "producer_stage"),
        (generation_mode, "generation_mode"),
        (model_key, "model_key"),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"generated pose manifest artifact {name} must be non-empty.")
    return ModelStageArtifactRef(
        role="generated_pose_manifest",
        path=Path(path),
        kind=kind,
        description=description,
        metadata={
            "split": split.value,
            "producer_stage": producer_stage,
            "generation_mode": generation_mode,
            "model_key": model_key,
            "artifact_subtype": subtype.value,
        },
    )


def generated_pose_sample_artifact_ref(
    *,
    path: Path,
    split: SampleSplit,
    sample_id: str,
    generation_index: int,
    producer_stage: str,
    generation_mode: str,
    model_key: str,
) -> ModelStageArtifactRef:
    """Build the canonical provider-stage reference for one generated pose payload."""

    if not isinstance(split, SampleSplit):
        split = SampleSplit(split)
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError("generated pose artifact sample_id must be non-empty.")
    if (
        not isinstance(generation_index, int)
        or isinstance(generation_index, bool)
        or generation_index < 0
    ):
        raise ValueError("generated pose artifact generation_index must be non-negative.")
    for value, name in (
        (producer_stage, "producer_stage"),
        (generation_mode, "generation_mode"),
        (model_key, "model_key"),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"generated pose artifact {name} must be non-empty.")
    return ModelStageArtifactRef(
        role="generated_pose_sample",
        path=Path(path),
        kind="generated_pose_sample",
        metadata={
            "split": split.value,
            "sample_id": sample_id,
            "generation_index": generation_index,
            "producer_stage": producer_stage,
            "generation_mode": generation_mode,
            "model_key": model_key,
        },
    )


def generated_pose_samples_archive_artifact_ref(
    *,
    path: Path,
    split: SampleSplit,
    producer_stage: str,
    model_key: str,
    member_count: int,
    archive_sha256: str,
) -> ModelStageArtifactRef:
    """Build the canonical reference for a generated-pose samples archive."""

    return _generated_pose_archive_artifact_ref(
        role="generated_pose_samples_archive",
        path=path,
        kind="generated_pose_samples_archive",
        split=split,
        producer_stage=producer_stage,
        model_key=model_key,
        member_count=member_count,
        archive_sha256=archive_sha256,
    )


def generated_pose_samples_archive_manifest_artifact_ref(
    *,
    path: Path,
    split: SampleSplit,
    producer_stage: str,
    model_key: str,
    member_count: int,
    archive_sha256: str,
) -> ModelStageArtifactRef:
    """Build the canonical reference for a generated-pose samples archive manifest."""

    return _generated_pose_archive_artifact_ref(
        role="generated_pose_samples_archive_manifest",
        path=path,
        kind="generated_pose_samples_archive_manifest",
        split=split,
        producer_stage=producer_stage,
        model_key=model_key,
        member_count=member_count,
        archive_sha256=archive_sha256,
    )


def generated_pose_samples_archive_sha256_artifact_ref(
    *,
    path: Path,
    split: SampleSplit,
    producer_stage: str,
    model_key: str,
    member_count: int,
    archive_sha256: str,
) -> ModelStageArtifactRef:
    """Build the canonical reference for a generated-pose samples archive digest file."""

    return _generated_pose_archive_artifact_ref(
        role="generated_pose_samples_archive_sha256",
        path=path,
        kind="generated_pose_samples_archive_sha256",
        split=split,
        producer_stage=producer_stage,
        model_key=model_key,
        member_count=member_count,
        archive_sha256=archive_sha256,
    )


def _generated_pose_archive_artifact_ref(
    *,
    role: str,
    path: Path,
    kind: str,
    split: SampleSplit,
    producer_stage: str,
    model_key: str,
    member_count: int,
    archive_sha256: str,
) -> ModelStageArtifactRef:
    if not isinstance(split, SampleSplit):
        split = SampleSplit(split)
    if member_count < 0:
        raise ValueError("generated pose archive member_count must be non-negative.")
    for value, name in (
        (producer_stage, "producer_stage"),
        (model_key, "model_key"),
        (archive_sha256, "archive_sha256"),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"generated pose archive artifact {name} must be non-empty.")
    return ModelStageArtifactRef(
        role=role,
        path=Path(path),
        kind=kind,
        metadata={
            "split": split.value,
            "producer_stage": producer_stage,
            "model_key": model_key,
            "member_count": member_count,
            "archive_sha256": archive_sha256,
            "individual_sample_publish": False,
        },
    )


__all__ = [
    "GeneratedPoseManifestArtifactSubtype",
    "GeneratedPosePublishPolicy",
    "GeneratedPoseSurface",
    "generated_pose_surface",
    "generated_pose_manifest_artifact_ref",
    "generated_pose_sample_artifact_ref",
    "generated_pose_samples_archive_artifact_ref",
    "generated_pose_samples_archive_manifest_artifact_ref",
    "generated_pose_samples_archive_sha256_artifact_ref",
]
