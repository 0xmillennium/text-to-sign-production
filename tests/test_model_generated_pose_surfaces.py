from __future__ import annotations

from pathlib import Path

import pytest

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.candidates.artifacts import (
    GeneratedPoseManifestArtifactSubtype,
    GeneratedPosePublishPolicy,
    GeneratedPoseSurface,
    generated_pose_surface,
)


@pytest.mark.unit
def test_final_validation_surface_requires_archive(tmp_path: Path) -> None:
    surface = generated_pose_surface(
        manifest_path=tmp_path / "manifest.jsonl",
        samples_dir=tmp_path / "samples",
        split=SampleSplit.VAL,
        model_key="base_direct",
        model_run_name="run001",
        manifest_family="tiered:clean:included",
        producer_stage_id="export_generated_pose",
        artifact_subtype=GeneratedPoseManifestArtifactSubtype.FINAL_VALIDATION,
    )

    assert surface.publish_policy is GeneratedPosePublishPolicy.PERSISTENT
    assert surface.archive_required is True


@pytest.mark.unit
def test_runtime_only_surfaces_are_not_archive_required(tmp_path: Path) -> None:
    for subtype in (
        GeneratedPoseManifestArtifactSubtype.RECONSTRUCTION,
        GeneratedPoseManifestArtifactSubtype.DECODED_INTERMEDIATE,
    ):
        surface = generated_pose_surface(
            manifest_path=tmp_path / subtype.value / "manifest.jsonl",
            samples_dir=tmp_path / subtype.value / "samples",
            split=SampleSplit.VAL,
            model_key="learned_pose_token",
            model_run_name="run001",
            manifest_family="tiered:clean:included",
            producer_stage_id=subtype.value,
            artifact_subtype=subtype,
        )
        assert surface.publish_policy is GeneratedPosePublishPolicy.RUNTIME_ONLY
        assert surface.archive_required is False


@pytest.mark.unit
def test_surface_rejects_mismatched_policy(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="policy mismatch"):
        GeneratedPoseSurface(
            manifest_path=tmp_path / "manifest.jsonl",
            samples_dir=tmp_path / "samples",
            split=SampleSplit.VAL,
            model_key="base_direct",
            model_run_name="run001",
            manifest_family="tiered:clean:included",
            producer_stage_id="export_generated_pose",
            artifact_subtype=GeneratedPoseManifestArtifactSubtype.FINAL_VALIDATION,
            publish_policy=GeneratedPosePublishPolicy.RUNTIME_ONLY,
            archive_required=False,
        )
