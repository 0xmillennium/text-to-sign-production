"""Render generated pose and source-vs-generated videos for test_model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from text_to_sign_production.core.progress import ProgressSession, ProgressTaskHandle
from text_to_sign_production.modeling.data import BfhPoseArrays
from text_to_sign_production.visualization import (
    render_pose_pair_video,
    render_side_by_side_video,
    render_skeleton_video,
)
from text_to_sign_production.visualization.pose import PoseSample
from text_to_sign_production.workflows.foundation.provenance import written_file_receipt
from text_to_sign_production.workflows.test_model.contracts import (
    TestModelInferenceResult,
    TestModelSampleEvidence,
    TestModelVisualArtifact,
    TestModelVisualizationResult,
)
from text_to_sign_production.workflows.test_model.constants import (
    TEST_MODEL_STAGE_VISUALIZATION_RENDER,
)
from text_to_sign_production.workflows.test_model.layout import runtime_test_model_sample_run_root
from text_to_sign_production.workflows.test_model.progress import test_model_progress_stage


def render_test_model_visualization(
    layout,
    *,
    evidence: TestModelSampleEvidence,
    inference: TestModelInferenceResult,
    execution_id: str,
    progress_session: ProgressSession | None = None,
) -> TestModelVisualizationResult:
    output_root = (
        runtime_test_model_sample_run_root(
            layout,
            evidence.model_context["model_run_name"],
            evidence.target.target_sentence_name,
            execution_id,
        )
        / "videos"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    artifacts: list[TestModelVisualArtifact] = []
    warnings: list[str] = []
    errors: list[str] = []
    generated_sample = (
        None if inference.provider_result is None else inference.provider_result.generated_sample
    )
    pose_sample = None if generated_sample is None else _pose_sample_from_generated(generated_sample)
    progress_task = (
        progress_session.task(
            test_model_progress_stage(
                stage_id=TEST_MODEL_STAGE_VISUALIZATION_RENDER,
                label="test_model visualization render",
                unit="artifact",
                owner_module=__name__,
                operation_kind="visualization_render",
                total_semantics="single-sample visualization artifacts",
                allowed_counters=("created", "skipped", "failed"),
            ),
            total=3,
        )
        if progress_session is not None
        else None
    )
    skeleton_path = output_root / "generated_pose_skeleton.mp4"
    try:
        if pose_sample is None:
            skeleton = TestModelVisualArtifact(
                "generated_pose_skeleton",
                skeleton_path,
                False,
                None,
                "generated sample missing",
            )
        else:
            skeleton = _render_artifact(
                "generated_pose_skeleton",
                skeleton_path,
                lambda: render_skeleton_video(
                    pose_sample=pose_sample,
                    output_path=skeleton_path,
                    fps=30.0,
                    label=str(evidence.target.resolved_sample_id),
                ),
            )
        artifacts.append(skeleton)
        _advance_visualization_progress(progress_task, skeleton)
        side_by_side_path = output_root / "source_vs_generated_pose.mp4"
        if pose_sample is None:
            side_by_side = TestModelVisualArtifact(
                "source_vs_generated_pose",
                side_by_side_path,
                False,
                None,
                "generated sample missing",
            )
        elif evidence.source_video_exists:
            side_by_side = _render_artifact(
                "source_vs_generated_pose",
                side_by_side_path,
                lambda: render_side_by_side_video(
                    source_video_path=evidence.source_video_path,
                    pose_sample=pose_sample,
                    output_path=side_by_side_path,
                    fps=30.0,
                    label=str(evidence.target.resolved_sample_id),
                ),
            )
        else:
            warnings.append(
                f"source video missing; side-by-side not rendered: {evidence.source_video_path}"
            )
            side_by_side = TestModelVisualArtifact(
                "source_vs_generated_pose",
                side_by_side_path,
                False,
                None,
                "source video missing",
            )
        artifacts.append(side_by_side)
        _advance_visualization_progress(progress_task, side_by_side)
        reference_pair_path = output_root / "reference_vs_generated_pose.mp4"
        if evidence.target.manifest_sample is None:
            reference_pair = TestModelVisualArtifact(
                "reference_vs_generated_pose",
                reference_pair_path,
                False,
                None,
                "reference sample missing",
            )
        elif pose_sample is None:
            reference_pair = TestModelVisualArtifact(
                "reference_vs_generated_pose",
                reference_pair_path,
                False,
                None,
                "generated sample missing",
            )
        else:
            reference_sample = _pose_sample_from_bfh(
                pose=evidence.target.manifest_sample.pose,
                path=Path(f"{evidence.target.manifest_sample.sample_id}__reference.npz"),
                schema_version=evidence.target.manifest_sample.sample.schema_version,
            )
            reference_pair = _render_artifact(
                "reference_vs_generated_pose",
                reference_pair_path,
                lambda: render_pose_pair_video(
                    left_pose_sample=reference_sample,
                    right_pose_sample=pose_sample,
                    output_path=reference_pair_path,
                    fps=30.0,
                    left_label=str(evidence.target.resolved_sample_id),
                    right_label=str(generated_sample.sample_id),
                ),
            )
        artifacts.append(reference_pair)
        _advance_visualization_progress(progress_task, reference_pair)
    finally:
        if progress_task is not None:
            progress_task.close()
    required_labels = {"generated_pose_skeleton", "reference_vs_generated_pose"}
    errors.extend(
        artifact.error
        for artifact in artifacts
        if artifact.label in required_labels and artifact.error
    )
    receipts = tuple(
        written_file_receipt(
            f"test_model visual {artifact.label}",
            artifact.path,
            execution_id=execution_id,
            kind="test_model_visualization",
        )
        for artifact in artifacts
        if artifact.created
    )
    required_created = all(
        artifact.created for artifact in artifacts if artifact.label in required_labels
    )
    source_artifact = next(
        artifact for artifact in artifacts if artifact.label == "source_vs_generated_pose"
    )
    source_ok = source_artifact.created or source_artifact.error == "source video missing"
    status = "completed" if required_created and source_ok and not errors else "failed"
    return TestModelVisualizationResult(
        output_root=output_root,
        artifacts=tuple(artifacts),
        receipts=receipts,
        status=status,
        warnings=tuple(warnings),
        errors=tuple(errors),
    )


def _pose_sample_from_generated(sample) -> PoseSample:
    return _pose_sample_from_bfh(
        pose=sample.pose,
        path=Path(f"{sample.sample_id}__generated.npz"),
        schema_version=sample.schema_version,
    )


def _pose_sample_from_bfh(
    *,
    pose: BfhPoseArrays,
    path: Path,
    schema_version: str,
) -> PoseSample:
    frame_count = pose.frame_count
    return PoseSample(
        path=path,
        schema_version=schema_version,
        body=np.asarray(pose.body_xyc[..., :2], dtype=np.float32),
        body_confidence=np.asarray(pose.body_xyc[..., 2], dtype=np.float32),
        left_hand=np.asarray(pose.left_hand_xyc[..., :2], dtype=np.float32),
        left_hand_confidence=np.asarray(pose.left_hand_xyc[..., 2], dtype=np.float32),
        right_hand=np.asarray(pose.right_hand_xyc[..., :2], dtype=np.float32),
        right_hand_confidence=np.asarray(pose.right_hand_xyc[..., 2], dtype=np.float32),
        face=np.asarray(pose.face_xyc[..., :2], dtype=np.float32),
        face_confidence=np.asarray(pose.face_xyc[..., 2], dtype=np.float32),
        people_per_frame=np.ones((frame_count,), dtype=np.int16),
        selected_person_index=0,
        frame_valid_mask=np.asarray(pose.valid_frame_mask, dtype=np.bool_),
    )


def _render_artifact(label: str, path: Path, render) -> TestModelVisualArtifact:
    try:
        metadata = render()
    except (OSError, RuntimeError, ValueError, FileNotFoundError) as exc:
        return TestModelVisualArtifact(label, path, False, None, str(exc))
    return TestModelVisualArtifact(
        label,
        path,
        path.is_file(),
        json.dumps(metadata, sort_keys=True),
        None if path.is_file() else "render completed but output file was not created",
    )


def _advance_visualization_progress(
    task: ProgressTaskHandle | None,
    artifact: TestModelVisualArtifact,
) -> None:
    if task is None:
        return
    skipped = artifact.error == "source video missing"
    task.advance(
        1,
        counters={
            "created": 1 if artifact.created else 0,
            "skipped": 1 if skipped else 0,
            "failed": 1 if artifact.error and not skipped else 0,
        },
    )


__all__ = ["render_test_model_visualization"]
