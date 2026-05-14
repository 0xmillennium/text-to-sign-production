from __future__ import annotations

import json
from pathlib import Path

from text_to_sign_production.core.progress import ProgressSession, ProgressStageSpec
from text_to_sign_production.visualization import (
    load_pose_sample,
    render_side_by_side_video,
    render_skeleton_video,
)
from text_to_sign_production.workflows.debug.constants import DEBUG_STAGE_VISUALIZATION_RENDER
from text_to_sign_production.workflows.debug.contracts import (
    DebugGateResult,
    DebugSampleDossier,
    DebugTierResult,
    DebugVisualArtifact,
    DebugVisualizationResult,
)
from text_to_sign_production.workflows.debug.contracts.verdicts import DebugVisualizationVerdict
from text_to_sign_production.workflows.debug.layout import DebugLayout, debug_sample_run_root
from text_to_sign_production.workflows.debug.progress import debug_progress_stage


def debug_visualization(
    layout: DebugLayout,
    *,
    run_id: str,
    dossier: DebugSampleDossier,
    gate_result: DebugGateResult,
    tier_result: DebugTierResult,
    progress_session: ProgressSession | None = None,
) -> DebugVisualizationResult:
    del gate_result, tier_result
    output_root = debug_sample_run_root(
        layout,
        sentence_name=dossier.translation.sentence_name,
        run_id=run_id,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    payload_path = dossier.existing_gate.passed_payload_path
    artifacts: list[DebugVisualArtifact] = []
    warnings: list[str] = []
    errors: list[str] = []
    if payload_path is None or not payload_path.is_file():
        return DebugVisualizationResult(
            output_root=output_root,
            artifacts=(),
            verdict=DebugVisualizationVerdict.SKIPPED,
            errors=(),
            warnings=("existing passed payload is unavailable; visualization skipped",),
        )
    try:
        pose_sample = load_pose_sample(payload_path)
    except (OSError, ValueError, FileNotFoundError) as exc:
        return DebugVisualizationResult(
            output_root=output_root,
            artifacts=(),
            verdict=DebugVisualizationVerdict.FAILED,
            errors=(str(exc),),
            warnings=(),
        )
    if progress_session is not None:
        with progress_session.task(
            _visualization_render_progress_spec(),
            total=2,
        ) as progress_task:
            _render_visual_artifacts(
                output_root=output_root,
                dossier=dossier,
                pose_sample=pose_sample,
                artifacts=artifacts,
                warnings=warnings,
                progress_task=progress_task,
            )
    else:
        _render_visual_artifacts(
            output_root=output_root,
            dossier=dossier,
            pose_sample=pose_sample,
            artifacts=artifacts,
            warnings=warnings,
            progress_task=None,
        )
    errors.extend(artifact.error for artifact in artifacts if artifact.error)
    created_count = sum(1 for artifact in artifacts if artifact.created)
    if errors and created_count:
        verdict = DebugVisualizationVerdict.PARTIAL
    elif errors:
        verdict = DebugVisualizationVerdict.FAILED
    elif warnings or created_count < 2:
        verdict = DebugVisualizationVerdict.PARTIAL
    else:
        verdict = DebugVisualizationVerdict.PASS
    return DebugVisualizationResult(
        output_root=output_root,
        artifacts=tuple(artifacts),
        verdict=verdict,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def _render_visual_artifacts(
    *,
    output_root: Path,
    dossier: DebugSampleDossier,
    pose_sample,
    artifacts: list[DebugVisualArtifact],
    warnings: list[str],
    progress_task,
) -> None:
    skeleton_path = output_root / "existing_payload_skeleton.mp4"
    artifacts.append(
        _render_artifact(
            "existing_payload_skeleton",
            skeleton_path,
            lambda: render_skeleton_video(
                pose_sample=pose_sample,
                output_path=skeleton_path,
                fps=30.0,
                label=dossier.translation.sentence_name,
            ),
        )
    )
    _advance(progress_task, artifacts[-1])
    if not dossier.source.raw_video_exists:
        warnings.append("raw source video missing; side-by-side visualization skipped")
        artifacts.append(
            DebugVisualArtifact(
                label="existing_payload_side_by_side",
                path=output_root / "existing_payload_side_by_side.mp4",
                created=False,
                metadata_json=None,
                error="raw source video missing",
            )
        )
        _advance(progress_task, artifacts[-1])
        return
    side_by_side_path = output_root / "existing_payload_side_by_side.mp4"
    artifacts.append(
        _render_artifact(
            "existing_payload_side_by_side",
            side_by_side_path,
            lambda: render_side_by_side_video(
                source_video_path=dossier.source.raw_video_path,
                pose_sample=pose_sample,
                output_path=side_by_side_path,
                fps=30.0,
                label=dossier.translation.sentence_name,
            ),
        )
    )
    _advance(progress_task, artifacts[-1])


def _render_artifact(label: str, path: Path, render):
    try:
        metadata = render()
    except (OSError, RuntimeError, ValueError, FileNotFoundError) as exc:
        return DebugVisualArtifact(
            label=label,
            path=path,
            created=False,
            metadata_json=None,
            error=str(exc),
        )
    return DebugVisualArtifact(
        label=label,
        path=path,
        created=path.is_file(),
        metadata_json=json.dumps(metadata, sort_keys=True),
        error=None if path.is_file() else "render completed but output file was not created",
    )


def _advance(progress_task, artifact: DebugVisualArtifact) -> None:
    if progress_task is None:
        return
    counters = {
        "created": 1 if artifact.created else 0,
        "skipped": 1 if artifact.error == "raw source video missing" else 0,
        "failed": 1 if artifact.error and artifact.error != "raw source video missing" else 0,
    }
    progress_task.advance(counters=counters)


def _visualization_render_progress_spec() -> ProgressStageSpec:
    return debug_progress_stage(
        stage_id=DEBUG_STAGE_VISUALIZATION_RENDER,
        label="debug visualization render",
        unit="artifact",
        owner_module=__name__,
        split_behavior="per_split",
        operation_kind="visualization_render",
        total_semantics="debug visualization artifacts rendered or skipped",
        allowed_counters=("created", "skipped", "failed"),
    )


__all__ = ["debug_visualization"]
