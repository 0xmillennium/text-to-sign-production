"""Visualization workflow output verification helpers."""

from __future__ import annotations

from pathlib import Path

from text_to_sign_production.core.files import verify_output_file
from text_to_sign_production.core.paths import repo_relative_path
from text_to_sign_production.visualization.video import CANONICAL_MP4_CODEC
from text_to_sign_production.workflows._shared.metadata import verify_portable_json_file
from text_to_sign_production.workflows._visualization.types import (
    VisualizationWorkflowError,
    VisualizationWorkflowInputError,
    VisualizationWorkflowOutputSummary,
    VisualizationWorkflowResult,
)
from text_to_sign_production.workflows._visualization.validate import _validate_codec


def _verify_visualization_result(
    result: VisualizationWorkflowResult,
) -> VisualizationWorkflowOutputSummary:
    """Verify and summarize a completed visualization render."""

    _verify_run_root_policy(result)
    try:
        output_path = verify_output_file(result.output_path, label="Runtime visualization MP4")
        render_plan_path = verify_output_file(
            result.render_plan_metadata_path,
            label="Visualization render plan metadata",
        )
        render_result_path = verify_output_file(
            result.render_result_metadata_path,
            label="Visualization render result metadata",
        )
    except (FileNotFoundError, IsADirectoryError, ValueError) as exc:
        raise VisualizationWorkflowError(str(exc)) from exc
    mode = str(result.render_metadata.get("mode", ""))
    if mode not in ("skeleton_only", "side_by_side"):
        raise VisualizationWorkflowError(f"Visualization result has invalid mode metadata: {mode}")
    try:
        codec = _validate_codec(str(result.render_metadata.get("codec", "")))
    except VisualizationWorkflowInputError as exc:
        raise VisualizationWorkflowError(
            f"Visualization result has invalid codec metadata: {exc}"
        ) from exc
    if codec != CANONICAL_MP4_CODEC:
        raise VisualizationWorkflowError(
            f"Visualization result codec must be {CANONICAL_MP4_CODEC!r}, observed {codec!r}."
        )
    _verify_portable_json_file(render_plan_path, project_root=result.project_root)
    _verify_portable_json_file(render_result_path, project_root=result.project_root)
    return VisualizationWorkflowOutputSummary(
        sample_id=result.selected_sample.record.sample_id,
        split=result.selected_sample.record.split,
        mode=mode,
        output_path=output_path,
        output_size=output_path.stat().st_size,
        codec=codec,
        render_plan_metadata_path=render_plan_path,
        render_result_metadata_path=render_result_path,
    )


def _verify_run_root_policy(result: VisualizationWorkflowResult) -> None:
    relative_run_root = Path(repo_relative_path(result.run_root, repo_root=result.project_root))
    expected_prefix = Path("runs") / "visualization" / "visual-debug"
    if relative_run_root.parent != expected_prefix:
        raise VisualizationWorkflowError(
            "Visualization run root must live under the canonical visual-debug runs tree: "
            f"{result.run_root}"
        )


def _verify_portable_json_file(path: Path, *, project_root: Path) -> None:
    verify_portable_json_file(
        path,
        project_root=project_root,
        error_factory=VisualizationWorkflowError,
    )


__all__: list[str] = []
