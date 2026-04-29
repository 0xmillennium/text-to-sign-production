"""Visualization workflow input and policy validation."""

from __future__ import annotations

from text_to_sign_production.core.files import OutputExistsPolicy, require_file
from text_to_sign_production.workflows._visualization.constants import (
    MISSING_SAMPLE_FILE_MESSAGE,
    MISSING_SOURCE_VIDEO_MESSAGE,
    VISUALIZATION_ALLOWED_SPLITS,
    VISUALIZATION_OUTPUTS_DIRNAME,
    VISUALIZATION_RENDER_MODES,
    VISUALIZATION_SELECTION_MODES,
)
from text_to_sign_production.workflows._visualization.layout import (
    _validated_render_output_filename,
)
from text_to_sign_production.workflows._visualization.types import (
    SelectedVisualizationSample,
    VisualizationRenderMode,
    VisualizationRenderPlan,
    VisualizationSelectionMode,
    VisualizationWorkflowInputError,
    _SampleSelectionConfig,
)


def validate_visualization_inputs(
    plan: VisualizationRenderPlan,
) -> None:
    """Validate notebook/runtime visualization controls without rendering."""

    _validate_render_plan(plan)


def _validate_render_plan(plan: VisualizationRenderPlan) -> None:
    _validate_selected_runtime_inputs(plan.selected_sample, mode=plan.mode)
    _output_policy(plan.output_exists_policy)
    if plan.fps is not None and plan.fps <= 0:
        raise VisualizationWorkflowInputError("fps must be positive when provided.")
    if plan.output_filename is not None:
        _validate_output_filename(plan.output_filename)
    _validate_output_path_under_canonical_outputs(plan)


def _validate_output_path_under_canonical_outputs(plan: VisualizationRenderPlan) -> None:
    expected_outputs_root = (plan.run_root / VISUALIZATION_OUTPUTS_DIRNAME).resolve()
    observed_outputs_root = plan.outputs_root.resolve()
    output_path = plan.output_path.resolve()
    if observed_outputs_root != expected_outputs_root:
        raise VisualizationWorkflowInputError(
            "Visualization outputs_root must be the canonical outputs directory: "
            f"{expected_outputs_root}"
        )
    if output_path.parent != expected_outputs_root:
        raise VisualizationWorkflowInputError(
            "Visualization output_path must stay directly under the canonical outputs root: "
            f"{plan.output_path}"
        )
    _validated_render_output_filename(plan.output_path.name)
    if observed_outputs_root.exists() and not observed_outputs_root.is_dir():
        raise VisualizationWorkflowInputError(
            f"Visualization outputs_root is not a directory: {observed_outputs_root}"
        )


def _validate_selected_runtime_inputs(
    selected_sample: SelectedVisualizationSample,
    *,
    mode: VisualizationRenderMode,
) -> None:
    try:
        require_file(selected_sample.sample_path, label="Selected processed sample file")
    except (FileNotFoundError, IsADirectoryError, ValueError) as exc:
        raise VisualizationWorkflowInputError(
            MISSING_SAMPLE_FILE_MESSAGE.format(split=selected_sample.record.split)
            + f" Missing path: {selected_sample.sample_path}"
        ) from exc
    if mode == "side_by_side":
        if selected_sample.source_video_path is None:
            raise VisualizationWorkflowInputError(MISSING_SOURCE_VIDEO_MESSAGE)
        try:
            require_file(selected_sample.source_video_path, label="Source video")
        except (FileNotFoundError, IsADirectoryError, ValueError) as exc:
            raise VisualizationWorkflowInputError(MISSING_SOURCE_VIDEO_MESSAGE) from exc


def _validate_selection_modes(config: _SampleSelectionConfig) -> None:
    sample_id = config.sample_id
    has_sample_id = sample_id is not None
    if sample_id is not None and not sample_id.strip():
        raise VisualizationWorkflowInputError("sample_id must not be blank when provided.")
    if has_sample_id == config.random_selection:
        raise VisualizationWorkflowInputError(
            "Exactly one visualization selection mode is required: sample_id or random_selection."
        )


def _validate_splits(splits: tuple[str, ...]) -> tuple[str, ...]:
    if not splits:
        raise VisualizationWorkflowInputError("At least one visualization split is required.")
    duplicates = sorted({split for split in splits if splits.count(split) > 1})
    if duplicates:
        raise VisualizationWorkflowInputError(f"Duplicate visualization split(s): {duplicates}")
    invalid = [split for split in splits if split not in VISUALIZATION_ALLOWED_SPLITS]
    if invalid:
        expected = ", ".join(VISUALIZATION_ALLOWED_SPLITS)
        raise VisualizationWorkflowInputError(
            f"Unsupported visualization split(s) {invalid!r}; expected values from: {expected}."
        )
    requested = set(splits)
    return tuple(split for split in VISUALIZATION_ALLOWED_SPLITS if split in requested)


def _render_mode(value: str) -> VisualizationRenderMode:
    if value not in VISUALIZATION_RENDER_MODES:
        raise VisualizationWorkflowInputError(f"Unsupported visualization mode: {value}")
    return value  # type: ignore[return-value]


def _selection_mode(value: str) -> VisualizationSelectionMode:
    if value not in VISUALIZATION_SELECTION_MODES:
        raise VisualizationWorkflowInputError(
            f"Unsupported visualization selection mode {value!r}; expected 'sample_id' or 'random'."
        )
    return value  # type: ignore[return-value]


def _validate_codec(value: str) -> str:
    codec = value.strip()
    if len(codec) != 4:
        raise VisualizationWorkflowInputError(
            f"video codec must be exactly 4 characters; received {value!r}."
        )
    return codec


def _output_policy(value: OutputExistsPolicy | str) -> OutputExistsPolicy:
    try:
        return value if isinstance(value, OutputExistsPolicy) else OutputExistsPolicy(value)
    except ValueError as exc:
        expected = ", ".join(policy.value for policy in OutputExistsPolicy)
        raise VisualizationWorkflowInputError(
            f"Unsupported output_exists_policy {value!r}; expected one of: {expected}."
        ) from exc


def _validate_output_filename(value: str) -> None:
    _validated_render_output_filename(value)


__all__ = ["validate_visualization_inputs"]
