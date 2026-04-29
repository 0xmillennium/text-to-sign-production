"""Constants owned by the Visualization workflow."""

from __future__ import annotations

DEFAULT_VISUALIZATION_RUN_NAME = "default"

DEFAULT_VISUALIZATION_OUTPUT_FILENAME = "output.mp4"

VISUALIZATION_OUTPUTS_DIRNAME = "outputs"

VISUALIZATION_METADATA_DIRNAME = "metadata"

VISUALIZATION_PUBLISH_DIRNAME = "publish"

VISUALIZATION_RENDER_PLAN_FILENAME = "render-plan.json"

VISUALIZATION_RENDER_RESULT_FILENAME = "render-result.json"

VISUALIZATION_ALLOWED_SPLITS = ("train", "val", "test")

VISUALIZATION_RENDER_MODES = ("skeleton_only", "side_by_side")

VISUALIZATION_SELECTION_MODES = ("sample_id", "random")

MISSING_SOURCE_VIDEO_MESSAGE = (
    "Source video is missing. Extract the required raw BFH archive in the notebook/runtime first."
)

MISSING_SAMPLE_FILE_MESSAGE = (
    "Selected processed sample file is missing. "
    "Extract dataset_build_samples_{split}.tar.zst in the notebook/runtime first."
)

__all__ = [
    "DEFAULT_VISUALIZATION_OUTPUT_FILENAME",
    "DEFAULT_VISUALIZATION_RUN_NAME",
    "MISSING_SAMPLE_FILE_MESSAGE",
    "MISSING_SOURCE_VIDEO_MESSAGE",
    "VISUALIZATION_ALLOWED_SPLITS",
    "VISUALIZATION_METADATA_DIRNAME",
    "VISUALIZATION_OUTPUTS_DIRNAME",
    "VISUALIZATION_PUBLISH_DIRNAME",
    "VISUALIZATION_RENDER_MODES",
    "VISUALIZATION_RENDER_PLAN_FILENAME",
    "VISUALIZATION_RENDER_RESULT_FILENAME",
    "VISUALIZATION_SELECTION_MODES",
]
