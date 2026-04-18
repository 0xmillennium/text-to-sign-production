"""Public notebook and script contract tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.integration


def _load_notebook_sources() -> tuple[str, str]:
    notebook_path = (
        Path(__file__).resolve().parents[3] / "notebooks/colab/dataset_build_colab.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    markdown_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    )
    code_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    return markdown_source, code_source


def _load_notebook_cells() -> list[dict[str, Any]]:
    notebook_path = (
        Path(__file__).resolve().parents[3] / "notebooks/colab/dataset_build_colab.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    return list(notebook["cells"])


def test_colab_notebook_contains_only_the_supported_fixed_workflow() -> None:
    markdown_source, code_source = _load_notebook_sources()

    expected_headings = (
        "# Text-To-Sign Production Colab",
        "## 1. Environment And Repository Setup",
        "## 2. Mount Google Drive And Select Run Settings",
        "## 3. Run Dataset Build",
        "## 4. Locate Dataset Build Processed Outputs",
        "## 5. Prepare Sprint 3 Baseline Run Layout",
        "## 6. Resolve Or Run Sprint 3 Training Outputs",
        "## 7. Resolve Or Run Sprint 3 Qualitative Export",
        "## 8. Resolve Or Build Sprint 3 Record Package",
    )
    for heading in expected_headings:
        assert heading in markdown_source

    required_snippets = (
        'PIPELINE_SPLITS = ["train", "val", "test"]',
        'BASELINE_RUN_NAME = "baseline-default"',
        "BASELINE_PANEL_SIZE = 8",
        'drive.mount("/content/drive", force_remount=False)',
        'if str(WORKTREE_ROOT / "src") not in sys.path:',
        'sys.path.insert(0, str(WORKTREE_ROOT / "src"))',
        "from text_to_sign_production.data.constants import DEFAULT_FILTER_CONFIG_RELATIVE_PATH",
        "from text_to_sign_production.workflows.dataset_build import run_dataset_build",
        "from text_to_sign_production.workflows.baseline_modeling import",
        "COLAB_BASELINE_ARTIFACT_RUNS_ROOT",
        "run_baseline_modeling",
        "run_dataset_build(",
        "run_baseline_modeling(",
        'input_mode="fixed_colab_drive"',
        'output_mode="fixed_colab_drive"',
        "filter_config_path=WORKTREE_ROOT / DEFAULT_FILTER_CONFIG_RELATIVE_PATH",
        "/content/how2sign_downloads",
        "/content/drive/MyDrive/text-to-sign-production/artifacts/dataset-build/processed-v1/",
        "/content/drive/MyDrive/text-to-sign-production/artifacts/baseline-modeling/runs/",
        "baseline_training_outputs.tar.zst",
        "baseline_qualitative_outputs.tar.zst",
        "baseline_record_package.tar.zst",
        "train_2D_keypoints.tar.zst",
        "val_2D_keypoints.tar.zst",
        "test_2D_keypoints.tar.zst",
    )
    for snippet in required_snippets:
        assert snippet in (markdown_source + "\n" + code_source)

    removed_snippets = (
        "public_urls",
        "mounted_paths",
        "RAW_INPUT_MODE",
        "TRANSLATION_URLS",
        "KEYPOINT_ARCHIVE_URLS",
        "MOUNTED_TRANSLATION_FILES",
        "MOUNTED_KEYPOINT_SPLIT_DIRS",
        "gdown",
        "storage.local.yaml",
        "storage.example.yaml",
        "T2SP_PROGRESS_MODE",
        ".tar.gz",
        "scripts/package_sprint2_outputs.py",
        "scripts/prepare_raw.py",
        "scripts/normalize_keypoints.py",
        "scripts/filter_samples.py",
        "scripts/export_training_manifest.py",
        "from text_to_sign_production.ops.colab_workflow import stage_colab_inputs",
        "from text_to_sign_production.ops.colab_workflow import publish_colab_outputs",
        "from text_to_sign_production.data.raw import build_raw_manifests",
        "from text_to_sign_production.data.normalize import normalize_all_splits",
        "from text_to_sign_production.data.filtering import filter_all_splits",
        "from text_to_sign_production.data.manifests import export_final_manifests",
        "Sprint 2",
        "sprint2",
        "google.colab.files",
    )
    for snippet in removed_snippets:
        assert snippet not in (markdown_source + "\n" + code_source)


def test_colab_baseline_cells_redeclare_fixed_config_path() -> None:
    cells = _load_notebook_cells()
    baseline_cell_ids = {
        "baseline-prepare-code",
        "baseline-train-code",
        "baseline-qualitative-code",
        "baseline-package-code",
    }

    for cell in cells:
        if cell.get("id") not in baseline_cell_ids:
            continue
        source = "".join(cell.get("source", []))
        assert (
            "from text_to_sign_production.modeling.config import DEFAULT_BASELINE_CONFIG_PATH"
        ) in source
        assert "BASELINE_CONFIG_PATH = DEFAULT_BASELINE_CONFIG_PATH" in source
        assert "config_path=BASELINE_CONFIG_PATH" in source


def test_public_workflow_surface_has_one_main_notebook_and_stage_clis() -> None:
    project_root = Path(__file__).resolve().parents[3]
    colab_notebooks = sorted(
        path.name for path in (project_root / "notebooks/colab").glob("*.ipynb")
    )
    script_names = sorted(path.name for path in (project_root / "scripts").glob("*.py"))

    assert colab_notebooks == ["dataset_build_colab.ipynb"]
    assert "dataset_build.py" in script_names
    assert "baseline_modeling.py" in script_names
    assert "train_baseline.py" in script_names
    assert "export_qualitative_panel.py" in script_names
    assert "evaluate_baseline.py" in script_names
    assert "validate_manifest.py" in script_names
    assert "view_sample.py" in script_names

    removed_scripts = {
        "prepare_raw.py",
        "normalize_keypoints.py",
        "filter_samples.py",
        "export_training_manifest.py",
        "stage_colab_inputs.py",
        "publish_colab_outputs.py",
        "package_sprint2_outputs.py",
    }
    assert removed_scripts.isdisjoint(script_names)
