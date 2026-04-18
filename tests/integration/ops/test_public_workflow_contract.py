"""Public notebook, docs, and script contract tests."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import pytest

pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_NAME = "text_to_sign_production_colab.ipynb"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "colab" / NOTEBOOK_NAME
CURRENT_NOTEBOOK_REFERENCE = f"notebooks/colab/{NOTEBOOK_NAME}"
OLD_NOTEBOOK_NAME = "dataset_build_colab.ipynb"
PUBLIC_STATUS_LINES = (
    "Current public stage: Dataset Build",
    "Implemented internal downstream surface: Baseline Modeling",
    "Not yet implemented: broader evaluation, contribution modeling, playback/demo",
)


def _load_notebook() -> dict[str, Any]:
    loaded = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError(f"Notebook root must be a JSON object: {NOTEBOOK_PATH}")
    return cast(dict[str, Any], loaded)


def _load_notebook_sources() -> tuple[str, str]:
    notebook = _load_notebook()
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
    return list(_load_notebook()["cells"])


def _read_repo_file(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _durable_public_text_paths() -> list[Path]:
    return [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "mkdocs.yml",
        *sorted((PROJECT_ROOT / "docs").rglob("*.md")),
        *sorted((PROJECT_ROOT / "tests/operational").rglob("*.md")),
    ]


def test_colab_notebook_contains_stage_oriented_supported_workflow() -> None:
    markdown_source, code_source = _load_notebook_sources()
    combined_source = markdown_source + "\n" + code_source

    expected_headings = (
        "# Text-To-Sign Production Colab",
        "## Section 0: Runtime And Session Setup",
        "## Section 1: Drive Mount",
        "## Section 2: Repo Acquisition And Install",
        "### 2.3 Load Shared Workflow Helpers",
        "## Section 3: Dataset Build Outputs",
        "### 3.1 Reuse Extracted Dataset Build Outputs",
        "### 3.2 Extract Archived Dataset Build Outputs",
        "### 3.3 Run Dataset Build And Publish Outputs",
        "## Section 4: Baseline Modeling Training Outputs",
        "### 4.1 Prepare Baseline Run Root",
        "### 4.2 Reuse Extracted Training Outputs",
        "### 4.3 Extract Archived Training Outputs",
        "### 4.4 Run Training And Publish Outputs",
        "## Section 5: Qualitative Panel Outputs",
        "### 5.1 Reuse Extracted Qualitative Panel Outputs",
        "### 5.2 Extract Archived Qualitative Panel Outputs",
        "### 5.3 Run Qualitative Panel Export And Publish Outputs",
        "## Section 6: Record/Package Outputs",
        "### 6.1 Reuse Extracted Record/Package Outputs",
        "### 6.2 Extract Archived Record/Package Outputs",
        "### 6.3 Assemble/Package And Publish Record Outputs",
        "## Section 7: Final Artifact Inspection",
    )
    for heading in expected_headings:
        assert heading in markdown_source

    required_snippets = (
        *PUBLIC_STATUS_LINES,
        'PIPELINE_SPLITS = ["train", "val", "test"]',
        'BASELINE_RUN_NAME = "baseline-default"',
        "BASELINE_PANEL_SIZE = 8",
        'drive.mount("/content/drive", force_remount=False)',
        'if str(WORKTREE_ROOT / "src") not in sys.path:',
        'sys.path.insert(0, str(WORKTREE_ROOT / "src"))',
        "DEFAULT_FILTER_CONFIG_RELATIVE_PATH",
        "from text_to_sign_production.workflows.dataset_build import run_dataset_build",
        "from text_to_sign_production.workflows.baseline_modeling import",
        "COLAB_BASELINE_ARTIFACT_RUNS_ROOT",
        "run_baseline_modeling",
        "run_dataset_build(",
        "run_baseline_modeling(",
        'input_mode="fixed_colab_drive"',
        'output_mode="fixed_colab_drive"',
        "filter_config_path=WORKTREE_ROOT / DEFAULT_FILTER_CONFIG_RELATIVE_PATH",
        "/content/drive/MyDrive/text-to-sign-production/artifacts/dataset-build/processed-v1/",
        "/content/drive/MyDrive/text-to-sign-production/artifacts/baseline-modeling/runs/",
        "dataset_build_manifests_reports.tar.zst",
        "dataset_build_samples_",
        "baseline_training_outputs.tar.zst",
        "baseline_qualitative_outputs.tar.zst",
        "baseline_record_package.tar.zst",
    )
    for snippet in required_snippets:
        assert snippet in combined_source

    removed_snippets = (
        OLD_NOTEBOOK_NAME,
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
        "google.colab.files",
    )
    for snippet in removed_snippets:
        assert snippet not in combined_source


def test_colab_code_cells_are_preceded_by_operational_markdown() -> None:
    cells = _load_notebook_cells()
    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        assert index > 0
        previous = cells[index - 1]
        assert previous.get("cell_type") == "markdown"
        markdown = "".join(previous.get("source", []))
        for required_phrase in (
            "What this step does:",
            "Required inputs:",
            "Produced outputs:",
            "When this step may be skipped:",
        ):
            assert required_phrase in markdown


def test_colab_notebook_separates_reuse_extract_and_run_cells() -> None:
    cell_ids = {str(cell.get("id")) for cell in _load_notebook_cells()}

    required_ids = {
        "runtime-settings-code",
        "runtime-tools-code",
        "drive-mount-code",
        "repo-acquisition-code",
        "repo-install-code",
        "shared-helper-code",
        "dataset-reuse-code",
        "dataset-extract-code",
        "dataset-run-code",
        "baseline-prepare-code",
        "training-reuse-code",
        "training-extract-code",
        "training-run-code",
        "qualitative-reuse-code",
        "qualitative-extract-code",
        "qualitative-run-code",
        "record-reuse-code",
        "record-extract-code",
        "record-run-code",
        "final-inspection-code",
    }
    assert required_ids.issubset(cell_ids)

    cells_by_id = {
        str(cell.get("id")): "".join(cell.get("source", [])) for cell in _load_notebook_cells()
    }
    assert "def extract_baseline_archive(" in cells_by_id["shared-helper-code"]
    assert "def merge_extracted_tree(" in cells_by_id["shared-helper-code"]
    assert "run_dataset_build(" not in cells_by_id["dataset-extract-code"]
    assert "run_dataset_build(" in cells_by_id["dataset-run-code"]
    assert "def extract_baseline_archive(" not in cells_by_id["training-extract-code"]
    assert "run_baseline_modeling(" not in cells_by_id["training-extract-code"]
    assert "run_baseline_modeling(" in cells_by_id["training-run-code"]
    assert "run_baseline_modeling(" not in cells_by_id["qualitative-extract-code"]
    assert "run_baseline_modeling(" in cells_by_id["qualitative-run-code"]
    assert "run_baseline_modeling(" not in cells_by_id["record-extract-code"]
    assert "run_baseline_modeling(" in cells_by_id["record-run-code"]


def test_public_workflow_surface_has_one_main_notebook_and_stage_clis() -> None:
    colab_notebooks = sorted(
        path.name for path in (PROJECT_ROOT / "notebooks/colab").glob("*.ipynb")
    )
    script_names = sorted(path.name for path in (PROJECT_ROOT / "scripts").glob("*.py"))

    assert colab_notebooks == [NOTEBOOK_NAME]
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


def test_public_docs_reference_current_notebook_only() -> None:
    for path in _durable_public_text_paths():
        source = path.read_text(encoding="utf-8")
        assert OLD_NOTEBOOK_NAME not in source, path

    docs_expected_to_reference_notebook = (
        "README.md",
        "docs/index.md",
        "docs/getting-started.md",
        "docs/repository-structure.md",
        "docs/execution/dataset-build.md",
        "docs/execution/baseline-modeling.md",
    )
    for relative_path in docs_expected_to_reference_notebook:
        assert CURRENT_NOTEBOOK_REFERENCE in _read_repo_file(relative_path), relative_path


def test_public_docs_preserve_current_status_wording() -> None:
    docs_expected_to_state_public_status = (
        "README.md",
        "docs/index.md",
        "docs/repository-structure.md",
        "docs/experiments.md",
    )

    for relative_path in docs_expected_to_state_public_status:
        source = _read_repo_file(relative_path)
        for expected_line in PUBLIC_STATUS_LINES:
            assert expected_line in source, relative_path


def test_readme_public_surface_structure_is_stable() -> None:
    source = _read_repo_file("README.md")

    required_headings = (
        "## What This Repository Is",
        "## Current Public Workflow Surfaces",
        "## Current Maturity And Boundaries",
        "## Installation Matrix",
        "## Quick Starts",
        "### Dataset Build",
        "### Baseline Modeling",
        "### Colab Heavy-Run Path",
        "## Docs Map",
    )
    for heading in required_headings:
        assert heading in source

    required_snippets = (
        *PUBLIC_STATUS_LINES,
        "Dataset Build CLI: `python scripts/dataset_build.py`",
        "Baseline Modeling CLI: `python scripts/baseline_modeling.py "
        "[prepare|train|export-panel|package|all]`",
        "`python -m pip install -e .`",
        '`python -m pip install -e ".[dev]"`',
        '`python -m pip install -e ".[docs]"`',
        '`python -m pip install -e ".[modeling]"`',
        '`python -m pip install -e ".[dev,docs,modeling]"`',
        CURRENT_NOTEBOOK_REFERENCE,
    )
    for snippet in required_snippets:
        assert snippet in source


def test_public_docs_use_baseline_modeling_record_names() -> None:
    old_record_names = (
        "sprint3-baseline-record-guide.md",
        "sprint3-baseline-record-template.md",
        "experiments/sprint3-baseline-record-guide.md",
        "experiments/sprint3-baseline-record-template.md",
        "Sprint 3 Baseline Record Guide",
        "Sprint 3 Baseline Record Template",
        "Sprint 3 baseline record guide",
        "Sprint 3 baseline record template",
    )
    for path in _durable_public_text_paths():
        source = path.read_text(encoding="utf-8")
        for old_record_name in old_record_names:
            assert old_record_name not in source, path

    old_guide = PROJECT_ROOT / "docs/experiments/sprint3-baseline-record-guide.md"
    old_template = PROJECT_ROOT / "docs/experiments/sprint3-baseline-record-template.md"
    new_guide = PROJECT_ROOT / "docs/experiments/baseline-modeling-record-guide.md"
    new_template = PROJECT_ROOT / "docs/experiments/baseline-modeling-record-template.md"
    assert not old_guide.exists()
    assert not old_template.exists()
    assert new_guide.is_file()
    assert new_template.is_file()

    mkdocs_source = _read_repo_file("mkdocs.yml")
    experiments_source = _read_repo_file("docs/experiments.md")
    for source in (mkdocs_source, experiments_source):
        has_record_guide_label = (
            "Baseline Modeling Record Guide" in source or "Baseline Modeling record guide" in source
        )
        assert has_record_guide_label
        assert (
            "Baseline Modeling Record Template" in source
            or "Baseline Modeling record template" in source
        )
        assert "experiments/baseline-modeling-record-guide.md" in source
        assert "experiments/baseline-modeling-record-template.md" in source

    assert "# Baseline Modeling Record Guide" in new_guide.read_text(encoding="utf-8")
    assert "# Baseline Modeling Record Template" in new_template.read_text(encoding="utf-8")
