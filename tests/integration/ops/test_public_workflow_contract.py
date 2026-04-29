"""Transition contract for the public notebook-facing workflow surface.

This file remains under ``tests/integration/ops`` temporarily for continuity; the
assertions are about the public workflow contract, not an ``ops`` Python package.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import pytest

pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[3]
COLAB_ROOT = PROJECT_ROOT / "notebooks" / "colab"
TARGET_NOTEBOOK_NAMES = (
    "text_to_sign_dataset_colab.ipynb",
    "text_to_sign_visualization_colab.ipynb",
    "text_to_sign_base_colab.ipynb",
)
TARGET_NOTEBOOK_PATHS = tuple(COLAB_ROOT / name for name in TARGET_NOTEBOOK_NAMES)
OLD_PUBLIC_NOTEBOOK_NAMES = (
    "text_to_sign_production_colab.ipynb",
    "text_to_sign_visual_debug_colab.ipynb",
)
OLD_NOTEBOOK_NAME = "dataset_build_colab.ipynb"
ACTIVE_PUBLIC_SURFACES = (
    "README.md",
    "docs/index.md",
    "docs/execution.md",
    "docs/experiments/index.md",
    "docs/decisions/index.md",
)
FORBIDDEN_ACTIVE_PUBLIC_SNIPPETS = (
    "C1 = Dynamic VQ Pose Tokens",
    "C2 = Channel-Aware Loss Reweighting",
    "current selected pair",
    "selected pair",
    "selected-pair implementation",
    "current thesis-facing direction is fixed",
    "Sprint 5 through Sprint 8",
    "Current public stage: Dataset Build",
    "Current public stage: Baseline Modeling",
    "Implemented internal downstream surface",
    "bibliography",
)


def test_public_notebooks_are_final_three_notebook_surface_and_valid_json() -> None:
    colab_notebooks = sorted(path.name for path in COLAB_ROOT.glob("*.ipynb"))

    assert colab_notebooks == sorted(TARGET_NOTEBOOK_NAMES)

    for path in TARGET_NOTEBOOK_PATHS:
        loaded = json.loads(path.read_text(encoding="utf-8"))

        assert isinstance(loaded, dict)
        assert isinstance(loaded.get("cells"), list)

    for old_name in OLD_PUBLIC_NOTEBOOK_NAMES:
        assert old_name not in colab_notebooks


def test_public_surface_has_notebook_workflows_without_script_or_ops_packages() -> None:
    colab_notebooks = sorted(path.name for path in COLAB_ROOT.glob("*.ipynb"))
    scripts_dir = PROJECT_ROOT / "scripts"

    assert not scripts_dir.exists()
    for target_name in TARGET_NOTEBOOK_NAMES:
        assert target_name in colab_notebooks
    assert OLD_NOTEBOOK_NAME not in colab_notebooks

    assert (PROJECT_ROOT / "src/text_to_sign_production/workflows/dataset.py").is_file()
    assert (PROJECT_ROOT / "src/text_to_sign_production/workflows/visualization.py").is_file()
    assert (PROJECT_ROOT / "src/text_to_sign_production/workflows/base.py").is_file()

    removed_workflow_modules = (
        PROJECT_ROOT / "src/text_to_sign_production/workflows/baseline_modeling.py",
        PROJECT_ROOT / "src/text_to_sign_production/workflows/dataset_build.py",
        PROJECT_ROOT / "src/text_to_sign_production/workflows/_baseline_archive_ops.py",
    )
    for module_path in removed_workflow_modules:
        assert not module_path.exists(), module_path

    assert not (PROJECT_ROOT / "src/text_to_sign_production/ops").exists()

    removed_python_archive_helpers = (
        PROJECT_ROOT / "src/text_to_sign_production/ops/archive_ops.py",
        PROJECT_ROOT / "src/text_to_sign_production/ops/colab_workflow.py",
        PROJECT_ROOT / "src/text_to_sign_production/workflows/_baseline_archive_ops.py",
        PROJECT_ROOT / "tests/support/builders/archives.py",
    )
    for helper_path in removed_python_archive_helpers:
        assert not helper_path.exists(), helper_path


def test_target_notebooks_call_public_workflows_and_not_old_helpers() -> None:
    notebook_code = _joined_notebook_code_sources(TARGET_NOTEBOOK_PATHS)
    required_snippets = (
        "text_to_sign_production.workflows.dataset",
        "DatasetWorkflowConfig",
        "validate_dataset_inputs",
        "run_dataset_workflow",
        "text_to_sign_production.workflows.visualization",
        "VisualizationSelectionConfig",
        "VisualizationWorkflowConfig",
        "select_visualization_sample",
        "validate_visualization_inputs",
        "run_visualization_workflow",
        "text_to_sign_production.workflows.base",
        "BaseWorkflowConfig",
        "validate_base_inputs",
        "run_base_workflow",
    )
    forbidden_snippets = (
        "text_to_sign_production.ops",
        "archive_ops",
        "colab_workflow",
        "scripts.",
        "workflows.dataset_build",
        "workflows.baseline_modeling",
        "_baseline_archive_ops",
        "extract_tar_zst_with_progress",
        "restore_tar_zst_members",
        "restore_processed_visual_debug_inputs",
        "ensure_bfh_clip_videos",
        "merge_extracted_tree",
        "extract_baseline_archive",
    )

    for snippet in required_snippets:
        assert snippet in notebook_code, snippet
    for snippet in forbidden_snippets:
        assert snippet not in notebook_code, snippet


def test_public_surface_uses_core_layout_and_mirrored_data_tree() -> None:
    notebook_code = _joined_notebook_code_sources(TARGET_NOTEBOOK_PATHS)
    src_source = _joined_python_sources(PROJECT_ROOT / "src")

    for snippet in (
        "ProjectLayout",
        "runtime_layout = ProjectLayout(WORKTREE_ROOT)",
        "drive_layout = ProjectLayout(DRIVE_PROJECT_ROOT)",
        "drive_layout.raw_bfh_keypoints_archive",
        "drive_layout.processed_sample_archive",
        "drive_layout.processed_manifests_reports_archive",
        '-C "{WORKTREE_ROOT}"',
    ):
        assert snippet in notebook_code, snippet

    for snippet in (
        "class ProjectLayout",
        "def processed_sample_archive",
        "def processed_manifests_reports_archive",
        "def base_run_archive",
        "OutputExistsPolicy",
        "prepare_output_file",
        "prepare_output_dir",
    ):
        assert snippet in src_source, snippet


def test_public_surface_has_no_old_runtime_or_drive_layout_strings() -> None:
    source = "\n".join(
        [
            _joined_notebook_code_sources(TARGET_NOTEBOOK_PATHS),
            _joined_python_sources(PROJECT_ROOT / "src"),
        ]
    )
    forbidden_snippets = (
        "/content/t2sp-runtime",
        "/content/text_to_sign_runtime",
        "raw/how2sign/archives",
        "artifacts/dataset-build",
        'DRIVE_PROJECT_ROOT / "artifacts"',
        'DRIVE_PROJECT_ROOT / "raw"',
    )
    for snippet in forbidden_snippets:
        assert snippet not in source, snippet


def test_public_contract_has_no_test_imports_from_deleted_surfaces() -> None:
    tests_source = _joined_python_import_lines(PROJECT_ROOT / "tests")
    forbidden_import_patterns = (
        re.compile(r"\bfrom\s+scripts\b"),
        re.compile(r"\bimport\s+scripts\b"),
        re.compile(r"\bscripts[.]"),
        re.compile(r"\bworkflows[.]dataset_build\b"),
        re.compile(r"\bworkflows[.]baseline_modeling\b"),
        re.compile(r"\bworkflows[.]_baseline_archive_ops\b"),
    )
    for pattern in forbidden_import_patterns:
        assert pattern.search(tests_source) is None, pattern.pattern


def test_public_contract_has_no_archive_helper_source_surface() -> None:
    src_source = _joined_python_sources(PROJECT_ROOT / "src")
    forbidden_snippets = (
        "archive_ops",
        "colab_workflow",
        "extract_tar_zst",
        "restore_tar_zst",
        "create_tar_zst",
        "copy_archive",
        "stage_colab_inputs",
        "package_dataset_build_outputs",
        "publish_colab_outputs",
        "restore_processed_visual_debug_inputs",
        "ensure_bfh_clip_videos",
    )
    for snippet in forbidden_snippets:
        assert snippet not in src_source, snippet


def test_public_docs_keep_current_navigation_without_legacy_dataset_build_notebook() -> None:
    for path in _durable_public_text_paths():
        source = path.read_text(encoding="utf-8")
        assert OLD_NOTEBOOK_NAME not in source, path

    docs_expected_to_route_to_execution = {
        "README.md": "[Execution](docs/execution.md)",
        "docs/index.md": "[Execution](execution.md)",
    }
    for relative_path, expected_link in docs_expected_to_route_to_execution.items():
        source = _read_repo_file(relative_path)
        assert expected_link in source, relative_path


def test_public_surfaces_reject_stale_governance_framing_during_transition() -> None:
    sources = {
        relative_path: _read_repo_file(relative_path) for relative_path in ACTIVE_PUBLIC_SURFACES
    }
    for notebook_path in TARGET_NOTEBOOK_PATHS:
        notebook_markdown, notebook_code = _load_notebook_sources(notebook_path)
        notebook_reference = f"notebooks/colab/{notebook_path.name}"
        sources[f"notebook markdown: {notebook_reference}"] = notebook_markdown
        sources[f"notebook code: {notebook_reference}"] = notebook_code

    for surface_name, source in sources.items():
        normalized_source = source.lower()
        for forbidden_snippet in FORBIDDEN_ACTIVE_PUBLIC_SNIPPETS:
            assert forbidden_snippet.lower() not in normalized_source, surface_name


def test_public_surfaces_explain_operational_not_stage_status() -> None:
    execution_source = _read_repo_file("docs/execution.md")
    experiments_source = _read_repo_file("docs/experiments/index.md")

    assert "Implemented operational surfaces:" in execution_source
    assert "- Dataset Build" in execution_source
    assert "- Baseline Modeling" in execution_source
    assert (
        "Operational documentation currently covers Dataset Build and Baseline Modeling workflows."
        in experiments_source
    )


def test_public_docs_reference_repository_map_surface() -> None:
    readme_source = _read_repo_file("README.md")
    getting_started_source = _read_repo_file("docs/getting-started.md")
    mkdocs_source = _read_repo_file("mkdocs.yml")

    assert "Repository Map" in readme_source
    assert "docs/repository-map.md" in readme_source
    assert "Repository Map" in getting_started_source
    assert "repository-map.md" in getting_started_source
    assert "Repository Map: repository-map.md" in mkdocs_source


def test_readme_public_surface_structure_is_stable_landing_page() -> None:
    source = _read_repo_file("README.md")

    required_headings = (
        "# text-to-sign-production",
        "## Documentation",
        "## Current Research Boundary",
        "## Scope",
        "## Development",
        "## License",
    )
    for heading in required_headings:
        assert heading in source

    required_snippets = (
        "stable repository landing page",
        "[Docs Home](docs/index.md)",
        "[Research Context](docs/research/index.md)",
        "[Research Roadmap](docs/research/roadmap.md)",
        "[Source Corpus](docs/research/source-corpus.md)",
        "[Contribution Audit Result](docs/research/contribution-audit/audit-result.md)",
        "[Execution](docs/execution.md)",
        "[Data](docs/data/index.md)",
        "[Experiments](docs/experiments/index.md)",
        "[Testing](docs/testing/index.md)",
        "[Decisions](docs/decisions/index.md)",
        "[Repository Map](docs/repository-map.md)",
        "No final implementation model is selected.",
        "No final candidate ranking is assigned.",
        "No experimental proof of sign-language intelligibility is claimed.",
        "No released public model artifact is claimed",
        "Development Setup",
    )
    for snippet in required_snippets:
        assert snippet in source

    snapshot_headings = (
        "## Current Status",
        "## Implemented Surfaces",
        "## Selected Current Research Direction",
        "## Supported Workflow",
        "## Artifact / Publication / Reproducibility",
    )
    for heading in snapshot_headings:
        assert heading not in source


def test_docs_index_public_surface_is_navigation_overview() -> None:
    source = _read_repo_file("docs/index.md")

    required_snippets = (
        "# text-to-sign-production Docs",
        "## What These Docs Cover",
        "Use this page for navigation.",
        "Phase-specific status, experiment evidence, and operational",
        "instructions live in the canonical leaf docs",
        "does not select a final model",
        "assign a final candidate ranking",
        "or claim",
        "experimental proof of sign-language intelligibility",
        "[Research Context](research/index.md)",
        "[Research Roadmap](research/roadmap.md)",
        "[Source Corpus](research/source-corpus.md)",
        "[Contribution Audit Result](research/contribution-audit/audit-result.md)",
        "[Execution](execution.md)",
        "[Data](data/index.md)",
        "[Experiments](experiments/index.md)",
        "[Testing](testing/index.md)",
        "[Decisions](decisions/index.md)",
        "[Development Setup](development-setup.md)",
        "[Repository Map](repository-map.md)",
    )
    for snippet in required_snippets:
        assert snippet in source

    status_dashboard_snippets = (
        "## Current Repository State",
        "Dataset Build is completed",
        "Baseline Modeling is completed as the implemented baseline milestone",
        "## Current Research Direction",
    )
    for snippet in status_dashboard_snippets:
        assert snippet not in source


def test_decisions_index_routes_to_current_governance_and_superseded_history() -> None:
    source = _read_repo_file("docs/decisions/index.md")

    assert "[ADR Template](template.md)" in source
    assert "ADRs are historical decision records, not the active operator guide." in source
    assert "ADR-0023: Phase-Based Research Governance And Public Documentation Boundaries" in source
    assert "current Phase 0-12 research frame and public documentation boundaries" in source

    superseded_records = (
        "ADR-0013",
        "ADR-0018",
        "ADR-0019",
        "ADR-0020",
        "ADR-0021",
        "ADR-0022",
    )
    for record_id in superseded_records:
        line = next(line for line in source.splitlines() if record_id in line)
        assert "superseded historical" in line


def test_public_docs_use_canonical_experiments_surface() -> None:
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

    experiments_root = PROJECT_ROOT / "docs" / "experiments"
    obsolete_paths = (
        PROJECT_ROOT / "docs" / "experiments.md",
        experiments_root / "README.md",
        experiments_root / ("baseline-modeling-record" + "-guide.md"),
        experiments_root / ("baseline-modeling-record" + "-template.md"),
        experiments_root / ("sprint3-baseline-record" + "-guide.md"),
        experiments_root / ("sprint3-baseline-record" + "-template.md"),
    )
    for obsolete_path in obsolete_paths:
        assert not obsolete_path.exists()

    experiments_dir = PROJECT_ROOT / "docs/experiments"
    record_pattern = re.compile(r"\d{4}-\d{2}-[a-z0-9-]+-experiment-record-[a-z0-9-]+\.md")
    experiment_files = sorted(path.name for path in experiments_dir.glob("*.md"))
    assert "index.md" in experiment_files
    assert "template.md" in experiment_files
    assert all(
        name in {"index.md", "template.md"} or record_pattern.fullmatch(name)
        for name in experiment_files
    )
    record_files = [name for name in experiment_files if name not in {"index.md", "template.md"}]
    assert record_files

    mkdocs_source = _read_repo_file("mkdocs.yml")
    experiments_index_source = _read_repo_file("docs/experiments/index.md")
    assert "experiments/index.md" in mkdocs_source
    assert "experiments/template.md" in mkdocs_source

    assert "template.md" in experiments_index_source
    for record_file in record_files:
        assert f"experiments/{record_file}" in mkdocs_source
        assert record_file in experiments_index_source

    new_index = PROJECT_ROOT / "docs/experiments/index.md"
    new_template = PROJECT_ROOT / "docs/experiments/template.md"
    assert "# Experiment Records" in new_index.read_text(encoding="utf-8")
    assert "# Experiment Record Template" in new_template.read_text(encoding="utf-8")


def test_public_contract_notes_notebook_rewrite_is_no_longer_deferred() -> None:
    for notebook_path in TARGET_NOTEBOOK_PATHS:
        notebook = _load_notebook(notebook_path)

        assert notebook["cells"]

    for old_name in OLD_PUBLIC_NOTEBOOK_NAMES:
        assert not (COLAB_ROOT / old_name).exists(), old_name


def _load_notebook_sources(path: Path) -> tuple[str, str]:
    notebook = _load_notebook(path)
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


def _load_notebook(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError(f"Notebook root must be a JSON object: {path}")
    return cast(dict[str, Any], loaded)


def _read_repo_file(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def _joined_python_sources(root: Path) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(root.rglob("*.py"))
        if "__pycache__" not in path.parts
    )


def _joined_python_import_lines(root: Path) -> str:
    return "\n".join(
        line
        for path in sorted(root.rglob("*.py"))
        if "__pycache__" not in path.parts
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith(("import ", "from "))
    )


def _joined_notebook_code_sources(paths: tuple[Path, ...]) -> str:
    return "\n".join(_load_notebook_sources(path)[1] for path in paths)


@lru_cache(maxsize=1)
def _durable_public_text_paths() -> list[Path]:
    return [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "mkdocs.yml",
        *sorted((PROJECT_ROOT / "docs").rglob("*.md")),
        *sorted((PROJECT_ROOT / "tests/operational").rglob("*.md")),
    ]
