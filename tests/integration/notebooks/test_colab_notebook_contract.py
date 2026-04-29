"""Contract tests for the public Colab notebook surface."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import nbformat
import pytest

pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[3]
COLAB_ROOT = PROJECT_ROOT / "notebooks" / "colab"

TARGET_NOTEBOOKS = {
    "dataset": COLAB_ROOT / "text_to_sign_dataset_colab.ipynb",
    "visualization": COLAB_ROOT / "text_to_sign_visualization_colab.ipynb",
    "base": COLAB_ROOT / "text_to_sign_base_colab.ipynb",
}
OLD_NOTEBOOKS = (
    COLAB_ROOT / "text_to_sign_production_colab.ipynb",
    COLAB_ROOT / "text_to_sign_visual_debug_colab.ipynb",
)
CONTRACT_PHRASES = (
    "What this step does",
    "Required inputs",
    "Produced outputs",
    "When this step may be skipped",
)
FORBIDDEN_STRINGS = (
    "%%bash",
    "subprocess",
    "/content/t2sp-runtime",
    "/content/text_to_sign_runtime",
    "/content/runtime",
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
FORBIDDEN_PUBLIC_INPUTS = (
    "DATA_ROOT",
    "RUNTIME_ROOT",
    "OUTPUT_ROOT",
    "DATASET_ARTIFACT_ROOT",
    "RAW_BFH_ARCHIVE_ROOT",
    "BASE_ARTIFACT_ROOT",
)
CRITICAL_COMMAND_PREFIXES = (
    "!mkdir -p",
    "!rm -rf",
    "!git clone",
    "!git checkout",
    "!git -C",
    "!tar --zstd -xf",
    "!tar --zstd -cf",
    "!cp",
)


def test_required_colab_notebooks_exist_and_are_valid_nbformat() -> None:
    for path in TARGET_NOTEBOOKS.values():
        assert path.is_file(), path
        notebook = _read_notebook(path)

        nbformat.validate(notebook)


def test_old_public_colab_notebooks_are_absent() -> None:
    for path in OLD_NOTEBOOKS:
        assert not path.exists(), path


@pytest.mark.parametrize("path", TARGET_NOTEBOOKS.values(), ids=TARGET_NOTEBOOKS.keys())
def test_code_cells_have_previous_markdown_contract(path: Path) -> None:
    cells = _read_notebook(path).cells
    for index, cell in enumerate(cells):
        if cell.cell_type != "code":
            continue
        assert index > 0, path
        previous = cells[index - 1]
        assert previous.cell_type == "markdown", path
        for phrase in CONTRACT_PHRASES:
            assert phrase in previous.source, (path, index, phrase)


def test_required_workflow_and_core_imports_are_present() -> None:
    expected = {
        "dataset": (
            "text_to_sign_production.workflows.dataset",
            "DatasetWorkflowConfig",
            "validate_dataset_inputs",
            "run_dataset_workflow",
        ),
        "visualization": (
            "text_to_sign_production.workflows.visualization",
            "VisualizationSelectionConfig",
            "VisualizationWorkflowConfig",
            "select_visualization_sample",
            "validate_visualization_inputs",
            "run_visualization_workflow",
        ),
        "base": (
            "text_to_sign_production.workflows.base",
            "BaseWorkflowConfig",
            "validate_base_inputs",
            "run_base_workflow",
        ),
    }
    for notebook_name, snippets in expected.items():
        code = _joined_code(TARGET_NOTEBOOKS[notebook_name])
        assert "from text_to_sign_production.core.paths import ProjectLayout" in code
        assert "from text_to_sign_production.core.files import" in code
        assert "runtime_layout = ProjectLayout(WORKTREE_ROOT)" in code
        assert "drive_layout = ProjectLayout(DRIVE_PROJECT_ROOT)" in code
        for snippet in snippets:
            assert snippet in code, (notebook_name, snippet)


@pytest.mark.parametrize("path", TARGET_NOTEBOOKS.values(), ids=TARGET_NOTEBOOKS.keys())
def test_target_notebooks_avoid_deleted_or_forbidden_surfaces(path: Path) -> None:
    source = _joined_source(path)

    for snippet in FORBIDDEN_STRINGS:
        assert snippet not in source, (path, snippet)


@pytest.mark.parametrize("path", TARGET_NOTEBOOKS.values(), ids=TARGET_NOTEBOOKS.keys())
def test_target_notebooks_use_colab_command_style(path: Path) -> None:
    code = _joined_code(path)

    for snippet in ("%cd", "%pip", "!git", "!tar --zstd -xf"):
        assert snippet in code, (path, snippet)
    assert "%%bash" not in code
    assert "subprocess" not in code


@pytest.mark.parametrize("path", TARGET_NOTEBOOKS.values(), ids=TARGET_NOTEBOOKS.keys())
def test_critical_colab_commands_have_exit_code_checks(path: Path) -> None:
    notebook = _read_notebook(path)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue

        critical_commands = [
            command for command in _bang_commands(str(cell.source)) if _is_critical_command(command)
        ]
        if not critical_commands:
            continue

        assert "_exit_code" in cell.source, (path, index, critical_commands)
        assert "RuntimeError" in cell.source, (path, index, critical_commands)


def test_notebooks_expose_only_allowed_public_inputs() -> None:
    allowed = {
        "dataset": {"REPO_URL", "REPO_REF", "DATASET_SPLITS"},
        "visualization": {
            "REPO_URL",
            "REPO_REF",
            "RENDER_MODE",
            "CONFIDENCE_THRESHOLD",
            "OUTPUT_FILENAME",
        },
        "base": {
            "REPO_URL",
            "REPO_REF",
            "BASE_RUN_NAME",
            "RUN_QUALITATIVE_EXPORT",
            "PANEL_SIZE",
        },
    }
    derived = {"WORKTREE_ROOT", "DRIVE_PROJECT_ROOT"}
    for notebook_name, path in TARGET_NOTEBOOKS.items():
        first_code = next(
            cell.source for cell in _read_notebook(path).cells if cell.cell_type == "code"
        )
        assignments = set(re.findall(r"^([A-Z][A-Z0-9_]+)\s*=", first_code, re.MULTILINE))
        assert assignments <= allowed[notebook_name] | derived, (notebook_name, assignments)
        for forbidden in FORBIDDEN_PUBLIC_INPUTS:
            assert forbidden not in _joined_code(path), (notebook_name, forbidden)


def test_dataset_notebook_uses_mirrored_layout_for_raw_restore_and_publish() -> None:
    code = _joined_code(TARGET_NOTEBOOKS["dataset"])

    for snippet in (
        "drive_layout.how2sign_translation_file(split)",
        "runtime_layout.how2sign_translation_file(split)",
        "drive_layout.raw_bfh_keypoints_archive(split)",
        "runtime_layout.raw_bfh_keypoints_split_root(split)",
        "DatasetWorkflowConfig(",
        "project_root=WORKTREE_ROOT",
        "drive_layout.processed_manifests_reports_archive()",
        "drive_layout.processed_sample_archive(split)",
        '!tar --zstd -xf "{archive}" -C "{split_root}"',
        "openpose_root = runtime_layout.raw_bfh_keypoints_split_root(split)",
        "split_root = openpose_root.parent",
        'require_dir_contains(openpose_root / "json"',
        'require_dir_contains(openpose_root / "video"',
        "dataset_result.processed_sample_archive_members[split]",
        "runtime_layout.dataset_artifacts_root",
        '!tar --zstd -cf "{sample_archive}" -C "{WORKTREE_ROOT}" -T "{archive_member_list}"',
    ):
        assert snippet in code, snippet


def test_dataset_notebook_command_checks_fail_closed() -> None:
    code = _joined_code(TARGET_NOTEBOOKS["dataset"])

    assert 'getattr(_output, "exit_code", 0)' not in code
    assert 'getattr(_output, "exit_code", 1)' in code


def test_dataset_notebook_preflights_zstd_before_tar_usage() -> None:
    code = _joined_code(TARGET_NOTEBOOKS["dataset"])

    zstd_preflight = code.index('shutil.which("zstd")')
    first_tar = code.index("!tar --zstd")

    assert zstd_preflight < first_tar
    for snippet in (
        'shutil.which("zstd")',
        "!apt-get install -y zstd",
        "_exit_code",
    ):
        assert snippet in code, snippet


def test_dataset_notebook_restores_raw_bfh_source_archives_without_member_preflight() -> None:
    code = _joined_code(TARGET_NOTEBOOKS["dataset"])
    raw_extraction = code.index('!tar --zstd -xf "{archive}" -C "{split_root}"')
    first_archive_inspection = code.index("!tar --zstd -tf")

    assert '!tar --zstd -tf "{archive}"' not in code
    assert raw_extraction < first_archive_inspection
    for snippet in (
        "drive_layout.raw_bfh_keypoints_archive",
        "runtime_layout.raw_bfh_keypoints_split_root",
        'openpose_root.name != "openpose_output"',
        'require_dir_contains(openpose_root / "json"',
        'require_dir_contains(openpose_root / "video"',
    ):
        assert snippet in code, snippet


def test_dataset_notebook_raw_input_cells_are_single_responsibility() -> None:
    path = TARGET_NOTEBOOKS["dataset"]

    translation_copy_cell = _single_code_cell_containing(path, "!cp", "RAW_TRANSLATION_SOURCES")
    assert "!tar" not in translation_copy_cell

    raw_archive_locate_cell = _single_code_cell_containing(
        path,
        "RAW_BFH_ARCHIVES",
        "drive_layout.raw_bfh_keypoints_archive",
    )
    assert "!tar --zstd -xf" not in raw_archive_locate_cell
    assert "!tar --zstd -tf" not in raw_archive_locate_cell

    raw_extraction_cell = _single_code_cell_containing(
        path,
        "RAW_BFH_ARCHIVES.items()",
        '!tar --zstd -xf "{archive}" -C "{split_root}"',
    )
    assert "!tar --zstd -tf" not in raw_extraction_cell
    assert 'require_dir_contains(openpose_root / "json"' not in raw_extraction_cell
    assert "Archive member paths must be project-root-relative" not in raw_extraction_cell

    raw_layout_verification_cell = _single_code_cell_containing(
        path,
        'require_dir_contains(openpose_root / "json"',
        'require_dir_contains(openpose_root / "video"',
    )
    assert "!tar" not in raw_layout_verification_cell


def test_dataset_notebook_verifies_processed_sample_archives_from_manifest_members() -> None:
    code = _joined_code(TARGET_NOTEBOOKS["dataset"])

    for snippet in (
        "dataset_result.processed_sample_archive_members[split]",
        "PROCESSED_SAMPLE_ARCHIVE_MEMBER_LISTS",
        'not member.startswith("data/processed/v1/samples/")',
        '!tar --zstd -cf "{sample_archive}" -C "{WORKTREE_ROOT}" -T "{archive_member_list}"',
        '!tar --zstd -tf "{sample_archive}"',
        "observed_members != expected_members",
        "sample archive members do not match the final processed manifest",
    ):
        assert snippet in code, snippet


def test_visualization_notebook_restores_in_manifest_selection_order() -> None:
    code = _joined_code(TARGET_NOTEBOOKS["visualization"])

    manifest_restore = code.index("drive_layout.processed_manifests_reports_archive()")
    selection_call = code.index("selected_sample = select_visualization_sample(selection_config)")
    split_binding = code.index("selected_split = selected_sample.record.split")
    sample_archive = code.index("drive_layout.processed_sample_archive(selected_split)")
    sample_restore = code.index('!tar --zstd -xf "{SAMPLE_ARCHIVE}" -C "{WORKTREE_ROOT}"')

    assert manifest_restore < selection_call < split_binding < sample_archive < sample_restore


def test_visualization_notebook_uses_layout_for_restore_render_and_publish() -> None:
    code = _joined_code(TARGET_NOTEBOOKS["visualization"])

    for snippet in (
        "drive_layout.raw_bfh_keypoints_archive(selected_split)",
        "VisualizationSelectionConfig(",
        "project_root=WORKTREE_ROOT",
        "runtime_layout.visualization_artifacts_root",
        "drive_layout.visualization_artifacts_root",
        "PUBLISHED_VISUALIZATION_PATH.stat().st_size <= 0",
    ):
        assert snippet in code, snippet


def test_base_notebook_uses_layout_for_restore_run_and_publish() -> None:
    code = _joined_code(TARGET_NOTEBOOKS["base"])

    for snippet in (
        "drive_layout.processed_manifests_reports_archive()",
        'drive_layout.processed_sample_archive("train")',
        'drive_layout.processed_sample_archive("val")',
        'runtime_layout.processed_manifest_path("train")',
        'runtime_layout.processed_samples_split_root("train")',
        "BaseWorkflowConfig(",
        "project_root=WORKTREE_ROOT",
        "runtime_layout.base_run_root(BASE_RUN_NAME)",
        "drive_layout.base_run_archive(BASE_RUN_NAME)",
        '!tar --zstd -cf "{PUBLISHED_BASE_ARCHIVE}" -C "{WORKTREE_ROOT}" "{member}"',
        "PUBLISHED_BASE_ARCHIVE.stat().st_size <= 0",
    ):
        assert snippet in code, snippet


def _joined_source(path: Path) -> str:
    notebook = _read_notebook(path)
    return "\n".join(str(cell.source) for cell in notebook.cells)


def _joined_code(path: Path) -> str:
    notebook = _read_notebook(path)
    return "\n".join(str(cell.source) for cell in notebook.cells if cell.cell_type == "code")


def _code_cells(path: Path) -> list[str]:
    notebook = _read_notebook(path)
    return [str(cell.source) for cell in notebook.cells if cell.cell_type == "code"]


def _read_notebook(path: Path) -> Any:
    return nbformat.read(path, as_version=4)  # type: ignore[no-untyped-call]


def _single_code_cell_containing(path: Path, *snippets: str) -> str:
    matches = [
        source for source in _code_cells(path) if all(snippet in source for snippet in snippets)
    ]
    assert len(matches) == 1, (path, snippets, len(matches))
    return matches[0]


def _bang_commands(source: str) -> list[str]:
    return [line.lstrip() for line in source.splitlines() if line.lstrip().startswith("!")]


def _is_critical_command(command: str) -> bool:
    return any(command.startswith(prefix) for prefix in CRITICAL_COMMAND_PREFIXES)
