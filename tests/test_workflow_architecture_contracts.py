"""Static workflow architecture contract checks."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import fields
from pathlib import Path

from text_to_sign_production import workflows
from text_to_sign_production.workflows import base, commands, dataset, visualization

REPO_ROOT = Path(__file__).resolve().parents[1]

BASE_PUBLIC_FUNCTIONS = {
    "build_base_runtime_plan",
    "verify_base_runtime_inputs",
    "validate_base_inputs",
    "run_base_workflow",
    "summarize_base_workflow_outputs",
    "build_base_publish_plan",
    "write_base_checkpoint_manifest",
    "write_base_prediction_sample_archive_member_list",
    "write_base_qualitative_sample_archive_member_list",
    "summarize_base_publish_plan",
}

BASE_PUBLIC_TYPES = {
    "BaseWorkflowConfig",
    "BaseWorkflowResult",
    "BaseRuntimePlan",
    "BaseRunLayout",
    "BaseProcessedManifestRestoreSpec",
    "BaseProcessedSampleArchiveRestoreSpec",
    "BaseProcessedRestoreVerificationSummary",
    "BaseProcessedSplitInputVerification",
    "BasePredictionSplitVerification",
    "BaseRunVerificationSummary",
    "BaseQualitativeVerificationSummary",
    "BaseWorkflowOutputSummary",
    "BaseFilePublishSpec",
    "BaseCheckpointPublishSpec",
    "BasePredictionSampleArchiveSpec",
    "BaseQualitativeSampleArchiveSpec",
    "BasePublishPlan",
    "BasePublishedFileSummary",
    "BasePublishedCheckpointSummary",
    "BasePublishedArchiveSummary",
    "BasePublishSummary",
    "BaseWorkflowError",
    "BaseWorkflowInputError",
}

BASE_REMOVED_LOW_LEVEL_FUNCTIONS = {
    "build_base_run_layout",
    "verify_base_processed_manifest_restore",
    "verify_base_processed_sample_restore",
    "verify_base_processed_restore",
    "verify_base_file_publish",
    "verify_base_checkpoint_publish",
    "verify_base_prediction_sample_archive_file",
    "verify_base_prediction_sample_archive_members",
    "verify_base_qualitative_sample_archive_file",
    "verify_base_qualitative_sample_archive_members",
    "verify_base_run_outputs",
    "write_base_config_snapshot",
    "write_base_run_summary",
    "write_baseline_report",
    "write_failure_mode_report",
}

DATASET_PUBLIC_FUNCTIONS = {
    "build_dataset_runtime_plan",
    "verify_dataset_runtime_inputs",
    "validate_dataset_inputs",
    "run_dataset_workflow",
    "summarize_dataset_workflow_outputs",
    "build_dataset_publish_plan",
    "write_dataset_archive_member_list",
    "summarize_dataset_publish_plan",
}

DATASET_PUBLIC_TYPES = {
    "DatasetWorkflowConfig",
    "DatasetWorkflowResult",
    "DatasetWorkflowOutputSummary",
    "DatasetPublishSummary",
    "DatasetFileTransferSpec",
    "DatasetRawArchiveRestoreSpec",
    "DatasetRuntimePlan",
    "DatasetArchiveSpec",
    "DatasetFilePublishSpec",
    "DatasetPublishPlan",
    "DatasetWorkflowError",
    "DatasetWorkflowInputError",
}

DATASET_REMOVED_LOW_LEVEL_FUNCTIONS = {
    "verify_dataset_file_transfer",
    "verify_dataset_raw_restore",
    "verify_dataset_workflow_outputs",
    "verify_dataset_file_publish",
    "verify_dataset_archive_file",
    "verify_dataset_archive_members",
}

DATASET_NOTEBOOK_EXPECTED_IMPORTS = DATASET_PUBLIC_FUNCTIONS

VISUALIZATION_PUBLIC_LIFECYCLE_FUNCTIONS = {
    "build_visualization_runtime_plan",
    "verify_visualization_runtime_inputs",
    "validate_visualization_inputs",
    "run_visualization_workflow",
    "summarize_visualization_workflow_outputs",
    "build_visualization_publish_plan",
    "summarize_visualization_publish_plan",
}

VISUALIZATION_PUBLIC_PLAN_FUNCTIONS = {
    "build_visualization_sample_plan",
    "summarize_visualization_sample",
    "build_visualization_render_plan",
}

VISUALIZATION_PUBLIC_TYPES = {
    "VisualizationWorkflowConfig",
    "VisualizationRuntimePlan",
    "VisualizationRuntimeInputSummary",
    "VisualizationWorkflowResult",
    "VisualizationWorkflowOutputSummary",
    "VisualizationPublishPlan",
    "VisualizationPublishSummary",
    "VisualizationWorkflowError",
    "VisualizationWorkflowInputError",
    "VisualizationArchiveRestoreSpec",
    "VisualizationFileRestoreSpec",
    "VisualizationRawVideoRestoreSpec",
    "VisualizationSamplePlan",
    "VisualizationSampleSummary",
    "VisualizationRenderPlan",
    "VisualizationPublishSpec",
}

VISUALIZATION_REMOVED_LOW_LEVEL_FUNCTIONS = {
    "verify_visualization_processed_restore",
    "verify_visualization_sample_restore",
    "verify_visualization_raw_restore",
    "verify_visualization_raw_video_restore",
    "verify_visualization_publish",
    "verify_visualization_manifest_restore",
    "verify_visualization_workflow_outputs",
    "build_visualization_publish_spec",
    "summarize_visualization_publish_spec",
    "summarize_visualization_publish",
}

VISUALIZATION_NOTEBOOK_EXPECTED_IMPORTS = (
    VISUALIZATION_PUBLIC_LIFECYCLE_FUNCTIONS | VISUALIZATION_PUBLIC_PLAN_FUNCTIONS
)

PRIVATE_WORKFLOW_PACKAGES = ("_base", "_dataset", "_visualization", "_shared")

OLD_WORKFLOW_ARTIFACT_ROOT_PATTERNS = {
    "data/artifacts/base",
    "data/artifacts/dataset",
    "data/artifacts/visualization",
    'artifacts_root / "base"',
    'artifacts_root / "dataset"',
    'artifacts_root / "visualization"',
    "base_run.tar.zst",
}


def test_workflows_package_public_modules_remain_minimal() -> None:
    assert workflows.__all__ == ["base", "dataset", "visualization"]


def test_commands_contract_remains_visible_and_small() -> None:
    assert commands.__all__ == ["CommandSpec", "command"]
    assert [field.name for field in fields(commands.CommandSpec)] == [
        "label",
        "bash",
        "arg",
        "failure",
    ]


def test_base_public_facade_exports_required_symbols_only() -> None:
    required = BASE_PUBLIC_TYPES | BASE_PUBLIC_FUNCTIONS
    assert required.issubset(set(base.__all__))
    for name in required:
        assert hasattr(base, name), name
    for name in BASE_REMOVED_LOW_LEVEL_FUNCTIONS:
        assert name not in base.__all__
        assert not hasattr(base, name), name


def test_base_public_facade_contains_no_implementation_definitions() -> None:
    _assert_facade_pure(REPO_ROOT / "src/text_to_sign_production/workflows/base.py")


def test_dataset_public_facade_exports_required_symbols_only() -> None:
    required = DATASET_PUBLIC_TYPES | DATASET_PUBLIC_FUNCTIONS
    assert required.issubset(set(dataset.__all__))
    for name in required:
        assert hasattr(dataset, name), name
    for name in DATASET_REMOVED_LOW_LEVEL_FUNCTIONS:
        assert name not in dataset.__all__
        assert not hasattr(dataset, name), name


def test_dataset_public_facade_contains_no_implementation_definitions() -> None:
    _assert_facade_pure(REPO_ROOT / "src/text_to_sign_production/workflows/dataset.py")


def test_visualization_public_facade_exports_required_symbols_only() -> None:
    required = (
        VISUALIZATION_PUBLIC_TYPES
        | VISUALIZATION_PUBLIC_LIFECYCLE_FUNCTIONS
        | VISUALIZATION_PUBLIC_PLAN_FUNCTIONS
    )
    assert required.issubset(set(visualization.__all__))
    for name in required:
        assert hasattr(visualization, name), name
    for name in VISUALIZATION_REMOVED_LOW_LEVEL_FUNCTIONS:
        assert name not in visualization.__all__
        assert not hasattr(visualization, name), name


def test_workflow_public_facades_contain_no_implementation_definitions() -> None:
    workflow_root = REPO_ROOT / "src/text_to_sign_production/workflows"
    for filename in ("base.py", "dataset.py", "visualization.py"):
        _assert_facade_pure(workflow_root / filename)


def test_private_workflow_package_skeletons_are_complete() -> None:
    required = {
        "_base": {
            "constants.py",
            "types.py",
            "layout.py",
            "runtime_plan.py",
            "restore.py",
            "validate.py",
            "execute.py",
            "verify.py",
            "summary.py",
            "publish.py",
            "reports.py",
        },
        "_dataset": {
            "constants.py",
            "types.py",
            "layout.py",
            "runtime_plan.py",
            "restore.py",
            "validate.py",
            "execute.py",
            "verify.py",
            "summary.py",
            "publish.py",
            "reports.py",
        },
        "_visualization": {
            "constants.py",
            "types.py",
            "layout.py",
            "runtime_plan.py",
            "restore.py",
            "selection.py",
            "validate.py",
            "execute.py",
            "verify.py",
            "summary.py",
            "publish.py",
        },
    }
    workflow_root = REPO_ROOT / "src/text_to_sign_production/workflows"
    for package, expected in required.items():
        observed = {path.name for path in (workflow_root / package).glob("*.py")}
        assert expected <= observed, f"{package} missing modules: {sorted(expected - observed)}"


def test_private_package_init_files_are_markers_only() -> None:
    workflow_root = REPO_ROOT / "src/text_to_sign_production/workflows"
    for package in PRIVATE_WORKFLOW_PACKAGES:
        path = workflow_root / package / "__init__.py"
        text = path.read_text(encoding="utf-8")
        assert "import *" not in text
        assert "from ." not in text
        assert "__all__ = [" not in text


def test_colab_notebooks_do_not_import_private_workflow_packages() -> None:
    for path in sorted((REPO_ROOT / "notebooks/colab").glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "code"
        )
        assert "text_to_sign_production.workflows._" not in source, path
        assert "workflows._" not in source, path


def test_dataset_notebook_imports_standard_public_lifecycle_api() -> None:
    source = _notebook_code_source(REPO_ROOT / "notebooks/colab/text_to_sign_dataset_colab.ipynb")
    imported = _imported_workflow_symbols(source, "text_to_sign_production.workflows.dataset")
    assert imported == DATASET_NOTEBOOK_EXPECTED_IMPORTS
    for name in DATASET_REMOVED_LOW_LEVEL_FUNCTIONS:
        assert name not in source


def test_visualization_notebook_imports_standard_public_lifecycle_api() -> None:
    source = _notebook_code_source(
        REPO_ROOT / "notebooks/colab/text_to_sign_visualization_colab.ipynb"
    )
    imported = _imported_workflow_symbols(
        source,
        "text_to_sign_production.workflows.visualization",
    )
    assert imported == VISUALIZATION_NOTEBOOK_EXPECTED_IMPORTS
    for name in VISUALIZATION_REMOVED_LOW_LEVEL_FUNCTIONS:
        assert re.search(rf"(?<!\w){re.escape(name)}(?!\w)", source) is None


def test_notebooks_import_public_workflow_facades_only() -> None:
    expected_modules = {
        "text_to_sign_base_colab.ipynb": {"text_to_sign_production.workflows.base"},
        "text_to_sign_dataset_colab.ipynb": {"text_to_sign_production.workflows.dataset"},
        "text_to_sign_visualization_colab.ipynb": {
            "text_to_sign_production.workflows.visualization"
        },
    }
    for filename, modules in expected_modules.items():
        source = _notebook_code_source(REPO_ROOT / "notebooks/colab" / filename)
        observed_modules = set(
            re.findall(r"from (text_to_sign_production\.workflows(?:\.[\w_]+)*) import", source)
        )
        assert observed_modules <= modules


def test_shared_metadata_does_not_import_data_layer() -> None:
    shared_root = REPO_ROOT / "src/text_to_sign_production/workflows/_shared"
    for path in shared_root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "text_to_sign_production.data" not in text
        assert "text_to_sign_production.modeling" not in text
        assert "text_to_sign_production.visualization" not in text
        assert "MyDrive" not in text
        assert "/content/drive" not in text
        assert "/content/text-to-sign-production" not in text


def test_old_public_workflow_surfaces_are_absent() -> None:
    for module, removed in (
        (base, BASE_REMOVED_LOW_LEVEL_FUNCTIONS),
        (dataset, DATASET_REMOVED_LOW_LEVEL_FUNCTIONS),
        (visualization, VISUALIZATION_REMOVED_LOW_LEVEL_FUNCTIONS),
    ):
        for name in removed:
            assert name not in module.__all__
            assert not hasattr(module, name), name


def test_old_workflow_artifact_roots_are_absent_from_active_code_and_notebooks() -> None:
    roots = [
        REPO_ROOT / "src/text_to_sign_production",
        REPO_ROOT / "notebooks/colab",
    ]
    for root in roots:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".ipynb"}:
                text = path.read_text(encoding="utf-8")
                for pattern in OLD_WORKFLOW_ARTIFACT_ROOT_PATTERNS:
                    assert pattern not in text, f"{pattern!r} found in {path}"


def test_dataset_report_artifact_writing_is_not_owned_by_data_layer() -> None:
    data_reports = (REPO_ROOT / "src/text_to_sign_production/data/reports.py").read_text(
        encoding="utf-8"
    )
    forbidden = {
        "write_markdown_reports",
        "processed_reports_root",
        "interim_reports_root",
        "reports_root",
        "write_json(",
    }
    for item in forbidden:
        assert item not in data_reports
    dataset_reports = (
        REPO_ROOT / "src/text_to_sign_production/workflows/_dataset/reports.py"
    ).read_text(encoding="utf-8")
    assert "write_processed_reports" in dataset_reports
    assert "write_interim_assumption_reports" in dataset_reports


def test_notebook_workflow_command_cells_use_command_spec_arg_pattern() -> None:
    for path in sorted((REPO_ROOT / "notebooks/colab").glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            if "!bash -o pipefail -c" not in source:
                continue
            assert "!bash -o pipefail -c {spec.arg}" in source, path
            assert 'if globals().get("_exit_code", 1) != 0:' in source, path
            assert "raise RuntimeError(spec.failure)" in source, path


def test_base_publish_does_not_import_base_verify() -> None:
    publish_source = (
        REPO_ROOT / "src/text_to_sign_production/workflows/_base/publish.py"
    ).read_text(encoding="utf-8")
    assert "text_to_sign_production.workflows._base.verify" not in publish_source
    assert "_base.verify" not in publish_source


def test_base_split_validation_has_single_authoritative_definition() -> None:
    base_root = REPO_ROOT / "src/text_to_sign_production/workflows/_base"
    definitions: list[tuple[Path, str]] = []
    for path in base_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        definitions.extend(
            (path, node.name)
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"_validate_split", "validate_split"}
        )
    assert definitions == [(base_root / "validate.py", "_validate_split")]


def _notebook_code_source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def _imported_workflow_symbols(source: str, module: str) -> set[str]:
    pattern = rf"from {re.escape(module)} import \((.*?)\)"
    matches = re.findall(pattern, source, flags=re.DOTALL)
    imported: set[str] = set()
    for match in matches:
        imported.update(
            line.strip().rstrip(",")
            for line in match.splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    return imported


def _assert_facade_pure(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert not any(isinstance(node, ast.FunctionDef) for node in tree.body)
    assert not any(isinstance(node, ast.ClassDef) for node in tree.body)
    assert "__all__" in path.read_text(encoding="utf-8")
