"""Artifact-topology-backed layout for the single-sample test-model workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from text_to_sign_production.artifacts.store import ArtifactStores, build_artifact_stores
from text_to_sign_production.core import build_repo_roots
from text_to_sign_production.workflows.test_model.contracts import TestModelWorkflowConfig


@dataclass(frozen=True, slots=True)
class TestModelLayout:
    config: TestModelWorkflowConfig
    stores: ArtifactStores


def build_test_model_layout(config: TestModelWorkflowConfig) -> TestModelLayout:
    stores = build_artifact_stores(
        build_repo_roots(config.runtime_root),
        build_repo_roots(config.drive_project_root),
    )
    return TestModelLayout(config=config, stores=stores)


def runtime_test_model_sample_run_root(
    layout: TestModelLayout,
    model_run_name: str,
    target_sentence_name: str,
    execution_id: str,
) -> Path:
    """Runtime output root for one test_model diagnostic execution."""

    return (
        layout.config.runtime_root
        / "reports"
        / "test_model"
        / _safe_path_token(model_run_name, "model_run_name")
        / _sentence_name_slug(target_sentence_name)
        / _safe_path_token(execution_id, "execution_id")
    )


def drive_test_model_sample_run_root(
    layout: TestModelLayout,
    model_run_name: str,
    target_sentence_name: str,
    execution_id: str,
) -> Path:
    """Drive output root for one test_model diagnostic execution."""

    return (
        layout.config.drive_project_root
        / "reports"
        / "test_model"
        / _safe_path_token(model_run_name, "model_run_name")
        / _sentence_name_slug(target_sentence_name)
        / _safe_path_token(execution_id, "execution_id")
    )


def _safe_path_token(value: str, label: str) -> str:
    token = str(value).strip()
    if not token:
        raise ValueError(f"{label} must be a non-empty path token")
    path = Path(token)
    if path.is_absolute() or len(path.parts) != 1 or token in {".", ".."}:
        raise ValueError(f"{label} must not contain path traversal or separators: {value!r}")
    if "/" in token or "\\" in token:
        raise ValueError(f"{label} must not contain path separators: {value!r}")
    return token


def _sentence_name_slug(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("target_sentence_name must be non-empty")
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_")
    if not slug:
        raise ValueError("target_sentence_name does not contain path-safe slug characters")
    return slug


__all__ = [
    "TestModelLayout",
    "build_test_model_layout",
    "drive_test_model_sample_run_root",
    "runtime_test_model_sample_run_root",
]
