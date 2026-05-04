"""Runtime input verification for the samples workflow."""

from __future__ import annotations

from pathlib import Path

from text_to_sign_production.artifacts.store import build_artifact_topology
from text_to_sign_production.foundation.roots import build_repo_roots
from text_to_sign_production.workflows.samples.types import (
    SamplesWorkflowConfig,
    SamplesWorkflowInputError,
)


def verify_samples_runtime_inputs(config: SamplesWorkflowConfig) -> None:
    topology = build_artifact_topology(build_repo_roots(config.project_root))
    missing: list[str] = []

    gates_config_path = config.project_root / config.gates_config_relative_path
    _require_file(gates_config_path, "gates config", missing)

    for split in config.splits:
        _require_file(
            topology.assets.translation_csv(split).path,
            f"{split.value} translation CSV",
            missing,
        )
        _require_dir(
            topology.assets.keypoint_split_root(split).path,
            f"{split.value} keypoint split root",
            missing,
        )
        _require_dir(
            topology.assets.keypoint_json_dir(split).path,
            f"{split.value} keypoint json dir",
            missing,
        )
        _require_dir(
            topology.assets.keypoint_video_dir(split).path,
            f"{split.value} keypoint video dir",
            missing,
        )

    if missing:
        raise SamplesWorkflowInputError(
            "Samples runtime inputs are incomplete: " + "; ".join(missing)
        )


def _require_file(path: Path, label: str, missing: list[str]) -> None:
    if not path.is_file():
        missing.append(f"{label} is not a file: {path}")


def _require_dir(path: Path, label: str, missing: list[str]) -> None:
    if not path.is_dir():
        missing.append(f"{label} is not a directory: {path}")


__all__ = ["verify_samples_runtime_inputs"]
