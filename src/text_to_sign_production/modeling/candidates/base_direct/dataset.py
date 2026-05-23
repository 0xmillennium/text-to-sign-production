"""Runtime data resolution for the M0 direct baseline provider."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.candidates import (
    ModelStageExecutionContext,
    ModelStageExecutionError,
)
from text_to_sign_production.modeling.data import resolve_modeling_manifest_path


@dataclass(frozen=True, slots=True)
class BaseDirectRuntimeDataPaths:
    train_manifest: Path
    validation_manifest: Path
    prediction_manifests: Mapping[SampleSplit, Path]
    samples_root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "train_manifest", Path(self.train_manifest))
        object.__setattr__(self, "validation_manifest", Path(self.validation_manifest))
        object.__setattr__(self, "samples_root", Path(self.samples_root))
        object.__setattr__(
            self,
            "prediction_manifests",
            MappingProxyType(
                {
                    SampleSplit(split): Path(path)
                    for split, path in self.prediction_manifests.items()
                }
            ),
        )


def resolve_base_direct_runtime_data_paths(
    context: ModelStageExecutionContext,
) -> BaseDirectRuntimeDataPaths:
    """Resolve requested data inputs through the runtime artifact topology."""

    request = context.request
    return BaseDirectRuntimeDataPaths(
        train_manifest=resolve_modeling_manifest_path(
            context.topology,
            request.manifest_family,
            request.train_split,
        ),
        validation_manifest=resolve_modeling_manifest_path(
            context.topology,
            request.manifest_family,
            request.validation_split,
        ),
        prediction_manifests={
            split: resolve_modeling_manifest_path(
                context.topology,
                request.manifest_family,
                split,
            )
            for split in request.prediction_splits
        },
        samples_root=context.topology.samples_root,
    )


def validate_base_direct_runtime_data_paths(paths: BaseDirectRuntimeDataPaths) -> None:
    """Require restored input manifests and the passed-sample payload tree."""

    required_files = {
        "training manifest": paths.train_manifest,
        "validation manifest": paths.validation_manifest,
        **{
            f"prediction manifest [{split.value}]": path
            for split, path in paths.prediction_manifests.items()
        },
    }
    for label, path in required_files.items():
        if not path.is_file():
            raise ModelStageExecutionError(f"base_direct {label} does not exist: {path}")
    if not paths.samples_root.is_dir():
        raise ModelStageExecutionError(
            f"base_direct runtime samples root does not exist: {paths.samples_root}"
        )


__all__ = [
    "BaseDirectRuntimeDataPaths",
    "resolve_base_direct_runtime_data_paths",
    "validate_base_direct_runtime_data_paths",
]
