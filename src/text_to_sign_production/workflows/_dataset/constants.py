"""Constants owned by the Dataset workflow."""

from __future__ import annotations

from pathlib import Path

DEFAULT_DATASET_FILTER_CONFIG_RELATIVE_PATH = Path("configs") / "data" / "filter-v2.yaml"

DEFAULT_DATASET_QUALITY_CONFIG_RELATIVE_PATH = Path("configs") / "data" / "quality-tier-v1.yaml"

DEFAULT_DATASET_RUN_NAME = "dataset_build"

DATASET_PUBLISH_DIRNAME = "publish"

__all__ = [
    "DATASET_PUBLISH_DIRNAME",
    "DEFAULT_DATASET_FILTER_CONFIG_RELATIVE_PATH",
    "DEFAULT_DATASET_QUALITY_CONFIG_RELATIVE_PATH",
    "DEFAULT_DATASET_RUN_NAME",
]
