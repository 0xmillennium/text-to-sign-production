"""Configuration files for Sprint 3 baseline modeling."""

from __future__ import annotations

from pathlib import Path
from typing import Final

DEFAULT_BASELINE_CONFIG_PATH: Final[Path] = Path(__file__).resolve().with_name("baseline.yaml")

__all__ = ["DEFAULT_BASELINE_CONFIG_PATH"]
