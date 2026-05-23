"""Articulator-aware contracts without implicit provider registration."""

from __future__ import annotations

from text_to_sign_production.modeling.candidates.articulator_aware.config import (
    ArticulatorAwareConfig,
    ArticulatorAwareLengthConfig,
    load_articulator_aware_config,
)
from text_to_sign_production.modeling.candidates.articulator_aware.spec import (
    ARTICULATOR_AWARE_CANONICAL_ID,
    ARTICULATOR_AWARE_MODEL_KEY,
    ARTICULATOR_AWARE_PHASE_NUMBER,
    ARTICULATOR_AWARE_RESEARCH_ROLE,
)

__all__ = [
    "ARTICULATOR_AWARE_CANONICAL_ID",
    "ARTICULATOR_AWARE_MODEL_KEY",
    "ARTICULATOR_AWARE_PHASE_NUMBER",
    "ARTICULATOR_AWARE_RESEARCH_ROLE",
    "ArticulatorAwareConfig",
    "ArticulatorAwareLengthConfig",
    "load_articulator_aware_config",
]
