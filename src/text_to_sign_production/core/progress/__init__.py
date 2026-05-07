"""Canonical progress contract surface."""

from text_to_sign_production.core.progress.session import (
    NoOpProgressSink,
    ProgressSession,
    ProgressSink,
    ProgressTaskHandle,
)
from text_to_sign_production.core.progress.sinks import TqdmProgressSink
from text_to_sign_production.core.progress.specs import (
    ProgressEvent,
    ProgressSplitBehavior,
    ProgressStageSpec,
)

__all__ = [
    "NoOpProgressSink",
    "ProgressEvent",
    "ProgressSession",
    "ProgressSink",
    "ProgressSplitBehavior",
    "ProgressStageSpec",
    "ProgressTaskHandle",
    "TqdmProgressSink",
]
