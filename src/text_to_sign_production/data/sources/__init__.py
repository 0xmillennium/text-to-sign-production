"""Source-level functionality for raw ingestion."""

from text_to_sign_production.data.sources.candidates import (
    assemble_candidate,
    sample_id_from_translation,
)
from text_to_sign_production.data.sources.keypoints import resolve_keypoint_source
from text_to_sign_production.data.sources.matching import match_sources
from text_to_sign_production.data.sources.translations import load_translation_rows
from text_to_sign_production.data.sources.types import (
    KeypointSourceFacts,
    SourceCandidate,
    SourceMatchResult,
    SourceValidationIssue,
    TranslationRow,
    VideoMetadataFacts,
)
from text_to_sign_production.data.sources.validate import validate_candidate
from text_to_sign_production.data.sources.videos import read_video_metadata

__all__ = [
    "KeypointSourceFacts",
    "SourceCandidate",
    "SourceMatchResult",
    "SourceValidationIssue",
    "TranslationRow",
    "VideoMetadataFacts",
    "assemble_candidate",
    "load_translation_rows",
    "match_sources",
    "read_video_metadata",
    "resolve_keypoint_source",
    "sample_id_from_translation",
    "validate_candidate",
]
