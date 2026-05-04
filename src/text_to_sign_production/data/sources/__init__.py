"""Source-level functionality for raw ingestion."""

from text_to_sign_production.data.sources.candidates import assemble_candidate
from text_to_sign_production.data.sources.keypoints import resolve_keypoint_source
from text_to_sign_production.data.sources.matching import match_sources
from text_to_sign_production.data.sources.translations import load_translation_rows
from text_to_sign_production.data.sources.types import (
    KeypointSourceFacts,
    SourceCandidate,
    SourceMatchResult,
    TranslationRow,
    VideoMetadataFacts,
)
from text_to_sign_production.data.sources.validate import validate_candidate
from text_to_sign_production.data.sources.videos import read_video_metadata

__all__ = [
    "KeypointSourceFacts",
    "SourceCandidate",
    "SourceMatchResult",
    "TranslationRow",
    "VideoMetadataFacts",
    "assemble_candidate",
    "load_translation_rows",
    "match_sources",
    "read_video_metadata",
    "resolve_keypoint_source",
    "validate_candidate",
]
