"""Representative calibration surface contracts shared by providers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RepresentativeCalibrationSurfaceResult:
    """Provider-built representative data for one calibration candidate."""

    provider_key: str
    candidate_key: str
    surface_kind: str
    split: str
    source_manifest_path: Path
    source_manifest_sha256: str
    provider_config_sha256: str
    surface_schema_hash: str
    feature_dim: int | None
    sample_count: int
    unit_count: int
    surface_root: Path | None
    surface_metadata_path: Path | None
    dataloader_kind: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_manifest_path", Path(self.source_manifest_path))
        if self.surface_root is not None:
            object.__setattr__(self, "surface_root", Path(self.surface_root))
        if self.surface_metadata_path is not None:
            object.__setattr__(self, "surface_metadata_path", Path(self.surface_metadata_path))


__all__ = ["RepresentativeCalibrationSurfaceResult"]
