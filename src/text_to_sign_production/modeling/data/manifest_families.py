"""Canonical modeling manifest-family selection over dataset-owned readers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.artifacts.store import ArtifactTopology
from text_to_sign_production.core.ids import SampleSplit, TierMembership
from text_to_sign_production.core.models import PassedManifestEntry
from text_to_sign_production.data.dataset import (
    read_passed_manifest_header,
    read_passed_manifest_json,
    read_tier_manifest_header,
    read_tier_manifest_json,
)
from text_to_sign_production.modeling.data.errors import ModelingDataError

_TIER_TOKEN_PATTERN = re.compile(r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$")
_ALIASES = {
    "untiered_passed": "untiered:passed",
    "tiered_loose": "tiered:loose:included",
    "tiered_clean": "tiered:clean:included",
    "tiered_tight": "tiered:tight:included",
}
_DROPPED_ERROR = (
    "untiered:dropped is not a valid modeling manifest family because dropped samples "
    "do not provide the passed prepared-payload contract required for model training."
)


@dataclass(frozen=True, slots=True)
class ModelingManifestFamily:
    """Canonical input-manifest identity selected for modeling."""

    family_id: str
    kind: str
    status: str | None
    tier: str | None
    membership: str | None

    def __post_init__(self) -> None:
        canonical = _components_from_value(self.family_id)
        if canonical != (
            self.family_id,
            self.kind,
            self.status,
            self.tier,
            self.membership,
        ):
            raise ModelingDataError(
                "modeling manifest family fields must match its canonical family_id."
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_family": self.family_id,
            "manifest_kind": self.kind,
            "manifest_status": self.status,
            "manifest_tier": self.tier,
            "manifest_membership": self.membership,
        }

    @property
    def path_slug(self) -> str:
        return self.family_id.replace(":", "_")


@dataclass(frozen=True, slots=True)
class ModelingManifest:
    """Resolved model-ready manifest rows for one family and split."""

    manifest_family: ModelingManifestFamily
    split: SampleSplit
    manifest_path: Path
    entries: tuple[PassedManifestEntry, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "split", SampleSplit(self.split))
        object.__setattr__(self, "manifest_path", Path(self.manifest_path))
        if not isinstance(self.entries, tuple):
            raise ModelingDataError("modeling manifest entries must be a tuple.")
        seen: set[str] = set()
        for entry in self.entries:
            if not isinstance(entry, PassedManifestEntry):
                raise ModelingDataError(
                    "modeling manifest entries must contain PassedManifestEntry values."
                )
            if entry.split is not self.split:
                raise ModelingDataError("modeling manifest entry split does not match split.")
            if entry.sample_id in seen:
                raise ModelingDataError(f"duplicate modeling manifest sample_id: {entry.sample_id}")
            seen.add(entry.sample_id)


def parse_modeling_manifest_family(value: str) -> ModelingManifestFamily:
    """Parse a family string or migration alias into its canonical contract."""

    family_id, kind, status, tier, membership = _components_from_value(value)
    return ModelingManifestFamily(family_id, kind, status, tier, membership)


def resolve_modeling_manifest_path(
    topology: ArtifactTopology,
    family: ModelingManifestFamily,
    split: SampleSplit | str,
) -> Path:
    """Resolve one canonical manifest-family path through artifact topology."""

    resolved_family = _require_family(family)
    resolved_split = SampleSplit(split)
    if resolved_family.kind == "untiered":
        return topology.manifests.untiered_passed_manifest(resolved_split).path
    return topology.manifests.tiered_manifest(
        resolved_family.tier or "",
        resolved_family.membership or "",
        resolved_split,
    ).path


def read_modeling_manifest(
    topology: ArtifactTopology,
    family: ModelingManifestFamily,
    split: SampleSplit | str,
) -> ModelingManifest:
    """Read a model-ready family through existing passed/tier manifest readers."""

    resolved_family = _require_family(family)
    resolved_split = SampleSplit(split)
    path = resolve_modeling_manifest_path(topology, resolved_family, resolved_split)
    try:
        if resolved_family.kind == "untiered":
            header = read_passed_manifest_header(path)
            if header.split is not resolved_split:
                raise ModelingDataError(
                    "passed manifest split does not match requested split: "
                    f"expected {resolved_split.value!r}, observed {header.split.value!r}."
                )
            entries = read_passed_manifest_json(path)
        else:
            header = read_tier_manifest_header(path)
            if header.tier != resolved_family.tier:
                raise ModelingDataError(
                    "tier manifest tier does not match requested manifest family: "
                    f"expected {resolved_family.tier!r}, observed {header.tier!r}."
                )
            if header.membership.value != resolved_family.membership:
                raise ModelingDataError(
                    "tier manifest membership does not match requested manifest family: "
                    f"expected {resolved_family.membership!r}, observed {header.membership.value!r}."
                )
            if header.split is not resolved_split:
                raise ModelingDataError(
                    "tier manifest split does not match requested split: "
                    f"expected {resolved_split.value!r}, observed {header.split.value!r}."
                )
            entries = read_tier_manifest_json(path)
    except FileNotFoundError:
        raise
    except ModelingDataError:
        raise
    except ValueError as exc:
        raise ModelingDataError(f"modeling manifest is invalid: {path}: {exc}") from exc
    return ModelingManifest(resolved_family, resolved_split, path, entries)


def _components_from_value(
    value: object,
) -> tuple[str, str, str | None, str | None, str | None]:
    if not isinstance(value, str) or not value.strip():
        raise ModelingDataError("modeling manifest family must be non-empty.")
    canonical = _ALIASES.get(value.strip(), value.strip())
    if canonical == "untiered:dropped":
        raise ModelingDataError(_DROPPED_ERROR)
    parts = canonical.split(":")
    if parts[0] == "untiered":
        if len(parts) != 2 or parts[1] != "passed":
            raise ModelingDataError(
                "untiered modeling manifest family must be 'untiered:passed'."
            )
        return canonical, "untiered", "passed", None, None
    if parts[0] == "tiered":
        if len(parts) != 3 or not parts[1]:
            raise ModelingDataError(
                "tiered modeling manifest family must be 'tiered:<tier>:included' "
                "or 'tiered:<tier>:excluded'."
            )
        if not _TIER_TOKEN_PATTERN.fullmatch(parts[1]):
            raise ModelingDataError(
                "tiered modeling manifest family tier must be a lowercase safe token."
            )
        if parts[2] not in {member.value for member in TierMembership}:
            raise ModelingDataError(
                "tiered modeling manifest family membership must be 'included' or 'excluded'."
            )
        return canonical, "tiered", None, parts[1], parts[2]
    raise ModelingDataError(
        "modeling manifest family kind must be 'untiered' or 'tiered'."
    )


def _require_family(family: ModelingManifestFamily) -> ModelingManifestFamily:
    if not isinstance(family, ModelingManifestFamily):
        raise ModelingDataError("family must be a ModelingManifestFamily.")
    return family


__all__ = [
    "ModelingManifest",
    "ModelingManifestFamily",
    "parse_modeling_manifest_family",
    "read_modeling_manifest",
    "resolve_modeling_manifest_path",
]
