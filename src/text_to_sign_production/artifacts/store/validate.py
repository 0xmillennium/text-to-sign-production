"""Topology and path-shape validation for the physical artifact store."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path, PurePosixPath

from text_to_sign_production.artifacts.store.topology import ArtifactStores, ArtifactTopology
from text_to_sign_production.artifacts.store.types import (
    SampleStatus,
    SplitName,
    TierMembership,
    TierName,
)


def validate_artifact_topology(topology: ArtifactTopology) -> list[str]:
    """Validate physical topology invariants."""

    errors: list[str] = []
    _expect_path(errors, topology.assets_root, topology.repo_root / "assets", "assets_root")
    _expect_path(
        errors,
        topology.manifests_root,
        topology.repo_root / "manifests",
        "manifests_root",
    )
    _expect_path(errors, topology.samples_root, topology.repo_root / "samples", "samples_root")
    _expect_path(errors, topology.models_root, topology.repo_root / "models", "models_root")
    _expect_path(
        errors,
        topology.evaluations_root,
        topology.repo_root / "evaluations",
        "evaluations_root",
    )
    _expect_path(errors, topology.reports_root, topology.repo_root / "reports", "reports_root")

    _expect_path(
        errors,
        topology.assets.how2sign_root,
        topology.assets_root / "how2sign",
        "assets.how2sign_root",
    )
    _expect_path(
        errors,
        topology.assets.how2sign_bfh_keypoints_root,
        topology.assets_root / "how2sign" / "bfh_keypoints",
        "assets.how2sign_bfh_keypoints_root",
    )
    _expect_path(
        errors,
        topology.assets.how2sign_translations_root,
        topology.assets_root / "how2sign" / "translations",
        "assets.how2sign_translations_root",
    )

    _expect_path(
        errors,
        topology.manifests.untiered_root,
        topology.manifests_root / "untiered",
        "manifests.untiered_root",
    )
    _expect_path(
        errors,
        topology.manifests.tiered_root,
        topology.manifests_root / "tiered",
        "manifests.tiered_root",
    )

    _expect_path(
        errors,
        topology.samples.passed_root,
        topology.samples_root / "passed",
        "samples.passed_root",
    )
    _expect_path(
        errors,
        topology.samples.dropped_root,
        topology.samples_root / "dropped",
        "samples.dropped_root",
    )
    _expect_path(errors, topology.models.root, topology.models_root, "models.root")
    _expect_path(
        errors,
        topology.evaluations.root,
        topology.evaluations_root,
        "evaluations.root",
    )
    _expect_path(errors, topology.reports.root, topology.reports_root, "reports.root")
    _expect_path(
        errors,
        topology.reports.samples_root,
        topology.reports_root / "samples",
        "reports.samples_root",
    )
    _expect_path(
        errors,
        topology.reports.tiers_root,
        topology.reports_root / "tiers",
        "reports.tiers_root",
    )
    _expect_path(
        errors,
        topology.reports.modeling_root,
        topology.reports_root / "modeling",
        "reports.modeling_root",
    )
    _expect_path(
        errors,
        topology.reports.visualization_root,
        topology.reports_root / "visualization",
        "reports.visualization_root",
    )
    _expect_path(
        errors,
        topology.reports.evaluation_root,
        topology.reports_root / "evaluation",
        "reports.evaluation_root",
    )
    return errors


def validate_artifact_stores(stores: ArtifactStores) -> list[str]:
    """Validate runtime and Drive physical topology projections."""

    errors: list[str] = []
    for label, topology in (("runtime", stores.runtime), ("drive", stores.drive)):
        errors.extend(f"{label}: {error}" for error in validate_artifact_topology(topology))
    return errors


def validate_samples_relative_path(relative_path: str | Path) -> list[str]:
    """Validate a samples-root-relative physical sample path shape."""

    parts_errors, parts = _relative_parts(relative_path)
    if parts_errors:
        return parts_errors

    if len(parts) != 3:
        return [
            "Sample relative path must have shape "
            "passed/<split>/<filename> or dropped/<split>/<filename>."
        ]

    status, split, filename = parts
    errors: list[str] = []
    if status not in _values(SampleStatus):
        errors.append("Sample relative path status must be passed or dropped.")
    if split not in _values(SplitName):
        errors.append("Sample relative path split must be train, val, or test.")
    if filename in {"", ".", ".."}:
        errors.append("Sample relative path filename must be a concrete file name.")
    elif Path(filename).suffix != ".npz" or not Path(filename).stem:
        errors.append("Sample relative path filename must be a concrete .npz file name.")
    return errors


def validate_keypoint_archive_relative_path(relative_path: str | Path) -> list[str]:
    """Validate a repo-relative How2Sign 2D keypoint archive path shape."""

    parts_errors, parts = _relative_parts(relative_path)
    if parts_errors:
        return parts_errors

    if len(parts) != 4 or parts[:3] != ("assets", "how2sign", "bfh_keypoints"):
        return [
            "Keypoint archive relative path must have shape "
            "assets/how2sign/bfh_keypoints/<split>_2D_keypoints.tar.zst."
        ]

    filename = parts[3]
    errors: list[str] = []
    if not filename.endswith("_2D_keypoints.tar.zst"):
        errors.append("Keypoint archive filename must end with _2D_keypoints.tar.zst.")
        return errors

    split = filename.removesuffix("_2D_keypoints.tar.zst")
    if split not in _values(SplitName):
        errors.append("Keypoint archive split must be train, val, or test.")
    return errors


def validate_sample_archive_relative_path(relative_path: str | Path) -> list[str]:
    """Validate a repo-relative processed sample split archive path shape."""

    parts_errors, parts = _relative_parts(relative_path)
    if parts_errors:
        return parts_errors

    if len(parts) != 3 or parts[0] != "samples":
        return [
            "Sample archive relative path must have shape "
            "samples/<passed|dropped>/<split>.tar.zst."
        ]

    _, status, filename = parts
    errors: list[str] = []
    if status not in _values(SampleStatus):
        errors.append("Sample archive status must be passed or dropped.")
    if not filename.endswith(".tar.zst"):
        errors.append("Sample archive filename must end with .tar.zst.")
        return errors

    split = filename.removesuffix(".tar.zst")
    if split not in _values(SplitName):
        errors.append("Sample archive split must be train, val, or test.")
    return errors


def validate_sample_archive_member_path(relative_path: str | PurePosixPath) -> list[str]:
    """Validate a processed sample archive member path shape."""

    parts_errors, parts = _posix_relative_parts(relative_path)
    if parts_errors:
        return parts_errors

    if len(parts) != 2:
        return ["Sample archive member path must have shape <split>/<sample_id>.npz."]

    split, filename = parts
    errors: list[str] = []
    if split not in _values(SplitName):
        errors.append("Sample archive member split must be train, val, or test.")
    if filename in {"", ".", ".."}:
        errors.append("Sample archive member filename must be a concrete file name.")
    elif PurePosixPath(filename).suffix != ".npz" or not PurePosixPath(filename).stem:
        errors.append("Sample archive member filename must be a concrete .npz file name.")
    return errors


def validate_checkpoint_runtime_file_relative_path(relative_path: str | Path) -> list[str]:
    """Validate a repo-relative runtime model checkpoint file path shape."""

    return _validate_checkpoint_relative_path(
        relative_path,
        expected_suffix=".pt",
        label="Checkpoint runtime file",
    )


def validate_checkpoint_drive_archive_relative_path(relative_path: str | Path) -> list[str]:
    """Validate a repo-relative Drive model checkpoint archive path shape."""

    return _validate_checkpoint_relative_path(
        relative_path,
        expected_suffix=".pt.zst",
        label="Checkpoint Drive archive",
    )


def validate_manifests_relative_path(relative_path: str | Path) -> list[str]:
    """Validate a manifests-root-relative physical manifest path shape."""

    parts_errors, parts = _relative_parts(relative_path)
    if parts_errors:
        return parts_errors

    if len(parts) == 3 and parts[0] == "untiered":
        _, status, filename = parts
        errors: list[str] = []
        if status not in _values(SampleStatus):
            errors.append("Untiered manifest status must be passed or dropped.")
        errors.extend(_validate_split_jsonl_filename(filename, "Untiered manifest"))
        return errors

    if len(parts) == 4 and parts[0] == "tiered":
        _, tier, membership, filename = parts
        errors = []
        if tier not in _values(TierName):
            errors.append("Tiered manifest tier must be loose, clean, or tight.")
        if membership not in _values(TierMembership):
            errors.append("Tiered manifest membership must be included or excluded.")
        errors.extend(_validate_split_jsonl_filename(filename, "Tiered manifest"))
        return errors

    return [
        "Manifest relative path must have shape untiered/passed/<split>.jsonl, "
        "untiered/dropped/<split>.jsonl, or tiered/<tier>/<membership>/<split>.jsonl."
    ]


def _values(enum_type: type[StrEnum]) -> set[str]:
    return {entry.value for entry in enum_type}


def _expect_path(errors: list[str], actual: Path, expected: Path, label: str) -> None:
    if actual != expected:
        errors.append(f"{label} must be {expected}, got {actual}.")


def _relative_parts(relative_path: str | Path) -> tuple[list[str], tuple[str, ...]]:
    path = Path(relative_path)
    if path.is_absolute():
        return ([f"Expected a relative path, got absolute path: {path}"], ())
    if not path.parts:
        return (["Expected a non-empty relative path."], ())
    if ".." in path.parts:
        return ([f"Relative path must not contain parent directory references: {path}"], ())
    return ([], path.parts)


def _posix_relative_parts(relative_path: str | PurePosixPath) -> tuple[list[str], tuple[str, ...]]:
    path = PurePosixPath(relative_path)
    if path.is_absolute():
        return ([f"Expected a relative path, got absolute path: {path}"], ())
    if not path.parts:
        return (["Expected a non-empty relative path."], ())
    if ".." in path.parts:
        return ([f"Relative path must not contain parent directory references: {path}"], ())
    return ([], path.parts)


def _validate_split_jsonl_filename(filename: str, label: str) -> list[str]:
    errors: list[str] = []
    if not filename.endswith(".jsonl"):
        errors.append(f"{label} filename must end with .jsonl.")
        return errors

    split = filename.removesuffix(".jsonl")
    if split not in _values(SplitName):
        errors.append(f"{label} filename must be train.jsonl, val.jsonl, or test.jsonl.")
    return errors


def _validate_checkpoint_relative_path(
    relative_path: str | Path,
    *,
    expected_suffix: str,
    label: str,
) -> list[str]:
    parts_errors, parts = _relative_parts(relative_path)
    if parts_errors:
        return parts_errors

    if len(parts) < 2 or parts[0] != "models":
        return [f"{label} relative path must be under models/."]

    filename = parts[-1]
    if filename in {"", ".", ".."}:
        return [f"{label} filename must be a concrete file name."]
    if not filename.endswith(expected_suffix) or filename == expected_suffix:
        return [f"{label} filename must end with {expected_suffix}."]
    return []


__all__ = [
    "validate_artifact_topology",
    "validate_artifact_stores",
    "validate_checkpoint_drive_archive_relative_path",
    "validate_checkpoint_runtime_file_relative_path",
    "validate_keypoint_archive_relative_path",
    "validate_manifests_relative_path",
    "validate_sample_archive_member_path",
    "validate_sample_archive_relative_path",
    "validate_samples_relative_path",
]
