"""Topology validation for the physical artifact store."""

from __future__ import annotations

from pathlib import Path

from text_to_sign_production.artifacts.store.topology import ArtifactStores, ArtifactTopology


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
    return errors


def validate_artifact_stores(stores: ArtifactStores) -> list[str]:
    """Validate runtime and Drive physical topology projections."""

    errors: list[str] = []
    for label, topology in (("runtime", stores.runtime), ("drive", stores.drive)):
        errors.extend(f"{label}: {error}" for error in validate_artifact_topology(topology))
    return errors


def _expect_path(errors: list[str], actual: Path, expected: Path, label: str) -> None:
    if actual != expected:
        errors.append(f"{label} must be {expected}, got {actual}.")


__all__ = [
    "validate_artifact_topology",
    "validate_artifact_stores",
]
