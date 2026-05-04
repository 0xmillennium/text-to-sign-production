"""Publish planning for tiers workflow outputs."""

from __future__ import annotations

from pathlib import Path

from text_to_sign_production.artifacts.store import build_artifact_stores
from text_to_sign_production.foundation.roots import build_repo_roots
from text_to_sign_production.workflows._shared.transfers import build_byte_progress_copy_spec
from text_to_sign_production.workflows.tiers.types import (
    TiersPublishPlan,
    TiersWorkflowConfig,
    TiersWorkflowResult,
)


def build_tiers_publish_plan(
    config: TiersWorkflowConfig,
    result: TiersWorkflowResult,
) -> TiersPublishPlan:
    stores = build_artifact_stores(
        build_repo_roots(config.project_root),
        build_repo_roots(config.drive_project_root),
    )

    file_transfers = []
    for runtime_path in (
        *result.output_summary.tiered_manifest_paths,
        *result.output_summary.report_paths,
    ):
        file_transfers.append(
            build_byte_progress_copy_spec(
                label=_publish_label(runtime_path, stores.runtime.repo_root),
                source_path=runtime_path,
                target_path=stores.drive.repo_root
                / runtime_path.relative_to(stores.runtime.repo_root),
            )
        )

    return TiersPublishPlan(file_transfers=tuple(file_transfers))


def _publish_label(runtime_path: Path, runtime_root: Path) -> str:
    relative = runtime_path.relative_to(runtime_root)
    parts = relative.parts
    if len(parts) == 5 and parts[:2] == ("manifests", "tiered"):
        _, _, tier, membership, filename = parts
        return f"publish_tiers_manifest_{tier}_{membership}_{Path(filename).stem}"
    if len(parts) >= 3 and parts[:2] == ("reports", "tiers"):
        return f"publish_tiers_report_{runtime_path.stem}"
    return "publish_tiers_" + "_".join(Path(part).stem for part in parts)


__all__ = ["build_tiers_publish_plan"]
