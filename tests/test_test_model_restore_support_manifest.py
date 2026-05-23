from __future__ import annotations

import json
from pathlib import Path

import pytest

from text_to_sign_production.modeling.candidates import (
    MODEL_RUNTIME_SUPPORT_SCHEMA_VERSION,
    ModelRuntimeSupportManifest,
    support_artifact_from_model_run_file,
    write_runtime_support_manifest,
)
from text_to_sign_production.workflows.test_model.contracts import (
    CheckpointPolicy,
    TestModelRequest,
    TestModelWorkflowConfig,
)
from text_to_sign_production.workflows.test_model.layout import build_test_model_layout
from text_to_sign_production.workflows.test_model.runtime.plan import (
    build_test_model_restore_plan,
    validate_test_model_restore_plan,
)


@pytest.mark.unit
def test_restore_plan_reads_runtime_support_manifest(tmp_path: Path) -> None:
    config = TestModelWorkflowConfig(
        project_root=tmp_path,
        drive_project_root=tmp_path / "drive",
        runtime_root=tmp_path / "runtime",
    )
    layout = build_test_model_layout(config)
    run_root = layout.stores.drive.models.model_run_root("latent_diffusion", "run001").path
    run_root.mkdir(parents=True)
    (run_root / "run_metadata.json").write_text(
        json.dumps(
            {
                "model_key": "latent_diffusion",
                "model_run_name": "run001",
                "manifest_family": "tiered:clean:included",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    support_path = run_root / "intermediates" / "latent" / "autoencoder" / "window_autoencoder.pt"
    support_path.parent.mkdir(parents=True)
    support_path.write_bytes(b"checkpoint")
    artifact = support_artifact_from_model_run_file(
        model_run_root=run_root,
        path=support_path,
        role="latent_diffusion_autoencoder_checkpoint",
        provider_key="latent_diffusion",
    )
    write_runtime_support_manifest(
        run_root / "runtime_support_manifest.json",
        ModelRuntimeSupportManifest(
            schema_version=MODEL_RUNTIME_SUPPORT_SCHEMA_VERSION,
            model_key="latent_diffusion",
            model_run_name="run001",
            manifest_family="tiered:clean:included",
            provider_config_kind="latent_diffusion.effective_config",
            artifacts=(artifact,),
        ),
    )

    plan = build_test_model_restore_plan(
        config,
        TestModelRequest("run001", CheckpointPolicy.BEST, "sentence-001"),
        layout=layout,
    )

    support_ops = [
        operation
        for operation in plan.operations
        if operation.group == "latent_diffusion_autoencoder_checkpoint"
    ]
    assert not plan.blocking_errors
    assert len(support_ops) == 1
    assert support_ops[0].source == support_path
    assert support_ops[0].target == (
        layout.stores.runtime.models.model_run_root("latent_diffusion", "run001").path
        / artifact.relative_path
    )


@pytest.mark.unit
def test_missing_runtime_support_manifest_is_blocking(tmp_path: Path) -> None:
    config = TestModelWorkflowConfig(
        project_root=tmp_path,
        drive_project_root=tmp_path / "drive",
        runtime_root=tmp_path / "runtime",
    )
    layout = build_test_model_layout(config)
    run_root = layout.stores.drive.models.model_run_root("base_direct", "run001").path
    run_root.mkdir(parents=True)
    (run_root / "run_metadata.json").write_text(
        json.dumps(
            {
                "model_key": "base_direct",
                "model_run_name": "run001",
                "manifest_family": "tiered:clean:included",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    plan = build_test_model_restore_plan(
        config,
        TestModelRequest("run001", CheckpointPolicy.BEST, "sentence-001"),
        layout=layout,
    )

    assert any(issue.code == "runtime_support_manifest_missing" for issue in plan.blocking_errors)


@pytest.mark.unit
def test_restore_validation_rejects_duplicate_targets(tmp_path: Path) -> None:
    config = TestModelWorkflowConfig(
        project_root=tmp_path,
        drive_project_root=tmp_path / "drive",
        runtime_root=tmp_path / "runtime",
    )
    layout = build_test_model_layout(config)
    run_root = layout.stores.drive.models.model_run_root("base_direct", "run001").path
    run_root.mkdir(parents=True)
    duplicate_path = run_root / "effective_config.json"
    duplicate_path.write_text("{}\n", encoding="utf-8")
    artifact = support_artifact_from_model_run_file(
        model_run_root=run_root,
        path=duplicate_path,
        role="base_direct_compatibility_config",
        provider_key="base_direct",
    )
    (run_root / "run_metadata.json").write_text(
        json.dumps(
            {
                "model_key": "base_direct",
                "model_run_name": "run001",
                "manifest_family": "tiered:clean:included",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    write_runtime_support_manifest(
        run_root / "runtime_support_manifest.json",
        ModelRuntimeSupportManifest(
            schema_version=MODEL_RUNTIME_SUPPORT_SCHEMA_VERSION,
            model_key="base_direct",
            model_run_name="run001",
            manifest_family="tiered:clean:included",
            provider_config_kind="base_direct.effective_config",
            artifacts=(artifact,),
        ),
    )

    plan = build_test_model_restore_plan(
        config,
        TestModelRequest("run001", CheckpointPolicy.BEST, "sentence-001"),
        layout=layout,
    )
    validation = validate_test_model_restore_plan(layout, plan)

    assert any(issue.code == "duplicate_target_path" for issue in validation.blocking_errors)
