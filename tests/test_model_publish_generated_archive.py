from __future__ import annotations

from pathlib import Path

import pytest

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.candidates import ModelRunMode
from text_to_sign_production.modeling.data import parse_modeling_manifest_family
from text_to_sign_production.modeling.research import ModelKey
from text_to_sign_production.workflows.foundation.provenance import written_file_receipt
from text_to_sign_production.modeling.candidates import (
    MODEL_RUNTIME_SUPPORT_SCHEMA_VERSION,
    ModelRuntimeSupportManifest,
    support_artifact_from_model_run_file,
    write_runtime_support_manifest,
)
from text_to_sign_production.workflows.model.contracts import (
    ModelRunMetadataArtifacts,
    ModelPublishSourceBundle,
    ModelStageArtifactReceiptResult,
    ModelWorkflowInvariantError,
)
from text_to_sign_production.workflows.model.contracts.config import ModelWorkflowConfig
from text_to_sign_production.workflows.model.layout import build_model_layout
from text_to_sign_production.workflows.model.publish.plan import build_model_publish_plan


@pytest.mark.unit
def test_model_publish_plan_uses_generated_pose_archive_not_individual_samples(
    tmp_path: Path,
) -> None:
    config = ModelWorkflowConfig(
        project_root=tmp_path,
        drive_project_root=tmp_path / "drive",
        model_key=ModelKey.BASE_DIRECT,
        manifest_family=parse_modeling_manifest_family("tiered:clean:included"),
        run_mode=ModelRunMode.SMOKE,
        run_name="run001",
    )
    layout = build_model_layout(config)
    split_root = layout.outputs.generated_pose_split_outputs[0].manifest_path.parent
    samples_root = split_root / "samples"
    samples_root.mkdir(parents=True)
    sample_path = samples_root / "a__g0.npz"
    sample_path.write_bytes(b"sample-a")
    for name in (
        "manifest.jsonl",
        "samples.tar.zst",
        "samples_archive_manifest.json",
        "samples_archive_sha256.txt",
    ):
        (split_root / name).write_text(f"{name}\n", encoding="utf-8")
    index_path = layout.outputs.stage_artifacts_index_path
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("{}\n", encoding="utf-8")
    receipts = ModelStageArtifactReceiptResult(
        execution_id="exec001",
        receipts=(
            written_file_receipt(
                "model stage artifact generated_pose_sample",
                sample_path,
                execution_id="exec001",
                kind="generated_pose_sample",
            ),
        ),
        skipped_artifacts=(),
        index_path=index_path,
        index_receipt=written_file_receipt(
            "model stage artifacts index",
            index_path,
            execution_id="exec001",
            kind="model_stage_artifacts_index",
        ),
    )

    plan = build_model_publish_plan(
        layout=layout,
        sources=ModelPublishSourceBundle(stage_artifact_receipts=receipts),
    )

    target_names = {target.target_path.name for target in plan.targets}
    assert "manifest.jsonl" in target_names
    assert "samples.tar.zst" in target_names
    assert "samples_archive_manifest.json" in target_names
    assert "samples_archive_sha256.txt" in target_names
    assert not any(
        "generated_pose" in target.target_path.as_posix()
        and "/samples/" in target.target_path.as_posix()
        and target.target_path.suffix == ".npz"
        for target in plan.targets
    )
    assert all(skipped.path != sample_path for skipped in plan.skipped_sources)


@pytest.mark.unit
def test_model_publish_plan_rejects_duplicate_targets_from_support_manifest(
    tmp_path: Path,
) -> None:
    config = ModelWorkflowConfig(
        project_root=tmp_path,
        drive_project_root=tmp_path / "drive",
        model_key=ModelKey.BASE_DIRECT,
        manifest_family=parse_modeling_manifest_family("tiered:clean:included"),
        run_mode=ModelRunMode.SMOKE,
        run_name="run001",
    )
    layout = build_model_layout(config)
    run_root = layout.outputs.model_run_root
    run_root.mkdir(parents=True, exist_ok=True)
    for path in (
        layout.outputs.effective_config_path,
        layout.outputs.research_spec_path,
        layout.outputs.run_metadata_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    artifact = support_artifact_from_model_run_file(
        model_run_root=run_root,
        path=layout.outputs.run_metadata_path,
        role="base_direct_compatibility_config",
        provider_key="base_direct",
    )
    write_runtime_support_manifest(
        layout.outputs.runtime_support_manifest_path,
        ModelRuntimeSupportManifest(
            schema_version=MODEL_RUNTIME_SUPPORT_SCHEMA_VERSION,
            model_key="base_direct",
            model_run_name="run001",
            manifest_family="tiered:clean:included",
            provider_config_kind="base_direct.effective_config",
            artifacts=(artifact,),
        ),
    )
    metadata = ModelRunMetadataArtifacts(
        execution_id="exec001",
        effective_config_path=layout.outputs.effective_config_path,
        research_spec_path=layout.outputs.research_spec_path,
        run_metadata_path=layout.outputs.run_metadata_path,
        runtime_support_manifest_path=layout.outputs.runtime_support_manifest_path,
        effective_config=written_file_receipt(
            "model effective config",
            layout.outputs.effective_config_path,
            execution_id="exec001",
            kind="model_effective_config",
        ),
        research_spec=written_file_receipt(
            "model research spec",
            layout.outputs.research_spec_path,
            execution_id="exec001",
            kind="model_research_spec",
        ),
        run_metadata=written_file_receipt(
            "model run metadata",
            layout.outputs.run_metadata_path,
            execution_id="exec001",
            kind="model_run_metadata",
        ),
        runtime_support_manifest=written_file_receipt(
            "model runtime support manifest",
            layout.outputs.runtime_support_manifest_path,
            execution_id="exec001",
            kind="model_runtime_support_manifest",
        ),
    )

    with pytest.raises(ModelWorkflowInvariantError, match="duplicate model publish target"):
        build_model_publish_plan(
            layout=layout,
            sources=ModelPublishSourceBundle(metadata_artifacts=metadata),
        )
