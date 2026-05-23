"""Run provider-owned single-sample inference and receipt generated-pose outputs."""

from __future__ import annotations

from text_to_sign_production.core.progress import ProgressSession
from text_to_sign_production.modeling.candidates import (
    ModelProviderLoadedConfig,
    ModelRunMode,
    ModelRunRequest,
    ModelSingleSampleInferenceContext,
)
from text_to_sign_production.workflows.test_model.constants import TEST_MODEL_STAGE_INFERENCE_RUN
from text_to_sign_production.modeling.research import ModelKey
from text_to_sign_production.workflows.foundation.provenance import written_file_receipt
from text_to_sign_production.workflows.foundation.review import write_json
from text_to_sign_production.workflows.test_model.contracts import (
    TestModelCheckpointSelection,
    TestModelInferenceResult,
    TestModelRunResolution,
    TestModelTargetResolution,
    TestModelWorkflowInvariantError,
)
from text_to_sign_production.workflows.test_model.layout import (
    TestModelLayout,
    runtime_test_model_sample_run_root,
)
from text_to_sign_production.workflows.test_model.progress import test_model_progress_stage


def run_test_model_sample_inference(
    layout: TestModelLayout,
    *,
    model_run: TestModelRunResolution,
    checkpoint: TestModelCheckpointSelection,
    target: TestModelTargetResolution,
    provider_registry,
    execution_id: str,
    progress_session: ProgressSession | None = None,
) -> TestModelInferenceResult:
    if progress_session is None:
        return _run_test_model_sample_inference(
            layout,
            model_run=model_run,
            checkpoint=checkpoint,
            target=target,
            provider_registry=provider_registry,
            execution_id=execution_id,
        )
    with progress_session.task(
        test_model_progress_stage(
            stage_id=TEST_MODEL_STAGE_INFERENCE_RUN,
            label="test_model inference",
            unit="sample",
            owner_module=__name__,
            operation_kind="inference_run",
            total_semantics="single target sample inference",
            allowed_counters=("generated", "failed"),
        ),
        total=1,
    ) as task:
        try:
            result = _run_test_model_sample_inference(
                layout,
                model_run=model_run,
                checkpoint=checkpoint,
                target=target,
                provider_registry=provider_registry,
                execution_id=execution_id,
            )
        except Exception:
            task.advance(1, counters={"failed": 1})
            raise
        task.advance(
            1,
            counters={
                "generated": 1 if result.status == "completed" else 0,
                "failed": 0 if result.status == "completed" else 1,
            },
        )
        return result


def _run_test_model_sample_inference(
    layout: TestModelLayout,
    *,
    model_run: TestModelRunResolution,
    checkpoint: TestModelCheckpointSelection,
    target: TestModelTargetResolution,
    provider_registry,
    execution_id: str,
) -> TestModelInferenceResult:
    errors: list[str] = []
    if not checkpoint.succeeded:
        errors.extend(checkpoint.errors)
    if not target.succeeded or target.manifest_sample is None:
        errors.extend(target.issues or ("target sample was not resolved",))
    if errors:
        return TestModelInferenceResult(None, None, (), "not_ready", (), tuple(errors))
    try:
        request = ModelRunRequest(
            model_key=ModelKey(model_run.model_key),
            run_name=model_run.model_run_name,
            manifest_family=target.manifest_family,
            train_split=model_run.train_split,
            validation_split=model_run.validation_split,
            prediction_splits=(model_run.validation_split,),
            auxiliary_objectives=(),
            run_mode=ModelRunMode(model_run.run_mode),
            seed=None,
            config_path=None,
        )
        provider = provider_registry.require(request.model_key)
        effective_config = _effective_config(model_run)
        loaded_config = ModelProviderLoadedConfig(
            model_key=request.model_key,
            source_path=None,
            raw_config={},
            effective_config=effective_config,
        )
        output_root = (
            runtime_test_model_sample_run_root(
                layout,
                model_run.model_run_name,
                target.target_sentence_name,
                execution_id,
            )
            / "generated_pose"
        )
        context = ModelSingleSampleInferenceContext(
            request=request,
            loaded_config=loaded_config,
            topology=layout.stores.runtime,
            checkpoint_path=checkpoint.checkpoint_path,
            sample=target.manifest_sample,
            output_root=output_root,
        )
        if not callable(getattr(provider, "infer_single_sample", None)):
            raise TestModelWorkflowInvariantError(
                f"provider {model_run.model_key!r} does not implement single-sample inference"
            )
        provider_result = provider.infer_single_sample(context)
        metadata_path = output_root / "inference_metadata.json"
        write_json(
            metadata_path,
            {
                "schema_version": "test_model_inference_result.v1",
                "model_run_name": model_run.model_run_name,
                "model_key": model_run.model_key,
                "checkpoint_policy": checkpoint.policy.value,
                "checkpoint_path": str(checkpoint.checkpoint_path),
                "target_sentence_name": target.target_sentence_name,
                "resolved_sample_id": target.resolved_sample_id,
                "source_sentence_name": target.source_sentence_name,
                "manifest_family": (
                    None
                    if target.manifest_family is None
                    else target.manifest_family.family_id
                ),
                "split": target.split.value,
                "generated_payload_path": str(provider_result.generated_payload_path),
                "generated_manifest_path": str(provider_result.generated_manifest_path),
                "provider_metadata": dict(provider_result.metadata),
            },
        )
        receipts = (
            written_file_receipt(
                "test_model generated pose sample",
                provider_result.generated_payload_path,
                execution_id=execution_id,
                kind="test_model_generated_pose_sample",
            ),
            written_file_receipt(
                "test_model generated pose manifest",
                provider_result.generated_manifest_path,
                execution_id=execution_id,
                kind="test_model_generated_pose_manifest",
            ),
            written_file_receipt(
                "test_model inference metadata",
                metadata_path,
                execution_id=execution_id,
                kind="test_model_inference_metadata",
            ),
        )
        return TestModelInferenceResult(
            provider_result=provider_result,
            metadata_path=metadata_path,
            receipts=receipts,
            status="completed",
            warnings=(),
            errors=(),
        )
    except Exception as exc:
        return TestModelInferenceResult(None, None, (), "failed", (), (str(exc),))


def _effective_config(model_run: TestModelRunResolution) -> dict[str, object]:
    path = model_run.run_metadata_path.parent / "effective_config.json"
    try:
        import json

        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TestModelWorkflowInvariantError(f"effective config could not be read: {path}: {exc}") from exc
    if not isinstance(loaded, dict) or not isinstance(loaded.get("effective_config"), dict):
        raise TestModelWorkflowInvariantError(f"effective config is missing provider config: {path}")
    return dict(loaded["effective_config"])


__all__ = ["run_test_model_sample_inference"]
