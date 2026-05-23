from text_to_sign_production.workflows.model.processing.performance import (
    collect_stage_performance_telemetry,
)
from text_to_sign_production.workflows.model.processing.reports import (
    _performance_summary_markdown,
    _performance_summary_payload,
)
from text_to_sign_production.modeling.candidates import (
    ModelExecutionResult,
    ModelStageResult,
    ModelStageStatus,
    default_stage_plan_for_request,
    ModelRunRequest,
)
from text_to_sign_production.modeling.data import parse_modeling_manifest_family
from text_to_sign_production.modeling.research import ModelKey


def test_gpu_memory_fractions_and_underutilized_class_with_total_memory(monkeypatch) -> None:
    import torch

    class Props:
        name = "NVIDIA A100-SXM4-80GB"
        total_memory = 80 * 1024 ** 3

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: Props())
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 1 * 1024 ** 3)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 2 * 1024 ** 3)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 0)

    telemetry = collect_stage_performance_telemetry(
        provider_key="learned_pose_token",
        provider_stage_id="learned_pose_token.fit_representation",
        stage_kind="fit_representation",
        elapsed_seconds=2.0,
        metadata={"unit_count": 10, "unit_type": "token_unit"},
    )

    assert telemetry["peak_cuda_memory_reserved_fraction"] == 0.025
    assert telemetry["peak_cuda_memory_allocated_fraction"] == 0.0125
    assert telemetry["gpu_utilization_class"] == "underutilized"
    assert telemetry["bottleneck_hint"] == "batch_too_small_or_python_loop_bound"


def test_learned_pose_token_provider_batch_fields_passthrough() -> None:
    telemetry = collect_stage_performance_telemetry(
        provider_key="learned_pose_token",
        provider_stage_id="learned_pose_token.fit_representation",
        stage_kind="fit_representation",
        elapsed_seconds=1.0,
        metadata={
            "tokenizer_batch_size": 256,
            "text_to_token_batch_size": 128,
            "reconstruction_batch_size": 128,
            "decode_batch_size": 128,
            "materialization_workers": 4,
            "cache_materialized_sources": True,
            "unit_count": 4,
            "unit_type": "token_unit",
        },
    )
    assert telemetry["physical_batch_size"] == "not_applicable"
    assert telemetry["effective_batch_size"] == "not_applicable"
    assert telemetry["tokenizer_batch_size"] == 256
    assert telemetry["text_to_token_batch_size"] == 128


def test_performance_summary_includes_bottleneck_hint() -> None:
    request = ModelRunRequest(
        model_key=ModelKey.LEARNED_POSE_TOKEN,
        run_name="run",
        manifest_family=parse_modeling_manifest_family("tiered:clean:included"),
    )
    stage = default_stage_plan_for_request(request).stages[0]
    result = ModelExecutionResult(
        model_key=ModelKey.LEARNED_POSE_TOKEN,
        run_name="run",
        stages=(
            ModelStageResult(
                stage=stage,
                status=ModelStageStatus.COMPLETED,
                metadata={
                    "performance": {
                        "elapsed_seconds": 1.0,
                        "provider_stage_id": stage.provider_stage_id,
                        "peak_cuda_memory_reserved_gb": 0.1,
                        "peak_cuda_memory_reserved_fraction": 0.001,
                        "gpu_utilization_class": "underutilized",
                        "bottleneck_hint": "batch_too_small_or_python_loop_bound",
                    }
                },
            ),
        ),
    )
    payload = _performance_summary_payload(result)
    markdown = _performance_summary_markdown(payload)
    assert payload["stages"][0]["bottleneck_hint"] == "batch_too_small_or_python_loop_bound"
    assert "bottleneck=batch_too_small_or_python_loop_bound" in markdown
