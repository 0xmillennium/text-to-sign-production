from text_to_sign_production.workflows.model.processing.performance import (
    begin_stage_performance,
    collect_model_device_telemetry,
    collect_stage_performance_telemetry,
    finish_stage_performance,
    stage_performance_to_metadata,
)


def test_device_telemetry_has_cpu_safe_keys() -> None:
    telemetry = collect_model_device_telemetry()
    for key in (
        "device_type",
        "device_name",
        "cuda_available",
        "cuda_device_count",
        "cuda_total_memory_gb",
        "torch_version",
        "bf16_supported",
        "tf32_matmul_allowed",
        "tf32_cudnn_allowed",
        "float32_matmul_precision",
    ):
        assert key in telemetry


def test_stage_telemetry_cpu_safe_memory_fields() -> None:
    telemetry = collect_stage_performance_telemetry(
        provider_key="provider",
        provider_stage_id="provider.stage",
        stage_kind="train",
        elapsed_seconds=0.5,
        metadata={"batch_size": 4, "unit_count": 2},
    )
    assert telemetry["elapsed_seconds"] == 0.5
    assert "peak_cuda_memory_allocated_gb" in telemetry
    assert "end_cuda_memory_reserved_gb" in telemetry
    assert "cuda_available" in telemetry
    assert "device_name" in telemetry
    assert "requested_precision_policy" in telemetry
    assert "resolved_precision_policy" in telemetry
    assert "autocast_enabled" in telemetry
    assert "autocast_dtype" in telemetry
    assert "precision_applied" in telemetry
    assert telemetry["physical_batch_size"] == 4
    assert telemetry["units_per_second"] == 4.0


def test_stage_performance_timer_metadata_conversion_is_stable() -> None:
    timer = begin_stage_performance(
        provider_key="provider",
        provider_stage_id="provider.stage",
        stage_kind="train",
    )
    performance = finish_stage_performance(
        timer,
        metadata={
            "requested_precision_policy": "bf16",
            "resolved_precision_policy": "fp32",
            "autocast_enabled": False,
            "autocast_dtype": None,
            "precision_applied": False,
        },
    )

    metadata = stage_performance_to_metadata(performance)

    assert metadata["elapsed_seconds"] >= 0.0
    assert metadata["requested_precision_policy"] == "bf16"
    assert metadata["precision_applied"] is False
