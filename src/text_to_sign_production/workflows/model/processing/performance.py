"""Device and stage performance telemetry for model workflow execution."""

from __future__ import annotations

import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


def collect_model_device_telemetry() -> dict[str, object]:
    """Collect torch/CUDA facts with CPU-safe defaults."""

    import torch

    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    total_memory_gb = None
    device_name = None
    if cuda_available and device_count:
        props = torch.cuda.get_device_properties(0)
        total_memory_gb = props.total_memory / (1024 ** 3)
        device_name = props.name
    return {
        "device_type": "cuda" if cuda_available else "cpu",
        "device_name": device_name,
        "cuda_available": cuda_available,
        "cuda_device_count": device_count,
        "cuda_total_memory_gb": total_memory_gb,
        "torch_version": torch.__version__,
        "bf16_supported": bool(cuda_available and torch.cuda.is_bf16_supported()),
        "tf32_matmul_allowed": bool(
            cuda_available and torch.backends.cuda.matmul.allow_tf32
        ),
        "tf32_cudnn_allowed": bool(cuda_available and torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def reset_cuda_peak_memory_stats_if_available() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def collect_stage_performance_telemetry(
    *,
    provider_key: str,
    provider_stage_id: str,
    stage_kind: str,
    elapsed_seconds: float,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Collect per-stage timing and CUDA memory facts."""

    import torch

    cuda_available = torch.cuda.is_available()
    device_name = None
    if cuda_available and torch.cuda.device_count():
        props = torch.cuda.get_device_properties(0)
        device_name = props.name
        total_memory_gb = _gb(props.total_memory)
    else:
        total_memory_gb = None
    perf = {
        "provider_key": provider_key,
        "provider_stage_id": provider_stage_id,
        "stage_kind": stage_kind,
        "elapsed_seconds": float(elapsed_seconds),
        "stage_elapsed_seconds": float(elapsed_seconds),
        "peak_cuda_memory_allocated_gb": None,
        "peak_cuda_memory_reserved_gb": None,
        "peak_cuda_memory_allocated_fraction": None,
        "peak_cuda_memory_reserved_fraction": None,
        "end_cuda_memory_allocated_gb": None,
        "end_cuda_memory_reserved_gb": None,
        "cuda_available": cuda_available,
        "device_name": device_name,
        "cuda_total_memory_gb": total_memory_gb,
        "physical_batch_size": None,
        "effective_batch_size": None,
        "gradient_accumulation_steps": None,
        "precision_policy": None,
        "requested_precision_policy": None,
        "resolved_precision_policy": None,
        "autocast_enabled": None,
        "autocast_dtype": None,
        "precision_applied": None,
        "num_workers": None,
        "unit_count": None,
        "units_per_second": None,
        "unit_type": "unknown",
        "gpu_utilization_class": "no_cuda" if not cuda_available else "unknown",
        "bottleneck_hint": "unknown",
    }
    if cuda_available:
        allocated = _gb(torch.cuda.max_memory_allocated())
        reserved = _gb(torch.cuda.max_memory_reserved())
        perf.update(
            {
                "peak_cuda_memory_allocated_gb": allocated,
                "peak_cuda_memory_reserved_gb": reserved,
                "end_cuda_memory_allocated_gb": _gb(torch.cuda.memory_allocated()),
                "end_cuda_memory_reserved_gb": _gb(torch.cuda.memory_reserved()),
                "peak_cuda_memory_allocated_fraction": _fraction(allocated, total_memory_gb),
                "peak_cuda_memory_reserved_fraction": _fraction(reserved, total_memory_gb),
            }
        )
    if metadata is not None:
        perf.update(_optional_stage_fields(metadata, elapsed_seconds=float(elapsed_seconds)))
        if _has_provider_specific_batch_fields(metadata):
            if perf.get("physical_batch_size") is None:
                perf["physical_batch_size"] = "not_applicable"
            if perf.get("effective_batch_size") is None:
                perf["effective_batch_size"] = "not_applicable"
    perf["gpu_utilization_class"] = _gpu_utilization_class(
        cuda_available=bool(perf.get("cuda_available")),
        reserved_fraction=perf.get("peak_cuda_memory_reserved_fraction"),
    )
    perf["bottleneck_hint"] = _bottleneck_hint(perf)
    return perf


@dataclass(slots=True)
class StagePerformanceTimer:
    provider_key: str
    provider_stage_id: str
    stage_kind: str
    started_at: float

    @classmethod
    def start(cls, *, provider_key: str, provider_stage_id: str, stage_kind: str):
        reset_cuda_peak_memory_stats_if_available()
        return cls(
            provider_key=provider_key,
            provider_stage_id=provider_stage_id,
            stage_kind=stage_kind,
            started_at=time.perf_counter(),
        )

    def stop(self, *, metadata: Mapping[str, object] | None = None) -> dict[str, object]:
        return collect_stage_performance_telemetry(
            provider_key=self.provider_key,
            provider_stage_id=self.provider_stage_id,
            stage_kind=self.stage_kind,
            elapsed_seconds=time.perf_counter() - self.started_at,
            metadata=metadata,
        )


def begin_stage_performance(
    *,
    provider_key: str,
    provider_stage_id: str,
    stage_kind: str,
) -> StagePerformanceTimer:
    return StagePerformanceTimer.start(
        provider_key=provider_key,
        provider_stage_id=provider_stage_id,
        stage_kind=stage_kind,
    )


def finish_stage_performance(
    timer: StagePerformanceTimer,
    *,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return timer.stop(metadata=metadata)


def stage_performance_to_metadata(performance: Mapping[str, object]) -> dict[str, object]:
    return dict(performance)


@contextmanager
def measure_stage_performance(
    *,
    provider_key: str,
    provider_stage_id: str,
    stage_kind: str,
) -> Iterator[StagePerformanceTimer]:
    timer = StagePerformanceTimer.start(
        provider_key=provider_key,
        provider_stage_id=provider_stage_id,
        stage_kind=stage_kind,
    )
    yield timer


def _optional_stage_fields(
    metadata: Mapping[str, object],
    *,
    elapsed_seconds: float,
) -> dict[str, object]:
    result = {}
    aliases = {
        "physical_batch_size": ("physical_batch_size", "batch_size"),
        "effective_batch_size": ("effective_batch_size",),
        "gradient_accumulation_steps": ("gradient_accumulation_steps",),
        "precision_policy": ("precision_policy",),
        "requested_precision_policy": ("requested_precision_policy",),
        "resolved_precision_policy": ("resolved_precision_policy",),
        "autocast_enabled": ("autocast_enabled",),
        "autocast_dtype": ("autocast_dtype",),
        "precision_applied": ("precision_applied",),
        "num_workers": ("num_workers",),
        "unit_type": ("unit_type",),
        "unit_count": (
            "unit_count",
            "train_token_unit_count",
            "token_unit_count_train",
            "train_latent_unit_count",
            "generated_sample_count",
            "total_sample_count",
        ),
    }
    for output_key, keys in aliases.items():
        for key in keys:
            if key in metadata:
                result[output_key] = metadata[key]
                break
    passthrough_keys = (
        "batch_size",
        "tokenizer_batch_size",
        "text_to_token_batch_size",
        "reconstruction_batch_size",
        "decode_batch_size",
        "denoiser_batch_size",
        "source_batch_size",
        "frame_batch_size",
        "num_workers",
        "surface_reader_num_workers_used",
        "surface_reader_worker_mode",
        "surface_reader_prefetch_factor_used",
        "surface_reader_persistent_workers_used",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
        "non_blocking_transfers",
        "mixed_precision",
        "cache_materialized_sources",
        "latent_autoencoder_active",
        "configured_sampling_steps",
        "effective_sampling_steps",
        "candidate_count",
        "token_unit_count_train",
        "token_unit_count_validation",
        "train_sequence_count",
        "validation_sequence_count",
        "average_token_units_per_sample",
        "max_token_units_per_sample",
        "cache_enabled",
        "cache_hit_count",
        "cache_miss_count",
        "cache_read_seconds",
        "cache_write_seconds",
        "materialization_elapsed_seconds",
        "autoencoder_batch_size",
        "sampling_steps_source",
        "window_count",
        "latent_sequence_count",
        "frame_training_sample_count",
        "architecture_variant",
        "decode_batching_limited",
        "batching_limited_by_variable_length",
    )
    for key in passthrough_keys:
        if key in metadata:
            result[key] = metadata[key]
    unit_count = result.get("unit_count")
    if (
        isinstance(unit_count, int | float)
        and not isinstance(unit_count, bool)
        and elapsed_seconds > 0.0
    ):
        result["units_per_second"] = float(unit_count) / elapsed_seconds
    return result


def _fraction(value: object, total: object) -> float | None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    if not isinstance(total, int | float) or isinstance(total, bool) or total <= 0:
        return None
    return float(value) / float(total)


def _gpu_utilization_class(
    *,
    cuda_available: bool,
    reserved_fraction: object,
) -> str:
    if not cuda_available:
        return "no_cuda"
    if not isinstance(reserved_fraction, int | float) or isinstance(reserved_fraction, bool):
        return "unknown"
    fraction = float(reserved_fraction)
    if fraction < 0.10:
        return "underutilized"
    if fraction < 0.50:
        return "moderate"
    if fraction < 0.85:
        return "high"
    return "near_limit"


def _bottleneck_hint(perf: Mapping[str, object]) -> str:
    unit_count = perf.get("unit_count")
    utilization = perf.get("gpu_utilization_class")
    if utilization == "no_cuda":
        return "unknown"
    if not isinstance(unit_count, int | float) or isinstance(unit_count, bool) or unit_count <= 0:
        return "not_enough_data"
    if utilization == "near_limit":
        return "memory_limited"
    if utilization == "underutilized":
        return "batch_too_small_or_python_loop_bound"
    if utilization in {"moderate", "high"}:
        return "gpu_utilized"
    return "unknown"


def _has_provider_specific_batch_fields(metadata: Mapping[str, object]) -> bool:
    return any(
        key in metadata
        for key in (
            "tokenizer_batch_size",
            "text_to_token_batch_size",
            "reconstruction_batch_size",
            "decode_batch_size",
            "denoiser_batch_size",
            "autoencoder_batch_size",
            "source_batch_size",
            "frame_batch_size",
        )
    )


def _gb(bytes_value: int) -> float:
    return float(bytes_value) / float(1024 ** 3)


__all__ = [
    "StagePerformanceTimer",
    "begin_stage_performance",
    "collect_model_device_telemetry",
    "collect_stage_performance_telemetry",
    "finish_stage_performance",
    "measure_stage_performance",
    "reset_cuda_peak_memory_stats_if_available",
    "stage_performance_to_metadata",
]
