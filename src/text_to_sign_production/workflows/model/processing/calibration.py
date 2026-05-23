"""Compute-profile calibration selection for model workflow runs."""

from __future__ import annotations

import json
import hashlib
import importlib
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol

from text_to_sign_production.artifacts.store import ArtifactTopology
from text_to_sign_production.core.progress import ProgressSession
from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.candidates import (
    ModelProvider,
    ModelProviderLoadedConfig,
    ModelRunRequest,
    ModelStagePlan,
    ProviderCalibrationPolicy,
)
from text_to_sign_production.modeling.candidates.calibration_surfaces import (
    RepresentativeCalibrationSurfaceResult,
)
from text_to_sign_production.modeling.data import read_modeling_manifest, resolve_modeling_manifest_path
from text_to_sign_production.workflows.model.contracts.compute_profiles import (
    provider_candidate_overrides,
)
from text_to_sign_production.workflows.model.contracts.config import ModelWorkflowInvariantError


@dataclass(frozen=True, slots=True)
class ComputeCalibrationResult:
    json_path: Path
    markdown_path: Path
    payload: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class PreExecutionCalibrationResult:
    required: bool
    executed: bool
    reused: bool
    reason: str
    compute_calibration_path: Path | None
    compute_calibration_report_path: Path | None
    selected_overrides_path: Path | None
    calibrated_effective_config_path: Path | None
    selected_overrides: Mapping[str, object]
    applied_overrides: Mapping[str, object]
    base_effective_config_hash: str
    selected_overrides_hash: str | None
    calibrated_effective_config_hash: str
    policy: ProviderCalibrationPolicy | None = None
    representative_surfaces: tuple[RepresentativeCalibrationSurfaceResult, ...] = ()
    blocking_issues: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ProviderRealCalibrationArtifactStatus:
    provider_real_calibration_required: bool
    provider_real_calibration_found: bool
    selected_overrides_found: bool
    calibration_artifact_path: Path | None
    selected_overrides_path: Path | None
    calibration_authoritative: bool
    calibration_benchmark_type: str | None
    provider_mismatch: bool = False
    config_hash_mismatch: bool = False
    effective_config_hash_mismatch: bool = False
    model_build_spec_hash_mismatch: bool = False
    surface_schema_mismatch: bool = False
    not_bound_to_effective_config: bool = False
    provider_real_hook_not_shared_training_step: bool = False
    synthetic_authoritative_attempt: bool = False
    selected_overrides_not_provider_real: bool = False
    selected_trace_id_mismatch: bool = False

    @property
    def blocking_issues(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.provider_real_calibration_found:
            issues.append("provider_real_calibration_missing")
        if self.synthetic_authoritative_attempt:
            issues.append("synthetic_calibration_used_as_authoritative")
        if not self.selected_overrides_found:
            issues.append("selected_overrides_missing")
        if self.provider_mismatch:
            issues.append("calibration_provider_mismatch")
        if self.config_hash_mismatch:
            issues.append("calibration_config_hash_mismatch")
        if self.effective_config_hash_mismatch:
            issues.append("calibration_effective_config_hash_mismatch")
        if self.model_build_spec_hash_mismatch:
            issues.append("calibration_model_build_spec_hash_mismatch")
        if self.surface_schema_mismatch:
            issues.append("calibration_surface_schema_mismatch")
        if self.not_bound_to_effective_config:
            issues.append("calibration_not_bound_to_effective_config")
        if self.provider_real_hook_not_shared_training_step:
            issues.append("provider_real_hook_not_shared_training_step")
        if self.selected_overrides_not_provider_real:
            issues.append("selected_overrides_not_provider_real")
        if self.selected_trace_id_mismatch:
            issues.append("selected_overrides_trace_id_mismatch")
        if (
            self.provider_real_calibration_found
            and not self.calibration_authoritative
            and not self.synthetic_authoritative_attempt
        ):
            issues.append("provider_real_calibration_missing")
        return tuple(dict.fromkeys(issues))


@dataclass(frozen=True, slots=True)
class ProviderBenchmarkSurfaceBundle:
    provider_key: str
    surface_kind: str
    unit_count: int
    surface_schema: Mapping[str, object]
    surface_schema_hash: str
    surface_provider_config_sha256: str
    surface_feature_dim: int | None
    surface_root: Path | None = None


@dataclass(frozen=True, slots=True)
class ProviderRealCalibrationContext:
    provider_key: str
    model_key: str
    run_name: str | None
    run_mode: str
    manifest_family: str

    loaded_config: ModelProviderLoadedConfig
    effective_config: Mapping[str, object]
    stage_plan: ModelStagePlan | None

    train_manifest_path: Path | None
    validation_manifest_path: Path | None
    topology: ArtifactTopology | None

    working_dir: Path
    layout_root: Path | None

    compute_profile: str | None
    precision_policy: str | None

    provider_config_hash: str
    model_build_spec_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.loaded_config, ModelProviderLoadedConfig):
            raise ModelWorkflowInvariantError("provider-real calibration context requires loaded_config.")
        if self.loaded_config.model_key.value != self.provider_key:
            raise ModelWorkflowInvariantError("provider-real calibration context provider/config mismatch.")
        if self.provider_config_hash != stable_config_hash(self.effective_config):
            raise ModelWorkflowInvariantError("provider-real calibration context effective_config_hash mismatch.")
        if not _is_sha256(self.model_build_spec_hash):
            raise ModelWorkflowInvariantError("provider-real calibration context missing model_build_spec_hash.")
        object.__setattr__(self, "working_dir", Path(self.working_dir))
        if self.layout_root is not None:
            object.__setattr__(self, "layout_root", Path(self.layout_root))
        if self.train_manifest_path is not None:
            object.__setattr__(self, "train_manifest_path", Path(self.train_manifest_path))
        if self.validation_manifest_path is not None:
            object.__setattr__(self, "validation_manifest_path", Path(self.validation_manifest_path))
        if self.topology is not None and not isinstance(self.topology, ArtifactTopology):
            raise ModelWorkflowInvariantError("provider-real calibration context topology must be ArtifactTopology.")


@dataclass(frozen=True, slots=True)
class ProviderRealCalibrationTrace:
    provider_key: str
    provider_hook: str
    benchmark_name: str
    benchmark_type: Literal["provider_real"]
    benchmark_kind: Literal["training", "generation"]
    forward_only_reason: str | None

    surface_kind: str
    surface_metadata_path: Path
    surface_metadata_sha256: str
    surface_batch_fingerprint: str

    batch_adapter: str
    model_input_fingerprint: str
    target_fingerprint: str | None

    model_class: str
    model_module: str
    model_config_hash: str
    effective_config_hash: str
    model_build_spec_hash: str
    surface_schema_hash: str
    surface_provider_config_sha256: str
    surface_feature_dim: int | None
    model_expected_input_dim: int | None
    model_factory: str
    model_build_spec: Mapping[str, object]

    loss_name: str | None
    loss_module: str | None
    training_step_helper: str | None

    forward_executed: bool
    backward_executed: bool
    gradient_observed: bool
    finite_loss_observed: bool

    surface_derived_primary_input: bool
    random_used: bool
    random_role: str | None

    units_processed: int
    elapsed_seconds: float
    peak_cuda_memory_reserved_fraction: float | None
    representative_source_manifest_path: Path | None = None
    representative_source_manifest_sha256: str | None = None
    representative_surface_kind: str | None = None
    representative_provider_config_sha256: str | None = None
    representative_surface_schema_hash: str | None = None
    representative_sample_count: int | None = None
    representative_unit_count: int | None = None
    calibration_surface_reader_num_workers_used: int | None = None
    calibration_surface_reader_worker_mode: str | None = None
    calibration_surface_reader_prefetch_factor_used: int | None = None
    calibration_surface_reader_persistent_workers_used: bool | None = None


class ProviderRealBenchmark(Protocol):
    provider_key: str

    def run_candidate(
        self,
        *,
        context: ProviderRealCalibrationContext,
        candidate_key: str,
        candidate_value: int,
        representative: RepresentativeCalibrationSurfaceResult,
        max_batches_per_candidate: int,
        warmup_batches: int,
        progress_callbacks: "CalibrationProgressCallbacks | None" = None,
    ) -> ProviderRealCalibrationTrace:
        ...


@dataclass(frozen=True, slots=True)
class CalibrationProgressCallbacks:
    advance_warmup_batch: Any
    advance_measured_batch: Any


_PROVIDER_MODULE_PREFIXES = {
    "learned_pose_token": "text_to_sign_production.modeling.candidates.learned_pose_token",
    "latent_diffusion": "text_to_sign_production.modeling.candidates.latent_diffusion",
    "articulator_aware": "text_to_sign_production.modeling.candidates.articulator_aware",
    "base_direct": "text_to_sign_production.modeling.candidates.base_direct",
}


def tensor_fingerprint(tensor: Any, *, max_values: int = 256) -> str:
    """Stable bounded fingerprint for a tensor's value lineage."""

    import torch

    if not isinstance(tensor, torch.Tensor):
        raise ModelWorkflowInvariantError("tensor_fingerprint requires a torch.Tensor.")
    value = tensor.detach().cpu().contiguous()
    flat = value.reshape(-1)
    prefix = flat[: max(0, int(max_values))]
    payload = {
        "dtype": str(value.dtype),
        "shape": tuple(int(item) for item in value.shape),
        "values": prefix.tolist(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def batch_fingerprint(batch: Mapping[str, Any], *, max_values: int = 256) -> str:
    """Stable bounded fingerprint for nested tensor batches."""

    def encode(value: Any) -> Any:
        import torch

        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            flat = tensor.reshape(-1)
            return {
                "kind": "tensor",
                "dtype": str(tensor.dtype),
                "shape": tuple(int(item) for item in tensor.shape),
                "values": flat[: max(0, int(max_values))].tolist(),
            }
        if isinstance(value, Mapping):
            return {
                str(key): encode(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, tuple):
            return {"kind": "tuple", "items": [encode(item) for item in value]}
        if isinstance(value, list):
            return {"kind": "list", "items": [encode(item) for item in value]}
        if isinstance(value, str | int | float | bool) or value is None:
            return value
        return {"kind": type(value).__name__}

    payload = encode(batch)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def provider_real_trace_id(trace: ProviderRealCalibrationTrace) -> str:
    payload = _trace_to_jsonable(trace)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def validate_provider_real_trace(
    trace: ProviderRealCalibrationTrace,
    *,
    context: ProviderRealCalibrationContext | None = None,
) -> None:
    if not isinstance(trace, ProviderRealCalibrationTrace):
        raise ModelWorkflowInvariantError("provider-real calibration requires a trace object.")
    if trace.benchmark_type != "provider_real":
        raise ModelWorkflowInvariantError("provider-real trace benchmark_type must be provider_real.")
    if trace.benchmark_kind not in {"training", "generation"}:
        raise ModelWorkflowInvariantError("provider-real trace benchmark_kind is unsupported.")
    for name in (
        "provider_key",
        "provider_hook",
        "benchmark_name",
        "surface_kind",
        "surface_metadata_sha256",
        "surface_batch_fingerprint",
        "batch_adapter",
        "model_input_fingerprint",
        "model_class",
        "model_module",
        "model_config_hash",
        "effective_config_hash",
        "model_build_spec_hash",
        "surface_schema_hash",
        "surface_provider_config_sha256",
        "model_factory",
    ):
        if not isinstance(getattr(trace, name), str) or not getattr(trace, name).strip():
            raise ModelWorkflowInvariantError(f"provider-real trace missing {name}.")
    for name in (
        "effective_config_hash",
        "model_build_spec_hash",
        "surface_schema_hash",
        "surface_provider_config_sha256",
    ):
        if not _is_sha256(getattr(trace, name)):
            raise ModelWorkflowInvariantError(f"provider-real trace {name} must be a sha256 digest.")
    if trace.provider_key not in _PROVIDER_MODULE_PREFIXES:
        raise ModelWorkflowInvariantError("provider-real trace provider_key is unsupported.")
    provider_prefix = _PROVIDER_MODULE_PREFIXES[trace.provider_key]
    if not trace.model_module.startswith(provider_prefix) and not (
        trace.provider_key == "base_direct"
        and trace.model_factory
        == "text_to_sign_production.modeling.candidates.base_direct.model.build_base_direct_model"
    ):
        raise ModelWorkflowInvariantError("provider-real trace model module is not provider-owned.")
    if not trace.model_factory.startswith(provider_prefix):
        raise ModelWorkflowInvariantError("provider-real trace model factory is not provider-owned.")
    if not isinstance(trace.model_build_spec, Mapping) or not trace.model_build_spec:
        raise ModelWorkflowInvariantError("provider-real trace missing model_build_spec.")
    if stable_config_hash(trace.model_build_spec) != trace.model_build_spec_hash:
        raise ModelWorkflowInvariantError("provider-real trace model_build_spec_hash mismatch.")
    if trace.surface_provider_config_sha256 != trace.effective_config_hash:
        raise ModelWorkflowInvariantError("provider-real trace surface provider config hash mismatch.")
    if trace.representative_surface_kind is not None and trace.representative_surface_kind != trace.surface_kind:
        raise ModelWorkflowInvariantError("provider-real trace representative surface kind mismatch.")
    if (
        trace.representative_provider_config_sha256 is not None
        and trace.representative_provider_config_sha256 != trace.effective_config_hash
    ):
        raise ModelWorkflowInvariantError(
            "provider-real trace representative provider config effective_config_hash mismatch."
        )
    if (
        trace.representative_surface_schema_hash is not None
        and trace.representative_surface_schema_hash != trace.surface_schema_hash
    ):
        raise ModelWorkflowInvariantError(
            "provider-real trace representative surface schema hash mismatch."
        )
    if trace.representative_sample_count is not None and trace.representative_sample_count <= 0:
        raise ModelWorkflowInvariantError("provider-real trace representative_sample_count must be positive.")
    if trace.representative_unit_count is not None and trace.representative_unit_count <= 0:
        raise ModelWorkflowInvariantError("provider-real trace representative_unit_count must be positive.")
    if trace.surface_feature_dim is not None:
        if not isinstance(trace.surface_feature_dim, int) or isinstance(trace.surface_feature_dim, bool) or trace.surface_feature_dim <= 0:
            raise ModelWorkflowInvariantError("provider-real trace surface_feature_dim must be positive.")
    if trace.model_expected_input_dim is not None:
        if not isinstance(trace.model_expected_input_dim, int) or isinstance(trace.model_expected_input_dim, bool) or trace.model_expected_input_dim <= 0:
            raise ModelWorkflowInvariantError("provider-real trace model_expected_input_dim must be positive.")
    if (
        trace.surface_feature_dim is not None
        and trace.model_expected_input_dim is not None
        and trace.surface_feature_dim != trace.model_expected_input_dim
    ):
        raise ModelWorkflowInvariantError("provider-real trace surface/model input dimension mismatch.")
    if context is not None:
        from text_to_sign_production.modeling.candidates.calibration_runtime import (
            calibration_reader_runtime_options,
        )
        runtime_options = calibration_reader_runtime_options(context.effective_config)
        if trace.provider_key != context.provider_key:
            raise ModelWorkflowInvariantError("provider-real trace context provider mismatch.")
        if trace.effective_config_hash != context.provider_config_hash:
            raise ModelWorkflowInvariantError("provider-real trace effective_config_hash mismatch.")
        if trace.model_build_spec_hash != context.model_build_spec_hash:
            raise ModelWorkflowInvariantError("provider-real trace model_build_spec_hash mismatch.")
        if trace.surface_provider_config_sha256 != context.provider_config_hash:
            raise ModelWorkflowInvariantError("provider-real trace surface provider config hash mismatch.")
        if trace.calibration_surface_reader_num_workers_used is not None:
            if trace.calibration_surface_reader_num_workers_used != runtime_options.num_workers:
                raise ModelWorkflowInvariantError(
                    "provider-real trace reader num_workers mismatch with effective config."
                )
            expected_mode = "multiprocess" if runtime_options.num_workers > 0 else "single_process"
            if trace.calibration_surface_reader_worker_mode != expected_mode:
                raise ModelWorkflowInvariantError(
                    "provider-real trace reader worker_mode mismatch with effective config."
                )
            if trace.calibration_surface_reader_prefetch_factor_used != runtime_options.prefetch_factor:
                raise ModelWorkflowInvariantError(
                    "provider-real trace reader prefetch_factor mismatch with effective config."
                )
            if trace.calibration_surface_reader_persistent_workers_used != runtime_options.persistent_workers:
                raise ModelWorkflowInvariantError(
                    "provider-real trace reader persistent_workers mismatch with effective config."
                )
    if trace.forward_executed is not True:
        raise ModelWorkflowInvariantError("provider-real trace did not execute forward.")
    if trace.surface_derived_primary_input is not True:
        raise ModelWorkflowInvariantError("provider-real trace primary input is not surface-derived.")
    if trace.random_used and trace.random_role not in {"noise", "augmentation"}:
        raise ModelWorkflowInvariantError("provider-real trace used random primary input.")
    if trace.random_used and not trace.surface_derived_primary_input:
        raise ModelWorkflowInvariantError("provider-real trace random input replaced surface primary input.")
    if trace.benchmark_kind == "training":
        if trace.finite_loss_observed is not True:
            raise ModelWorkflowInvariantError("provider-real training trace did not observe finite loss.")
        if not isinstance(trace.training_step_helper, str) or not trace.training_step_helper.startswith(provider_prefix):
            raise ModelWorkflowInvariantError("provider-real training trace missing provider training step helper.")
        if not isinstance(trace.loss_name, str) or not trace.loss_name.strip():
            raise ModelWorkflowInvariantError("provider-real training trace missing loss name.")
        if not isinstance(trace.loss_module, str) or not trace.loss_module.startswith(provider_prefix):
            raise ModelWorkflowInvariantError("provider-real training trace loss module is not provider-owned.")
        if trace.backward_executed is not True:
            raise ModelWorkflowInvariantError("provider-real training trace did not execute backward.")
        if trace.gradient_observed is not True:
            raise ModelWorkflowInvariantError("provider-real training trace did not observe finite gradients.")
    else:
        if trace.forward_only_reason != "generation":
            raise ModelWorkflowInvariantError("provider-real generation trace missing forward_only_reason.")
        if trace.backward_executed:
            raise ModelWorkflowInvariantError("provider-real generation trace must be forward-only.")
        if trace.training_step_helper is None and not trace.forward_only_reason:
            raise ModelWorkflowInvariantError("provider-real generation trace missing forward-only reason.")
    if not isinstance(trace.units_processed, int) or trace.units_processed <= 0:
        raise ModelWorkflowInvariantError("provider-real trace units_processed must be positive.")
    if not isinstance(trace.elapsed_seconds, int | float) or trace.elapsed_seconds < 0:
        raise ModelWorkflowInvariantError("provider-real trace elapsed_seconds must be non-negative.")


def calibration_row_from_provider_real_trace(
    trace: ProviderRealCalibrationTrace,
    *,
    candidate_name: str,
    candidate_value: int,
) -> dict[str, object]:
    validate_provider_real_trace(trace)
    elapsed = float(trace.elapsed_seconds)
    trace_id = provider_real_trace_id(trace)
    row = {
        "provider_key": trace.provider_key,
        "parameter": candidate_name,
        "value": candidate_value,
        "candidate_name": candidate_name,
        "candidate_value": candidate_value,
        "candidate": {candidate_name: candidate_value},
        "benchmark_path": trace.benchmark_name,
        "benchmark_type": "provider_real",
        "benchmark_kind": trace.benchmark_kind,
        "trace_id": trace_id,
        "trace_validation_passed": True,
        "provider_real_trace": _trace_to_jsonable(trace),
        "provider_hook": trace.provider_hook,
        "surface_kind": trace.surface_kind,
        "surface_metadata_path": str(trace.surface_metadata_path),
        "surface_metadata_sha256": trace.surface_metadata_sha256,
        "surface_batch_fingerprint": trace.surface_batch_fingerprint,
        "batch_adapter": trace.batch_adapter,
        "model_input_fingerprint": trace.model_input_fingerprint,
        "target_fingerprint": trace.target_fingerprint,
        "model_class": trace.model_class,
        "model_module": trace.model_module,
        "model_config_hash": trace.model_config_hash,
        "effective_config_hash": trace.effective_config_hash,
        "model_build_spec_hash": trace.model_build_spec_hash,
        "surface_schema_hash": trace.surface_schema_hash,
        "surface_provider_config_sha256": trace.surface_provider_config_sha256,
        "surface_feature_dim": trace.surface_feature_dim,
        "representative_source_manifest_path": (
            None
            if trace.representative_source_manifest_path is None
            else str(trace.representative_source_manifest_path)
        ),
        "representative_source_manifest_sha256": trace.representative_source_manifest_sha256,
        "representative_surface_kind": trace.representative_surface_kind,
        "representative_provider_config_sha256": trace.representative_provider_config_sha256,
        "representative_surface_schema_hash": trace.representative_surface_schema_hash,
        "representative_sample_count": trace.representative_sample_count,
        "representative_unit_count": trace.representative_unit_count,
        "model_expected_input_dim": trace.model_expected_input_dim,
        "model_factory": trace.model_factory,
        "model_build_spec": dict(trace.model_build_spec),
        "loss_name": trace.loss_name,
        "loss_module": trace.loss_module,
        "training_step_helper": trace.training_step_helper,
        "surface_reader_used": True,
        "forward_executed": trace.forward_executed,
        "backward_executed": trace.backward_executed,
        "gradient_observed": trace.gradient_observed,
        "finite_loss_observed": trace.finite_loss_observed,
        "surface_derived_primary_input": trace.surface_derived_primary_input,
        "random_used": trace.random_used,
        "random_role": trace.random_role,
        "units_processed": trace.units_processed,
        "elapsed_seconds": elapsed,
        "units_per_second": trace.units_processed / elapsed if elapsed > 0 else float(trace.units_processed),
        "peak_cuda_memory_reserved_fraction": trace.peak_cuda_memory_reserved_fraction,
        "oom": False,
        "error": None,
        "selected": False,
        "selection_reason": None,
        "synthetic_device_smoke_fallback": False,
        "calibration_surface_reader_num_workers_used": trace.calibration_surface_reader_num_workers_used,
        "calibration_surface_reader_worker_mode": trace.calibration_surface_reader_worker_mode,
        "calibration_surface_reader_prefetch_factor_used": trace.calibration_surface_reader_prefetch_factor_used,
        "calibration_surface_reader_persistent_workers_used": trace.calibration_surface_reader_persistent_workers_used,
    }
    if trace.forward_only_reason is not None:
        row["forward_only_reason"] = trace.forward_only_reason
    return row


def synthetic_device_smoke_row(
    *,
    parameter: str,
    value: int,
) -> dict[str, object]:
    return {
        "parameter": parameter,
        "value": value,
        "candidate": {parameter: value},
        "elapsed_seconds": None,
        "units_processed": 0,
        "units_per_second": None,
        "peak_cuda_memory_reserved_gb": None,
        "peak_cuda_memory_reserved_fraction": None,
        "oom": False,
        "error": None,
        "selected": False,
        "selection_reason": None,
        "benchmark_type": "synthetic_device_smoke",
        "synthetic_device_smoke_fallback": True,
    }


def _trace_to_jsonable(trace: ProviderRealCalibrationTrace) -> dict[str, object]:
    payload = asdict(trace)
    payload["surface_metadata_path"] = str(trace.surface_metadata_path)
    if trace.representative_source_manifest_path is not None:
        payload["representative_source_manifest_path"] = str(
            trace.representative_source_manifest_path
        )
    return payload


def _trace_from_mapping(payload: Mapping[str, object]) -> ProviderRealCalibrationTrace:
    values = dict(payload)
    values["surface_metadata_path"] = Path(str(values["surface_metadata_path"]))
    if values.get("representative_source_manifest_path") is not None:
        values["representative_source_manifest_path"] = Path(
            str(values["representative_source_manifest_path"])
        )
    return ProviderRealCalibrationTrace(**values)  # type: ignore[arg-type]


def _selected_overrides_artifact_payload(
    *,
    provider_key: str,
    model_key: str,
    run_name: str | None,
    effective_config_hash: str,
    model_build_spec_hash: str,
    calibration_artifact_path: Path,
    overrides: Mapping[str, int],
    trace_ids: Sequence[str],
) -> dict[str, object]:
    return {
        "source": "provider_real_calibration",
        "provider_key": provider_key,
        "model_key": model_key,
        "run_name": run_name,
        "effective_config_hash": effective_config_hash,
        "model_build_spec_hash": model_build_spec_hash,
        "calibration_artifact_path": str(calibration_artifact_path),
        "selected_candidate_trace_ids": tuple(trace_ids),
        "overrides": dict(overrides),
    }


def _model_config_hash(payload: Mapping[str, object]) -> str:
    return stable_config_hash(payload)


def stable_config_hash(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            _jsonable_config(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


_PERFORMANCE_OVERRIDE_KEYS = frozenset(
    {
        "batch_size",
        "frame_batch_size",
        "source_batch_size",
        "tokenizer_batch_size",
        "text_to_token_batch_size",
        "denoiser_batch_size",
        "decode_batch_size",
        "reconstruction_batch_size",
    }
)

_FORBIDDEN_OVERRIDE_PATH_PARTS = frozenset(
    {
        "architecture",
        "identity",
        "manifest",
        "split",
        "surface",
        "schema",
        "data",
        "pose_representation",
        "latent_target",
    }
)

_ALLOWED_OVERRIDE_PREFIXES = ("training.", "generation.", "runtime.performance.")


def apply_selected_overrides_to_effective_config(
    *,
    effective_config: Mapping[str, object],
    selected_overrides: Mapping[str, object],
    policy: ProviderCalibrationPolicy,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    """Apply provider-real selected performance overrides to an effective config."""

    if not isinstance(effective_config, Mapping):
        raise ModelWorkflowInvariantError("effective_config must be a mapping.")
    if not isinstance(selected_overrides, Mapping):
        raise ModelWorkflowInvariantError("selected_overrides must be a mapping.")
    calibrated = _mutable_json_mapping(effective_config)
    for key, value in selected_overrides.items():
        if key not in policy.candidate_keys:
            raise ModelWorkflowInvariantError(f"unknown calibration selected_override key: {key!r}.")
        if key not in policy.override_targets:
            raise ModelWorkflowInvariantError(f"calibration selected_override has no target: {key!r}.")
        if key not in _PERFORMANCE_OVERRIDE_KEYS:
            raise ModelWorkflowInvariantError(f"calibration override key is not performance-scoped: {key!r}.")
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ModelWorkflowInvariantError(
                f"calibration override {key!r} must be a positive integer."
            )
        target = policy.override_targets[key]
        _reject_forbidden_override_target(target)
        if not target.startswith(_ALLOWED_OVERRIDE_PREFIXES):
            raise ModelWorkflowInvariantError(
                f"calibration override target is not in an allowed runtime prefix: {target!r}."
            )
        existing = _get_config_path(calibrated, target)
        if not _type_compatible(existing, value):
            raise ModelWorkflowInvariantError(
                f"calibration override {key!r} value is not type-compatible with {target!r}."
            )
        _set_config_path(calibrated, target, value)
    applied = {policy.override_targets[str(key)]: value for key, value in selected_overrides.items()}
    bindings = calibrated.get("calibration_override_bindings")
    if isinstance(bindings, Mapping):
        for key, value in selected_overrides.items():
            binding = bindings.get(str(key))
            if isinstance(binding, str) and binding.strip():
                if _config_path_exists(calibrated, binding):
                    _set_config_path(calibrated, binding, value)
                    applied[binding] = value
    base_application = effective_config.get("compute_profile_application")
    calibrated["compute_calibration_application"] = {
        "schema_version": "model.compute_calibration_application.v1",
        "source": "provider_real_calibration",
        "selected_overrides": dict(selected_overrides),
        "applied_override_paths": dict(applied),
        "base_compute_profile_application": (
            dict(base_application) if isinstance(base_application, Mapping) else {}
        ),
        "base_effective_config_hash": stable_config_hash(effective_config),
        "selected_overrides_hash": stable_config_hash(dict(selected_overrides)),
        # calibrated_effective_config_hash_ref and _scope describe where the
        # self-referential-free calibrated config hash lives in the artifact.
        # The hash is written at the top-level artifact key, not inside this
        # mapping, so that hashing the payload can exclude the hash field itself.
        "calibrated_effective_config_hash_ref": "calibrated_effective_config_hash",
        "calibrated_effective_config_hash_scope": (
            "artifact_payload_without_calibrated_effective_config_hash_field"
        ),
    }
    return calibrated, applied


def _calibration_candidates_from_application(
    effective_config: Mapping[str, object],
) -> Mapping[str, tuple[object, ...]]:
    application = effective_config.get("compute_profile_application")
    if not isinstance(application, Mapping):
        raise ModelWorkflowInvariantError(
            "calibration requires effective_config.compute_profile_application."
        )

    candidates_applicable = application.get("candidates_applicable")
    calibration_candidate_keys = application.get("calibration_candidate_keys")

    if not isinstance(candidates_applicable, Mapping):
        raise ModelWorkflowInvariantError(
            "calibration requires compute_profile_application.candidates_applicable."
        )

    if (
        not isinstance(calibration_candidate_keys, Sequence)
        or isinstance(calibration_candidate_keys, str | bytes | bytearray)
    ):
        raise ModelWorkflowInvariantError(
            "calibration requires compute_profile_application.calibration_candidate_keys."
        )

    result: dict[str, tuple[object, ...]] = {}
    for key in calibration_candidate_keys:
        if not isinstance(key, str):
            raise ModelWorkflowInvariantError("calibration candidate keys must be strings.")
        values = candidates_applicable.get(key)
        if (
            not isinstance(values, Sequence)
            or isinstance(values, str | bytes | bytearray)
        ):
            raise ModelWorkflowInvariantError(
                f"calibration candidate {key!r} must have a sequence of values."
            )
        result[key] = tuple(values)

    extra = set(candidates_applicable) - set(result)
    if extra:
        raise ModelWorkflowInvariantError(
            "applicable candidate keys are not present in calibration_candidate_keys: "
            f"{sorted(extra)}"
        )

    return MappingProxyType(result)


def build_calibrated_loaded_config(
    loaded_config: ModelProviderLoadedConfig,
    pre_calibration: PreExecutionCalibrationResult,
) -> ModelProviderLoadedConfig:
    if pre_calibration.calibrated_effective_config_path is None:
        return loaded_config
    if not pre_calibration.calibrated_effective_config_path.is_file():
        raise ModelWorkflowInvariantError("calibrated effective config artifact is missing.")
    payload = json.loads(pre_calibration.calibrated_effective_config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ModelWorkflowInvariantError("calibrated effective config artifact must be an object.")
    effective = payload.get("effective_config")
    if not isinstance(effective, Mapping):
        raise ModelWorkflowInvariantError("calibrated effective config artifact missing effective_config.")
    if stable_config_hash(effective) != pre_calibration.calibrated_effective_config_hash:
        raise ModelWorkflowInvariantError("calibrated effective config hash mismatch.")
    return replace(loaded_config, effective_config=effective)


def ensure_pre_execution_calibration(
    *,
    provider: ModelProvider,
    request: ModelRunRequest,
    loaded_config: ModelProviderLoadedConfig,
    stage_plan: ModelStagePlan,
    output_root: Path,
    working_dir: Path,
    topology: ArtifactTopology | None = None,
    progress_session: ProgressSession | None,
) -> PreExecutionCalibrationResult:
    """Ensure required provider-real calibration exists before provider stages run."""

    policy = provider.calibration_policy(loaded_config)
    if policy.provider_key != request.model_key.value:
        raise ModelWorkflowInvariantError("provider calibration policy provider_key mismatch.")
    base_hash = stable_config_hash(loaded_config.effective_config)
    required = (
        request.run_mode.value in policy.required_run_modes
        and _compute_profile_name(request) in policy.required_compute_profiles
    )
    root = Path(output_root)
    calibrated_path = root / "calibrated_effective_config.json"
    if not required:
        _write_calibrated_effective_config(
            path=calibrated_path,
            provider_key=policy.provider_key,
            model_key=request.model_key.value,
            run_name=request.run_name,
            base_hash=base_hash,
            selected_hash=None,
            calibrated=loaded_config.effective_config,
            applied={},
            progress_session=progress_session,
        )
        return PreExecutionCalibrationResult(
            required=False,
            executed=False,
            reused=False,
            reason="calibration_not_required",
            compute_calibration_path=None,
            compute_calibration_report_path=None,
            selected_overrides_path=None,
            calibrated_effective_config_path=calibrated_path,
            selected_overrides={},
            applied_overrides={},
            base_effective_config_hash=base_hash,
            selected_overrides_hash=None,
            calibrated_effective_config_hash=base_hash,
            policy=policy,
            blocking_issues=(),
        )
    if not policy.supports_provider_real:
        return PreExecutionCalibrationResult(
            required=True,
            executed=False,
            reused=False,
            reason="provider_real_calibration_not_supported",
            compute_calibration_path=None,
            compute_calibration_report_path=None,
            selected_overrides_path=None,
            calibrated_effective_config_path=None,
            selected_overrides={},
            applied_overrides={},
            base_effective_config_hash=base_hash,
            selected_overrides_hash=None,
            calibrated_effective_config_hash=base_hash,
            policy=policy,
            blocking_issues=("provider_real_calibration_not_supported",),
        )

    candidates = _policy_candidates_from_application(
        loaded_config.effective_config,
        policy,
    )
    expected_build_hash = provider_model_build_spec_hash(
        request.model_key.value,
        effective_config=loaded_config.effective_config,
    )
    status = inspect_provider_real_calibration_artifacts(
        calibration_root=root,
        provider_key=request.model_key.value,
        expected_provider_config_hash=base_hash,
        expected_model_build_spec_hash=expected_build_hash,
        required=True,
    )
    reused = bool(policy.reuse_if_artifacts_match and not status.blocking_issues)
    if not reused:
        result = run_compute_calibration(
            provider_config=loaded_config,
            stage_plan=stage_plan,
            output_root=root,
            max_batches_per_candidate=policy.max_batches_per_candidate,
            max_samples=policy.max_samples,
            warmup_batches=policy.warmup_batches,
            topology=topology,
            progress_session=progress_session,
        )
        status = inspect_provider_real_calibration_artifacts(
            calibration_root=root,
            provider_key=request.model_key.value,
            expected_provider_config_hash=base_hash,
            expected_model_build_spec_hash=expected_build_hash,
            required=True,
        )
        if status.blocking_issues:
            return PreExecutionCalibrationResult(
                required=True,
                executed=True,
                reused=False,
                reason="calibration_artifacts_invalid",
                compute_calibration_path=result.json_path,
                compute_calibration_report_path=result.markdown_path,
                selected_overrides_path=status.selected_overrides_path,
                calibrated_effective_config_path=None,
                selected_overrides={},
                applied_overrides={},
                base_effective_config_hash=base_hash,
                selected_overrides_hash=None,
                calibrated_effective_config_hash=base_hash,
                policy=policy,
                blocking_issues=status.blocking_issues,
            )
    selected_path = root / "selected_overrides.json"
    selected_payload = _load_selected_overrides_payload(selected_path)
    selected = selected_payload["overrides"]
    if not isinstance(selected, Mapping) or not selected:
        raise ModelWorkflowInvariantError("required calibration produced no selected_overrides.")
    unknown = set(selected) - set(candidates)
    if unknown:
        raise ModelWorkflowInvariantError(
            f"selected_overrides contain keys outside compute-profile candidates: {sorted(unknown)}."
        )
    calibrated, applied = apply_selected_overrides_to_effective_config(
        effective_config=loaded_config.effective_config,
        selected_overrides=selected,
        policy=policy,
    )
    selected_hash = stable_config_hash(selected)
    calibrated_hash = stable_config_hash(calibrated)
    _write_calibrated_effective_config(
        path=calibrated_path,
        provider_key=policy.provider_key,
        model_key=request.model_key.value,
        run_name=request.run_name,
        base_hash=base_hash,
        selected_hash=selected_hash,
        calibrated=calibrated,
        applied=applied,
        progress_session=progress_session,
    )
    return PreExecutionCalibrationResult(
        required=True,
        executed=not reused,
        reused=reused,
        reason="reused_matching_artifacts" if reused else "executed_provider_real_calibration",
        compute_calibration_path=root / "compute_calibration.json",
        compute_calibration_report_path=root / "compute_calibration.md",
        selected_overrides_path=selected_path,
        calibrated_effective_config_path=calibrated_path,
        selected_overrides=dict(selected),
        applied_overrides=applied,
        base_effective_config_hash=base_hash,
        selected_overrides_hash=selected_hash,
        calibrated_effective_config_hash=calibrated_hash,
        policy=policy,
        representative_surfaces=_representative_results_from_payload(
            json.loads((root / "compute_calibration.json").read_text(encoding="utf-8"))
        ),
        blocking_issues=(),
    )


def run_compute_calibration(
    *,
    provider_config: ModelProviderLoadedConfig,
    stage_plan: ModelStagePlan,
    output_root: Path,
    measurements: Sequence[Mapping[str, object]] | None = None,
    max_batches_per_candidate: int = 50,
    max_samples: int | None = None,
    warmup_batches: int = 1,
    topology: ArtifactTopology | None = None,
    progress_session: ProgressSession | None = None,
) -> ComputeCalibrationResult:
    """Write deterministic calibration artifacts from measured candidate rows."""

    if provider_config.model_key is not stage_plan.model.key:
        raise ModelWorkflowInvariantError("calibration provider config does not match stage plan.")
    from text_to_sign_production.workflows.model.processing.provider_progress import (
        ModelProviderProgress,
    )

    progress = ModelProviderProgress(
        progress_session=progress_session,
        provider_key=provider_config.model_key.value,
        provider_stage_id="compute_calibration",
        run_mode=stage_plan.request.run_mode.value,
    )
    effective = provider_config.effective_config
    compute_profile = effective.get("compute_profile")
    if not isinstance(compute_profile, Mapping):
        raise ModelWorkflowInvariantError("calibration requires effective compute_profile.")
    candidates = _calibration_candidates_from_application(effective)
    context = build_provider_real_calibration_context(
        loaded_config=provider_config,
        stage_plan=stage_plan,
        working_dir=Path(output_root),
        layout_root=Path(output_root).parent,
        topology=topology,
    )
    rows = tuple(dict(row) for row in (measurements or ()))
    _reject_provider_named_synthetic_rows(rows)
    if not rows:
        candidate_total = sum(
            len(values) if isinstance(values, Sequence) and not isinstance(values, str | bytes | bytearray) else 0
            for values in candidates.values()
        )
        candidate_task = progress.task(
            operation="candidates",
            label="model compute calibration candidates",
            unit="candidate",
            total=candidate_total,
            total_semantics="provider-real calibration candidate values",
        )
        rows = tuple(
            run_provider_real_calibration(
                context=context,
                candidates=candidates,
                max_batches_per_candidate=max_batches_per_candidate,
                max_samples=max_samples,
                warmup_batches=warmup_batches,
                progress_session=progress_session,
            )
        )
        candidate_task.advance(candidate_total, counters={"candidate": candidate_total})
        candidate_task.close()
    authoritative = bool(rows) and all(
        _is_authoritative_provider_real_row(row)
        and _provider_real_row_matches_context(row, context=context)
        for row in rows
    )
    selected_overrides, flags = (
        select_compute_calibration_overrides(rows)
        if authoritative
        else ({}, {"provider_real_calibration_missing": True})
    )
    selected_trace_ids = tuple(
        str(row["trace_id"])
        for row in rows
        if row.get("selected") is True and isinstance(row.get("trace_id"), str)
    )
    if authoritative and not selected_trace_ids:
        selected_trace_ids = tuple(
            str(row["trace_id"])
            for row in rows
            if isinstance(row.get("trace_id"), str)
            and selected_overrides.get(str(row.get("parameter"))) == row.get("value")
        )
    selected_artifact = _selected_overrides_artifact_payload(
        provider_key=provider_config.model_key.value,
        model_key=stage_plan.model.key.value,
        run_name=stage_plan.request.run_name,
        effective_config_hash=context.provider_config_hash,
        model_build_spec_hash=context.model_build_spec_hash,
        calibration_artifact_path=Path(output_root) / "compute_calibration.json",
        overrides=selected_overrides,
        trace_ids=selected_trace_ids,
    )
    payload = {
        "schema_version": "model.compute_calibration.v1",
        "provider_key": provider_config.model_key.value,
        "model_key": stage_plan.model.key.value,
        "run_name": stage_plan.request.run_name,
        "provider_config_hash": context.provider_config_hash,
        "effective_config_hash": context.provider_config_hash,
        "model_build_spec_hash": context.model_build_spec_hash,
        "compute_profile": compute_profile.get("name"),
        "candidate_values": dict(candidates),
        "max_batches_per_candidate": max_batches_per_candidate,
        "warmup_batches": warmup_batches,
        "max_samples": max_samples,
        "measurements": rows,
        "selected_overrides": selected_overrides,
        "selected_candidate_trace_ids": selected_trace_ids,
        "flags": flags,
    }
    root = Path(output_root)
    json_path = root / "compute_calibration.json"
    markdown_path = root / "compute_calibration.md"
    selected_path = root / "selected_overrides.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_calibration_markdown(payload), encoding="utf-8")
    if authoritative:
        selected_task = progress.task(
            operation="selected_overrides_write",
            label="model selected overrides write",
            unit="artifact",
            total=1,
            total_semantics="selected_overrides artifact",
        )
        selected_path.write_text(
            json.dumps(selected_artifact, sort_keys=True, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        selected_task.advance(1, counters={"artifact": 1})
        selected_task.close()
    return ComputeCalibrationResult(
        json_path=json_path,
        markdown_path=markdown_path,
        payload=payload,
    )


def build_provider_real_calibration_context(
    *,
    loaded_config: ModelProviderLoadedConfig,
    stage_plan: ModelStagePlan,
    working_dir: Path,
    layout_root: Path | None = None,
    train_manifest_path: Path | None = None,
    validation_manifest_path: Path | None = None,
    topology: ArtifactTopology | None = None,
) -> ProviderRealCalibrationContext:
    effective = loaded_config.effective_config
    surface_schema = _provider_calibration_surface_schema(
        loaded_config.model_key.value,
        effective_config=effective,
    )
    model_build_spec = _provider_calibration_model_build_spec(
        loaded_config.model_key.value,
        effective_config=effective,
        surface_schema=surface_schema,
    )
    compute_profile = effective.get("compute_profile")
    compute_profile_name = (
        str(compute_profile.get("name"))
        if isinstance(compute_profile, Mapping) and compute_profile.get("name") is not None
        else None
    )
    precision_policy = effective.get("precision_policy")
    resolved_train = train_manifest_path
    resolved_validation = validation_manifest_path
    if topology is not None:
        if resolved_train is None:
            resolved_train = resolve_modeling_manifest_path(
                topology,
                stage_plan.request.manifest_family,
                stage_plan.request.train_split,
            )
        if resolved_validation is None:
            resolved_validation = resolve_modeling_manifest_path(
                topology,
                stage_plan.request.manifest_family,
                stage_plan.request.validation_split,
            )
    return ProviderRealCalibrationContext(
        provider_key=loaded_config.model_key.value,
        model_key=stage_plan.model.key.value,
        run_name=stage_plan.request.run_name,
        run_mode=stage_plan.request.run_mode.value,
        manifest_family=stage_plan.request.manifest_family.family_id,
        loaded_config=loaded_config,
        effective_config=effective,
        stage_plan=stage_plan,
        train_manifest_path=resolved_train,
        validation_manifest_path=resolved_validation,
        topology=topology,
        working_dir=working_dir,
        layout_root=layout_root,
        compute_profile=compute_profile_name,
        precision_policy=str(precision_policy) if precision_policy is not None else None,
        provider_config_hash=stable_config_hash(effective),
        model_build_spec_hash=stable_config_hash(model_build_spec),
    )


def provider_config_hash(config: Mapping[str, object]) -> str:
    """Stable hash for runtime calibration/config matching."""

    return stable_config_hash(config)


def provider_model_build_spec_hash(
    provider_key: str,
    *,
    effective_config: Mapping[str, object],
) -> str:
    surface_schema = _provider_calibration_surface_schema(
        provider_key,
        effective_config=effective_config,
    )
    spec = _provider_calibration_model_build_spec(
        provider_key,
        effective_config=effective_config,
        surface_schema=surface_schema,
    )
    return stable_config_hash(spec)


def _compute_profile_name(request: ModelRunRequest) -> str | None:
    value = request.compute_profile.get("name")
    return str(value) if value is not None else None


def _policy_candidates(
    compute_profile: Mapping[str, object],
    policy: ProviderCalibrationPolicy,
) -> Mapping[str, object]:
    candidates = provider_candidate_overrides(
        compute_profile,
        provider_key=policy.provider_key,
    )
    unknown = set(candidates) - set(policy.candidate_keys)
    if unknown:
        raise ModelWorkflowInvariantError(
            f"compute profile candidates are not in provider calibration policy: {sorted(unknown)}."
        )
    missing = tuple(key for key in policy.candidate_keys if key not in candidates)
    if missing:
        raise ModelWorkflowInvariantError(
            f"compute profile is missing required calibration candidates: {missing!r}."
        )
    return candidates


def _policy_candidates_from_application(
    effective_config: Mapping[str, object],
    policy: ProviderCalibrationPolicy,
) -> Mapping[str, object]:
    candidates = _calibration_candidates_from_application(effective_config)
    unknown = set(candidates) - set(policy.candidate_keys)
    if unknown:
        raise ModelWorkflowInvariantError(
            f"compute profile application candidates are not in provider calibration policy: {sorted(unknown)}."
        )
    missing = tuple(key for key in policy.candidate_keys if key not in candidates)
    if missing:
        raise ModelWorkflowInvariantError(
            f"compute profile application is missing required calibration candidates: {missing!r}."
        )
    return candidates


def representative_manifest_for_policy(
    context: ProviderRealCalibrationContext,
    policy: ProviderCalibrationPolicy,
) -> tuple[SampleSplit, Path, int]:
    if context.topology is None:
        raise ModelWorkflowInvariantError(
            "provider-real representative calibration requires artifact topology."
        )
    split = (
        context.stage_plan.request.train_split
        if policy.representative_split == "train" and context.stage_plan is not None
        else context.stage_plan.request.validation_split
        if context.stage_plan is not None
        else SampleSplit.TRAIN
    )
    manifest = read_modeling_manifest(
        context.topology,
        context.stage_plan.request.manifest_family if context.stage_plan is not None else context.manifest_family,
        split,
    )
    return split, manifest.manifest_path, len(manifest.entries)


def representative_result_from_surface(
    *,
    context: ProviderRealCalibrationContext,
    candidate_key: str,
    surface_kind: str,
    split: SampleSplit,
    surface,
    surface_schema: Mapping[str, object],
    feature_dim: int | None,
) -> RepresentativeCalibrationSurfaceResult:
    metadata_path = surface.root / "metadata.json"
    return RepresentativeCalibrationSurfaceResult(
        provider_key=context.provider_key,
        candidate_key=candidate_key,
        surface_kind=surface_kind,
        split=split.value,
        source_manifest_path=Path(surface.metadata.source_manifest_path),
        source_manifest_sha256=surface.metadata.source_manifest_sha256,
        provider_config_sha256=surface.metadata.provider_config_sha256,
        surface_schema_hash=stable_config_hash(surface_schema),
        feature_dim=feature_dim,
        sample_count=int(surface.metadata.loaded_sample_count),
        unit_count=int(surface.metadata.unit_count),
        surface_root=surface.root,
        surface_metadata_path=metadata_path,
        dataloader_kind=None,
    )


def _representative_results_from_payload(
    payload: Mapping[str, object],
) -> tuple[RepresentativeCalibrationSurfaceResult, ...]:
    results: list[RepresentativeCalibrationSurfaceResult] = []
    seen: set[tuple[str, str]] = set()
    measurements = payload.get("measurements")
    if not isinstance(measurements, Sequence) or isinstance(measurements, str | bytes | bytearray):
        return ()
    for row in measurements:
        if not isinstance(row, Mapping):
            continue
        candidate = row.get("candidate_name") or row.get("parameter")
        surface_kind = row.get("representative_surface_kind") or row.get("surface_kind")
        manifest_path = row.get("representative_source_manifest_path")
        manifest_sha = row.get("representative_source_manifest_sha256")
        config_sha = row.get("representative_provider_config_sha256") or row.get("surface_provider_config_sha256")
        schema_hash = row.get("representative_surface_schema_hash") or row.get("surface_schema_hash")
        if not all(isinstance(item, str) and item for item in (candidate, surface_kind, manifest_path, manifest_sha, config_sha, schema_hash)):
            continue
        key = (str(candidate), str(surface_kind))
        if key in seen:
            continue
        seen.add(key)
        results.append(
            RepresentativeCalibrationSurfaceResult(
                provider_key=str(payload.get("provider_key")),
                candidate_key=str(candidate),
                surface_kind=str(surface_kind),
                split="train",
                source_manifest_path=Path(str(manifest_path)),
                source_manifest_sha256=str(manifest_sha),
                provider_config_sha256=str(config_sha),
                surface_schema_hash=str(schema_hash),
                feature_dim=row.get("surface_feature_dim") if isinstance(row.get("surface_feature_dim"), int) else None,
                sample_count=int(row.get("representative_sample_count") or 1),
                unit_count=int(row.get("representative_unit_count") or row.get("units_processed") or 1),
                surface_root=None,
                surface_metadata_path=Path(str(row["surface_metadata_path"])) if isinstance(row.get("surface_metadata_path"), str) else None,
                dataloader_kind=None,
            )
        )
    return tuple(results)


def _load_selected_overrides_payload(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        raise ModelWorkflowInvariantError("selected_overrides artifact is missing.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ModelWorkflowInvariantError("selected_overrides artifact must be an object.")
    if payload.get("source") != "provider_real_calibration":
        raise ModelWorkflowInvariantError("selected_overrides must come from provider_real_calibration.")
    overrides = payload.get("overrides")
    if not isinstance(overrides, Mapping):
        raise ModelWorkflowInvariantError("selected_overrides artifact missing overrides mapping.")
    return payload


def _write_calibrated_effective_config(
    *,
    path: Path,
    provider_key: str,
    model_key: str,
    run_name: str | None,
    base_hash: str,
    selected_hash: str | None,
    calibrated: Mapping[str, object],
    applied: Mapping[str, object],
    progress_session: ProgressSession | None,
) -> None:
    from text_to_sign_production.workflows.model.processing.provider_progress import (
        ModelProviderProgress,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    calibrated_hash = stable_config_hash(calibrated)
    payload_without_hash: dict[str, object] = {
        "schema_version": "model.calibrated_effective_config.v1",
        "provider_key": provider_key,
        "model_key": model_key,
        "run_name": run_name,
        "base_effective_config_hash": base_hash,
        "selected_overrides_hash": selected_hash,
        "applied_overrides": dict(applied),
        "source": "provider_real_calibration" if applied else "base_effective_config",
        "effective_config": _jsonable_config(calibrated),
    }
    # Write calibrated_effective_config_hash at artifact top-level to avoid
    # self-referential hashing: the hash covers the payload *without* this field.
    payload = {**payload_without_hash, "calibrated_effective_config_hash": calibrated_hash}
    progress = ModelProviderProgress(
        progress_session=progress_session,
        provider_key=provider_key,
        provider_stage_id="pre_execution_calibration",
        run_mode="full",
    )
    task = progress.task(
        operation="calibrated_effective_config_write",
        label="model calibrated effective config write",
        unit="artifact",
        total=1,
        total_semantics="calibrated effective config artifact",
    )
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    task.advance(1, counters={"artifact": 1})
    task.close()


def _mutable_json_mapping(value: Mapping[str, object]) -> dict[str, object]:
    payload = _jsonable_config(value)
    if not isinstance(payload, Mapping):
        raise ModelWorkflowInvariantError("effective_config must be JSON object compatible.")
    return dict(payload)


def _get_config_path(config: Mapping[str, object], target: str) -> object:
    current: object = config
    for part in target.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ModelWorkflowInvariantError(f"calibration override target path does not exist: {target!r}.")
        current = current[part]
    return current


def _config_path_exists(config: Mapping[str, object], target: str) -> bool:
    try:
        _get_config_path(config, target)
    except ModelWorkflowInvariantError:
        return False
    return True


def _set_config_path(config: dict[str, object], target: str, value: object) -> None:
    current: object = config
    parts = target.split(".")
    for part in parts[:-1]:
        if not isinstance(current, dict):
            raise ModelWorkflowInvariantError(f"calibration override target path is not mutable: {target!r}.")
        current = current[part]
    if not isinstance(current, dict):
        raise ModelWorkflowInvariantError(f"calibration override target path is not mutable: {target!r}.")
    current[parts[-1]] = value


def _type_compatible(existing: object, value: object) -> bool:
    if isinstance(existing, bool):
        return isinstance(value, bool)
    if isinstance(existing, int) and not isinstance(existing, bool):
        return isinstance(value, int) and not isinstance(value, bool)
    if isinstance(existing, float):
        return isinstance(value, int | float) and not isinstance(value, bool)
    if existing is None:
        return isinstance(value, int | float | str | bool) or value is None
    return isinstance(value, type(existing))


def _reject_forbidden_override_target(target: str) -> None:
    parts = set(target.split("."))
    if parts & _FORBIDDEN_OVERRIDE_PATH_PARTS:
        raise ModelWorkflowInvariantError(
            f"calibration override target is not performance-scoped: {target!r}."
        )


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _jsonable_config(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable_config(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_jsonable_config(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def inspect_provider_real_calibration_artifacts(
    *,
    calibration_root: Path | None,
    provider_key: str,
    expected_provider_config_hash: str | None = None,
    expected_model_build_spec_hash: str | None = None,
    required: bool = True,
) -> ProviderRealCalibrationArtifactStatus:
    """Inspect calibration artifacts without treating synthetic rows as evidence."""

    if calibration_root is None:
        return ProviderRealCalibrationArtifactStatus(
            provider_real_calibration_required=required,
            provider_real_calibration_found=False,
            selected_overrides_found=False,
            calibration_artifact_path=None,
            selected_overrides_path=None,
            calibration_authoritative=False,
            calibration_benchmark_type=None,
        )
    root = Path(calibration_root)
    calibration_path = root / "compute_calibration.json"
    selected_path = root / "selected_overrides.json"
    selected_found = selected_path.is_file()
    if not calibration_path.is_file():
        return ProviderRealCalibrationArtifactStatus(
            provider_real_calibration_required=required,
            provider_real_calibration_found=False,
            selected_overrides_found=selected_found,
            calibration_artifact_path=calibration_path,
            selected_overrides_path=selected_path if selected_found else None,
            calibration_authoritative=False,
            calibration_benchmark_type=None,
        )
    try:
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ModelWorkflowInvariantError(
            f"compute calibration artifact could not be read: {calibration_path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ModelWorkflowInvariantError("compute calibration artifact must contain a JSON object.")
    measurements = payload.get("measurements")
    rows = tuple(row for row in measurements if isinstance(row, Mapping)) if isinstance(measurements, Sequence) else ()
    authoritative_rows = tuple(row for row in rows if _is_authoritative_provider_real_row(row))
    selected_not_provider_real, selected_trace_mismatch = _selected_overrides_linkage_issues(
        selected_path=selected_path,
        selected_found=selected_found,
        payload=payload,
        authoritative_rows=authoritative_rows,
        provider_key=provider_key,
        expected_provider_config_hash=expected_provider_config_hash,
        expected_model_build_spec_hash=expected_model_build_spec_hash,
    )
    benchmark_types = tuple(
        str(row.get("benchmark_type"))
        for row in rows
        if isinstance(row.get("benchmark_type"), str)
    )
    synthetic_authoritative = any(
        row.get("benchmark_type") == "synthetic_device_smoke"
        and (
            row.get("authoritative_for_provider") is True
            or row.get("provider_key") == provider_key
        )
        for row in rows
    )
    payload_hash = payload.get("provider_config_hash")
    config_hash_mismatch = (
        expected_provider_config_hash is not None
        and (
            not isinstance(payload_hash, str)
            or payload_hash != expected_provider_config_hash
        )
    )
    payload_effective_hash = payload.get("effective_config_hash")
    payload_build_hash = payload.get("model_build_spec_hash")
    effective_config_hash_mismatch = (
        expected_provider_config_hash is not None
        and (
            not isinstance(payload_effective_hash, str)
            or payload_effective_hash != expected_provider_config_hash
        )
    )
    model_build_spec_hash_mismatch = (
        expected_model_build_spec_hash is not None
        and (
            not isinstance(payload_build_hash, str)
            or payload_build_hash != expected_model_build_spec_hash
        )
    )
    not_bound_to_effective_config = any(
        row.get("effective_config_hash") != expected_provider_config_hash
        or row.get("surface_provider_config_sha256") != expected_provider_config_hash
        for row in authoritative_rows
    ) if expected_provider_config_hash is not None else False
    row_build_mismatch = any(
        row.get("model_build_spec_hash") != expected_model_build_spec_hash
        for row in authoritative_rows
    ) if expected_model_build_spec_hash is not None else False
    provider_real_hook_not_shared_training_step = any(
        row.get("benchmark_type") == "provider_real"
        and row.get("benchmark_kind") != "generation"
        and not isinstance(row.get("training_step_helper"), str)
        for row in rows
    )
    return ProviderRealCalibrationArtifactStatus(
        provider_real_calibration_required=required,
        provider_real_calibration_found=bool(authoritative_rows),
        selected_overrides_found=selected_found,
        calibration_artifact_path=calibration_path,
        selected_overrides_path=selected_path if selected_found else None,
        calibration_authoritative=bool(authoritative_rows),
        calibration_benchmark_type="provider_real" if authoritative_rows else (benchmark_types[0] if benchmark_types else None),
        provider_mismatch=payload.get("provider_key") != provider_key,
        config_hash_mismatch=config_hash_mismatch,
        effective_config_hash_mismatch=effective_config_hash_mismatch,
        model_build_spec_hash_mismatch=model_build_spec_hash_mismatch or row_build_mismatch,
        surface_schema_mismatch=any(
            not isinstance(row.get("surface_schema_hash"), str)
            for row in authoritative_rows
        ),
        not_bound_to_effective_config=not_bound_to_effective_config,
        provider_real_hook_not_shared_training_step=provider_real_hook_not_shared_training_step,
        synthetic_authoritative_attempt=synthetic_authoritative,
        selected_overrides_not_provider_real=selected_not_provider_real,
        selected_trace_id_mismatch=selected_trace_mismatch,
    )


def benchmark_compute_candidates(
    candidates: Mapping[str, object],
    *,
    max_batches_per_candidate: int = 50,
    max_samples: int | None = None,
    warmup_batches: int = 1,
) -> tuple[dict[str, object], ...]:
    """Run a small real torch benchmark for each configured batch-size candidate."""

    import torch

    rows: list[dict[str, object]] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_dim = 128
    hidden_dim = 64
    batches = max(1, min(int(max_batches_per_candidate), 50))
    if max_samples is not None:
        batches = max(1, min(batches, int(max_samples)))
    for parameter, raw_values in candidates.items():
        if not isinstance(parameter, str):
            continue
        values = raw_values if isinstance(raw_values, Sequence) and not isinstance(raw_values, str) else ()
        for value in values:
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                continue
            row = synthetic_device_smoke_row(parameter=parameter, value=value)
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    total_memory = torch.cuda.get_device_properties(0).total_memory
                else:
                    total_memory = None
                weight = torch.randn((feature_dim, hidden_dim), device=device)
                for _ in range(warmup_batches):
                    x = torch.randn((value, feature_dim), device=device)
                    y = torch.relu(x @ weight)
                    del x, y
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                started = time.perf_counter()
                processed = 0
                for _ in range(batches):
                    x = torch.randn((value, feature_dim), device=device)
                    y = torch.relu(x @ weight)
                    processed += int(y.shape[0])
                    del x, y
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                row["elapsed_seconds"] = elapsed
                row["units_processed"] = processed
                row["units_per_second"] = processed / elapsed if elapsed > 0 else None
                if torch.cuda.is_available() and total_memory:
                    reserved = torch.cuda.max_memory_reserved()
                    row["peak_cuda_memory_reserved_gb"] = reserved / (1024 ** 3)
                    row["peak_cuda_memory_reserved_fraction"] = reserved / total_memory
            except RuntimeError as exc:
                message = str(exc)
                row["oom"] = "out of memory" in message.lower()
                row["error"] = message
            rows.append(row)
    selected, _flags = select_compute_calibration_overrides(rows)
    for row in rows:
        parameter = row["parameter"]
        value = row["value"]
        if selected.get(parameter) == value:
            row["selected"] = True
            row["selection_reason"] = "highest_safe_throughput_with_smaller_near_tie"
    return tuple(rows)


def _selected_overrides_linkage_issues(
    *,
    selected_path: Path,
    selected_found: bool,
    payload: Mapping[str, object],
    authoritative_rows: Sequence[Mapping[str, object]],
    provider_key: str,
    expected_provider_config_hash: str | None,
    expected_model_build_spec_hash: str | None,
) -> tuple[bool, bool]:
    if not selected_found:
        return False, False
    try:
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return True, False
    if not isinstance(selected, Mapping):
        return True, False
    if selected.get("source") != "provider_real_calibration":
        return True, False
    if selected.get("provider_key") != provider_key:
        return True, False
    if (
        expected_provider_config_hash is not None
        and selected.get("effective_config_hash") != expected_provider_config_hash
    ):
        return False, True
    if (
        expected_model_build_spec_hash is not None
        and selected.get("model_build_spec_hash") != expected_model_build_spec_hash
    ):
        return False, True
    calibration_ref = selected.get("calibration_artifact_path")
    if not isinstance(calibration_ref, str) or Path(calibration_ref).name != "compute_calibration.json":
        return True, False
    trace_ids = selected.get("selected_candidate_trace_ids")
    if not isinstance(trace_ids, Sequence) or isinstance(trace_ids, str | bytes | bytearray):
        return True, False
    selected_ids = {item for item in trace_ids if isinstance(item, str) and item.strip()}
    if not selected_ids:
        return True, False
    authoritative_ids = {
        str(row.get("trace_id"))
        for row in authoritative_rows
        if isinstance(row.get("trace_id"), str)
    }
    payload_ids = payload.get("selected_candidate_trace_ids")
    payload_selected_ids = {
        item for item in payload_ids
        if isinstance(item, str) and item.strip()
    } if isinstance(payload_ids, Sequence) and not isinstance(payload_ids, str | bytes | bytearray) else set()
    mismatch = not selected_ids.issubset(authoritative_ids) or (
        bool(payload_selected_ids) and selected_ids != payload_selected_ids
    )
    return False, mismatch


def run_synthetic_device_smoke_benchmark(
    candidates: Mapping[str, object],
    *,
    max_batches_per_candidate: int = 50,
    max_samples: int | None = None,
    warmup_batches: int = 1,
) -> tuple[dict[str, object], ...]:
    """Run non-authoritative device smoke rows.

    This alias makes the calibration contract explicit: rows produced here are
    never authoritative for provider batch-size selection.
    """

    return benchmark_compute_candidates(
        candidates,
        max_batches_per_candidate=max_batches_per_candidate,
        max_samples=max_samples,
        warmup_batches=warmup_batches,
    )


def run_provider_real_calibration(
    *,
    context: ProviderRealCalibrationContext,
    candidates: Mapping[str, object],
    benchmark: ProviderRealBenchmark | None = None,
    max_batches_per_candidate: int = 50,
    max_samples: int | None = None,
    warmup_batches: int = 1,
    progress_session: ProgressSession | None = None,
) -> tuple[dict[str, object], ...]:
    """Run authoritative provider-real calibration through provider hooks."""

    if not isinstance(context, ProviderRealCalibrationContext):
        raise ModelWorkflowInvariantError("provider-real calibration requires ProviderRealCalibrationContext.")
    provider_key = context.provider_key
    hook = benchmark or _provider_real_benchmark_for(provider_key)
    if hook.provider_key != provider_key:
        raise ModelWorkflowInvariantError("provider-real calibration hook provider_key mismatch.")
    from text_to_sign_production.modeling.candidates.registry import require_model_provider
    from text_to_sign_production.workflows.model.processing.provider_progress import (
        ModelProviderProgress,
    )

    provider = require_model_provider(context.loaded_config.model_key)
    policy = provider.calibration_policy(context.loaded_config)
    progress = ModelProviderProgress(
        progress_session=progress_session,
        provider_key=provider_key,
        provider_stage_id="provider_real_calibration",
        run_mode=context.run_mode,
    )
    rows: list[dict[str, object]] = []
    for parameter, raw_values in candidates.items():
        if not isinstance(parameter, str):
            continue
        values = raw_values if isinstance(raw_values, Sequence) and not isinstance(raw_values, str) else ()
        if parameter not in policy.candidate_keys:
            raise ModelWorkflowInvariantError(
                f"provider-real calibration candidate is not in provider policy: {parameter!r}."
            )
        for value in values:
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                continue
            split, _manifest_path, manifest_count = representative_manifest_for_policy(context, policy)
            surface_total = min(int(max_samples or policy.max_samples), manifest_count)
            surface_task = progress.task(
                operation="surface_build",
                label="model calibration surface build",
                unit="sample",
                total=max(1, surface_total),
                total_semantics="representative calibration surface samples",
            )
            representative = provider.build_representative_calibration_surface(
                context=context,
                candidate_key=parameter,
                policy=replace(policy, max_samples=int(max_samples or policy.max_samples)),
                progress_session=progress_session,
            )
            del split
            surface_task.advance(representative.sample_count, counters={"sample": representative.sample_count})
            surface_task.close()
            warmup_task = progress.task(
                operation="candidate_warmup_batches",
                label="model calibration candidate warmup",
                unit="batch",
                total=max(0, int(warmup_batches)),
                total_semantics="provider-real calibration warmup batches",
            )
            measured_task = progress.task(
                operation="candidate_measured_batches",
                label="model calibration candidate measured",
                unit="batch",
                total=max(1, int(max_batches_per_candidate)),
                total_semantics="provider-real calibration measured batches",
            )
            callbacks = CalibrationProgressCallbacks(
                advance_warmup_batch=lambda amount: warmup_task.advance(
                    int(amount),
                    counters={"batch": int(amount)},
                ),
                advance_measured_batch=lambda amount: measured_task.advance(
                    int(amount),
                    counters={"batch": int(amount)},
                ),
            )
            trace = hook.run_candidate(
                context=context,
                candidate_key=parameter,
                candidate_value=value,
                representative=representative,
                max_batches_per_candidate=max_batches_per_candidate,
                warmup_batches=warmup_batches,
                progress_callbacks=callbacks,
            )
            warmup_task.close()
            measured_task.close()
            row = calibration_row_from_provider_real_trace(
                trace,
                candidate_name=parameter,
                candidate_value=value,
            )
            rows.append(row)
    selected, _flags = select_compute_calibration_overrides(rows)
    for row in rows:
        parameter = row["parameter"]
        value = row["value"]
        if selected.get(parameter) == value:
            row["selected"] = True
            row["selection_reason"] = "highest_safe_throughput_with_smaller_near_tie"
    return tuple(rows)


def benchmark_provider_compute_candidates(
    context: ProviderRealCalibrationContext,
    candidates: Mapping[str, object],
    *,
    max_batches_per_candidate: int = 50,
    max_samples: int | None = None,
    warmup_batches: int = 1,
) -> tuple[dict[str, object], ...]:
    """Backward-compatible provider-real calibration entry point."""

    return run_provider_real_calibration(
        context=context,
        candidates=candidates,
        max_batches_per_candidate=max_batches_per_candidate,
        max_samples=max_samples,
        warmup_batches=warmup_batches,
    )


def _surface_reader_provider_real_runtime_evidence_required(
    row: Mapping[str, object],
) -> bool:
    """Return True for providers that must have calibration reader runtime evidence."""
    return row.get("provider_key") in {
        "learned_pose_token",
        "latent_diffusion",
        "articulator_aware",
    }


def _is_authoritative_provider_real_row(row: Mapping[str, object]) -> bool:
    if row.get("benchmark_type") != "provider_real":
        return False
    trace_payload = row.get("provider_real_trace")
    if not isinstance(trace_payload, Mapping):
        return False
    try:
        trace = _trace_from_mapping(trace_payload)
        validate_provider_real_trace(trace)
    except (TypeError, KeyError, ModelWorkflowInvariantError, ValueError):
        return False
    if _surface_reader_provider_real_runtime_evidence_required(row):
        if row.get("calibration_surface_reader_num_workers_used") is None:
            return False
        if row.get("calibration_surface_reader_worker_mode") not in {
            "single_process",
            "multiprocess",
        }:
            return False
    return row.get("trace_id") == provider_real_trace_id(trace) and row.get("trace_validation_passed") is True


def _provider_real_row_matches_context(
    row: Mapping[str, object],
    *,
    context: ProviderRealCalibrationContext,
) -> bool:
    if row.get("effective_config_hash") != context.provider_config_hash:
        return False
    if row.get("model_build_spec_hash") != context.model_build_spec_hash:
        return False
    if row.get("surface_provider_config_sha256") != context.provider_config_hash:
        return False
    trace_payload = row.get("provider_real_trace")
    if not isinstance(trace_payload, Mapping):
        return False
    try:
        validate_provider_real_trace(_trace_from_mapping(trace_payload), context=context)
    except (TypeError, KeyError, ModelWorkflowInvariantError, ValueError):
        return False
    return True


def _reject_provider_named_synthetic_rows(rows: Sequence[Mapping[str, object]]) -> None:
    for row in rows:
        if (
            row.get("benchmark_type") == "synthetic_device_smoke"
            and isinstance(row.get("provider_key"), str)
        ):
            raise ModelWorkflowInvariantError(
                "provider-named synthetic calibration rows are not provider-real evidence."
            )
        if (
            row.get("benchmark_type") == "synthetic_device_smoke"
            and row.get("authoritative_for_provider") is True
        ):
            raise ModelWorkflowInvariantError(
                "synthetic calibration rows cannot be authoritative_for_provider."
            )


def _provider_real_benchmark_for(provider_key: str) -> ProviderRealBenchmark:
    from text_to_sign_production.modeling.candidates.articulator_aware.calibration import (
        ArticulatorAwareProviderRealBenchmark,
    )
    from text_to_sign_production.modeling.candidates.base_direct.calibration import (
        BaseDirectProviderRealBenchmark,
    )
    from text_to_sign_production.modeling.candidates.latent_diffusion.calibration import (
        LatentDiffusionProviderRealBenchmark,
    )
    from text_to_sign_production.modeling.candidates.learned_pose_token.calibration import (
        LearnedPoseTokenProviderRealBenchmark,
    )

    hooks: dict[str, type[ProviderRealBenchmark]] = {
        "learned_pose_token": LearnedPoseTokenProviderRealBenchmark,
        "latent_diffusion": LatentDiffusionProviderRealBenchmark,
        "articulator_aware": ArticulatorAwareProviderRealBenchmark,
        "base_direct": BaseDirectProviderRealBenchmark,
    }
    hook_type = hooks.get(provider_key)
    if hook_type is None:
        raise ModelWorkflowInvariantError(
            f"provider-real calibration hook is not implemented for {provider_key!r}."
        )
    return hook_type()


def _provider_calibration_module(provider_key: str):
    prefix = _PROVIDER_MODULE_PREFIXES.get(provider_key)
    if prefix is None:
        raise ModelWorkflowInvariantError(
            f"provider-real calibration hook is not implemented for {provider_key!r}."
        )
    return importlib.import_module(f"{prefix}.calibration")


def _provider_calibration_surface_schema(
    provider_key: str,
    *,
    effective_config: Mapping[str, object],
) -> Mapping[str, object]:
    module = _provider_calibration_module(provider_key)
    helper = getattr(module, "calibration_surface_schema", None)
    if helper is None:
        raise ModelWorkflowInvariantError(
            f"provider-real calibration surface schema helper missing for {provider_key!r}."
        )
    schema = helper(effective_config=effective_config)
    if not isinstance(schema, Mapping) or not schema:
        raise ModelWorkflowInvariantError("provider-real calibration surface schema must be a mapping.")
    return schema


def _provider_calibration_model_build_spec(
    provider_key: str,
    *,
    effective_config: Mapping[str, object],
    surface_schema: Mapping[str, object],
) -> Mapping[str, object]:
    module = _provider_calibration_module(provider_key)
    helper = getattr(module, "calibration_model_build_spec", None)
    if helper is None:
        raise ModelWorkflowInvariantError(
            f"provider-real calibration model build spec helper missing for {provider_key!r}."
        )
    spec = helper(effective_config=effective_config, surface_schema=surface_schema)
    if not isinstance(spec, Mapping) or not spec:
        raise ModelWorkflowInvariantError("provider-real calibration model build spec must be a mapping.")
    return spec


def _surface_kind_for_candidate(provider_key: str, candidate_name: str) -> str:
    if provider_key == "learned_pose_token":
        if "text_to_token" in candidate_name:
            return "text_to_token_sequences"
        return "pose_token_units"
    if provider_key == "latent_diffusion":
        if "autoencoder" in candidate_name:
            return "latent_windows"
        return "latent_sequences"
    if provider_key == "articulator_aware":
        return "articulator_frame_units"
    if provider_key == "base_direct":
        return "processed_pose_dataset"
    return "unknown"


def _provider_calibration_paths(provider_key: str) -> dict[str, str]:
    return {
        "learned_pose_token": {
            "tokenizer_batch_size": "learned_pose_token.tokenizer.forward_backward",
            "text_to_token_batch_size": "learned_pose_token.text_to_token.forward_backward",
            "decode_batch_size": "learned_pose_token.decode.forward",
            "reconstruction_batch_size": "learned_pose_token.reconstruction.forward",
        },
        "latent_diffusion": {
            "denoiser_batch_size": "latent_diffusion.denoiser.forward_backward",
            "autoencoder_batch_size": "latent_diffusion.autoencoder.forward_backward",
        },
        "articulator_aware": {
            "frame_batch_size": "articulator_aware.channel_fusion.forward_backward",
            "source_batch_size": "articulator_aware.length_predictor.forward_backward",
        },
        "base_direct": {
            "batch_size": "base_direct.lazy_dataloader.forward",
        },
    }.get(provider_key, {})


def select_compute_calibration_overrides(
    measurements: Sequence[Mapping[str, object]],
    *,
    memory_fraction_cap: float = 0.85,
    near_tie_fraction: float = 0.05,
) -> tuple[dict[str, int], dict[str, object]]:
    """Select highest-throughput non-OOM candidates with smaller-batch tie-breaks."""

    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in measurements:
        parameter = row.get("parameter")
        value = row.get("value")
        throughput = row.get("units_per_second")
        memory_fraction = row.get("peak_cuda_memory_reserved_fraction")
        if not isinstance(parameter, str) or not isinstance(value, int) or isinstance(value, bool):
            continue
        if row.get("oom") is True or row.get("error"):
            continue
        if not isinstance(throughput, int | float) or isinstance(throughput, bool):
            continue
        if (
            isinstance(memory_fraction, int | float)
            and not isinstance(memory_fraction, bool)
            and float(memory_fraction) > memory_fraction_cap
        ):
            continue
        grouped.setdefault(parameter, []).append(row)
    selected: dict[str, int] = {}
    cpu_or_io_bound: list[str] = []
    for parameter, rows in grouped.items():
        ordered = sorted(
            rows,
            key=lambda row: (
                float(row["units_per_second"]),
                -int(row["value"]),
            ),
            reverse=True,
        )
        best = ordered[0]
        best_tps = float(best["units_per_second"])
        near = [
            row for row in ordered
            if best_tps <= 0 or float(row["units_per_second"]) >= best_tps * (1.0 - near_tie_fraction)
        ]
        chosen = min(near, key=lambda row: int(row["value"]))
        selected[parameter] = int(chosen["value"])
        low_memory = all(
            isinstance(row.get("peak_cuda_memory_reserved_fraction"), int | float)
            and not isinstance(row.get("peak_cuda_memory_reserved_fraction"), bool)
            and float(row["peak_cuda_memory_reserved_fraction"]) < 0.10
            for row in rows
        )
        sorted_by_value = sorted(rows, key=lambda row: int(row["value"]))
        improving = any(
            float(right["units_per_second"]) > float(left["units_per_second"]) * 1.05
            for left, right in zip(sorted_by_value, sorted_by_value[1:], strict=False)
        )
        if low_memory and not improving:
            cpu_or_io_bound.append(parameter)
    return selected, {"cpu_or_io_bound": cpu_or_io_bound}


def _calibration_markdown(payload: Mapping[str, object]) -> str:
    selected = payload.get("selected_overrides")
    selected = selected if isinstance(selected, Mapping) else {}
    lines = [
        "# Compute Calibration",
        "",
        f"- provider_key: {payload.get('provider_key')}",
        f"- compute_profile: {payload.get('compute_profile')}",
        "",
        "## Selected Overrides",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in selected.items())
    if not selected:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "ComputeCalibrationResult",
    "PreExecutionCalibrationResult",
    "ProviderBenchmarkSurfaceBundle",
    "ProviderRealCalibrationContext",
    "ProviderRealCalibrationArtifactStatus",
    "ProviderRealCalibrationTrace",
    "ProviderRealBenchmark",
    "_calibration_candidates_from_application",
    "apply_selected_overrides_to_effective_config",
    "benchmark_compute_candidates",
    "build_calibrated_loaded_config",
    "benchmark_provider_compute_candidates",
    "ensure_pre_execution_calibration",
    "inspect_provider_real_calibration_artifacts",
    "build_provider_real_calibration_context",
    "provider_config_hash",
    "provider_model_build_spec_hash",
    "run_provider_real_calibration",
    "run_synthetic_device_smoke_benchmark",
    "run_compute_calibration",
    "select_compute_calibration_overrides",
    "tensor_fingerprint",
    "batch_fingerprint",
    "stable_config_hash",
    "validate_provider_real_trace",
    "calibration_row_from_provider_real_trace",
]
