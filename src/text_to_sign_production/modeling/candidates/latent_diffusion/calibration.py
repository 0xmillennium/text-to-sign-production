"""Provider-owned calibration hooks for latent_diffusion."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path

import torch

from text_to_sign_production.modeling.backbones.bfh_vectorization import default_bfh_tensor_layout
from text_to_sign_production.modeling.backbones.bfh_standardization import read_bfh_standardization_stats_json
from text_to_sign_production.modeling.candidates.calibration_config import (
    optional_config_path,
    require_positive_int_config_path,
)
from text_to_sign_production.modeling.candidates.calibration_runtime import (
    calibration_reader_runtime_options,
)
from text_to_sign_production.modeling.candidates.calibration_surfaces import (
    RepresentativeCalibrationSurfaceResult,
)
from text_to_sign_production.modeling.candidates.provider import ProviderCalibrationPolicy
from text_to_sign_production.modeling.candidates.latent_diffusion.config import (
    latent_diffusion_config_from_mapping,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.dataset import (
    build_latent_source_surface,
    build_latent_sequence_surface_from_source_surface,
    build_latent_target_spec,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.autoencoder import (
    TemporalWindowAutoencoder,
    compute_window_autoencoder_losses,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.model import (
    LatentDenoiser,
    compute_denoising_loss,
)
from text_to_sign_production.modeling.data_surfaces import (
    ModelDataSurfaceReader,
    ModelDataSurfaceWriter,
    model_data_surface_cache_key,
    sha256_file,
)
from text_to_sign_production.workflows.model.contracts.config import ModelWorkflowInvariantError
from text_to_sign_production.workflows.model.processing.calibration import (
    CalibrationProgressCallbacks,
    ProviderBenchmarkSurfaceBundle,
    ProviderRealCalibrationContext,
    ProviderRealCalibrationTrace,
    batch_fingerprint,
    representative_manifest_for_policy,
    representative_result_from_surface,
    stable_config_hash,
    tensor_fingerprint,
    _model_config_hash,
)


class LatentDiffusionProviderRealBenchmark:
    provider_key = "latent_diffusion"

    def run_candidate(
        self,
        *,
        context: ProviderRealCalibrationContext,
        candidate_key: str,
        candidate_value: int,
        representative: RepresentativeCalibrationSurfaceResult,
        max_batches_per_candidate: int,
        warmup_batches: int,
        progress_callbacks: CalibrationProgressCallbacks | None = None,
    ) -> ProviderRealCalibrationTrace:
        _require_context(context, self.provider_key)
        _validate_representative(context, candidate_key, representative)
        if representative.surface_root is None:
            raise ModelWorkflowInvariantError("provider-real calibration surface_root missing.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        reader_options = calibration_reader_runtime_options(context.effective_config)
        reader = ModelDataSurfaceReader(representative.surface_root)
        reader_runtime_trace: dict[str, object] = {}
        batch = next(
            reader.iter_batches(
                batch_size=int(candidate_value),
                num_workers=reader_options.num_workers,
                prefetch_factor=reader_options.prefetch_factor,
                persistent_workers=reader_options.persistent_workers,
                runtime_trace=reader_runtime_trace,
            )
        )
        batch = _values_batch_from_surface(representative.surface_kind, batch)
        surfaces = _bundle_from_representative(context, representative)
        surface_fp = batch_fingerprint(batch)
        if representative.surface_kind == "latent_windows":
            return self._run_autoencoder(
                candidate_name=candidate_key,
                context=context,
                surfaces=surfaces,
                batch=batch,
                surface_fingerprint=surface_fp,
                device=device,
                max_batches=max_batches_per_candidate,
                warmup_batches=warmup_batches,
                progress_callbacks=progress_callbacks,
                reader_runtime_trace=reader_runtime_trace,
            )
        return self._run_denoiser(
            candidate_name=candidate_key,
            context=context,
            surfaces=surfaces,
            batch=batch,
            surface_fingerprint=surface_fp,
            device=device,
            max_batches=max_batches_per_candidate,
            warmup_batches=warmup_batches,
            forward_only="generation" in candidate_key,
            progress_callbacks=progress_callbacks,
            reader_runtime_trace=reader_runtime_trace,
        )

    def _run_autoencoder(
        self,
        *,
        candidate_name: str,
        context: ProviderRealCalibrationContext,
        surfaces: ProviderBenchmarkSurfaceBundle,
        batch: dict[str, object],
        surface_fingerprint: str,
        device: torch.device,
        max_batches: int,
        warmup_batches: int,
        progress_callbacks: CalibrationProgressCallbacks | None,
        reader_runtime_trace: dict[str, object] | None = None,
    ) -> ProviderRealCalibrationTrace:
        values = batch["values"].to(device=device, dtype=torch.float32)  # type: ignore[index,union-attr]
        model = build_latent_autoencoder_model_from_config(
            effective_config=context.effective_config,
            surface_schema=surfaces.surface_schema,
        ).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        loss_value: torch.Tensor | None = None
        gradient_observed = False

        def step() -> int:
            nonlocal loss_value, gradient_observed
            loss_value = run_latent_autoencoder_training_step(
                model=model,
                values=values,
                optimizer=optimizer,
                effective_config=context.effective_config,
            )
            gradient_observed = gradient_observed or _finite_gradient_observed(model)
            return int(values.shape[0])

        for _ in range(max(0, int(warmup_batches))):
            step()
            if progress_callbacks is not None:
                progress_callbacks.advance_warmup_batch(1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        processed = 0
        for _ in range(max(1, min(int(max_batches), 50))):
            processed += step()
            if progress_callbacks is not None:
                progress_callbacks.advance_measured_batch(1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        return _trace(
            provider_hook=self.__class__.__name__,
            benchmark_name=_benchmark_path(candidate_name),
            benchmark_kind="training",
            forward_only_reason=None,
            surface_kind=surfaces.surface_kind,
            surface_root=surfaces.surface_root,
            context=context,
            surfaces=surfaces,
            surface_batch_fingerprint=surface_fingerprint,
            batch_adapter="latent_diffusion.calibration.window_values",
            model_input_fingerprint=tensor_fingerprint(values),
            target_fingerprint=tensor_fingerprint(values),
            model=model,
            model_config=calibration_model_build_spec(
                effective_config=context.effective_config,
                surface_schema=calibration_surface_schema(effective_config=context.effective_config),
            ),
            model_factory=f"{build_latent_autoencoder_model_from_config.__module__}.{build_latent_autoencoder_model_from_config.__name__}",
            model_expected_input_dim=model.input_dim,
            loss_name="compute_window_autoencoder_losses",
            loss_module=compute_window_autoencoder_losses.__module__,
            training_step_helper=f"{run_latent_autoencoder_training_step.__module__}.{run_latent_autoencoder_training_step.__name__}",
            backward_executed=True,
            gradient_observed=gradient_observed,
            finite_loss_observed=loss_value is not None and bool(torch.isfinite(loss_value.detach()).item()),
            random_used=False,
            random_role=None,
            units_processed=processed,
            elapsed_seconds=elapsed,
            reader_runtime_trace=reader_runtime_trace,
        )

    def _run_denoiser(
        self,
        *,
        candidate_name: str,
        context: ProviderRealCalibrationContext,
        surfaces: ProviderBenchmarkSurfaceBundle,
        batch: dict[str, object],
        surface_fingerprint: str,
        device: torch.device,
        max_batches: int,
        warmup_batches: int,
        forward_only: bool,
        progress_callbacks: CalibrationProgressCallbacks | None,
        reader_runtime_trace: dict[str, object] | None = None,
    ) -> ProviderRealCalibrationTrace:
        values = batch["values"].to(device=device, dtype=torch.float32)  # type: ignore[index,union-attr]
        spec = calibration_model_build_spec(
            effective_config=context.effective_config,
            surface_schema=calibration_surface_schema(effective_config=context.effective_config),
        )
        denoiser_spec = _mapping(spec.get("denoiser"), "denoiser")
        latent_dim = int(denoiser_spec["latent_dim"])
        text_dim = int(denoiser_spec["text_embedding_dim"])
        clean_latent = values[:, :latent_dim]
        text_embedding = values[:, latent_dim : latent_dim + text_dim]
        noise = torch.randn_like(clean_latent)
        noisy_latent = clean_latent + 0.05 * noise
        batch_size = int(clean_latent.shape[0])
        timestep = torch.zeros((batch_size,), dtype=torch.long, device=device)
        position = torch.arange(batch_size, dtype=torch.long, device=device)
        model = build_latent_denoiser_model_from_config(
            effective_config=context.effective_config,
            surface_schema=surfaces.surface_schema,
            min_positions=batch_size,
        ).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        loss_value: torch.Tensor | None = None
        gradient_observed = False

        def step(*, backward: bool) -> int:
            nonlocal loss_value, gradient_observed
            output = model(noisy_latent, timestep, text_embedding, position)
            if backward:
                loss_value = run_latent_denoiser_training_step(
                    model=model,
                    noisy_latent=noisy_latent,
                    timestep=timestep,
                    text_embedding=text_embedding,
                    position=position,
                    target_noise=noise,
                    optimizer=optimizer,
                )
                gradient_observed = gradient_observed or _finite_gradient_observed(model)
            return batch_size

        for _ in range(max(0, int(warmup_batches))):
            step(backward=not forward_only)
            if progress_callbacks is not None:
                progress_callbacks.advance_warmup_batch(1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        processed = 0
        for _ in range(max(1, min(int(max_batches), 50))):
            processed += step(backward=not forward_only)
            if progress_callbacks is not None:
                progress_callbacks.advance_measured_batch(1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        return _trace(
            provider_hook=self.__class__.__name__,
            benchmark_name=_benchmark_path(candidate_name),
            benchmark_kind="generation" if forward_only else "training",
            forward_only_reason="generation" if forward_only else None,
            surface_kind=surfaces.surface_kind,
            surface_root=surfaces.surface_root,
            context=context,
            surfaces=surfaces,
            surface_batch_fingerprint=surface_fingerprint,
            batch_adapter="latent_diffusion.calibration.sequence_latent_plus_noise",
            model_input_fingerprint=batch_fingerprint(
                {"noisy_latent": noisy_latent, "text_embedding": text_embedding}
            ),
            target_fingerprint=tensor_fingerprint(clean_latent),
            model=model,
            model_config=spec,
            model_factory=f"{build_latent_denoiser_model_from_config.__module__}.{build_latent_denoiser_model_from_config.__name__}",
            model_expected_input_dim=None,
            loss_name=None if forward_only else "compute_denoising_loss",
            loss_module=None if forward_only else compute_denoising_loss.__module__,
            training_step_helper=None if forward_only else f"{run_latent_denoiser_training_step.__module__}.{run_latent_denoiser_training_step.__name__}",
            backward_executed=not forward_only,
            gradient_observed=gradient_observed,
            finite_loss_observed=loss_value is not None and bool(torch.isfinite(loss_value.detach()).item()),
            random_used=True,
            random_role="noise",
            units_processed=processed,
            elapsed_seconds=elapsed,
            reader_runtime_trace=reader_runtime_trace,
        )


def _trace(**kwargs: object) -> ProviderRealCalibrationTrace:
    model = kwargs.pop("model")
    surface_root = Path(kwargs.pop("surface_root"))
    context = kwargs.pop("context")
    surfaces = kwargs.pop("surfaces")
    reader_runtime_trace: dict[str, object] | None = kwargs.pop("reader_runtime_trace", None)  # type: ignore[assignment]
    if not isinstance(context, ProviderRealCalibrationContext):
        raise ModelWorkflowInvariantError("provider-real trace requires context.")
    if not isinstance(surfaces, ProviderBenchmarkSurfaceBundle):
        raise ModelWorkflowInvariantError("provider-real trace requires surface bundle.")
    model_config = kwargs.pop("model_config")  # type: ignore[assignment]
    total_memory = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
    memory_fraction = torch.cuda.max_memory_reserved() / total_memory if total_memory else None
    metadata = json.loads((surface_root / "metadata.json").read_text(encoding="utf-8"))
    return ProviderRealCalibrationTrace(
        provider_key="latent_diffusion",
        benchmark_type="provider_real",
        surface_metadata_path=surface_root / "metadata.json",
        surface_metadata_sha256=sha256_file(surface_root / "metadata.json"),
        effective_config_hash=context.provider_config_hash,
        model_build_spec_hash=context.model_build_spec_hash,
        surface_schema_hash=surfaces.surface_schema_hash,
        surface_provider_config_sha256=surfaces.surface_provider_config_sha256,
        surface_feature_dim=surfaces.surface_feature_dim,
        representative_source_manifest_path=Path(str(metadata["source_manifest_path"])),
        representative_source_manifest_sha256=str(metadata["source_manifest_sha256"]),
        representative_surface_kind=surfaces.surface_kind,
        representative_provider_config_sha256=surfaces.surface_provider_config_sha256,
        representative_surface_schema_hash=surfaces.surface_schema_hash,
        representative_sample_count=int(metadata.get("sample_count", 1)),
        representative_unit_count=surfaces.unit_count,
        model_build_spec=model_config,  # type: ignore[arg-type]
        model_class=model.__class__.__name__,
        model_module=model.__class__.__module__,
        model_config_hash=_model_config_hash(model_config),  # type: ignore[arg-type]
        forward_executed=True,
        surface_derived_primary_input=True,
        peak_cuda_memory_reserved_fraction=memory_fraction,
        calibration_surface_reader_num_workers_used=(
            int(reader_runtime_trace["surface_reader_num_workers_used"])
            if reader_runtime_trace is not None and "surface_reader_num_workers_used" in reader_runtime_trace
            else None
        ),
        calibration_surface_reader_worker_mode=(
            str(reader_runtime_trace["surface_reader_worker_mode"])
            if reader_runtime_trace is not None and "surface_reader_worker_mode" in reader_runtime_trace
            else None
        ),
        calibration_surface_reader_prefetch_factor_used=(
            int(reader_runtime_trace["surface_reader_prefetch_factor_used"])  # type: ignore[arg-type]
            if reader_runtime_trace is not None
            and reader_runtime_trace.get("surface_reader_prefetch_factor_used") is not None
            else None
        ),
        calibration_surface_reader_persistent_workers_used=(
            bool(reader_runtime_trace["surface_reader_persistent_workers_used"])
            if reader_runtime_trace is not None and "surface_reader_persistent_workers_used" in reader_runtime_trace
            else None
        ),
        **kwargs,  # type: ignore[arg-type]
    )


def build_representative_calibration_surface(
    *,
    context: ProviderRealCalibrationContext,
    candidate_key: str,
    policy: ProviderCalibrationPolicy,
    progress_session,
) -> RepresentativeCalibrationSurfaceResult:
    del progress_session
    _require_context(context, "latent_diffusion")
    if candidate_key not in policy.candidate_keys:
        raise ModelWorkflowInvariantError(f"unknown calibration candidate: {candidate_key!r}.")
    surface_kind = policy.representative_surface_kinds.get(candidate_key)
    if not isinstance(surface_kind, str):
        raise ModelWorkflowInvariantError(f"candidate has no representative surface kind: {candidate_key!r}.")
    split, manifest_path, manifest_count = representative_manifest_for_policy(context, policy)
    manifest_sha = sha256_file(manifest_path)
    full_schema = calibration_surface_schema(effective_config=context.effective_config)
    surface_schema = _schema_for_kind(full_schema, surface_kind)
    feature_dim = _schema_feature_dim(surface_schema)
    root = (
        context.working_dir
        / "representative_calibration_surfaces"
        / "latent_diffusion"
        / candidate_key
    )
    if context.topology is None or context.stage_plan is None:
        raise ModelWorkflowInvariantError("latent representative surface requires topology and stage plan.")
    provider_config = context.effective_config.get("provider_config")
    if not isinstance(provider_config, Mapping):
        raise ModelWorkflowInvariantError("latent effective config missing provider_config.")
    config = latent_diffusion_config_from_mapping(provider_config)
    source_cache_key = model_data_surface_cache_key(
        provider_key="latent_diffusion",
        surface_kind="latent_source_sequences",
        split=split.value,
        source_manifest_sha256=manifest_sha,
        provider_config_sha256=context.provider_config_hash,
        run_mode=context.run_mode,
        limit_samples=policy.max_samples,
        data_version="provider_real_calibration:source:v1",
    )
    source_surface = build_latent_source_surface(
        topology=context.topology,
        manifest_family=context.stage_plan.request.manifest_family,
        split=split,
        surface_root=root / "source",
        source_manifest_sha256=manifest_sha,
        provider_config_sha256=context.provider_config_hash,
        cache_key=source_cache_key,
        run_mode=context.run_mode,
        manifest_entry_count=manifest_count,
        max_samples=policy.max_samples,
        max_units_per_shard=10_000,
        max_source_samples_per_shard=512,
        standardization_missing_observation_policy=config.latent_target.standardization_missing_observation_policy,
    )
    stats = read_bfh_standardization_stats_json(source_surface.root / "standardization_stats.json")
    target_spec = build_latent_target_spec(
        config=config.latent_target,
        layout=default_bfh_tensor_layout(),
        learned_latent_dim=resolve_latent_calibration_dim(context.effective_config),
    )
    cache_key = model_data_surface_cache_key(
        provider_key="latent_diffusion",
        surface_kind=surface_kind,
        split=split.value,
        source_manifest_sha256=manifest_sha,
        provider_config_sha256=context.provider_config_hash,
        run_mode=context.run_mode,
        limit_samples=policy.max_samples,
        data_version="provider_real_calibration:v1",
    )
    if surface_kind == "latent_sequences":
        surface = build_latent_sequence_surface_from_source_surface(
            source_surface=source_surface,
            stats=stats,
            surface_root=root / "latent_sequences",
            source_manifest_sha256=manifest_sha,
            provider_config_sha256=context.provider_config_hash,
            cache_key=cache_key,
            run_mode=context.run_mode,
            manifest_entry_count=manifest_count,
            target_spec=target_spec,
            max_units_per_shard=10_000,
            max_source_samples_per_shard=512,
        )
    elif surface_kind == "latent_windows":
        surface = _build_latent_window_calibration_surface(
            source_surface=source_surface,
            stats=stats,
            surface_root=root / "latent_windows",
            source_manifest_sha256=manifest_sha,
            provider_config_sha256=context.provider_config_hash,
            cache_key=cache_key,
            run_mode=context.run_mode,
            manifest_entry_count=manifest_count,
            target_spec=target_spec,
        )
    else:
        raise ModelWorkflowInvariantError(f"unsupported latent calibration surface kind: {surface_kind!r}.")
    return representative_result_from_surface(
        context=context,
        candidate_key=candidate_key,
        surface_kind=surface_kind,
        split=split,
        surface=surface,
        surface_schema=surface_schema,
        feature_dim=feature_dim,
    )


def _build_latent_window_calibration_surface(
    *,
    source_surface,
    stats,
    surface_root: Path,
    source_manifest_sha256: str,
    provider_config_sha256: str,
    cache_key: str,
    run_mode: str,
    manifest_entry_count: int,
    target_spec,
):
    reader = ModelDataSurfaceReader(source_surface)
    writer = ModelDataSurfaceWriter(
        root=surface_root,
        provider_key="latent_diffusion",
        surface_kind="latent_windows",
        split=reader.metadata.split,
        manifest_family=reader.metadata.manifest_family,
        source_manifest_path=reader.metadata.source_manifest_path,
        source_manifest_sha256=source_manifest_sha256,
        provider_config_sha256=provider_config_sha256,
        cache_key=cache_key,
        manifest_entry_count=manifest_entry_count,
        run_mode=run_mode,
        data_version="provider_real_calibration:latent_windows:v1",
        max_units_per_shard=10_000,
        max_source_samples_per_shard=512,
    )
    mean = torch.as_tensor(stats.mean, dtype=torch.float32).reshape(-1)
    std = torch.as_tensor(stats.std, dtype=torch.float32).reshape(-1)
    window_size = max(1, int(target_spec.window_size))
    for shard in reader.iter_shards():
        values = torch.where(
            shard["validity_mask"].bool(),
            (shard["pose_values"].float() - mean.unsqueeze(0)) / std.unsqueeze(0),
            torch.zeros_like(shard["pose_values"].float()),
        )
        rows = []
        for start in range(max(1, values.shape[0] - window_size + 1)):
            window = values[start : start + window_size]
            if window.shape[0] < window_size:
                pad = torch.zeros((window_size - window.shape[0], values.shape[1]), dtype=values.dtype)
                window = torch.cat([window, pad], dim=0)
            rows.append(window.reshape(-1))
        stacked = torch.stack(rows) if rows else values[:1].repeat(1, window_size).reshape(1, -1)
        writer.append_units(
            {"values": stacked},
            sample_count=int(shard["metadata"]["sample_count"]),
            frame_count=int(stacked.shape[0]),
        )
    return writer.close()


def _validate_representative(
    context: ProviderRealCalibrationContext,
    candidate_key: str,
    representative: RepresentativeCalibrationSurfaceResult,
) -> None:
    if representative.provider_key != "latent_diffusion":
        raise ModelWorkflowInvariantError("latent representative provider mismatch.")
    if representative.candidate_key != candidate_key:
        raise ModelWorkflowInvariantError("representative candidate key mismatch.")
    if representative.provider_config_sha256 != context.provider_config_hash:
        raise ModelWorkflowInvariantError("representative provider config hash mismatch.")
    if representative.surface_kind not in {"latent_windows", "latent_sequences"}:
        raise ModelWorkflowInvariantError("representative surface kind is unsupported.")
    if not representative.source_manifest_path.is_file():
        raise ModelWorkflowInvariantError("representative source manifest is missing.")
    if sha256_file(representative.source_manifest_path) != representative.source_manifest_sha256:
        raise ModelWorkflowInvariantError("representative source manifest hash mismatch.")


def _bundle_from_representative(
    context: ProviderRealCalibrationContext,
    representative: RepresentativeCalibrationSurfaceResult,
) -> ProviderBenchmarkSurfaceBundle:
    full_schema = calibration_surface_schema(effective_config=context.effective_config)
    surface_schema = _schema_for_kind(full_schema, representative.surface_kind)
    return ProviderBenchmarkSurfaceBundle(
        provider_key="latent_diffusion",
        surface_kind=representative.surface_kind,
        unit_count=representative.unit_count,
        surface_schema=surface_schema,
        surface_schema_hash=representative.surface_schema_hash,
        surface_provider_config_sha256=representative.provider_config_sha256,
        surface_feature_dim=representative.feature_dim,
        surface_root=representative.surface_root,
    )


def _values_batch_from_surface(surface_kind: str, batch: Mapping[str, object]) -> dict[str, object]:
    if isinstance(batch.get("values"), torch.Tensor):
        return dict(batch)
    if surface_kind == "latent_sequences" and isinstance(batch.get("latent_values"), torch.Tensor):
        latent = batch["latent_values"].float()  # type: ignore[index,union-attr]
        text = torch.zeros((latent.shape[0], 256), dtype=latent.dtype, device=latent.device)
        return {**dict(batch), "values": torch.cat([latent, text], dim=1)}
    if surface_kind == "latent_windows" and isinstance(batch.get("windows"), torch.Tensor):
        windows = batch["windows"].float()  # type: ignore[index,union-attr]
        return {**dict(batch), "values": windows.reshape(windows.shape[0], -1)}
    raise ModelWorkflowInvariantError("latent representative batch missing values.")


def calibration_surface_schema(*, effective_config: Mapping[str, object]) -> Mapping[str, object]:
    latent_dim = resolve_latent_calibration_dim(effective_config)
    text_dim = require_positive_int_config_path(
        effective_config,
        "provider_config.text_encoder.output_dim",
    )
    window_size = require_positive_int_config_path(effective_config, "provider_config.latent_target.window_size")
    feature_dim = _latent_window_feature_dim(effective_config)
    return {
        "provider_key": "latent_diffusion",
        "surfaces": {
            "latent_windows": {
                "surface_kind": "latent_windows",
                "fields": {"values": {"feature_dim": window_size * feature_dim}},
            },
            "latent_sequences": {
                "surface_kind": "latent_sequences",
                "fields": {"values": {"feature_dim": latent_dim + text_dim}},
            },
        },
    }


def calibration_model_build_spec(
    *,
    effective_config: Mapping[str, object],
    surface_schema: Mapping[str, object],
) -> Mapping[str, object]:
    window_schema = _schema_for_kind(surface_schema, "latent_windows")
    sequence_schema = _schema_for_kind(surface_schema, "latent_sequences")
    latent_dim = resolve_latent_calibration_dim(effective_config)
    text_dim = require_positive_int_config_path(
        effective_config,
        "provider_config.text_encoder.output_dim",
    )
    window_size = require_positive_int_config_path(effective_config, "provider_config.latent_target.window_size")
    return {
        "provider_key": "latent_diffusion",
        "autoencoder": {
            "input_dim": _schema_feature_dim(window_schema),
            "latent_dim": latent_dim,
            "window_size": window_size,
            "autoencoder_hidden_dim": require_positive_int_config_path(effective_config, "provider_config.denoiser.hidden_dim"),
            "feature_dim": max(1, _schema_feature_dim(window_schema) // window_size),
        },
        "denoiser": {
            "latent_dim": latent_dim,
            "text_embedding_dim": text_dim,
            "sequence_surface_feature_dim": _schema_feature_dim(sequence_schema),
            "denoiser_hidden_dim": require_positive_int_config_path(effective_config, "provider_config.denoiser.hidden_dim"),
            "timestep_embedding_dim": require_positive_int_config_path(effective_config, "provider_config.denoiser.timestep_embedding_dim"),
            "position_embedding_dim": require_positive_int_config_path(effective_config, "provider_config.denoiser.position_embedding_dim"),
            "max_positions": require_positive_int_config_path(effective_config, "provider_config.denoiser.max_positions"),
            "diffusion_steps": require_positive_int_config_path(effective_config, "provider_config.diffusion.timesteps"),
        },
    }


def resolve_latent_calibration_dim(effective_config: Mapping[str, object]) -> int:
    configured = optional_config_path(effective_config, "provider_config.denoiser.latent_dim")
    if isinstance(configured, int) and not isinstance(configured, bool) and configured > 0:
        return configured
    target_type = optional_config_path(effective_config, "provider_config.latent_target.target_type")
    if target_type == "standardized_bfh_frame":
        return default_bfh_tensor_layout().total_feature_dim
    if target_type == "learned_bfh_window_latent":
        return require_positive_int_config_path(effective_config, "provider_config.latent_autoencoder.latent_dim")
    raise ModelWorkflowInvariantError("latent_diffusion calibration cannot resolve latent dimension.")


def _latent_window_feature_dim(effective_config: Mapping[str, object]) -> int:
    target_type = optional_config_path(effective_config, "provider_config.latent_target.target_type")
    if target_type == "standardized_bfh_frame":
        return default_bfh_tensor_layout().total_feature_dim
    if target_type == "learned_bfh_window_latent":
        return require_positive_int_config_path(effective_config, "provider_config.latent_autoencoder.latent_dim")
    raise ModelWorkflowInvariantError("latent_diffusion calibration cannot resolve window feature dimension.")


def build_latent_autoencoder_model_from_config(
    *,
    effective_config: Mapping[str, object],
    surface_schema: Mapping[str, object],
) -> TemporalWindowAutoencoder:
    spec = _mapping(
        calibration_model_build_spec(
            effective_config=effective_config,
            surface_schema=calibration_surface_schema(effective_config=effective_config),
        ).get("autoencoder"),
        "autoencoder",
    )
    return TemporalWindowAutoencoder(
        input_dim=int(spec["input_dim"]),
        latent_dim=int(spec["latent_dim"]),
        hidden_dim=int(spec["autoencoder_hidden_dim"]),
    )


def build_latent_denoiser_model_from_config(
    *,
    effective_config: Mapping[str, object],
    surface_schema: Mapping[str, object],
    min_positions: int = 1,
) -> LatentDenoiser:
    del surface_schema
    spec = _mapping(
        calibration_model_build_spec(
            effective_config=effective_config,
            surface_schema=calibration_surface_schema(effective_config=effective_config),
        ).get("denoiser"),
        "denoiser",
    )
    return LatentDenoiser(
        latent_dim=int(spec["latent_dim"]),
        text_embedding_dim=int(spec["text_embedding_dim"]),
        hidden_dim=int(spec["denoiser_hidden_dim"]),
        timestep_embedding_dim=int(spec["timestep_embedding_dim"]),
        position_embedding_dim=int(spec["position_embedding_dim"]),
        dropout=float(_value_at(effective_config, "denoiser.dropout", 0.0)),
        max_positions=max(int(spec["max_positions"]), int(min_positions)),
    )


def run_latent_autoencoder_training_step(
    *,
    model: TemporalWindowAutoencoder,
    values: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    effective_config: Mapping[str, object],
) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)
    spec = _mapping(
        calibration_model_build_spec(
            effective_config=effective_config,
            surface_schema=calibration_surface_schema(effective_config=effective_config),
        ).get("autoencoder"),
        "autoencoder",
    )
    output = model(values)
    losses = compute_window_autoencoder_losses(
        output=output,
        target=values,
        validity_mask=torch.ones_like(values, dtype=torch.bool),
        window_size=int(spec["window_size"]),
        feature_dim=int(spec["feature_dim"]),
        velocity_loss_weight=float(_value_at(effective_config, "autoencoder.velocity_loss_weight", 0.0)),
    )
    loss_value = losses["total_loss"]
    loss_value.backward()
    optimizer.step()
    return loss_value


def run_latent_denoiser_training_step(
    *,
    model: LatentDenoiser,
    noisy_latent: torch.Tensor,
    timestep: torch.Tensor,
    text_embedding: torch.Tensor,
    position: torch.Tensor,
    target_noise: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)
    output = model(noisy_latent, timestep, text_embedding, position)
    loss_value = compute_denoising_loss(
        predicted_noise=output,
        target_noise=target_noise,
        validity_mask=torch.ones_like(output, dtype=torch.bool),
    )
    loss_value.backward()
    optimizer.step()
    return loss_value


def _require_context(context: ProviderRealCalibrationContext, provider_key: str) -> None:
    if not isinstance(context, ProviderRealCalibrationContext):
        raise ModelWorkflowInvariantError("provider-real calibration requires ProviderRealCalibrationContext.")
    if context.provider_key != provider_key:
        raise ModelWorkflowInvariantError("provider-real calibration context provider mismatch.")


def _schema_for_kind(surface_schema: Mapping[str, object], surface_kind: str) -> Mapping[str, object]:
    surfaces = _mapping(surface_schema.get("surfaces"), "surfaces")
    return _mapping(surfaces.get(surface_kind), surface_kind)


def _schema_feature_dim(surface_schema: Mapping[str, object]) -> int:
    fields = _mapping(surface_schema.get("fields"), "fields")
    values = _mapping(fields.get("values"), "values")
    feature_dim = values.get("feature_dim")
    if not isinstance(feature_dim, int) or isinstance(feature_dim, bool) or feature_dim <= 0:
        raise ModelWorkflowInvariantError("provider-real surface schema missing feature_dim.")
    return feature_dim


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ModelWorkflowInvariantError(f"{name} must be a mapping.")
    return value


def _int_from(effective_config: Mapping[str, object], paths: tuple[str, ...], default: int) -> int:
    for path in paths:
        value = _value_at(effective_config, path, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return default


def _value_at(config: Mapping[str, object], path: str, default: object) -> object:
    current: object = config
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _finite_gradient_observed(model: torch.nn.Module) -> bool:
    return any(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all().item())
        and bool((parameter.grad.detach().abs().sum() > 0).item())
        for parameter in model.parameters()
    )


def _benchmark_path(candidate_name: str) -> str:
    if "autoencoder" in candidate_name:
        return "latent_diffusion.autoencoder.forward_backward"
    if "generation" in candidate_name:
        return "latent_diffusion.generation.forward_only"
    return "latent_diffusion.denoiser.forward_backward"


__all__ = [
    "LatentDiffusionProviderRealBenchmark",
    "build_representative_calibration_surface",
    "build_latent_autoencoder_model_from_config",
    "build_latent_denoiser_model_from_config",
    "calibration_model_build_spec",
    "calibration_surface_schema",
    "resolve_latent_calibration_dim",
    "run_latent_autoencoder_training_step",
    "run_latent_denoiser_training_step",
]
