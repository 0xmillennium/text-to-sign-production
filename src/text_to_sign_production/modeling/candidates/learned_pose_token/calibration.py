"""Provider-owned calibration hooks for learned_pose_token."""

from __future__ import annotations

import time
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from text_to_sign_production.modeling.backbones.bfh_vectorization import default_bfh_tensor_layout
from text_to_sign_production.modeling.backbones.text_encoder import build_text_encoder
from text_to_sign_production.modeling.candidates.calibration_config import (
    optional_config_path,
    require_positive_int_config_path,
)
from text_to_sign_production.modeling.candidates.calibration_runtime import (
    calibration_reader_runtime_options,
)
from text_to_sign_production.modeling.candidates.provider import ProviderCalibrationPolicy
from text_to_sign_production.modeling.candidates.calibration_surfaces import (
    RepresentativeCalibrationSurfaceResult,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.config import (
    learned_pose_token_config_from_effective_dict,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.dataset import (
    build_pose_token_training_surface,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.model import (
    LearnedPoseTokenizer,
    compute_tokenizer_losses,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.predictor import (
    TextToTokenPredictor,
    compute_text_to_token_losses,
)
from text_to_sign_production.modeling.data_surfaces import (
    ModelDataSurfaceReader,
    ModelDataSurfaceWriter,
    model_data_surface_cache_key,
    sha256_file,
)
from text_to_sign_production.modeling.data import load_manifest_sample, read_modeling_manifest
from text_to_sign_production.modeling.backbones.bfh_vectorization import (
    flatten_bfh_vectorized_pose,
    vectorize_bfh_pose_arrays,
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


class LearnedPoseTokenProviderRealBenchmark:
    provider_key = "learned_pose_token"

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
        if representative.surface_kind == "text_to_token_sequences" and "values" not in batch:
            batch = _text_to_token_values_batch(batch)
        if not isinstance(batch.get("values"), torch.Tensor):
            raise ModelWorkflowInvariantError("learned_pose_token calibration batch missing values.")
        surfaces = _bundle_from_representative(context, representative)
        surface_fp = batch_fingerprint(batch)
        if representative.surface_kind == "text_to_token_sequences":
            return self._run_text_to_token(
                candidate_name=candidate_key,
                candidate_value=int(candidate_value),
                context=context,
                surfaces=surfaces,
                batch=batch,
                surface_fingerprint=surface_fp,
                device=device,
                max_batches=max_batches_per_candidate,
                warmup_batches=warmup_batches,
                progress_callbacks=progress_callbacks,
                forward_only="decode" in candidate_key,
                reader_runtime_trace=reader_runtime_trace,
            )
        return self._run_tokenizer(
            candidate_name=candidate_key,
            candidate_value=int(candidate_value),
            context=context,
            surfaces=surfaces,
            batch=batch,
            surface_fingerprint=surface_fp,
            device=device,
            max_batches=max_batches_per_candidate,
            warmup_batches=warmup_batches,
            forward_only="decode" in candidate_key,
            progress_callbacks=progress_callbacks,
            reader_runtime_trace=reader_runtime_trace,
        )

    def _run_tokenizer(
        self,
        *,
        candidate_name: str,
        candidate_value: int,
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
        model = build_learned_tokenizer_model_from_config(
            effective_config=context.effective_config,
            surface_schema=surfaces.surface_schema,
        ).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        loss_value: torch.Tensor | None = None
        gradient_observed = False

        def step(*, backward: bool) -> int:
            nonlocal loss_value, gradient_observed
            if backward:
                loss_value = run_learned_tokenizer_training_step(
                    model=model,
                    values=values,
                    optimizer=optimizer,
                    effective_config=context.effective_config,
                )
                gradient_observed = gradient_observed or _finite_gradient_observed(model)
            else:
                model(values)
            return int(values.shape[0])

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
            batch_adapter="learned_pose_token.calibration.tokenizer_values",
            model_input_fingerprint=tensor_fingerprint(values),
            target_fingerprint=None if forward_only else tensor_fingerprint(values),
            model=model,
            model_config=calibration_model_build_spec(
                effective_config=context.effective_config,
                surface_schema=calibration_surface_schema(effective_config=context.effective_config),
            ),
            model_factory=f"{build_learned_tokenizer_model_from_config.__module__}.{build_learned_tokenizer_model_from_config.__name__}",
            model_expected_input_dim=model.input_dim,
            loss_name=None if forward_only else "compute_tokenizer_losses",
            loss_module=None if forward_only else compute_tokenizer_losses.__module__,
            training_step_helper=None if forward_only else f"{run_learned_tokenizer_training_step.__module__}.{run_learned_tokenizer_training_step.__name__}",
            backward_executed=not forward_only,
            gradient_observed=gradient_observed,
            finite_loss_observed=loss_value is not None and bool(torch.isfinite(loss_value.detach()).item()),
            random_used=False,
            random_role=None,
            units_processed=processed,
            elapsed_seconds=elapsed,
            reader_runtime_trace=reader_runtime_trace,
        )

    def _run_text_to_token(
        self,
        *,
        candidate_name: str,
        candidate_value: int,
        context: ProviderRealCalibrationContext,
        surfaces: ProviderBenchmarkSurfaceBundle,
        batch: dict[str, object],
        surface_fingerprint: str,
        device: torch.device,
        max_batches: int,
        warmup_batches: int,
        progress_callbacks: CalibrationProgressCallbacks | None,
        forward_only: bool = False,
        reader_runtime_trace: dict[str, object] | None = None,
    ) -> ProviderRealCalibrationTrace:
        values = batch["values"].to(device=device, dtype=torch.float32)  # type: ignore[index,union-attr]
        spec = calibration_model_build_spec(
            effective_config=context.effective_config,
            surface_schema=calibration_surface_schema(effective_config=context.effective_config),
        )
        text_spec = _mapping(spec.get("text_to_token"), "text_to_token")
        text_dim = int(text_spec["text_embedding_dim"])
        max_positions = int(text_spec["max_pose_tokens"])
        codebook_size = int(text_spec["codebook_size"])
        text_embeddings = values[:, :text_dim]
        target_ids = values[:, text_dim : text_dim + max_positions].abs().to(dtype=torch.long) % codebook_size
        actual_batch_size = int(text_embeddings.shape[0])

        raw_mask = batch.get("target_mask")
        if isinstance(raw_mask, torch.Tensor):
            token_mask = raw_mask.to(device=device, dtype=torch.bool)
            token_mask = token_mask[:, : target_ids.shape[1]]
        else:
            token_mask = torch.ones_like(target_ids, dtype=torch.bool)

        raw_lengths = batch.get("sequence_length")
        if isinstance(raw_lengths, torch.Tensor):
            target_lengths = raw_lengths.to(device=device, dtype=torch.long).reshape(-1)
            target_lengths = torch.clamp(target_lengths, min=1, max=int(target_ids.shape[1]))
        else:
            target_lengths = token_mask.sum(dim=1).to(device=device, dtype=torch.long)
            target_lengths = torch.clamp(target_lengths, min=1, max=int(target_ids.shape[1]))

        if target_lengths.shape != (actual_batch_size,):
            raise ModelWorkflowInvariantError(
                "learned_pose_token text-to-token calibration target_lengths batch "
                f"shape mismatch: expected {(actual_batch_size,)}, got {tuple(target_lengths.shape)}."
            )
        if token_mask.shape != target_ids.shape:
            raise ModelWorkflowInvariantError(
                "learned_pose_token text-to-token calibration token_mask shape mismatch: "
                f"expected {tuple(target_ids.shape)}, got {tuple(token_mask.shape)}."
            )
        model = build_text_to_token_model_from_config(
            effective_config=context.effective_config,
            surface_schema=surfaces.surface_schema,
        ).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        loss_value: torch.Tensor | None = None
        gradient_observed = False

        def step() -> int:
            nonlocal loss_value, gradient_observed
            if forward_only:
                model(text_embeddings)
            else:
                loss_value = run_text_to_token_training_step(
                    model=model,
                    text_embeddings=text_embeddings,
                    target_token_ids=target_ids,
                    target_lengths=target_lengths,
                    token_mask=token_mask,
                    optimizer=optimizer,
                    effective_config=context.effective_config,
                )
                gradient_observed = gradient_observed or _finite_gradient_observed(model)
            return int(text_embeddings.shape[0])

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
            benchmark_kind="generation" if forward_only else "training",
            forward_only_reason="generation" if forward_only else None,
            surface_kind=surfaces.surface_kind,
            surface_root=surfaces.surface_root,
            context=context,
            surfaces=surfaces,
            surface_batch_fingerprint=surface_fingerprint,
            batch_adapter="learned_pose_token.calibration.text_to_token_values",
            model_input_fingerprint=tensor_fingerprint(text_embeddings),
            target_fingerprint=batch_fingerprint({"target_token_ids": target_ids, "target_lengths": target_lengths}),
            model=model,
            model_config=spec,
            model_factory=f"{build_text_to_token_model_from_config.__module__}.{build_text_to_token_model_from_config.__name__}",
            model_expected_input_dim=None,
            loss_name=None if forward_only else "compute_text_to_token_losses",
            loss_module=None if forward_only else compute_text_to_token_losses.__module__,
            training_step_helper=None if forward_only else f"{run_text_to_token_training_step.__module__}.{run_text_to_token_training_step.__name__}",
            backward_executed=not forward_only,
            gradient_observed=gradient_observed,
            finite_loss_observed=forward_only or (loss_value is not None and bool(torch.isfinite(loss_value.detach()).item())),
            random_used=False,
            random_role=None,
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
        provider_key="learned_pose_token",
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
    _require_context(context, "learned_pose_token")
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
        / "learned_pose_token"
        / candidate_key
        / surface_kind
    )
    cache_key = model_data_surface_cache_key(
        provider_key="learned_pose_token",
        surface_kind=surface_kind,
        split=split.value,
        source_manifest_sha256=manifest_sha,
        provider_config_sha256=context.provider_config_hash,
        run_mode=context.run_mode,
        limit_samples=policy.max_samples,
        data_version="provider_real_calibration:v1",
    )
    config = learned_pose_token_config_from_effective_dict(
        context.effective_config,
        source_path=context.loaded_config.source_path,
    )

    if surface_kind == "pose_token_units":
        if context.topology is None or context.stage_plan is None:
            raise ModelWorkflowInvariantError(
                "learned_pose_token representative surface requires topology and stage plan."
            )
        surface = build_pose_token_training_surface(
            topology=context.topology,
            manifest_family=context.stage_plan.request.manifest_family,
            split=split,
            surface_root=root,
            source_manifest_sha256=manifest_sha,
            provider_config_sha256=context.provider_config_hash,
            cache_key=cache_key,
            run_mode=context.run_mode,
            manifest_entry_count=manifest_count,
            max_samples=policy.max_samples,
            temporal_spec=config.tokenizer.temporal_window_spec(),
            max_units_per_shard=10_000,
            max_source_samples_per_shard=512,
            standardization_missing_observation_policy=(
                config.representation.standardization_missing_observation_policy
            ),
        )
    elif surface_kind == "text_to_token_sequences":
        surface = _build_text_to_token_calibration_surface(
            context=context,
            split=split,
            surface_root=root,
            manifest_path=manifest_path,
            manifest_sha=manifest_sha,
            manifest_count=manifest_count,
            max_samples=policy.max_samples,
            cache_key=cache_key,
        )
    else:
        raise ModelWorkflowInvariantError(f"unsupported learned calibration surface kind: {surface_kind!r}.")
    return representative_result_from_surface(
        context=context,
        candidate_key=candidate_key,
        surface_kind=surface_kind,
        split=split,
        surface=surface,
        surface_schema=surface_schema,
        feature_dim=feature_dim,
    )


def _build_text_to_token_calibration_surface(
    *,
    context: ProviderRealCalibrationContext,
    split,
    surface_root: Path,
    manifest_path: Path,
    manifest_sha: str,
    manifest_count: int,
    max_samples: int,
    cache_key: str,
):
    if context.topology is None or context.stage_plan is None:
        raise ModelWorkflowInvariantError(
            "learned_pose_token text-to-token representative surface requires topology and stage plan."
        )
    config = learned_pose_token_config_from_effective_dict(
        context.effective_config,
        source_path=context.loaded_config.source_path,
    )
    manifest = read_modeling_manifest(
        context.topology,
        context.stage_plan.request.manifest_family,
        split,
    )
    entries = manifest.entries[:max_samples]
    text_encoder = build_text_encoder(config.text_encoder)
    max_positions = _learned_max_pose_tokens(context.effective_config)
    text_dim = require_positive_int_config_path(
        context.effective_config,
        "text_encoder.output_dim",
    )
    writer = ModelDataSurfaceWriter(
        root=surface_root,
        provider_key="learned_pose_token",
        surface_kind="text_to_token_sequences",
        split=split.value,
        manifest_family=manifest.manifest_family.family_id,
        source_manifest_path=manifest_path,
        source_manifest_sha256=manifest_sha,
        provider_config_sha256=context.provider_config_hash,
        cache_key=cache_key,
        manifest_entry_count=manifest_count,
        run_mode=context.run_mode,
        data_version="manifest_text_to_pose_tokens:provider_real_calibration:v1",
        max_units_per_shard=10_000,
        max_source_samples_per_shard=512,
    )
    for source_index, entry in enumerate(entries):
        loaded = load_manifest_sample(
            context.topology,
            manifest.manifest_family,
            manifest.manifest_path,
            entry,
        )
        vectorized = vectorize_bfh_pose_arrays(loaded.pose, sample_id=loaded.sample_id)
        flat = flatten_bfh_vectorized_pose(vectorized)
        raw = np.abs(np.nan_to_num(flat[:max_positions, 0], nan=0.0))
        token_ids = (raw * 1000).astype(np.int64) % require_positive_int_config_path(context.effective_config, "codebook.size")
        count = max(1, min(max_positions, token_ids.shape[0]))
        target = np.zeros((1, max_positions), dtype=np.float32)
        target[0, :count] = token_ids[:count].astype(np.float32, copy=False)
        encoded = text_encoder.encode_one(
            loaded.text,
            sentence_name=loaded.source_sentence_name,
        ).pooled_embedding.astype(np.float32, copy=False)
        if encoded.shape[0] != text_dim:
            raise ModelWorkflowInvariantError("text encoder embedding dimension does not match effective config.")
        writer.append_units(
            {
                "values": np.concatenate([encoded.reshape(1, -1), target], axis=1),
                "text_embedding": encoded.reshape(1, -1),
                "target_token_ids": target.astype(np.int64, copy=False),
                "target_mask": np.arange(max_positions).reshape(1, -1) < count,
                "source_index": np.asarray([source_index], dtype=np.int64),
                "sequence_length": np.asarray([count], dtype=np.int64),
            },
            sample_count=1,
            frame_count=count,
        )
    return writer.close()


def calibration_surface_schema(*, effective_config: Mapping[str, object]) -> Mapping[str, object]:
    token_input_dim = _learned_tokenizer_input_dim(effective_config)
    text_dim = require_positive_int_config_path(effective_config, "text_encoder.output_dim")
    max_pose_tokens = _learned_max_pose_tokens(effective_config)
    return {
        "provider_key": "learned_pose_token",
        "surfaces": {
            "pose_token_units": {
                "surface_kind": "pose_token_units",
                "fields": {"values": {"feature_dim": token_input_dim}},
            },
            "text_to_token_sequences": {
                "surface_kind": "text_to_token_sequences",
                "fields": {"values": {"feature_dim": text_dim + max_pose_tokens}},
            },
        },
    }


def calibration_model_build_spec(
    *,
    effective_config: Mapping[str, object],
    surface_schema: Mapping[str, object],
) -> Mapping[str, object]:
    pose_schema = _schema_for_kind(surface_schema, "pose_token_units")
    text_schema = _schema_for_kind(surface_schema, "text_to_token_sequences")
    input_dim = _schema_feature_dim(pose_schema)
    text_feature_dim = _schema_feature_dim(text_schema)
    text_embedding_dim = require_positive_int_config_path(
        effective_config,
        "text_encoder.output_dim",
    )
    max_pose_tokens = max(1, text_feature_dim - text_embedding_dim)
    codebook_size = require_positive_int_config_path(effective_config, "codebook.size")
    return {
        "provider_key": "learned_pose_token",
        "tokenizer": {
            "input_dim": input_dim,
            "tokenizer_hidden_dim": require_positive_int_config_path(effective_config, "tokenizer.hidden_dim"),
            "tokenizer_latent_dim": require_positive_int_config_path(effective_config, "tokenizer.latent_dim"),
            "codebook_size": codebook_size,
        },
        "text_to_token": {
            "text_embedding_dim": text_embedding_dim,
            "text_to_token_hidden_dim": require_positive_int_config_path(effective_config, "text_to_token.hidden_dim"),
            "position_dim": require_positive_int_config_path(effective_config, "text_to_token.position_dim"),
            "codebook_size": codebook_size,
            "max_text_length": require_positive_int_config_path(effective_config, "text_encoder.max_length"),
            "max_pose_tokens": max_pose_tokens,
            "min_generated_tokens": require_positive_int_config_path(effective_config, "text_to_token.min_generated_tokens"),
        },
    }


def _learned_tokenizer_input_dim(effective_config: Mapping[str, object]) -> int:
    frame_dim = default_bfh_tensor_layout().total_feature_dim

    temporal_granularity = optional_config_path(
        effective_config,
        "tokenizer.temporal_granularity",
    )

    if temporal_granularity in (None, "frame"):
        return frame_dim

    if temporal_granularity == "window":
        window_size = require_positive_int_config_path(
            effective_config,
            "tokenizer.window_size",
        )
        return frame_dim * window_size

    raise ModelWorkflowInvariantError(
        "learned_pose_token calibration does not support tokenizer.temporal_granularity="
        f"{temporal_granularity!r}."
    )


def _learned_max_pose_tokens(effective_config: Mapping[str, object]) -> int:
    for path in ("text_to_token.max_generated_tokens", "generation.max_generation_tokens"):
        value = optional_config_path(effective_config, path)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return require_positive_int_config_path(effective_config, "text_encoder.max_length")


def _validate_representative(
    context: ProviderRealCalibrationContext,
    candidate_key: str,
    representative: RepresentativeCalibrationSurfaceResult,
) -> None:
    if representative.provider_key != "learned_pose_token":
        raise ModelWorkflowInvariantError("learned_pose_token representative provider mismatch.")
    if representative.candidate_key != candidate_key:
        raise ModelWorkflowInvariantError("representative candidate key mismatch.")
    if representative.provider_config_sha256 != context.provider_config_hash:
        raise ModelWorkflowInvariantError("representative provider config hash mismatch.")
    if representative.surface_kind not in {"pose_token_units", "text_to_token_sequences"}:
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
        provider_key="learned_pose_token",
        surface_kind=representative.surface_kind,
        unit_count=representative.unit_count,
        surface_schema=surface_schema,
        surface_schema_hash=representative.surface_schema_hash,
        surface_provider_config_sha256=representative.provider_config_sha256,
        surface_feature_dim=representative.feature_dim,
        surface_root=representative.surface_root,
    )


def _text_to_token_values_batch(batch: Mapping[str, object]) -> dict[str, object]:
    text = batch.get("text_embedding")
    target = batch.get("target_token_ids")
    if not isinstance(text, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise ModelWorkflowInvariantError("text-to-token representative batch missing fields.")
    values = torch.cat([text.float(), target.float()], dim=1)
    converted = dict(batch)
    converted["values"] = values
    return converted


def build_learned_tokenizer_model_from_config(
    *,
    effective_config: Mapping[str, object],
    surface_schema: Mapping[str, object],
) -> LearnedPoseTokenizer:
    del surface_schema
    full_schema = calibration_surface_schema(effective_config=effective_config)
    spec = _mapping(
        calibration_model_build_spec(
            effective_config=effective_config,
            surface_schema=full_schema,
        ).get("tokenizer"),
        "tokenizer",
    )
    return LearnedPoseTokenizer(
        input_dim=int(spec["input_dim"]),
        hidden_dim=int(spec["tokenizer_hidden_dim"]),
        latent_dim=int(spec["tokenizer_latent_dim"]),
        codebook_size=int(spec["codebook_size"]),
    )


def build_text_to_token_model_from_config(
    *,
    effective_config: Mapping[str, object],
    surface_schema: Mapping[str, object],
) -> TextToTokenPredictor:
    full_schema = calibration_surface_schema(effective_config=effective_config)
    spec = _mapping(
        calibration_model_build_spec(
            effective_config=effective_config,
            surface_schema=full_schema,
        ).get("text_to_token"),
        "text_to_token",
    )
    return TextToTokenPredictor(
        text_embedding_dim=int(spec["text_embedding_dim"]),
        codebook_size=int(spec["codebook_size"]),
        max_positions=int(spec["max_pose_tokens"]),
        hidden_dim=int(spec["text_to_token_hidden_dim"]),
        position_dim=int(spec["position_dim"]),
        dropout=float(_value_at(effective_config, "text_to_token.dropout", 0.0)),
        min_generated_tokens=int(spec["min_generated_tokens"]),
        max_generated_tokens=int(spec["max_pose_tokens"]),
    )


def run_learned_tokenizer_training_step(
    *,
    model: LearnedPoseTokenizer,
    values: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    effective_config: Mapping[str, object],
) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)
    output = model(values)
    losses = compute_tokenizer_losses(
        output=output,
        target=values,
        validity_mask=torch.ones_like(values, dtype=torch.bool),
        commitment_weight=float(_value_at(effective_config, "tokenizer.commitment_weight", 0.25)),
    )
    loss_value = losses["total_loss"]
    loss_value.backward()
    optimizer.step()
    return loss_value


def run_text_to_token_training_step(
    *,
    model: TextToTokenPredictor,
    text_embeddings: torch.Tensor,
    target_token_ids: torch.Tensor,
    target_lengths: torch.Tensor,
    token_mask: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    effective_config: Mapping[str, object],
) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)
    output = model(text_embeddings)

    target_lengths = _align_text_to_token_target_lengths(
        target_lengths=target_lengths,
        predicted_lengths=output.length_values,
    )

    losses = compute_text_to_token_losses(
        output=output,
        target_token_ids=target_token_ids,
        target_lengths=target_lengths,
        token_mask=token_mask,
        token_loss_weight=float(_value_at(effective_config, "text_to_token.token_loss_weight", 1.0)),
        length_loss_weight=float(_value_at(effective_config, "text_to_token.length_loss_weight", 0.1)),
    )
    loss_value = losses["total_loss"]
    loss_value.backward()
    optimizer.step()
    return loss_value

def _align_text_to_token_target_lengths(
    *,
    target_lengths: torch.Tensor,
    predicted_lengths: torch.Tensor,
) -> torch.Tensor:
    """Align calibration target lengths to the predictor output shape.

    Calibration batches can be smaller than the candidate batch size, and
    some predictor implementations may expose length values as either
    ``(batch,)`` or ``(batch, 1)``. The loss contract requires exact shape
    equality with ``output.length_values``.
    """

    if target_lengths.shape == predicted_lengths.shape:
        return target_lengths.to(device=predicted_lengths.device, dtype=torch.long)

    lengths = target_lengths.to(device=predicted_lengths.device, dtype=torch.long)

    if lengths.numel() != predicted_lengths.numel():
        raise ModelWorkflowInvariantError(
            "learned_pose_token text-to-token calibration target_lengths cannot be "
            "aligned to predicted length shape: "
            f"target_lengths shape={tuple(target_lengths.shape)}, "
            f"predicted shape={tuple(predicted_lengths.shape)}."
        )

    return lengths.reshape(predicted_lengths.shape)


def _require_context(context: ProviderRealCalibrationContext, provider_key: str) -> None:
    if not isinstance(context, ProviderRealCalibrationContext):
        raise ModelWorkflowInvariantError("provider-real calibration requires ProviderRealCalibrationContext.")
    if context.provider_key != provider_key:
        raise ModelWorkflowInvariantError("provider-real calibration context provider mismatch.")


def _schema_for_kind(surface_schema: Mapping[str, object], surface_kind: str) -> Mapping[str, object]:
    surfaces = _mapping(surface_schema.get("surfaces"), "surfaces")
    schema = surfaces.get(surface_kind)
    return _mapping(schema, surface_kind)


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
    if "text_to_token" in candidate_name:
        return "learned_pose_token.text_to_token.forward_backward"
    if "decode" in candidate_name:
        return "learned_pose_token.decode.forward_only"
    return "learned_pose_token.tokenizer.forward_backward"


__all__ = [
    "LearnedPoseTokenProviderRealBenchmark",
    "build_representative_calibration_surface",
    "build_learned_tokenizer_model_from_config",
    "build_text_to_token_model_from_config",
    "calibration_model_build_spec",
    "calibration_surface_schema",
    "run_learned_tokenizer_training_step",
    "run_text_to_token_training_step",
]
