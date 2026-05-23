"""Provider-owned calibration hook for articulator_aware."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path

import torch

from text_to_sign_production.data.gate.pose.types import PoseChannel
from text_to_sign_production.modeling.backbones.bfh_vectorization import default_bfh_tensor_layout
from text_to_sign_production.modeling.candidates.calibration_config import (
    require_positive_int_config_path,
    require_text_config_path,
)
from text_to_sign_production.modeling.candidates.calibration_runtime import (
    calibration_reader_runtime_options,
)
from text_to_sign_production.modeling.candidates.calibration_surfaces import (
    RepresentativeCalibrationSurfaceResult,
)
from text_to_sign_production.modeling.candidates.provider import ProviderCalibrationPolicy
from text_to_sign_production.modeling.candidates.articulator_aware.config import (
    articulator_aware_config_from_mapping,
)
from text_to_sign_production.modeling.candidates.articulator_aware.dataset import (
    build_articulator_frame_surface_from_source_surface,
    build_articulator_source_surface,
)
from text_to_sign_production.modeling.candidates.articulator_aware.config import (
    ChannelLossWeightingConfig,
    ChannelPartitionPolicyConfig,
)
from text_to_sign_production.modeling.candidates.articulator_aware.losses import (
    build_channel_loss_weighting_policy,
)
from text_to_sign_production.modeling.candidates.articulator_aware.model import (
    ChannelFusionPoseModel,
)
from text_to_sign_production.modeling.candidates.articulator_aware.partitions import (
    build_articulator_partition_policy,
)
from text_to_sign_production.modeling.candidates.articulator_aware.trainer import (
    compute_articulator_length_loss,
    compute_channel_weighted_reconstruction_loss_torch,
)
from text_to_sign_production.modeling.data_surfaces import (
    ModelDataSurfaceReader,
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


class ArticulatorAwareProviderRealBenchmark:
    provider_key = "articulator_aware"

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
        batch = _values_batch_from_surface(batch)
        if not isinstance(batch.get("values"), torch.Tensor):
            raise ModelWorkflowInvariantError("articulator_aware calibration batch missing values.")
        surfaces = _bundle_from_representative(context, representative)
        surface_fp = batch_fingerprint(batch)
        values = batch["values"].to(device=device, dtype=torch.float32)  # type: ignore[index,union-attr]
        spec = calibration_model_build_spec(
            effective_config=context.effective_config,
            surface_schema=calibration_surface_schema(effective_config=context.effective_config),
        )
        model_spec = _mapping(spec.get("frame_model"), "frame_model")
        text_dim = int(model_spec["text_embedding_dim"])
        text_embedding = values[:, :text_dim]
        layout = default_bfh_tensor_layout()
        target = values[:, text_dim : text_dim + layout.total_feature_dim]
        primary = (PoseChannel.BODY, PoseChannel.LEFT_HAND, PoseChannel.RIGHT_HAND, PoseChannel.FACE)
        policy = build_articulator_partition_policy(
            config=ChannelPartitionPolicyConfig(
                source="canonical_bfh_channels",
                primary_channels=primary,
            ),
            layout=layout,
        )
        weighting = build_channel_loss_weighting_policy(
            ChannelLossWeightingConfig(
                policy="static_channel_weights",
                normalize_weights=True,
                channel_weights={channel: 1.0 for channel in primary},
                velocity_weight=0.0,
                symmetry_weight=0.0,
                cross_channel_consistency_weight=0.0,
            )
        )
        model = build_articulator_model_from_config(
            effective_config=context.effective_config,
            surface_schema=surfaces.surface_schema,
            partition_policy=policy,
            min_positions=int(candidate_value),
        ).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        positions = torch.arange(int(candidate_value), dtype=torch.long, device=device)
        loss_value: torch.Tensor | None = None
        gradient_observed = False

        def step() -> int:
            nonlocal loss_value, gradient_observed
            loss_value = run_articulator_frame_training_step(
                model=model,
                text_embedding=text_embedding,
                positions=positions,
                target=target,
                target_length=float(candidate_value),
                weighting=weighting,
                channels=primary,
                optimizer=optimizer,
            )
            gradient_observed = gradient_observed or _finite_gradient_observed(model)
            return int(text_embedding.shape[0])

        for _ in range(max(0, int(warmup_batches))):
            step()
            if progress_callbacks is not None:
                progress_callbacks.advance_warmup_batch(1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        processed = 0
        for _ in range(max(1, min(int(max_batches_per_candidate), 50))):
            processed += step()
            if progress_callbacks is not None:
                progress_callbacks.advance_measured_batch(1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        total_memory = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
        memory_fraction = torch.cuda.max_memory_reserved() / total_memory if total_memory else None
        metadata = json.loads((surfaces.surface_root / "metadata.json").read_text(encoding="utf-8"))
        return ProviderRealCalibrationTrace(
            provider_key=self.provider_key,
            provider_hook=self.__class__.__name__,
            benchmark_name="articulator_aware.frame_model.forward_backward",
            benchmark_type="provider_real",
            benchmark_kind="training",
            forward_only_reason=None,
            surface_kind=surfaces.surface_kind,
            surface_metadata_path=surfaces.surface_root / "metadata.json",
            surface_metadata_sha256=sha256_file(surfaces.surface_root / "metadata.json"),
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
            surface_batch_fingerprint=surface_fp,
            batch_adapter="articulator_aware.calibration.frame_values",
            model_input_fingerprint=tensor_fingerprint(text_embedding),
            target_fingerprint=tensor_fingerprint(target),
            model_class=model.__class__.__name__,
            model_module=model.__class__.__module__,
            model_config_hash=_model_config_hash(spec),
            model_expected_input_dim=None,
            model_factory=f"{build_articulator_model_from_config.__module__}.{build_articulator_model_from_config.__name__}",
            model_build_spec=spec,
            loss_name="compute_channel_weighted_reconstruction_loss_torch+compute_articulator_length_loss",
            loss_module=compute_channel_weighted_reconstruction_loss_torch.__module__,
            training_step_helper=f"{run_articulator_frame_training_step.__module__}.{run_articulator_frame_training_step.__name__}",
            forward_executed=True,
            backward_executed=True,
            gradient_observed=gradient_observed,
            finite_loss_observed=loss_value is not None and bool(torch.isfinite(loss_value.detach()).item()),
            surface_derived_primary_input=True,
            random_used=False,
            random_role=None,
            units_processed=processed,
            elapsed_seconds=elapsed,
            peak_cuda_memory_reserved_fraction=memory_fraction,
            calibration_surface_reader_num_workers_used=(
                int(reader_runtime_trace["surface_reader_num_workers_used"])
                if "surface_reader_num_workers_used" in reader_runtime_trace
                else None
            ),
            calibration_surface_reader_worker_mode=(
                str(reader_runtime_trace["surface_reader_worker_mode"])
                if "surface_reader_worker_mode" in reader_runtime_trace
                else None
            ),
            calibration_surface_reader_prefetch_factor_used=(
                int(reader_runtime_trace["surface_reader_prefetch_factor_used"])  # type: ignore[arg-type]
                if reader_runtime_trace.get("surface_reader_prefetch_factor_used") is not None
                else None
            ),
            calibration_surface_reader_persistent_workers_used=(
                bool(reader_runtime_trace["surface_reader_persistent_workers_used"])
                if "surface_reader_persistent_workers_used" in reader_runtime_trace
                else None
            ),
        )


def calibration_surface_schema(*, effective_config: Mapping[str, object]) -> Mapping[str, object]:
    layout = default_bfh_tensor_layout()
    text_dim = require_positive_int_config_path(
        effective_config,
        "provider_config.text_encoder.output_dim",
    )
    return {
        "provider_key": "articulator_aware",
        "surfaces": {
            "articulator_source_sequences": {
                "surface_kind": "articulator_source_sequences",
                "fields": {"values": {"feature_dim": text_dim + layout.total_feature_dim}},
            },
            "articulator_frame_units": {
                "surface_kind": "articulator_frame_units",
                "fields": {"values": {"feature_dim": text_dim + layout.total_feature_dim}},
                "body_dim": _channel_feature_dim(layout, PoseChannel.BODY),
                "left_hand_dim": _channel_feature_dim(layout, PoseChannel.LEFT_HAND),
                "right_hand_dim": _channel_feature_dim(layout, PoseChannel.RIGHT_HAND),
                "face_dim": _channel_feature_dim(layout, PoseChannel.FACE),
            }
        },
    }


def build_representative_calibration_surface(
    *,
    context: ProviderRealCalibrationContext,
    candidate_key: str,
    policy: ProviderCalibrationPolicy,
    progress_session,
) -> RepresentativeCalibrationSurfaceResult:
    del progress_session
    _require_context(context, "articulator_aware")
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
    if context.topology is None or context.stage_plan is None:
        raise ModelWorkflowInvariantError("articulator representative surface requires topology and stage plan.")
    provider_config = context.effective_config.get("provider_config")
    if not isinstance(provider_config, Mapping):
        raise ModelWorkflowInvariantError("articulator effective config missing provider_config.")
    config = articulator_aware_config_from_mapping(provider_config)
    root = (
        context.working_dir
        / "representative_calibration_surfaces"
        / "articulator_aware"
        / candidate_key
    )
    source_cache_key = model_data_surface_cache_key(
        provider_key="articulator_aware",
        surface_kind="articulator_source_sequences",
        split=split.value,
        source_manifest_sha256=manifest_sha,
        provider_config_sha256=context.provider_config_hash,
        run_mode=context.run_mode,
        limit_samples=policy.max_samples,
        data_version="provider_real_calibration:source:v1",
    )
    source_surface = build_articulator_source_surface(
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
    )
    if surface_kind == "articulator_source_sequences":
        surface = source_surface
    elif surface_kind == "articulator_frame_units":
        cache_key = model_data_surface_cache_key(
            provider_key="articulator_aware",
            surface_kind=surface_kind,
            split=split.value,
            source_manifest_sha256=manifest_sha,
            provider_config_sha256=context.provider_config_hash,
            run_mode=context.run_mode,
            limit_samples=policy.max_samples,
            data_version="provider_real_calibration:v1",
        )
        surface = build_articulator_frame_surface_from_source_surface(
            source_surface=source_surface,
            partition_policy=build_articulator_partition_policy(
                config=config.partition_policy,
                layout=default_bfh_tensor_layout(),
            ),
            mask_config=config.mask_strategy,
            surface_root=root / "frame_units",
            source_manifest_sha256=manifest_sha,
            provider_config_sha256=context.provider_config_hash,
            cache_key=cache_key,
            run_mode=context.run_mode,
            manifest_entry_count=manifest_count,
            max_units_per_shard=10_000,
            max_source_samples_per_shard=512,
        )
    else:
        raise ModelWorkflowInvariantError(f"unsupported articulator surface kind: {surface_kind!r}.")
    return representative_result_from_surface(
        context=context,
        candidate_key=candidate_key,
        surface_kind=surface_kind,
        split=split,
        surface=surface,
        surface_schema=surface_schema,
        feature_dim=feature_dim,
    )


def _validate_representative(
    context: ProviderRealCalibrationContext,
    candidate_key: str,
    representative: RepresentativeCalibrationSurfaceResult,
) -> None:
    if representative.provider_key != "articulator_aware":
        raise ModelWorkflowInvariantError("articulator representative provider mismatch.")
    if representative.candidate_key != candidate_key:
        raise ModelWorkflowInvariantError("representative candidate key mismatch.")
    if representative.provider_config_sha256 != context.provider_config_hash:
        raise ModelWorkflowInvariantError("representative provider config hash mismatch.")
    if representative.surface_kind not in {"articulator_frame_units", "articulator_source_sequences"}:
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
        provider_key="articulator_aware",
        surface_kind=representative.surface_kind,
        unit_count=representative.unit_count,
        surface_schema=surface_schema,
        surface_schema_hash=representative.surface_schema_hash,
        surface_provider_config_sha256=representative.provider_config_sha256,
        surface_feature_dim=representative.feature_dim,
        surface_root=representative.surface_root,
    )


def _values_batch_from_surface(batch: Mapping[str, object]) -> dict[str, object]:
    if isinstance(batch.get("values"), torch.Tensor):
        return dict(batch)
    target = batch.get("target_values")
    if not isinstance(target, torch.Tensor):
        raise ModelWorkflowInvariantError("articulator representative batch missing target values.")
    text = torch.zeros((target.shape[0], 256), dtype=target.dtype, device=target.device)
    converted = dict(batch)
    converted["values"] = torch.cat([text, target.float()], dim=1)
    return converted


def calibration_model_build_spec(
    *,
    effective_config: Mapping[str, object],
    surface_schema: Mapping[str, object],
) -> Mapping[str, object]:
    frame_schema = _schema_for_kind(surface_schema, "articulator_frame_units")
    layout = default_bfh_tensor_layout()
    text_dim = require_positive_int_config_path(
        effective_config,
        "provider_config.text_encoder.output_dim",
    )
    return {
        "provider_key": "articulator_aware",
        "frame_model": {
            "feature_dim": _schema_feature_dim(frame_schema),
            "text_embedding_dim": text_dim,
            "hidden_dim": require_positive_int_config_path(effective_config, "provider_config.structure_variant.hidden_dim"),
            "architecture": require_text_config_path(effective_config, "provider_config.structure_variant.architecture"),
            "body_dim": _channel_feature_dim(layout, PoseChannel.BODY),
            "left_hand_dim": _channel_feature_dim(layout, PoseChannel.LEFT_HAND),
            "right_hand_dim": _channel_feature_dim(layout, PoseChannel.RIGHT_HAND),
            "face_dim": _channel_feature_dim(layout, PoseChannel.FACE),
        },
    }


def build_articulator_model_from_config(
    *,
    effective_config: Mapping[str, object],
    surface_schema: Mapping[str, object],
    partition_policy,
    min_positions: int = 1,
) -> ChannelFusionPoseModel:
    del surface_schema
    spec = _mapping(
        calibration_model_build_spec(
            effective_config=effective_config,
            surface_schema=calibration_surface_schema(effective_config=effective_config),
        ).get("frame_model"),
        "frame_model",
    )
    return ChannelFusionPoseModel(
        partition_policy=partition_policy,
        text_embedding_dim=int(spec["text_embedding_dim"]),
        hidden_dim=int(spec["hidden_dim"]),
        dropout=float(_value_at(effective_config, "dropout", 0.0)),
        max_positions=max(_int_from(effective_config, ("max_positions", "model.max_positions"), 5), int(min_positions)),
    )


def run_articulator_frame_training_step(
    *,
    model: ChannelFusionPoseModel,
    text_embedding: torch.Tensor,
    positions: torch.Tensor,
    target: torch.Tensor,
    target_length: float,
    weighting,
    channels: tuple[PoseChannel, ...],
    optimizer: torch.optim.Optimizer,
) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)
    output = model(text_embedding, positions)
    target_aligned = target[:, : output.full_pose_values.shape[1]]
    channel_masks = {
        channel: torch.ones_like(target_aligned, dtype=torch.bool)
        for channel in channels
    }
    reconstruction, _ = compute_channel_weighted_reconstruction_loss_torch(
        predicted=output.full_pose_values,
        target=target_aligned,
        channel_masks=channel_masks,
        weighting=weighting,
    )
    length = compute_articulator_length_loss(
        predicted_lengths=output.predicted_lengths,
        target_lengths=torch.full(
            (int(text_embedding.shape[0]),),
            float(target_length),
            dtype=torch.float32,
            device=text_embedding.device,
        ),
    )
    loss_value = reconstruction + length
    loss_value.backward()
    optimizer.step()
    return loss_value


def _require_context(context: ProviderRealCalibrationContext, provider_key: str) -> None:
    if not isinstance(context, ProviderRealCalibrationContext):
        raise ModelWorkflowInvariantError("provider-real calibration requires ProviderRealCalibrationContext.")
    if context.provider_key != provider_key:
        raise ModelWorkflowInvariantError("provider-real calibration context provider mismatch.")


def _channel_feature_dim(layout, channel: PoseChannel) -> int:
    item = layout.channel_slices[channel]
    return int((item.stop - item.start) * layout.coordinate_dimensions)


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


__all__ = [
    "ArticulatorAwareProviderRealBenchmark",
    "build_articulator_model_from_config",
    "build_representative_calibration_surface",
    "calibration_model_build_spec",
    "calibration_surface_schema",
    "run_articulator_frame_training_step",
]
