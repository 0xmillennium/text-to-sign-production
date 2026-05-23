"""Provider-owned calibration hook for base_direct."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
import json
import time

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from text_to_sign_production.modeling.candidates.calibration_config import (
    require_positive_int_config_path,
    require_text_config_path,
)
from text_to_sign_production.modeling.candidates.calibration_surfaces import (
    RepresentativeCalibrationSurfaceResult,
)
from text_to_sign_production.modeling.candidates.calibration_runtime import calibration_reader_runtime_options
from text_to_sign_production.modeling.candidates.provider import ProviderCalibrationPolicy
from text_to_sign_production.modeling.candidates.base_direct.config import (
    base_direct_config_from_effective_dict,
)
from text_to_sign_production.modeling.candidates.base_direct.losses import (
    compute_base_direct_loss,
)
from text_to_sign_production.modeling.candidates.base_direct.model import (
    build_base_direct_model,
)
from text_to_sign_production.modeling.data.schemas import ProcessedPoseBatch
from text_to_sign_production.modeling.data.legacy import M0_TARGET_CHANNEL_SHAPES
from text_to_sign_production.modeling.data import load_manifest_sample, read_modeling_manifest
from text_to_sign_production.modeling.data_surfaces import sha256_file
from text_to_sign_production.workflows.model.contracts.config import ModelWorkflowInvariantError
from text_to_sign_production.workflows.model.processing.calibration import (
    CalibrationProgressCallbacks,
    ProviderRealCalibrationContext,
    ProviderRealCalibrationTrace,
    batch_fingerprint,
    representative_manifest_for_policy,
    stable_config_hash,
    tensor_fingerprint,
    _model_config_hash,
)


class BaseDirectProviderRealBenchmark:
    provider_key = "base_direct"

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
        config = base_direct_config_from_effective_dict(
            context.effective_config,
            source_path=_effective_source_path(context.effective_config),
        )
        config = replace(
            config,
            raw_config={
                **dict(config.raw_config),
                "calibration": {
                    "provider_real": True,
                    "text_encoder_local_files_only": config.text_encoder.local_files_only,
                },
            },
        )
        model = build_base_direct_model(config)
        text_backbone = getattr(model, "backbone", None)
        text_backbone_class = text_backbone.__class__.__name__ if text_backbone is not None else None
        text_backbone_module = text_backbone.__class__.__module__ if text_backbone is not None else None
        runtime_options = calibration_reader_runtime_options(context.effective_config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
        loader = _representative_dataloader(
            context=context,
            representative=representative,
            batch_size=int(candidate_value),
            runtime_options=runtime_options,
        )
        batch = next(iter(loader))
        batch = _move_batch(batch, device)
        surface_fp = batch_fingerprint(
            {
                "padding_mask": batch.padding_mask,
                "frame_valid_mask": batch.frame_valid_mask,
                "body": batch.body,
                "left_hand": batch.left_hand,
                "right_hand": batch.right_hand,
                "face": batch.face,
            }
        )
        loss_value: torch.Tensor | None = None
        gradient_observed = False

        def step() -> int:
            nonlocal loss_value, gradient_observed
            model.train()
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch)
            loss_result = compute_base_direct_loss(
                predictions=predictions,
                targets=batch,
                padding_mask=batch.padding_mask,
                frame_valid_mask=batch.frame_valid_mask,
                channel_weights=config.loss.channel_weights,
            )
            loss_value = loss_result.total_loss
            torch.autograd.backward(loss_result.total_loss)
            optimizer.step()
            gradient_observed = gradient_observed or _finite_gradient_observed(model)
            return len(batch.texts)

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
        spec = calibration_model_build_spec(
            effective_config=context.effective_config,
            surface_schema=calibration_surface_schema(effective_config=context.effective_config),
        )
        metadata_path = representative.source_manifest_path
        return ProviderRealCalibrationTrace(
            provider_key=self.provider_key,
            provider_hook=self.__class__.__name__,
            benchmark_name="base_direct.lazy_dataloader.forward_backward",
            benchmark_type="provider_real",
            benchmark_kind="training",
            forward_only_reason=None,
            surface_kind=representative.surface_kind,
            surface_metadata_path=metadata_path,
            surface_metadata_sha256=sha256_file(metadata_path),
            surface_batch_fingerprint=surface_fp,
            batch_adapter="base_direct.calibration.lazy_dataloader_batch",
            model_input_fingerprint=tensor_fingerprint(batch.padding_mask.to(torch.float32)),
            target_fingerprint=batch_fingerprint(
                {
                    "body": batch.body,
                    "left_hand": batch.left_hand,
                    "right_hand": batch.right_hand,
                    "face": batch.face,
                }
            ),
            model_class=model.__class__.__name__,
            model_module=model.__class__.__module__,
            model_config_hash=_model_config_hash(spec),
            effective_config_hash=context.provider_config_hash,
            model_build_spec_hash=context.model_build_spec_hash,
            surface_schema_hash=representative.surface_schema_hash,
            surface_provider_config_sha256=representative.provider_config_sha256,
            surface_feature_dim=representative.feature_dim,
            representative_source_manifest_path=representative.source_manifest_path,
            representative_source_manifest_sha256=representative.source_manifest_sha256,
            representative_surface_kind=representative.surface_kind,
            representative_provider_config_sha256=representative.provider_config_sha256,
            representative_surface_schema_hash=representative.surface_schema_hash,
            representative_sample_count=representative.sample_count,
            representative_unit_count=representative.unit_count,
            model_expected_input_dim=representative.feature_dim,
            model_factory=f"{build_base_direct_model.__module__}.{build_base_direct_model.__name__}",
            model_build_spec=spec,
            calibration_surface_reader_num_workers_used=runtime_options.num_workers,
            calibration_surface_reader_worker_mode="multiprocess" if runtime_options.num_workers > 0 else "single_process",
            calibration_surface_reader_prefetch_factor_used=runtime_options.prefetch_factor,
            calibration_surface_reader_persistent_workers_used=runtime_options.persistent_workers,
            loss_name="compute_base_direct_loss",
            loss_module=compute_base_direct_loss.__module__,
            training_step_helper=f"{self.__class__.__module__}._base_direct_provider_real_step",
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
        )


def calibration_surface_schema(*, effective_config: Mapping[str, object]) -> Mapping[str, object]:
    input_dim = require_positive_int_config_path(effective_config, "text_encoder.max_length")
    target_dim = sum(shape[0] * shape[1] for shape in M0_TARGET_CHANNEL_SHAPES.values())
    return {
        "provider_key": "base_direct",
        "surfaces": {
            "base_direct_lazy_batch": {
                "surface_kind": "base_direct_lazy_batch",
                "fields": {"values": {"feature_dim": input_dim + target_dim}},
                "target_dim": target_dim,
            }
        },
    }


def calibration_model_build_spec(
    *,
    effective_config: Mapping[str, object],
    surface_schema: Mapping[str, object],
) -> Mapping[str, object]:
    schema = _schema_for_kind(surface_schema, "base_direct_lazy_batch")
    target_dim = int(schema.get("target_dim", sum(shape[0] * shape[1] for shape in M0_TARGET_CHANNEL_SHAPES.values())))
    feature_dim = _schema_feature_dim(schema)
    return {
        "provider_key": "base_direct",
        "direct_model": {
            "input_dim": max(1, feature_dim - target_dim),
            "hidden_dim": require_positive_int_config_path(effective_config, "model.decoder_hidden_dim"),
            "output_dim": target_dim,
            "sequence_length_policy": require_text_config_path(effective_config, "identity.length_policy"),
            "channels": tuple(M0_TARGET_CHANNEL_SHAPES),
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
    _require_context(context, "base_direct")
    if candidate_key not in policy.candidate_keys:
        raise ModelWorkflowInvariantError(f"unknown calibration candidate: {candidate_key!r}.")
    surface_kind = policy.representative_surface_kinds.get(candidate_key)
    if surface_kind != "base_direct_lazy_batch":
        raise ModelWorkflowInvariantError(f"unsupported base_direct surface kind: {surface_kind!r}.")
    _split, manifest_path, manifest_count = representative_manifest_for_policy(context, policy)
    schema = _schema_for_kind(calibration_surface_schema(effective_config=context.effective_config), surface_kind)
    selected_count = min(policy.max_samples, manifest_count)
    return RepresentativeCalibrationSurfaceResult(
        provider_key="base_direct",
        candidate_key=candidate_key,
        surface_kind=surface_kind,
        split=policy.representative_split,
        source_manifest_path=manifest_path,
        source_manifest_sha256=sha256_file(manifest_path),
        provider_config_sha256=context.provider_config_hash,
        surface_schema_hash=stable_config_hash(schema),
        feature_dim=_schema_feature_dim(schema),
        sample_count=selected_count,
        unit_count=selected_count,
        surface_root=None,
        surface_metadata_path=None,
        dataloader_kind="base_direct_manifest_lazy_dataloader",
    )


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


def _representative_dataloader(
    *,
    context: ProviderRealCalibrationContext,
    representative: RepresentativeCalibrationSurfaceResult,
    batch_size: int,
    runtime_options,
) -> DataLoader:
    if context.topology is None or context.stage_plan is None:
        raise ModelWorkflowInvariantError("base_direct representative dataloader requires topology and stage plan.")
    manifest = read_modeling_manifest(
        context.topology,
        context.stage_plan.request.manifest_family,
        context.stage_plan.request.train_split,
    )
    entries = manifest.entries[: representative.sample_count]
    dataset = _BaseDirectRepresentativeDataset(
        context=context,
        manifest_path=manifest.manifest_path,
        manifest_family=manifest.manifest_family,
        entries=entries,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=runtime_options.num_workers,
        prefetch_factor=runtime_options.prefetch_factor,
        persistent_workers=runtime_options.persistent_workers,
        collate_fn=_collate_samples,
    )


class _BaseDirectRepresentativeDataset(Dataset):
    def __init__(self, *, context, manifest_path, manifest_family, entries) -> None:
        self._context = context
        self._manifest_path = manifest_path
        self._manifest_family = manifest_family
        self._entries = tuple(entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> dict[str, object]:
        loaded = load_manifest_sample(
            self._context.topology,
            self._manifest_family,
            self._manifest_path,
            self._entries[index],
        )
        pose = loaded.pose
        frame_count = pose.frame_count
        return {
            "text": loaded.text,
            "sample_id": loaded.sample_id,
            "split": loaded.split.value,
            "length": frame_count,
            "body": torch.as_tensor(pose.coordinates("body"), dtype=torch.float32),
            "body_confidence": torch.as_tensor(pose.confidence("body"), dtype=torch.float32),
            "left_hand": torch.as_tensor(pose.coordinates("left_hand"), dtype=torch.float32),
            "left_hand_confidence": torch.as_tensor(pose.confidence("left_hand"), dtype=torch.float32),
            "right_hand": torch.as_tensor(pose.coordinates("right_hand"), dtype=torch.float32),
            "right_hand_confidence": torch.as_tensor(pose.confidence("right_hand"), dtype=torch.float32),
            "face": torch.as_tensor(pose.coordinates("face"), dtype=torch.float32),
            "face_confidence": torch.as_tensor(pose.confidence("face"), dtype=torch.float32),
            "padding_mask": torch.zeros((frame_count,), dtype=torch.bool),
            "frame_valid_mask": torch.as_tensor(np.asarray(pose.valid_frame_mask), dtype=torch.bool),
        }


def _collate_samples(samples: list[Mapping[str, object]]) -> ProcessedPoseBatch:
    max_frames = max(int(sample["length"]) for sample in samples)

    def pad_tensor(value: torch.Tensor, *, fill: float | bool = 0) -> torch.Tensor:
        if value.shape[0] == max_frames:
            return value
        pad_shape = (max_frames - value.shape[0], *value.shape[1:])
        pad = torch.full(pad_shape, fill, dtype=value.dtype)
        return torch.cat([value, pad], dim=0)

    return ProcessedPoseBatch(
        texts=[str(sample["text"]) for sample in samples],
        sample_ids=[str(sample["sample_id"]) for sample in samples],
        splits=[str(sample["split"]) for sample in samples],
        lengths=torch.tensor([int(sample["length"]) for sample in samples], dtype=torch.long),
        body=torch.stack([pad_tensor(sample["body"]) for sample in samples]),  # type: ignore[arg-type]
        body_confidence=torch.stack([pad_tensor(sample["body_confidence"]) for sample in samples]),  # type: ignore[arg-type]
        left_hand=torch.stack([pad_tensor(sample["left_hand"]) for sample in samples]),  # type: ignore[arg-type]
        left_hand_confidence=torch.stack([pad_tensor(sample["left_hand_confidence"]) for sample in samples]),  # type: ignore[arg-type]
        right_hand=torch.stack([pad_tensor(sample["right_hand"]) for sample in samples]),  # type: ignore[arg-type]
        right_hand_confidence=torch.stack([pad_tensor(sample["right_hand_confidence"]) for sample in samples]),  # type: ignore[arg-type]
        face=torch.stack([pad_tensor(sample["face"]) for sample in samples]),  # type: ignore[arg-type]
        face_confidence=torch.stack([pad_tensor(sample["face_confidence"]) for sample in samples]),  # type: ignore[arg-type]
        padding_mask=torch.stack([pad_tensor(sample["padding_mask"], fill=True) for sample in samples]),  # type: ignore[arg-type]
        frame_valid_mask=torch.stack([pad_tensor(sample["frame_valid_mask"]) for sample in samples]),  # type: ignore[arg-type]
        people_per_frame=torch.ones((len(samples), max_frames), dtype=torch.long),
        selected_person_indices=[0 for _ in samples],
        processed_schema_versions=["prepared_sample_payload" for _ in samples],
        fps=[None for _ in samples],
        num_frames=[int(sample["length"]) for sample in samples],
    )


def _move_batch(batch: ProcessedPoseBatch, device: torch.device) -> ProcessedPoseBatch:
    tensor_fields = {
        field: getattr(batch, field).to(device)
        for field in (
            "lengths",
            "body",
            "body_confidence",
            "left_hand",
            "left_hand_confidence",
            "right_hand",
            "right_hand_confidence",
            "face",
            "face_confidence",
            "padding_mask",
            "frame_valid_mask",
            "people_per_frame",
        )
    }
    return ProcessedPoseBatch(
        texts=batch.texts,
        sample_ids=batch.sample_ids,
        splits=batch.splits,
        selected_person_indices=batch.selected_person_indices,
        processed_schema_versions=batch.processed_schema_versions,
        fps=batch.fps,
        num_frames=batch.num_frames,
        **tensor_fields,
    )


def _finite_gradient_observed(model: torch.nn.Module) -> bool:
    return any(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all().item())
        for parameter in model.parameters()
    )


def _validate_representative(
    context: ProviderRealCalibrationContext,
    candidate_key: str,
    representative: RepresentativeCalibrationSurfaceResult,
) -> None:
    if representative.provider_key != "base_direct":
        raise ModelWorkflowInvariantError("base_direct representative provider mismatch.")
    if representative.candidate_key != candidate_key:
        raise ModelWorkflowInvariantError("representative candidate key mismatch.")
    if representative.surface_kind != "base_direct_lazy_batch":
        raise ModelWorkflowInvariantError("representative surface kind is unsupported.")
    if representative.provider_config_sha256 != context.provider_config_hash:
        raise ModelWorkflowInvariantError("representative provider config hash mismatch.")
    if representative.dataloader_kind != "base_direct_manifest_lazy_dataloader":
        raise ModelWorkflowInvariantError("base_direct representative dataloader kind mismatch.")
    if not representative.source_manifest_path.is_file():
        raise ModelWorkflowInvariantError("representative source manifest is missing.")
    if sha256_file(representative.source_manifest_path) != representative.source_manifest_sha256:
        raise ModelWorkflowInvariantError("representative source manifest hash mismatch.")


def _effective_source_path(effective_config: Mapping[str, object]) -> Path | None:
    value = effective_config.get("source_path")
    if value is None:
        return None
    if not isinstance(value, str):
        raise ModelWorkflowInvariantError("base_direct effective source_path must be string or null.")
    return Path(value)


def _require_context(context: ProviderRealCalibrationContext, provider_key: str) -> None:
    if context.provider_key != provider_key:
        raise ModelWorkflowInvariantError("base_direct calibration context provider mismatch.")


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ModelWorkflowInvariantError(f"{name} must be a mapping.")
    return value


__all__ = [
    "BaseDirectProviderRealBenchmark",
    "build_representative_calibration_surface",
    "calibration_model_build_spec",
    "calibration_surface_schema",
]
