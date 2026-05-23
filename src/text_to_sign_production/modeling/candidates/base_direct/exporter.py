"""Stage 3 generated-pose export for the M0 direct baseline provider."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.artifacts import (
    GENERATED_POSE_CHANNEL_POLICY,
    GENERATED_POSE_PAYLOAD_SCHEMA_VERSION,
    GeneratedPoseConfidencePolicy,
    GeneratedPoseGenerationMode,
    GeneratedPoseLengthPolicy,
    GeneratedPoseProducerType,
    GeneratedPoseSample,
    write_generated_pose_split,
    write_generated_pose_split_to_explicit_root,
)
from text_to_sign_production.modeling.candidates import (
    ModelSingleSampleInferenceContext,
    ModelSingleSampleInferenceResult,
    ModelStageExecutionContext,
    ModelStageExecutionError,
)
from text_to_sign_production.modeling.candidates.base_direct.config import BaseDirectRunConfig
from text_to_sign_production.modeling.candidates.base_direct.dataset import (
    resolve_base_direct_runtime_data_paths,
    validate_base_direct_runtime_data_paths,
)
from text_to_sign_production.modeling.candidates.base_direct.predictor import (
    load_base_direct_predictor,
    select_base_direct_checkpoint,
)
from text_to_sign_production.modeling.candidates.base_direct.trainer import (
    base_direct_compatibility_config_path,
)
from text_to_sign_production.modeling.data.bfh_schema import BfhPoseArrays
from text_to_sign_production.modeling.data.legacy import (
    ProcessedPoseItem,
    collate_processed_pose_samples,
    load_processed_pose_sample,
    processed_modeling_record_from_entry,
)
from text_to_sign_production.modeling.data import read_modeling_manifest


@dataclass(frozen=True, slots=True)
class BaseDirectGeneratedSplitResult:
    split: SampleSplit
    manifest_path: Path
    sample_paths: tuple[Path, ...]
    sample_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "split", SampleSplit(self.split))
        object.__setattr__(self, "manifest_path", Path(self.manifest_path))
        sample_paths = tuple(Path(path) for path in self.sample_paths)
        if not isinstance(self.sample_count, int) or isinstance(self.sample_count, bool):
            raise ModelStageExecutionError("sample_count must be an integer.")
        if self.sample_count < 0:
            raise ModelStageExecutionError("sample_count must be non-negative.")
        if self.sample_count != len(sample_paths):
            raise ModelStageExecutionError(
                "sample_count must match the number of generated sample paths."
            )
        object.__setattr__(self, "sample_paths", sample_paths)


@dataclass(frozen=True, slots=True)
class BaseDirectGeneratedPoseExportResult:
    checkpoint_path: Path
    splits: tuple[BaseDirectGeneratedSplitResult, ...]
    total_sample_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))
        splits = tuple(self.splits)
        if any(not isinstance(split, BaseDirectGeneratedSplitResult) for split in splits):
            raise ModelStageExecutionError(
                "splits must contain BaseDirectGeneratedSplitResult values."
            )
        if not isinstance(self.total_sample_count, int) or isinstance(
            self.total_sample_count,
            bool,
        ):
            raise ModelStageExecutionError("total_sample_count must be an integer.")
        expected_total = sum(split.sample_count for split in splits)
        if self.total_sample_count != expected_total:
            raise ModelStageExecutionError(
                "total_sample_count must equal the sum of split sample counts."
            )
        object.__setattr__(self, "splits", splits)


def export_base_direct_generated_pose(
    *,
    context: ModelStageExecutionContext,
    config: BaseDirectRunConfig,
) -> BaseDirectGeneratedPoseExportResult:
    """Predict requested splits and write only Stage 3 generated-pose artifacts."""

    paths = resolve_base_direct_runtime_data_paths(context)
    validate_base_direct_runtime_data_paths(paths)
    checkpoint_path = select_base_direct_checkpoint(context)
    predictor = load_base_direct_predictor(
        config=config,
        compatibility_config_path=base_direct_compatibility_config_path(context),
        checkpoint_path=checkpoint_path,
    )
    try:
        from text_to_sign_production.modeling.inference.predict import predict_baseline_batch
    except ModuleNotFoundError as exc:
        raise ModelStageExecutionError(
            "base_direct export requires the modeling torch/transformers dependencies."
        ) from exc

    split_results: list[BaseDirectGeneratedSplitResult] = []
    for split in config.data.prediction_splits:
        manifest = read_modeling_manifest(
            context.topology,
            context.request.manifest_family,
            split,
        )
        entries = manifest.entries
        if config.data.limit_prediction_samples is not None:
            entries = entries[: config.data.limit_prediction_samples]
        if not entries:
            raise ModelStageExecutionError(
                f"base_direct prediction manifest contains no selected samples: {split.value}"
            )
        generated: list[GeneratedPoseSample] = []
        for entry in entries:
            record = processed_modeling_record_from_entry(
                entry,
                manifest_path=manifest.manifest_path,
                expected_split=split,
                data_root=paths.samples_root,
            )
            reference = load_processed_pose_sample(record)
            item = ProcessedPoseItem.from_manifest_and_sample(record, reference)
            prediction = predict_baseline_batch(
                predictor.model,
                collate_processed_pose_samples([item]),
                device=predictor.device,
            )
            generated.append(
                GeneratedPoseSample(
                    schema_version=GENERATED_POSE_PAYLOAD_SCHEMA_VERSION,
                    producer_type=GeneratedPoseProducerType.MODEL,
                    producer_key="base_direct",
                    canonical_id="m0_direct_text_to_pose",
                    phase_number=4,
                    research_role="baseline_or_ablation_floor",
                    run_name=context.request.run_name,
                    split=split,
                    sample_id=entry.sample_id,
                    text=entry.text,
                    source_video_id=entry.source_video_id,
                    source_sentence_id=entry.source_sentence_id,
                    source_sentence_name=entry.source_sentence_name,
                    reference_payload_ref=entry.payload_ref,
                    generation_index=0,
                    num_candidates_for_sample=1,
                    generation_mode=GeneratedPoseGenerationMode.DETERMINISTIC,
                    length_policy=GeneratedPoseLengthPolicy.REFERENCE_LENGTH,
                    channel_policy=GENERATED_POSE_CHANNEL_POLICY,
                    confidence_policy=GeneratedPoseConfidencePolicy.SYNTHETIC_VALIDITY,
                    seed=context.request.seed,
                    failure_reason=None,
                    pose=base_direct_xy_predictions_to_bfh_pose_arrays(
                        body_xy=_single_prediction_array(prediction.body),
                        left_hand_xy=_single_prediction_array(prediction.left_hand),
                        right_hand_xy=_single_prediction_array(prediction.right_hand),
                        face_xy=_single_prediction_array(prediction.face),
                        valid_frame_mask=reference.frame_valid_mask,
                    ),
                )
            )
        written = write_generated_pose_split(
            context.topology,
            producer_key="base_direct",
            run_name=context.request.run_name,
            split=split,
            samples=generated,
        )
        split_results.append(
            BaseDirectGeneratedSplitResult(
                split=split,
                manifest_path=written.manifest_path,
                sample_paths=written.payload_paths,
                sample_count=len(written.payload_paths),
            )
        )
    return BaseDirectGeneratedPoseExportResult(
        checkpoint_path=checkpoint_path,
        splits=tuple(split_results),
        total_sample_count=sum(result.sample_count for result in split_results),
    )


def infer_base_direct_single_sample(
    *,
    context: ModelSingleSampleInferenceContext,
    config: BaseDirectRunConfig,
) -> ModelSingleSampleInferenceResult:
    """Predict one prepared surface sample and write Stage 3 generated-pose output."""

    if context.checkpoint_path.is_file() is not True:
        raise ModelStageExecutionError(
            f"base_direct selected checkpoint does not exist: {context.checkpoint_path}"
        )
    compatibility_config_path = (
        context.topology.models.model_intermediate_root(
            "base_direct",
            context.request.run_name,
            "config",
        ).path
        / "baseline_training_compat.yaml"
    )
    predictor = load_base_direct_predictor(
        config=config,
        compatibility_config_path=compatibility_config_path,
        checkpoint_path=context.checkpoint_path,
        validate_config_paths=False,
    )
    try:
        from text_to_sign_production.modeling.inference.predict import predict_baseline_batch
    except ModuleNotFoundError as exc:
        raise ModelStageExecutionError(
            "base_direct single-sample inference requires torch/transformers dependencies."
        ) from exc
    record = processed_modeling_record_from_entry(
        context.sample.entry,
        manifest_path=context.sample.manifest_path,
        expected_split=context.sample.split,
        data_root=context.topology.samples_root,
    )
    reference = load_processed_pose_sample(record)
    item = ProcessedPoseItem.from_manifest_and_sample(record, reference)
    prediction = predict_baseline_batch(
        predictor.model,
        collate_processed_pose_samples([item]),
        device=predictor.device,
    )
    generated = GeneratedPoseSample(
        schema_version=GENERATED_POSE_PAYLOAD_SCHEMA_VERSION,
        producer_type=GeneratedPoseProducerType.MODEL,
        producer_key="base_direct",
        canonical_id="m0_direct_text_to_pose",
        phase_number=4,
        research_role="baseline_or_ablation_floor",
        run_name=context.request.run_name,
        split=context.sample.split,
        sample_id=context.sample.entry.sample_id,
        text=context.sample.entry.text,
        source_video_id=context.sample.entry.source_video_id,
        source_sentence_id=context.sample.entry.source_sentence_id,
        source_sentence_name=context.sample.entry.source_sentence_name,
        reference_payload_ref=context.sample.entry.payload_ref,
        generation_index=0,
        num_candidates_for_sample=1,
        generation_mode=GeneratedPoseGenerationMode.DETERMINISTIC,
        length_policy=GeneratedPoseLengthPolicy.REFERENCE_LENGTH,
        channel_policy=GENERATED_POSE_CHANNEL_POLICY,
        confidence_policy=GeneratedPoseConfidencePolicy.SYNTHETIC_VALIDITY,
        seed=context.request.seed,
        failure_reason=None,
        pose=base_direct_xy_predictions_to_bfh_pose_arrays(
            body_xy=_single_prediction_array(prediction.body),
            left_hand_xy=_single_prediction_array(prediction.left_hand),
            right_hand_xy=_single_prediction_array(prediction.right_hand),
            face_xy=_single_prediction_array(prediction.face),
            valid_frame_mask=reference.frame_valid_mask,
        ),
    )
    written = write_generated_pose_split_to_explicit_root(
        manifest_path=context.output_root / "manifest.jsonl",
        samples_root=context.output_root / "samples",
        payload_ref_root=context.topology.repo_root,
        split=context.sample.split,
        samples=(generated,),
    )
    if len(written.payload_paths) != 1:
        raise ModelStageExecutionError("base_direct single-sample inference wrote no payload.")
    return ModelSingleSampleInferenceResult(
        model_key=context.request.model_key,
        run_name=context.request.run_name,
        checkpoint_path=context.checkpoint_path,
        sample_id=context.sample.entry.sample_id,
        generated_sample=generated,
        generated_payload_path=written.payload_paths[0],
        generated_manifest_path=written.manifest_path,
        metadata={
            "checkpoint_path": str(context.checkpoint_path),
            "generated_payload_path": str(written.payload_paths[0]),
            "generated_manifest_path": str(written.manifest_path),
            "producer_stage": "test_model_single_sample",
            "split": context.sample.split.value,
        },
    )


def base_direct_xy_predictions_to_bfh_pose_arrays(
    *,
    body_xy: np.ndarray,
    left_hand_xy: np.ndarray,
    right_hand_xy: np.ndarray,
    face_xy: np.ndarray,
    valid_frame_mask: np.ndarray,
) -> BfhPoseArrays:
    """Convert direct XY regressions to full-BFH XYC using validity confidence."""

    mask = np.asarray(valid_frame_mask, dtype=np.bool_)
    if mask.ndim != 1 or mask.shape[0] <= 0:
        raise ModelStageExecutionError("valid_frame_mask must be a non-empty 1D array.")
    return BfhPoseArrays(
        body_xyc=_with_synthetic_confidence(body_xy, mask, expected_joints=25),
        left_hand_xyc=_with_synthetic_confidence(left_hand_xy, mask, expected_joints=21),
        right_hand_xyc=_with_synthetic_confidence(right_hand_xy, mask, expected_joints=21),
        face_xyc=_with_synthetic_confidence(face_xy, mask, expected_joints=70),
        valid_frame_mask=mask,
    )


def _with_synthetic_confidence(
    xy: np.ndarray,
    valid_frame_mask: np.ndarray,
    *,
    expected_joints: int,
) -> np.ndarray:
    array = np.asarray(xy, dtype=np.float32).copy()
    expected = (int(valid_frame_mask.shape[0]), expected_joints, 2)
    if array.shape != expected:
        raise ModelStageExecutionError(f"base_direct XY prediction must have shape {expected}.")
    if not np.all(np.isfinite(array)):
        raise ModelStageExecutionError("base_direct XY prediction contains non-finite values.")
    array[~valid_frame_mask] = 0.0
    confidence = np.broadcast_to(
        valid_frame_mask[:, None, None],
        (expected[0], expected_joints, 1),
    ).astype(np.float32, copy=True)
    return np.concatenate((array, confidence), axis=-1)


def _single_prediction_array(value: object) -> np.ndarray:
    try:
        array = value.detach().cpu().numpy()
    except AttributeError as exc:
        raise ModelStageExecutionError("base_direct prediction channel must be a tensor.") from exc
    if array.ndim != 4 or array.shape[0] != 1:
        raise ModelStageExecutionError("base_direct prediction channel must have batch size one.")
    return np.asarray(array[0], dtype=np.float32)


__all__ = [
    "BaseDirectGeneratedPoseExportResult",
    "BaseDirectGeneratedSplitResult",
    "base_direct_xy_predictions_to_bfh_pose_arrays",
    "export_base_direct_generated_pose",
    "infer_base_direct_single_sample",
]
