from __future__ import annotations

import copy
import inspect
from pathlib import Path

import numpy as np
import pytest
import yaml

from text_to_sign_production.artifacts.store import build_artifact_topology
from text_to_sign_production.core import build_repo_roots
from text_to_sign_production.core.ids import CoordinateSpace, SampleSplit
from text_to_sign_production.core.models import PassedManifestEntry, PoseTruth, PreparedSample, SourceTruth
from text_to_sign_production.data.dataset import (
    PREPARED_SAMPLE_SCHEMA_VERSION,
    write_passed_manifest_json,
    write_prepared_sample_payload,
)
from text_to_sign_production.data.dataset.manifests import GATE_MANIFEST_SCHEMA_VERSION
from text_to_sign_production.data.gate.pose.types import PoseChannel
from text_to_sign_production.modeling.artifacts import (
    GENERATED_POSE_CHANNEL_POLICY,
    GENERATED_POSE_PAYLOAD_SCHEMA_VERSION,
    GeneratedPoseSample,
    write_generated_pose_split,
)
from text_to_sign_production.modeling.artifacts.run_metadata import (
    GeneratedPoseConfidencePolicy,
    GeneratedPoseGenerationMode,
    GeneratedPoseLengthPolicy,
    GeneratedPoseProducerType,
)
from text_to_sign_production.modeling.comparators.retrieval_pose.bank import (
    RetrievalBank,
    RetrievalBankItem,
)
from text_to_sign_production.modeling.comparators.retrieval_pose.config import (
    load_retrieval_pose_comparator_config,
)
from text_to_sign_production.modeling.comparators.retrieval_pose.io import (
    write_retrieval_generated_pose_split,
)
from text_to_sign_production.modeling.comparators.retrieval_pose.scoring import (
    RetrievalQuery,
    RetrievalQueryResult,
    score_retrieval_query,
)
from text_to_sign_production.modeling.data import parse_modeling_manifest_family
from text_to_sign_production.modeling.data.bfh_schema import (
    BFH_CHANNEL_SPECS,
    BfhPoseArrays,
)
from text_to_sign_production.modeling.validation import (
    ModelValidationError,
    ValidationComparisonConfig,
    ValidationEngineConfig,
    ValidationEngineDataConfig,
    ValidationEngineIdentityConfig,
    ValidationEngineReportsConfig,
    ValidationMetricSelectionConfig,
    ValidationSurfaceConfig,
    load_validation_engine_config,
    resolve_generated_surface,
    run_candidate_agnostic_validation,
)


@pytest.mark.unit
def test_validation_engine_config_parses_and_invalid_configs_fail(tmp_path: Path) -> None:
    path = Path("configs/modeling/validation_engine.yaml")
    assert load_validation_engine_config(path).data.split is SampleSplit.VAL
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    cases = (
        ("duplicate", ("surfaces", 1, "label", "base_direct"), "validation surface labels must be unique"),
        ("bad_split", ("data", "split", "test"), "candidate-agnostic validation currently supports split='val' only"),
        ("bad_producer", ("surfaces", 0, "producer_type", "baseline"), "producer_type"),
        ("overlap", ("comparison", "higher_is_better", ["valid_joint_coverage", "masked_l1_mean"]), "must not overlap"),
    )
    for name, (section, *rest), match in cases:
        candidate = copy.deepcopy(raw)
        if len(rest) == 3:
            index, key, value = rest
            candidate[section][index][key] = value
        else:
            key, value = rest
            candidate[section][key] = value
        invalid_path = tmp_path / f"{name}.yaml"
        invalid_path.write_text(yaml.safe_dump(candidate), encoding="utf-8")
        with pytest.raises(ModelValidationError, match=match):
            load_validation_engine_config(invalid_path)


@pytest.mark.unit
def test_validation_surface_resolution_handles_model_comparator_optional_required_and_failures(
    tmp_path: Path,
) -> None:
    topology = build_artifact_topology(build_repo_roots(tmp_path))
    _write_reference_split(topology, ("q1",))
    write_generated_pose_split(
        topology,
        producer_key="model_a",
        run_name="surface_run",
        split=SampleSplit.VAL,
        samples=(_generated_sample("model_a", "surface_run", "q1", shift=0.0),),
    )
    _write_retrieval_surface(topology, run_name="surface_run", include_failure=True)

    model_surface = resolve_generated_surface(
        topology=topology,
        surface_config=ValidationSurfaceConfig("model_a", "model", "model_a", required=True),
        run_name_override="surface_run",
    )
    comparator_surface = resolve_generated_surface(
        topology=topology,
        surface_config=ValidationSurfaceConfig(
            "retrieval_pose",
            "comparator",
            "retrieval_pose",
            canonical_id="retrieval_augmented_pose_comparator",
            research_role="counter_alternative_comparator",
            required=True,
        ),
        run_name_override="surface_run",
    )
    optional_missing = resolve_generated_surface(
        topology=topology,
        surface_config=ValidationSurfaceConfig("missing", "model", "missing", required=False),
        run_name_override="surface_run",
    )

    assert model_surface is not None and model_surface.is_primary_model
    assert comparator_surface is not None and comparator_surface.is_comparator
    assert any(entry.failure_reason is not None for entry in comparator_surface.entries)
    assert optional_missing is None
    with pytest.raises(
        ModelValidationError,
        match="required validation surface 'missing' is missing generated-pose manifest for run_name='surface_run'.",
    ):
        resolve_generated_surface(
            topology=topology,
            surface_config=ValidationSurfaceConfig("missing", "model", "missing", required=True),
            run_name_override="surface_run",
        )


@pytest.mark.unit
def test_candidate_agnostic_validation_engine_writes_metrics_comparison_and_reports(
    tmp_path: Path,
) -> None:
    topology = build_artifact_topology(build_repo_roots(tmp_path))
    family = parse_modeling_manifest_family("untiered:passed")
    _write_reference_split(topology, ("q1", "q2"))
    for producer_key, shift in (("model_a", 0.0), ("model_b", 0.25)):
        write_generated_pose_split(
            topology,
            producer_key=producer_key,
            run_name="engine_run",
            split=SampleSplit.VAL,
            samples=(
                _generated_sample(producer_key, "engine_run", "q1", shift=shift),
                _generated_sample(producer_key, "engine_run", "q2", shift=shift),
            ),
        )
    _write_retrieval_surface(topology, run_name="engine_run", include_failure=True)
    config = _engine_config(
        (
            ValidationSurfaceConfig("model_a", "model", "model_a", required=True),
            ValidationSurfaceConfig("model_b", "model", "model_b", required=True),
            ValidationSurfaceConfig(
                "retrieval_pose",
                "comparator",
                "retrieval_pose",
                canonical_id="retrieval_augmented_pose_comparator",
                research_role="counter_alternative_comparator",
                required=True,
            ),
        )
    )

    result = run_candidate_agnostic_validation(
        topology=topology,
        manifest_family=family,
        config=config,
        run_name="engine_run",
    )

    assert len(result.surface_results) == 3
    assert all(surface_result.pairing_path.exists() for surface_result in result.surface_results)
    retrieval_result = next(item for item in result.surface_results if item.surface.label == "retrieval_pose")
    assert retrieval_result.limitations.failed_generated_count == 1
    assert all(pair.status != "paired" for pair in retrieval_result.pairing if pair.key.sample_id == "q2")
    assert result.comparison is not None
    assert result.comparison.status == "ok"
    assert result.comparison.shared_pairing_subset_sample_ids == ("q1",)
    assert [key.to_dict() for key in result.comparison.shared_pairing_subset_keys] == [
        {"sample_id": "q1", "generation_index": 0}
    ]
    assert result.comparison.to_dict()["shared_pairing_subset_sample_ids"] == ["q1"]
    assert result.comparison.to_dict()["shared_pairing_subset_keys"] == [
        {"sample_id": "q1", "generation_index": 0}
    ]
    l1_rows = [row for row in result.comparison.rows if row.metric_key == "masked_l1_mean"]
    assert any(row.surface_label == "model_a" and row.rank == 1 and row.is_best_by_metric for row in l1_rows)
    coverage_rows = [row for row in result.comparison.rows if row.metric_key == "valid_joint_coverage"]
    assert all(row.rank is not None for row in coverage_rows if row.availability_status == "available")
    assert result.report_paths is not None
    assert result.report_paths.retrieval_metadata_report_path.exists()
    assert "Automatic pose/keypoint metrics do not prove sign intelligibility" in (
        result.report_paths.metric_interpretation_path.read_text(encoding="utf-8")
    )
    assert "best model" not in result.report_paths.comparison_report_path.read_text(encoding="utf-8")


@pytest.mark.unit
def test_validation_engine_rejects_manifest_family_mismatch(tmp_path: Path) -> None:
    topology = build_artifact_topology(build_repo_roots(tmp_path))
    family = parse_modeling_manifest_family("untiered:passed")
    config = _engine_config(
        (ValidationSurfaceConfig("model_a", "model", "model_a", required=False),),
        manifest_family="tiered:loose:included",
    )

    with pytest.raises(
        ModelValidationError,
        match="validation engine config manifest_family='tiered:loose:included' does not match runner manifest_family='untiered:passed'.",
    ):
        run_candidate_agnostic_validation(
            topology=topology,
            manifest_family=family,
            config=config,
            run_name="engine_run",
        )


@pytest.mark.unit
def test_workflow_validation_imports_and_public_constructor_signatures_remain_available() -> None:
    from text_to_sign_production.workflows.model.processing.validation import (
        write_model_validation_artifacts,
    )
    from text_to_sign_production.workflows.model.workflow import ModelWorkflow

    assert callable(write_model_validation_artifacts)
    assert "config" in inspect.signature(ModelWorkflow).parameters


def _engine_config(
    surfaces: tuple[ValidationSurfaceConfig, ...],
    *,
    manifest_family: str | None = None,
) -> ValidationEngineConfig:
    return ValidationEngineConfig(
        source_path=None,
        raw_config={},
        identity=ValidationEngineIdentityConfig(
            schema_version="t2sp-validation-engine-v1",
            manifest_family=(
                None if manifest_family is None else parse_modeling_manifest_family(manifest_family)
            ),
        ),
        data=ValidationEngineDataConfig(),
        surfaces=surfaces,
        metrics=ValidationMetricSelectionConfig(),
        comparison=ValidationComparisonConfig(
            lower_is_better=(
                "masked_l1_mean",
                "masked_l2_mean",
                "velocity_l1_mean",
                "velocity_l2_mean",
                "sequence_length_absolute_error",
                "channel_masked_l1_mean:*",
                "channel_masked_l2_mean:*",
                "channel_velocity_l1_mean:*",
                "channel_velocity_l2_mean:*",
            ),
            higher_is_better=("valid_joint_coverage", "channel_valid_joint_coverage:*"),
        ),
        reports=ValidationEngineReportsConfig(),
    )


def _write_reference_split(topology, sample_ids: tuple[str, ...]) -> None:
    entries = []
    for sample_id in sample_ids:
        sample = _prepared_sample(sample_id)
        payload_ref = f"passed/val/{sample_id}.npz"
        write_prepared_sample_payload(topology.samples_root / payload_ref, sample)
        entries.append(
            PassedManifestEntry(
                schema_version=GATE_MANIFEST_SCHEMA_VERSION,
                sample_id=sample_id,
                split=SampleSplit.VAL,
                payload_ref=payload_ref,
                text=f"text {sample_id}",
                fps=25.0,
                frame_count=sample.pose.frame_count,
                source_video_id=f"video-{sample_id}",
                source_sentence_id=f"sent-{sample_id}",
                source_sentence_name=f"sentence_{sample_id}",
                valid_frame_count=sample.pose.frame_count,
                body_nonzero_frame_count=sample.pose.frame_count,
                face_nonzero_frame_count=sample.pose.frame_count,
                left_hand_nonzero_frame_count=sample.pose.frame_count,
                right_hand_nonzero_frame_count=sample.pose.frame_count,
            )
        )
    write_passed_manifest_json(
        topology.manifests.untiered_passed_manifest(SampleSplit.VAL).path,
        entries,
        split=SampleSplit.VAL,
    )


def _prepared_sample(sample_id: str) -> PreparedSample:
    pose = _pose()
    return PreparedSample(
        schema_version=PREPARED_SAMPLE_SCHEMA_VERSION,
        source=SourceTruth(
            sample_id=sample_id,
            split=SampleSplit.VAL,
            text=f"text {sample_id}",
            fps=25.0,
            source_video_id=f"video-{sample_id}",
            source_sentence_id=f"sent-{sample_id}",
            source_sentence_name=f"sentence_{sample_id}",
            source_issue_codes=(),
        ),
        pose=PoseTruth(
            coordinate_space=CoordinateSpace.NORMALIZED_IMAGE,
            frame_count=pose.frame_count,
            valid_frame_mask=pose.valid_frame_mask,
            selected_person_indices=tuple(0 for _ in range(pose.frame_count)),
            tracked_target_missing_frame_count=0,
            continuity_break_count=0,
            reanchor_count=0,
            body_xyc=pose.body_xyc,
            face_xyc=pose.face_xyc,
            left_hand_xyc=pose.left_hand_xyc,
            right_hand_xyc=pose.right_hand_xyc,
            body_nonzero_frame_count=pose.frame_count,
            face_nonzero_frame_count=pose.frame_count,
            left_hand_nonzero_frame_count=pose.frame_count,
            right_hand_nonzero_frame_count=pose.frame_count,
        ),
    )


def _generated_sample(producer_key: str, run_name: str, sample_id: str, *, shift: float) -> GeneratedPoseSample:
    return GeneratedPoseSample(
        schema_version=GENERATED_POSE_PAYLOAD_SCHEMA_VERSION,
        producer_type=GeneratedPoseProducerType.MODEL,
        producer_key=producer_key,
        canonical_id=f"{producer_key}_canonical",
        phase_number=8,
        research_role="primary_model",
        run_name=run_name,
        split=SampleSplit.VAL,
        sample_id=sample_id,
        text=f"text {sample_id}",
        source_video_id=f"video-{sample_id}",
        source_sentence_id=f"sent-{sample_id}",
        source_sentence_name=f"sentence_{sample_id}",
        reference_payload_ref=f"passed/val/{sample_id}.npz",
        generation_index=0,
        num_candidates_for_sample=1,
        generation_mode=GeneratedPoseGenerationMode.DETERMINISTIC,
        length_policy=GeneratedPoseLengthPolicy.REFERENCE_LENGTH,
        channel_policy=GENERATED_POSE_CHANNEL_POLICY,
        confidence_policy=GeneratedPoseConfidencePolicy.SYNTHETIC_VALIDITY,
        seed=None,
        failure_reason=None,
        pose=_pose(shift=shift),
    )


def _write_retrieval_surface(topology, *, run_name: str, include_failure: bool) -> None:
    config = load_retrieval_pose_comparator_config(Path("configs/modeling/comparators/retrieval_pose.yaml"))
    bank_item = RetrievalBankItem(
        sample_id="bank1",
        split=SampleSplit.TRAIN,
        text="bank text",
        source_video_id="video-bank",
        source_sentence_id="sent-bank",
        source_sentence_name="sentence_bank",
        payload_ref="passed/train/bank1.npz",
        frame_count=_pose().frame_count,
        valid_frame_count=_pose().frame_count,
        pose=_pose(),
        embedding=np.asarray([1.0, 0.0], dtype=np.float32),
    )
    bank = RetrievalBank(
        split=SampleSplit.TRAIN,
        items=(bank_item,),
        embedding_matrix=np.asarray([[1.0, 0.0]], dtype=np.float32),
    )
    success = score_retrieval_query(
        query=RetrievalQuery(
            sample_id="q1",
            split=SampleSplit.VAL,
            text="text q1",
            source_video_id="video-q1",
            source_sentence_id="sent-q1",
            source_sentence_name="sentence_q1",
            reference_payload_ref="passed/val/q1.npz",
            embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        ),
        bank=bank,
        scoring=config.retrieval,
        leakage_policy=config.leakage,
    )
    results: list[RetrievalQueryResult] = [success]
    if include_failure:
        results.append(
            RetrievalQueryResult(
                query_sample_id="q2",
                query_split=SampleSplit.VAL,
                query_text="text q2",
                source_video_id="video-q2",
                source_sentence_id="sent-q2",
                source_sentence_name="sentence_q2",
                reference_payload_ref="passed/val/q2.npz",
                success=False,
                selected_candidate=None,
                selected_bank_item=None,
                top_k_safe_candidates=(),
                blocked_count=1,
                blocked_reasons=("same_sample_id",),
                failure_reason="no leakage-safe retrieval candidate remained for query sample_id='q2'.",
            )
        )
    write_retrieval_generated_pose_split(
        topology=topology,
        config=config,
        run_name=run_name,
        split=SampleSplit.VAL,
        results=results,
    )


def _pose(frame_count: int = 3, *, shift: float = 0.0) -> BfhPoseArrays:
    return BfhPoseArrays(
        body_xyc=_channel(frame_count, "body", shift=shift),
        left_hand_xyc=_channel(frame_count, "left_hand", shift=shift),
        right_hand_xyc=_channel(frame_count, "right_hand", shift=shift),
        face_xyc=_channel(frame_count, "face", shift=shift),
        valid_frame_mask=np.ones((frame_count,), dtype=np.bool_),
    )


def _channel(frame_count: int, channel: str, *, shift: float) -> np.ndarray:
    joints = BFH_CHANNEL_SPECS[PoseChannel(channel)].joint_count
    array = np.zeros((frame_count, joints, 3), dtype=np.float32)
    array[..., 0] = 0.5 + shift
    array[..., 1] = 0.5 + shift
    array[..., 2] = 1.0
    return array
