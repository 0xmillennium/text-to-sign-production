from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from test_validation_engine import (
    _generated_sample,
    _write_reference_split,
    _write_retrieval_surface,
)

from text_to_sign_production.artifacts.store import build_artifact_topology
from text_to_sign_production.core import build_repo_roots
from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.artifacts import write_generated_pose_split
from text_to_sign_production.modeling.data import parse_modeling_manifest_family
from text_to_sign_production.modeling.evaluation import (
    EvaluationProtocolConfig,
    EvaluationProtocolError,
    build_planned_run_requests,
    build_surface_inventory,
    load_evaluation_protocol_config,
    run_evaluation_protocol,
)
from text_to_sign_production.modeling.evaluation.config import (
    EvaluationContrastConfig,
    EvaluationProtocolExecutionConfig,
    EvaluationProtocolIdentityConfig,
    EvaluationProtocolReportsConfig,
    EvaluationValidationEngineConfig,
    EvaluationVariantConfig,
)


@pytest.mark.unit
def test_evaluation_protocol_config_parses_and_invalid_configs_fail(tmp_path: Path) -> None:
    path = Path("configs/modeling/evaluation_protocol.yaml")
    config = load_evaluation_protocol_config(path)
    assert config.split is SampleSplit.VAL
    assert {contrast.contrast_type for contrast in config.contrasts} >= {
        "ablation",
        "comparator_baseline",
    }
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    cases = (
        (
            "duplicate_variant",
            ("variants", 1, "variant_id", "learned_pose_token_frame"),
            "evaluation variant_id values must be unique",
        ),
        (
            "duplicate_contrast",
            ("contrasts", 1, "contrast_id", "learned_pose_token_window_vs_frame"),
            "evaluation contrast_id values must be unique",
        ),
        (
            "unknown_treatment",
            ("contrasts", 3, "treatment_variant_id", "unknown_variant"),
            "references unknown treatment_variant_id='unknown_variant'",
        ),
        (
            "bad_split",
            ("split", "test"),
            "evaluation protocol currently supports split='val' only",
        ),
        (
            "empty_run_name",
            ("variants", 0, "run_name", ""),
            "evaluation variant run_name is required because Aşama 9 validates existing generated surfaces and does not execute missing runs.",
        ),
        (
            "bad_retrieval_producer",
            ("variants", 7, "producer_type", "model"),
            "retrieval_pose_baseline must use producer_type='comparator'",
        ),
        (
            "bad_semantic_path",
            ("variants", 6, "objective_config_paths", {}),
            "semantic_consistency auxiliary objective requires",
        ),
    )
    for name, path_spec, match in cases:
        candidate = copy.deepcopy(raw)
        if len(path_spec) == 4:
            section, index, key, value = path_spec
            candidate[section][index][key] = value
        elif len(path_spec) == 3:
            section, key, value = path_spec
            candidate[section][key] = value
        else:
            key, value = path_spec
            candidate[key] = value
        invalid_path = tmp_path / f"{name}.yaml"
        invalid_path.write_text(yaml.safe_dump(candidate), encoding="utf-8")
        with pytest.raises(EvaluationProtocolError, match=match):
            load_evaluation_protocol_config(invalid_path)


@pytest.mark.unit
def test_surface_inventory_and_planned_run_requests_cover_ready_missing_and_failures(
    tmp_path: Path,
) -> None:
    topology = build_artifact_topology(build_repo_roots(tmp_path))
    _write_reference_split(topology, ("q1", "q2"))
    write_generated_pose_split(
        topology,
        producer_key="model_a",
        run_name="protocol_run",
        split=SampleSplit.VAL,
        samples=(_generated_sample("model_a", "protocol_run", "q1", shift=0.0),),
    )
    _write_retrieval_surface(topology, run_name="protocol_run", include_failure=True)
    config = _protocol_config(
        minimum_shared_paired_count=1,
        variants=(
            _variant("model_a", "model", "model_a", required=True),
            _variant("model_missing_optional", "model", "model_missing", required=False),
            _variant("model_missing_required", "model", "model_missing_required", required=True),
            _variant("retrieval_pose_baseline", "comparator", "retrieval_pose", required=True),
        ),
        contrasts=(
            _contrast("missing_treatment", "model_a", "model_missing_required"),
            _contrast(
                "retrieval_baseline_vs_model",
                "retrieval_pose_baseline",
                "model_a",
                contrast_type="comparator_baseline",
            ),
        ),
    )

    inventory = build_surface_inventory(
        topology=topology,
        config=config,
        manifest_family_id="untiered:passed",
    )
    statuses = {entry.variant_id: entry.status for entry in inventory.entries}
    assert statuses["model_a"] == "ready"
    assert statuses["retrieval_pose_baseline"] == "has_failures"
    assert statuses["model_missing_optional"] == "missing"
    assert statuses["model_missing_required"] == "missing"
    requests = build_planned_run_requests(inventory)
    assert any(record.get("model_key") == "model_missing" for record in requests)
    assert any(record.get("model_key") == "model_missing_required" for record in requests)
    assert all(record["executed"] is False for record in requests)


@pytest.mark.unit
def test_protocol_runner_invokes_validation_and_blocks_unready_claims(tmp_path: Path) -> None:
    topology = build_artifact_topology(build_repo_roots(tmp_path))
    family = parse_modeling_manifest_family("untiered:passed")
    _write_reference_split(topology, ("q1", "q2"))
    for producer_key, shift in (("model_a", 0.0), ("model_b", 0.25)):
        write_generated_pose_split(
            topology,
            producer_key=producer_key,
            run_name="protocol_run",
            split=SampleSplit.VAL,
            samples=(
                _generated_sample(producer_key, "protocol_run", "q1", shift=shift),
                _generated_sample(producer_key, "protocol_run", "q2", shift=shift),
            ),
        )
    _write_retrieval_surface(topology, run_name="protocol_run", include_failure=True)
    config = _protocol_config(
        minimum_shared_paired_count=2,
        variants=(
            _variant("model_a", "model", "model_a", required=True, config_path="same.yaml"),
            _variant("model_b", "model", "model_b", required=True, config_path="same.yaml"),
            _variant("model_missing", "model", "model_missing", required=True),
            _variant("retrieval_pose_baseline", "comparator", "retrieval_pose", required=True),
            _variant(
                "model_semantic",
                "model",
                "model_a",
                required=True,
                config_path="different.yaml",
                auxiliary_objectives=("semantic_consistency",),
                objective_config_paths={
                    "semantic_consistency": "configs/modeling/objectives/semantic_consistency.yaml"
                },
            ),
        ),
        contrasts=(
            _contrast("ready_ablation", "model_a", "model_b"),
            _contrast("missing_treatment", "model_a", "model_missing"),
            _contrast(
                "retrieval_baseline_vs_model",
                "retrieval_pose_baseline",
                "model_a",
                contrast_type="comparator_baseline",
            ),
            _contrast("semantic_mismatch", "model_a", "model_semantic"),
        ),
    )

    result = run_evaluation_protocol(
        topology=topology,
        manifest_family=family,
        config=config,
        run_name="protocol_run",
    )

    statuses = {entry.contrast_id: entry for entry in result.contrast_readiness}
    assert result.validation_engine_result_path is not None
    assert result.validation_engine_result_path.exists()
    assert statuses["ready_ablation"].readiness_status == "ready"
    assert statuses["ready_ablation"].shared_pairing_count == 2
    assert statuses["ready_ablation"].shared_pairing_keys == (("q1", 0), ("q2", 0))
    assert statuses["ready_ablation"].shared_pairing_sample_ids == ("q1", "q2")
    assert statuses["missing_treatment"].readiness_status == "not_ready_missing_surface"
    assert statuses["retrieval_baseline_vs_model"].contrast_type == "comparator_baseline"
    assert statuses["retrieval_baseline_vs_model"].readiness_status == "not_ready_insufficient_shared_subset"
    assert statuses["retrieval_baseline_vs_model"].shared_pairing_count == 1
    assert statuses["retrieval_baseline_vs_model"].shared_pairing_keys == (("q1", 0),)
    assert statuses["semantic_mismatch"].readiness_status == "not_ready_protocol_violation"
    assert "semantic objective ablation requires control and treatment to share the same model config_path." in (
        statuses["semantic_mismatch"].protocol_violations
    )
    assert result.report_paths is not None
    assert result.report_paths.evaluation_protocol_summary_path.exists()
    report_text = result.report_paths.claim_readiness_report_path.read_text(encoding="utf-8")
    assert "Retrieval baseline comparisons are counter-baselines, not ablations" in report_text
    assert "Automatic pose metrics do not prove sign intelligibility" in report_text
    assert "best model" not in result.report_paths.evaluation_protocol_summary_path.read_text(
        encoding="utf-8"
    )


@pytest.mark.unit
def test_protocol_runner_ready_ablation_passes_with_enough_shared_subset(tmp_path: Path) -> None:
    topology = build_artifact_topology(build_repo_roots(tmp_path))
    family = parse_modeling_manifest_family("untiered:passed")
    _write_reference_split(topology, ("q1", "q2"))
    for producer_key, shift in (("model_a", 0.0), ("model_b", 0.25)):
        write_generated_pose_split(
            topology,
            producer_key=producer_key,
            run_name="protocol_run",
            split=SampleSplit.VAL,
            samples=(
                _generated_sample(producer_key, "protocol_run", "q1", shift=shift),
                _generated_sample(producer_key, "protocol_run", "q2", shift=shift),
            ),
        )
    config = _protocol_config(
        minimum_shared_paired_count=2,
        variants=(
            _variant("model_a", "model", "model_a", required=True, config_path="same.yaml"),
            _variant("model_b", "model", "model_b", required=True, config_path="same.yaml"),
        ),
        contrasts=(_contrast("ready_ablation", "model_a", "model_b"),),
    )

    result = run_evaluation_protocol(
        topology=topology,
        manifest_family=family,
        config=config,
        run_name="protocol_run",
    )

    readiness = result.contrast_readiness[0]
    assert readiness.readiness_status == "ready"
    assert readiness.claim_ready is True
    assert readiness.shared_pairing_count == 2
    assert readiness.to_dict()["shared_pairing_keys"] == [
        {"sample_id": "q1", "generation_index": 0},
        {"sample_id": "q2", "generation_index": 0},
    ]
    assert readiness.to_dict()["shared_pairing_sample_ids"] == ["q1", "q2"]


@pytest.mark.unit
def test_unrelated_surface_failure_does_not_block_pairwise_ablation_readiness(
    tmp_path: Path,
) -> None:
    topology = build_artifact_topology(build_repo_roots(tmp_path))
    family = parse_modeling_manifest_family("untiered:passed")
    _write_reference_split(topology, ("q1", "q2"))
    for producer_key, shift in (("model_a", 0.0), ("model_b", 0.25)):
        write_generated_pose_split(
            topology,
            producer_key=producer_key,
            run_name="protocol_run",
            split=SampleSplit.VAL,
            samples=(
                _generated_sample(producer_key, "protocol_run", "q1", shift=shift),
                _generated_sample(producer_key, "protocol_run", "q2", shift=shift),
            ),
        )
    _write_retrieval_surface(topology, run_name="protocol_run", include_failure=True)
    config = _protocol_config(
        minimum_shared_paired_count=2,
        variants=(
            _variant("model_a", "model", "model_a", required=True, config_path="same.yaml"),
            _variant("model_b", "model", "model_b", required=True, config_path="same.yaml"),
            _variant("retrieval_pose_baseline", "comparator", "retrieval_pose", required=True),
        ),
        contrasts=(
            _contrast("model_a_vs_model_b", "model_a", "model_b"),
            _contrast(
                "retrieval_vs_model_a",
                "retrieval_pose_baseline",
                "model_a",
                contrast_type="comparator_baseline",
            ),
        ),
    )

    result = run_evaluation_protocol(
        topology=topology,
        manifest_family=family,
        config=config,
        run_name="protocol_run",
    )

    statuses = {entry.contrast_id: entry for entry in result.contrast_readiness}
    assert statuses["model_a_vs_model_b"].readiness_status == "ready"
    assert statuses["model_a_vs_model_b"].shared_pairing_count == 2
    assert statuses["model_a_vs_model_b"].claim_ready is True
    assert statuses["retrieval_vs_model_a"].readiness_status == (
        "not_ready_insufficient_shared_subset"
    )
    assert statuses["retrieval_vs_model_a"].shared_pairing_count == 1
    assert statuses["retrieval_vs_model_a"].claim_ready is False
    report_text = result.report_paths.claim_readiness_report_path.read_text(encoding="utf-8")
    assert "Contrast-specific shared subset count is below minimum_shared_paired_count" in report_text


@pytest.mark.unit
def test_protocol_runner_marks_ready_surfaces_comparison_unavailable_without_validation(
    tmp_path: Path,
) -> None:
    topology = build_artifact_topology(build_repo_roots(tmp_path))
    family = parse_modeling_manifest_family("untiered:passed")
    _write_reference_split(topology, ("q1", "q2"))
    for producer_key, shift in (("model_a", 0.0), ("model_b", 0.25)):
        write_generated_pose_split(
            topology,
            producer_key=producer_key,
            run_name="protocol_run",
            split=SampleSplit.VAL,
            samples=(
                _generated_sample(producer_key, "protocol_run", "q1", shift=shift),
                _generated_sample(producer_key, "protocol_run", "q2", shift=shift),
            ),
        )
    config = _protocol_config(
        minimum_shared_paired_count=2,
        variants=(
            _variant("model_a", "model", "model_a", required=True, config_path="same.yaml"),
            _variant("model_b", "model", "model_b", required=True, config_path="same.yaml"),
        ),
        contrasts=(_contrast("ready_surfaces_without_comparison", "model_a", "model_b"),),
        run_validation_engine=False,
    )

    result = run_evaluation_protocol(
        topology=topology,
        manifest_family=family,
        config=config,
        run_name="protocol_run",
    )

    readiness = result.contrast_readiness[0]
    assert readiness.readiness_status == "not_ready_comparison_unavailable"
    assert readiness.claim_ready is False


@pytest.mark.unit
def test_protocol_runner_raises_when_no_ready_surfaces(tmp_path: Path) -> None:
    topology = build_artifact_topology(build_repo_roots(tmp_path))
    family = parse_modeling_manifest_family("untiered:passed")
    config = _protocol_config(
        minimum_shared_paired_count=1,
        variants=(_variant("model_missing", "model", "model_missing", required=True),),
        contrasts=(),
    )

    with pytest.raises(
        EvaluationProtocolError,
        match="evaluation protocol did not find any ready generated surfaces",
    ):
        run_evaluation_protocol(
            topology=topology,
            manifest_family=family,
            config=config,
            run_name="protocol_run",
        )


def _protocol_config(
    *,
    minimum_shared_paired_count: int,
    variants: tuple[EvaluationVariantConfig, ...],
    contrasts: tuple[EvaluationContrastConfig, ...],
    run_validation_engine: bool = True,
) -> EvaluationProtocolConfig:
    return EvaluationProtocolConfig(
        source_path=None,
        raw_config={},
        identity=EvaluationProtocolIdentityConfig(
            schema_version="t2sp-evaluation-protocol-v1",
            manifest_family=parse_modeling_manifest_family("untiered:passed"),
        ),
        split=SampleSplit.VAL,
        execution=EvaluationProtocolExecutionConfig(run_validation_engine=run_validation_engine),
        variants=variants,
        contrasts=contrasts,
        validation_engine=EvaluationValidationEngineConfig(
            minimum_shared_paired_count=minimum_shared_paired_count,
        ),
        reports=EvaluationProtocolReportsConfig(),
    )


def _variant(
    variant_id: str,
    producer_type: str,
    producer_key: str,
    *,
    required: bool,
    config_path: str = "config.yaml",
    auxiliary_objectives: tuple[str, ...] = (),
    objective_config_paths: dict[str, str] | None = None,
) -> EvaluationVariantConfig:
    return EvaluationVariantConfig(
        variant_id=variant_id,
        label=variant_id,
        producer_type=producer_type,
        producer_key=producer_key,
        config_path=config_path,
        run_name="protocol_run",
        auxiliary_objectives=auxiliary_objectives,
        objective_config_paths={} if objective_config_paths is None else objective_config_paths,
        tags=(),
        required=required,
    )


def _contrast(
    contrast_id: str,
    control: str,
    treatment: str,
    *,
    contrast_type: str = "ablation",
) -> EvaluationContrastConfig:
    return EvaluationContrastConfig(
        contrast_id=contrast_id,
        contrast_type=contrast_type,
        control_variant_id=control,
        treatment_variant_id=treatment,
        intended_difference="controlled protocol difference",
        required=True,
    )
