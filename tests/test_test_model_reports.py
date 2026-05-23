from __future__ import annotations

import json
from pathlib import Path

import pytest

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.data import parse_modeling_manifest_family
from text_to_sign_production.workflows.test_model.contracts import (
    CheckpointPolicy,
    TestModelChannelComparisonMetric as _TestModelChannelComparisonMetric,
    TestModelCheckpointSelection as _TestModelCheckpointSelection,
    TestModelComparisonMetric as _TestModelComparisonMetric,
    TestModelInferenceResult as _TestModelInferenceResult,
    TestModelReferenceComparisonResult as _TestModelReferenceComparisonResult,
    TestModelRunResolution as _TestModelRunResolution,
    TestModelSampleEvidence as _TestModelSampleEvidence,
    TestModelTargetResolution as _TestModelTargetResolution,
    TestModelVisualizationResult as _TestModelVisualizationResult,
    TestModelWorkflowConfig as _TestModelWorkflowConfig,
)
from text_to_sign_production.workflows.test_model.layout import build_test_model_layout
from text_to_sign_production.workflows.test_model.processing.comparison import (
    REFERENCE_COMPARISON_SCHEMA_VERSION,
)
from text_to_sign_production.workflows.test_model.processing.reports import (
    write_test_model_reports,
)


@pytest.mark.unit
def test_test_model_reports_include_reference_comparison_files(tmp_path: Path) -> None:
    layout = build_test_model_layout(
        _TestModelWorkflowConfig(
            project_root=tmp_path,
            drive_project_root=tmp_path / "drive",
            runtime_root=tmp_path / "runtime",
        )
    )
    model_run = _model_run(tmp_path)
    checkpoint = _checkpoint(tmp_path)
    target = _target(tmp_path)
    evidence = _evidence(target)
    inference = _TestModelInferenceResult(
        provider_result=None,
        metadata_path=None,
        receipts=(),
        status="not_ready",
        warnings=(),
        errors=(),
    )
    comparison = _comparison()
    visualization = _TestModelVisualizationResult(
        output_root=tmp_path / "videos",
        artifacts=(),
        receipts=(),
        status="completed",
        warnings=(),
        errors=(),
    )

    result = write_test_model_reports(
        layout,
        model_run=model_run,
        checkpoint=checkpoint,
        target=target,
        evidence=evidence,
        inference=inference,
        comparison=comparison,
        visualization=visualization,
        execution_id="exec001",
    )

    names = {path.name for path in result.files}
    assert "reference_comparison.json" in names
    assert "reference_comparison_summary.md" in names
    index = json.loads((result.output_root / "index.json").read_text(encoding="utf-8"))
    assert "reference_comparison.json" in index["files"]
    assert "reference_comparison_summary.md" in index["files"]
    assert index["comparison_status"] == "completed"
    summary = (result.output_root / "test_model_summary.md").read_text(encoding="utf-8")
    assert "Reference-vs-generated diagnostic" in summary
    assert "single-sample diagnostic" in summary
    assert "not aggregate test performance" in summary
    assert "not human intelligibility evaluation" in summary
    comparison_payload = json.loads(
        (result.output_root / "reference_comparison.json").read_text(encoding="utf-8")
    )
    assert comparison_payload["status"] == "completed"
    assert comparison_payload["metrics"][0]["metric_key"] == "masked_l1_mean"
    assert len(result.files) == len(index["files"])


def _model_run(tmp_path: Path) -> _TestModelRunResolution:
    family = parse_modeling_manifest_family("untiered:passed")
    return _TestModelRunResolution(
        model_run_name="run001",
        model_key="base_direct",
        manifest_family=family,
        train_split=SampleSplit.TRAIN,
        validation_split=SampleSplit.VAL,
        test_split=SampleSplit.TEST,
        run_mode="smoke",
        run_metadata_path=tmp_path / "run_metadata.json",
        run_metadata={},
        status="resolved",
        warnings=(),
        errors=(),
    )


def _checkpoint(tmp_path: Path) -> _TestModelCheckpointSelection:
    return _TestModelCheckpointSelection(
        policy=CheckpointPolicy.BEST,
        checkpoint_role="best",
        checkpoint_path=tmp_path / "checkpoint.pt",
        checkpoint_exists=True,
        warnings=(),
        errors=(),
    )


def _target(tmp_path: Path) -> _TestModelTargetResolution:
    return _TestModelTargetResolution(
        status="found",
        target_sentence_name="target sentence",
        resolved_sample_id="reference-sample",
        source_sentence_name="source sentence",
        manifest_family=parse_modeling_manifest_family("untiered:passed"),
        split=SampleSplit.TEST,
        manifest_path=tmp_path / "manifest.jsonl",
        manifest_entry=None,
        manifest_sample=None,
        issues=(),
        warnings=(),
    )


def _evidence(target: _TestModelTargetResolution) -> _TestModelSampleEvidence:
    return _TestModelSampleEvidence(
        target=target,
        source_video_path=Path("source.mp4"),
        source_video_exists=False,
        prepared_payload_path=None,
        prepared_payload_exists=False,
        model_context={"model_run_name": "run001"},
        checkpoint_context={},
        consistency_notes=(),
        warnings=(),
        errors=(),
    )


def _comparison() -> _TestModelReferenceComparisonResult:
    metrics = tuple(
        _TestModelComparisonMetric(key, 0.0, 2, 137, 1, ())
        for key in (
            "masked_l1_mean",
            "masked_l2_mean",
            "velocity_l1_mean",
            "velocity_l2_mean",
            "sequence_length_absolute_error",
            "valid_joint_coverage",
        )
    )
    channel_metrics = tuple(
        _TestModelChannelComparisonMetric(key, channel, 0.0, 2, 1, 1, ())
        for channel in ("body", "left_hand", "right_hand", "face")
        for key in (
            "channel_masked_l1_mean",
            "channel_masked_l2_mean",
            "channel_velocity_l1_mean",
            "channel_velocity_l2_mean",
            "channel_valid_joint_coverage",
        )
    )
    return _TestModelReferenceComparisonResult(
        schema_version=REFERENCE_COMPARISON_SCHEMA_VERSION,
        status="completed",
        split=SampleSplit.TEST,
        reference_sample_id="reference-sample",
        generated_sample_id="generated-sample",
        reference_frame_count=2,
        generated_frame_count=2,
        aligned_frame_count=2,
        metrics=metrics,
        channel_metrics=channel_metrics,
        warnings=(),
        errors=(),
    )
