from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.core.models import PassedManifestEntry
from text_to_sign_production.modeling.data import ModelingManifest, parse_modeling_manifest_family
from text_to_sign_production.workflows.test_model.contracts import (
    CheckpointPolicy,
    TestModelRequest as _TestModelRequest,
    TestModelWorkflowConfig as _TestModelWorkflowConfig,
)
from text_to_sign_production.workflows.test_model.layout import build_test_model_layout
from text_to_sign_production.workflows.test_model.processing import preflight


@pytest.mark.unit
def test_preflight_expected_outputs_include_reference_comparison_artifacts(
    tmp_path: Path,
) -> None:
    layout = build_test_model_layout(
        _TestModelWorkflowConfig(
            project_root=tmp_path,
            drive_project_root=tmp_path / "drive",
            runtime_root=tmp_path / "runtime",
        )
    )
    request = _TestModelRequest(
        model_run_name="run001",
        checkpoint_policy=CheckpointPolicy.BEST,
        target_sentence_name="target",
    )

    outputs = preflight._expected_outputs(layout, request, "exec001")

    names = {output.path.name for output in outputs}
    assert "reference_vs_generated_pose.mp4" in names
    assert "reference_comparison.json" in names
    assert "reference_comparison_summary.md" in names


@pytest.mark.unit
def test_preflight_target_not_found_includes_manifest_candidate_suggestions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    family = parse_modeling_manifest_family("untiered:passed")
    entries = (
        _entry("sample-a", "sentence a", 45),
        _entry("sample-b", "sentence b", 80),
        _entry("sample-c", "sentence c", 160),
    )
    manifest = ModelingManifest(
        manifest_family=family,
        split=SampleSplit.TEST,
        manifest_path=tmp_path / "test.json",
        entries=entries,
    )
    monkeypatch.setattr(preflight, "read_modeling_manifest", lambda *_args: manifest)

    checks = preflight._target_checks(
        SimpleNamespace(stores=SimpleNamespace(drive=object())),
        _TestModelRequest(
            model_run_name="run001",
            checkpoint_policy=CheckpointPolicy.BEST,
            target_sentence_name="missing sentence",
        ),
        family,
    )

    assert len(checks) == 1
    assert checks[0].status == "fail"
    suggestions = checks[0].details["candidate_suggestions"]
    assert suggestions[0]["source_sentence_name"] == "sentence b"
    assert suggestions[0]["sample_id"] == "sample-b"
    assert suggestions[0]["frame_count"] == 80
    assert len(suggestions) == 3


def _entry(sample_id: str, sentence_name: str, frame_count: int) -> PassedManifestEntry:
    return PassedManifestEntry(
        schema_version="gate-manifest-v1",
        sample_id=sample_id,
        split=SampleSplit.TEST,
        payload_ref=f"passed/test/{sample_id}.npz",
        text=sentence_name,
        fps=25.0,
        frame_count=frame_count,
        source_video_id=f"video-{sample_id}",
        source_sentence_id=f"sent-{sample_id}",
        source_sentence_name=sentence_name,
        valid_frame_count=frame_count,
        body_nonzero_frame_count=frame_count,
        face_nonzero_frame_count=frame_count,
        left_hand_nonzero_frame_count=frame_count,
        right_hand_nonzero_frame_count=frame_count,
    )
