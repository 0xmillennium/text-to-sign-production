from __future__ import annotations

from pathlib import Path

import pytest

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.data import parse_modeling_manifest_family
from text_to_sign_production.workflows.test_model.contracts import (
    TestModelSampleEvidence as _TestModelSampleEvidence,
    TestModelTargetResolution as _TestModelTargetResolution,
)
from text_to_sign_production.workflows.test_model.review.sections import review_sample_evidence


@pytest.mark.unit
def test_review_sample_evidence_expands_context_dicts_to_scalar_fields(tmp_path: Path) -> None:
    target = _TestModelTargetResolution(
        status="found",
        target_sentence_name="target",
        resolved_sample_id="sample-1",
        source_sentence_name="sentence one",
        manifest_family=parse_modeling_manifest_family("untiered:passed"),
        split=SampleSplit.TEST,
        manifest_path=tmp_path / "manifest.json",
        manifest_entry=None,
        manifest_sample=None,
        issues=(),
        warnings=(),
    )
    evidence = _TestModelSampleEvidence(
        target=target,
        source_video_path=tmp_path / "source.mp4",
        source_video_exists=False,
        prepared_payload_path=None,
        prepared_payload_exists=False,
        model_context={"model_run_name": "run", "model_key": "base_direct"},
        checkpoint_context={"policy": "best", "checkpoint_exists": True},
        consistency_notes=(),
        warnings=(),
        errors=(),
    )

    sections = review_sample_evidence(evidence)

    values = [
        field.value
        for section in sections
        for item in section.items
        for field in item.fields
    ]
    assert values
    assert not any(isinstance(value, dict) for value in values)
