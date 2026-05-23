from __future__ import annotations

from pathlib import Path

import pytest

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.artifacts import (
    GENERATED_POSE_CHANNEL_POLICY,
    GENERATED_POSE_MANIFEST_SCHEMA_VERSION,
    GeneratedPoseConfidencePolicy,
    GeneratedPoseGenerationMode,
    GeneratedPoseManifestEntry,
    GeneratedPoseProducerType,
    build_generated_pose_samples_archive,
    write_generated_pose_manifest_jsonl,
)


@pytest.mark.unit
def test_generated_pose_archive_builder_emits_progress_callbacks(tmp_path: Path) -> None:
    split_root = tmp_path / "generated_pose" / "base_direct" / "run001" / "val"
    samples_root = split_root / "samples"
    samples_root.mkdir(parents=True)
    (samples_root / "a__g0.npz").write_bytes(b"sample-a")
    (samples_root / "b__g0.npz").write_bytes(b"sample-b")
    manifest_path = split_root / "manifest.jsonl"
    write_generated_pose_manifest_jsonl(
        manifest_path,
        (_entry("a", "sentence a"), _entry("b", "sentence b")),
        split=SampleSplit.VAL,
    )
    events: list[str] = []
    members: list[tuple[int, int, str, int]] = []

    build_generated_pose_samples_archive(
        manifest_path=manifest_path,
        samples_root=samples_root,
        model_run_name="run001",
        model_key="base_direct",
        manifest_family="tiered:clean:included",
        split=SampleSplit.VAL,
        on_event=lambda event, payload: events.append(event),
        on_member=lambda index, total, sample_id, size_bytes: members.append(
            (index, total, sample_id, size_bytes)
        ),
    )

    assert events[0] == "archive_start"
    assert "archive_done" in events
    assert [member[2] for member in members] == ["a", "b"]
    assert all(member[3] > 0 for member in members)


def _entry(sample_id: str, sentence_name: str) -> GeneratedPoseManifestEntry:
    return GeneratedPoseManifestEntry(
        schema_version=GENERATED_POSE_MANIFEST_SCHEMA_VERSION,
        producer_type=GeneratedPoseProducerType.MODEL,
        producer_key="base_direct",
        canonical_id="m0_direct_text_to_pose",
        phase_number=4,
        research_role="baseline_or_ablation_floor",
        run_name="run001",
        split=SampleSplit.VAL,
        sample_id=sample_id,
        source_video_id=f"video-{sample_id}",
        source_sentence_id=f"sent-{sample_id}",
        source_sentence_name=sentence_name,
        text=sentence_name,
        reference_payload_ref=f"passed/val/{sample_id}.npz",
        generated_payload_ref=(
            f"evaluations/generated_pose/base_direct/run001/val/samples/{sample_id}__g0.npz"
        ),
        generation_index=0,
        num_candidates_for_sample=1,
        generation_mode=GeneratedPoseGenerationMode.DETERMINISTIC,
        length_policy="reference_length",
        channel_policy=GENERATED_POSE_CHANNEL_POLICY,
        confidence_policy=GeneratedPoseConfidencePolicy.SYNTHETIC_VALIDITY.value,
        frame_count=2,
        valid_frame_count=2,
        seed=13,
        failure_reason=None,
    )
