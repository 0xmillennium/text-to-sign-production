from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import yaml

from text_to_sign_production.artifacts.store import build_artifact_topology
from text_to_sign_production.core import build_repo_roots
from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.artifacts import (
    load_generated_pose_payload,
    read_generated_pose_manifest_jsonl,
)
from text_to_sign_production.modeling.comparators.retrieval_pose.bank import (
    RetrievalBank,
    RetrievalBankItem,
)
from text_to_sign_production.modeling.comparators.retrieval_pose.config import (
    RetrievalLeakagePolicyConfig,
    RetrievalPoseComparatorConfig,
    RetrievalScoringConfig,
    load_retrieval_pose_comparator_config,
)
from text_to_sign_production.modeling.comparators.retrieval_pose.errors import (
    RetrievalPoseComparatorError,
)
from text_to_sign_production.modeling.comparators.retrieval_pose.io import (
    write_retrieval_generated_pose_split,
)
from text_to_sign_production.modeling.comparators.retrieval_pose.runner import (
    run_retrieval_pose_comparator,
)
from text_to_sign_production.modeling.comparators.retrieval_pose.leakage import (
    assert_bank_query_split_policy,
    detect_retrieval_leakage_reasons,
)
from text_to_sign_production.modeling.comparators.retrieval_pose.scoring import (
    RetrievalQuery,
    RetrievalQueryResult,
    cosine_similarity_scores,
    score_retrieval_query,
)
from text_to_sign_production.modeling.data.bfh_schema import (
    BFH_CHANNEL_SPECS,
    BfhPoseArrays,
)
from text_to_sign_production.modeling.data import parse_modeling_manifest_family
from text_to_sign_production.data.gate.pose.types import PoseChannel


@pytest.mark.unit
def test_retrieval_configs_parse_and_invalid_configs_fail(tmp_path: Path) -> None:
    for path in (
        Path("configs/modeling/comparators/retrieval_pose.yaml"),
        Path("configs/modeling/comparators/retrieval_pose_huggingface.yaml"),
    ):
        assert isinstance(load_retrieval_pose_comparator_config(path), RetrievalPoseComparatorConfig)

    raw = yaml.safe_load(Path("configs/modeling/comparators/retrieval_pose.yaml").read_text())
    invalid_cases = (
        ("bad_metric", ("retrieval", "score_metric", "dot"), "retrieval.score_metric"),
        ("bad_policy", ("retrieval", "selection_policy", "top3"), "retrieval.selection_policy"),
        ("bad_candidates", ("generation", "candidates_per_query", 2), "candidates_per_query"),
        ("trainable", ("text_encoder", "trainable", True), "trainable"),
        ("forbidden", ("dictionary", "enabled", True), "forbidden experimental key"),
    )
    for name, (section, key, value), match in invalid_cases:
        candidate = copy.deepcopy(raw)
        if name == "forbidden":
            candidate[section] = {key: value}
        else:
            candidate[section][key] = value
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(candidate), encoding="utf-8")
        with pytest.raises(RetrievalPoseComparatorError, match=match):
            load_retrieval_pose_comparator_config(path)


@pytest.mark.unit
def test_retrieval_leakage_policy_blocks_expected_metadata() -> None:
    policy = RetrievalLeakagePolicyConfig()
    query = _query()
    same_sample = _item(sample_id="q1", source_sentence_id="sent2")
    same_sentence = _item(sample_id="b2", source_sentence_id="sent1")
    safe = _item(sample_id="b3", source_sentence_id="sent3")

    assert "same_sample_id" in detect_retrieval_leakage_reasons(query, same_sample, policy)
    assert "same_source_sentence_id" in detect_retrieval_leakage_reasons(query, same_sentence, policy)
    assert detect_retrieval_leakage_reasons(query, safe, policy) == ()
    with pytest.raises(RetrievalPoseComparatorError, match="forbids using query split 'val'"):
        assert_bank_query_split_policy(
            bank_split=SampleSplit.VAL,
            query_split=SampleSplit.VAL,
            policy=policy,
        )

    identical_policy = RetrievalLeakagePolicyConfig(exclude_identical_text=True)
    identical = _item(sample_id="b4", source_sentence_id="sent4", text="hello world")
    assert "identical_text" in detect_retrieval_leakage_reasons(
        query,
        identical,
        identical_policy,
    )


@pytest.mark.unit
def test_retrieval_scoring_selects_nearest_sorts_top_k_and_fails_zero_norm() -> None:
    scoring = RetrievalScoringConfig(top_k=2)
    bank = RetrievalBank(
        split=SampleSplit.TRAIN,
        items=(
            _item(sample_id="b1", source_sentence_id="sent-b1", embedding=(1.0, 0.0)),
            _item(sample_id="b2", source_sentence_id="sent-b2", embedding=(0.0, 1.0)),
            _item(sample_id="b3", source_sentence_id="sent-b3", embedding=(0.7, 0.7)),
        ),
        embedding_matrix=np.asarray([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32),
    )
    result = score_retrieval_query(
        query=_query(embedding=(0.0, 1.0)),
        bank=bank,
        scoring=scoring,
        leakage_policy=RetrievalLeakagePolicyConfig(),
    )

    assert result.success
    assert result.selected_candidate is not None
    assert result.selected_candidate.retrieved_sample_id == "b2"
    assert [candidate.retrieved_sample_id for candidate in result.top_k_safe_candidates] == [
        "b2",
        "b3",
    ]
    assert [candidate.score for candidate in result.top_k_safe_candidates] == sorted(
        [candidate.score for candidate in result.top_k_safe_candidates],
        reverse=True,
    )
    with pytest.raises(RetrievalPoseComparatorError, match="zero-norm embeddings"):
        cosine_similarity_scores(
            np.asarray([0.0, 0.0], dtype=np.float32),
            bank.embedding_matrix,
            scoring=scoring,
        )


@pytest.mark.unit
def test_retrieval_scoring_no_safe_candidate_returns_failure() -> None:
    bank = RetrievalBank(
        split=SampleSplit.TRAIN,
        items=(_item(sample_id="q1", source_sentence_id="sent1", embedding=(1.0, 0.0)),),
        embedding_matrix=np.asarray([[1.0, 0.0]], dtype=np.float32),
    )
    result = score_retrieval_query(
        query=_query(embedding=(1.0, 0.0)),
        bank=bank,
        scoring=RetrievalScoringConfig(),
        leakage_policy=RetrievalLeakagePolicyConfig(),
    )

    assert not result.success
    assert result.failure_reason == "no leakage-safe retrieval candidate remained for query sample_id='q1'."


@pytest.mark.unit
def test_retrieval_generated_pose_writer_populates_manifest_fields(tmp_path: Path) -> None:
    topology = build_artifact_topology(build_repo_roots(tmp_path))
    config = load_retrieval_pose_comparator_config(
        Path("configs/modeling/comparators/retrieval_pose.yaml")
    )
    bank = RetrievalBank(
        split=SampleSplit.TRAIN,
        items=(_item(sample_id="b1", source_sentence_id="sent-b1", embedding=(1.0, 0.0)),),
        embedding_matrix=np.asarray([[1.0, 0.0]], dtype=np.float32),
    )
    result = score_retrieval_query(
        query=_query(embedding=(1.0, 0.0)),
        bank=bank,
        scoring=config.retrieval,
        leakage_policy=config.leakage,
    )

    written = write_retrieval_generated_pose_split(
        topology=topology,
        config=config,
        run_name="smoke_run",
        split=SampleSplit.VAL,
        results=(result,),
    )
    entries = read_generated_pose_manifest_jsonl(written.manifest_path, expected_split=SampleSplit.VAL)

    assert len(written.payload_paths) == 1
    assert written.payload_paths[0].exists()
    payload = load_generated_pose_payload(written.payload_paths[0])
    assert payload.producer_type.value == "comparator"
    assert payload.generation_mode.value == "retrieval"
    assert payload.length_policy.value == "retrieved_length"
    assert payload.confidence_policy.value == "retrieved_confidence"
    assert len(entries) == 1
    entry = entries[0]
    assert entry.retrieval_source_split is SampleSplit.TRAIN
    assert entry.retrieved_sample_id == "b1"
    assert entry.retrieval_score == pytest.approx(1.0)
    assert entry.leakage_checked is True
    assert entry.producer_type.value == "comparator"
    assert entry.generation_mode.value == "retrieval"
    assert entry.length_policy == "retrieved_length"
    assert entry.confidence_policy == "retrieved_confidence"


@pytest.mark.unit
def test_retrieval_generated_pose_writer_round_trips_failure_manifest_entry(tmp_path: Path) -> None:
    topology = build_artifact_topology(build_repo_roots(tmp_path))
    config = load_retrieval_pose_comparator_config(
        Path("configs/modeling/comparators/retrieval_pose.yaml")
    )
    result = RetrievalQueryResult(
        query_sample_id="q1",
        query_split=SampleSplit.VAL,
        query_text="hello world",
        source_video_id="video1",
        source_sentence_id="sent1",
        source_sentence_name="sentence_1",
        reference_payload_ref="passed/val/q1.npz",
        success=False,
        selected_candidate=None,
        selected_bank_item=None,
        top_k_safe_candidates=(),
        blocked_count=1,
        blocked_reasons=("same_sample_id",),
        failure_reason="no leakage-safe retrieval candidate remained for query sample_id='q1'.",
    )

    written = write_retrieval_generated_pose_split(
        topology=topology,
        config=config,
        run_name="failure_smoke",
        split=SampleSplit.VAL,
        results=(result,),
    )
    entries = read_generated_pose_manifest_jsonl(written.manifest_path, expected_split=SampleSplit.VAL)

    assert written.payload_paths == ()
    assert len(entries) == 1
    entry = entries[0]
    assert entry.generated_payload_ref is None
    assert entry.failure_reason is not None
    assert entry.leakage_checked is True


@pytest.mark.unit
def test_retrieval_runner_smoke_with_synthetic_manifest_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topology = build_artifact_topology(build_repo_roots(tmp_path))
    config = load_retrieval_pose_comparator_config(
        Path("configs/modeling/comparators/retrieval_pose.yaml")
    )
    train = _manifest_sample(
        sample_id="train1",
        split=SampleSplit.TRAIN,
        text="alpha beta",
        source_sentence_id="train-sent",
    )
    val = _manifest_sample(
        sample_id="val1",
        split=SampleSplit.VAL,
        text="alpha beta",
        source_sentence_id="val-sent",
    )

    def fake_load_manifest_samples(*args: object, **_kwargs: object):
        split = args[2]
        return (train,) if SampleSplit(split) is SampleSplit.TRAIN else (val,)

    import text_to_sign_production.modeling.comparators.retrieval_pose.bank as bank_module
    import text_to_sign_production.modeling.comparators.retrieval_pose.runner as runner_module

    monkeypatch.setattr(bank_module, "load_manifest_samples", fake_load_manifest_samples)
    monkeypatch.setattr(runner_module, "load_manifest_samples", fake_load_manifest_samples)

    result = run_retrieval_pose_comparator(
        topology=topology,
        manifest_family=config.data.manifest_family,
        config=config,
        run_name="runner_smoke",
    )

    assert result.records_count == 1
    assert result.safe_retrieval_count == 1
    assert result.failure_count == 0
    assert result.retrieval_results_path.exists()
    assert result.leakage_report_path.exists()
    assert all(path.exists() for path in result.report_paths)
    assert read_generated_pose_manifest_jsonl(result.manifest_paths[0], expected_split=SampleSplit.VAL)


@pytest.mark.unit
def test_retrieval_runner_rejects_manifest_family_mismatch(tmp_path: Path) -> None:
    topology = build_artifact_topology(build_repo_roots(tmp_path))
    config = load_retrieval_pose_comparator_config(
        Path("configs/modeling/comparators/retrieval_pose.yaml")
    )

    with pytest.raises(
        RetrievalPoseComparatorError,
        match="retrieval config manifest_family='untiered:passed' does not match runner manifest_family='tiered:loose:included'.",
    ):
        run_retrieval_pose_comparator(
            topology=topology,
            manifest_family=parse_modeling_manifest_family("tiered:loose:included"),
            config=config,
            run_name="mismatch",
        )


@dataclass(frozen=True, slots=True)
class _SyntheticManifestSample:
    sample_id: str
    split: SampleSplit
    text: str
    source_video_id: str
    source_sentence_id: str
    source_sentence_name: str
    payload_ref: str
    pose: BfhPoseArrays

    @property
    def frame_count(self) -> int:
        return self.pose.frame_count


def _manifest_sample(
    *,
    sample_id: str,
    split: SampleSplit,
    text: str,
    source_sentence_id: str,
) -> _SyntheticManifestSample:
    return _SyntheticManifestSample(
        sample_id=sample_id,
        split=split,
        text=text,
        source_video_id=f"video-{split.value}",
        source_sentence_id=source_sentence_id,
        source_sentence_name=source_sentence_id,
        payload_ref=f"passed/{split.value}/{sample_id}.npz",
        pose=_pose(),
    )


def _query(
    *,
    embedding: tuple[float, ...] = (1.0, 0.0),
) -> RetrievalQuery:
    return RetrievalQuery(
        sample_id="q1",
        split=SampleSplit.VAL,
        text="hello world",
        source_video_id="video1",
        source_sentence_id="sent1",
        source_sentence_name="sentence_1",
        reference_payload_ref="passed/val/q1.npz",
        embedding=np.asarray(embedding, dtype=np.float32),
    )


def _item(
    *,
    sample_id: str = "b1",
    source_sentence_id: str = "sent-b1",
    text: str = "retrieved text",
    embedding: tuple[float, ...] = (1.0, 0.0),
) -> RetrievalBankItem:
    pose = _pose()
    return RetrievalBankItem(
        sample_id=sample_id,
        split=SampleSplit.TRAIN,
        text=text,
        source_video_id="video-bank",
        source_sentence_id=source_sentence_id,
        source_sentence_name=f"name_{source_sentence_id}",
        payload_ref=f"passed/train/{sample_id}.npz",
        frame_count=pose.frame_count,
        valid_frame_count=int(np.count_nonzero(pose.valid_frame_mask)),
        pose=pose,
        embedding=np.asarray(embedding, dtype=np.float32),
    )


def _pose(frame_count: int = 3) -> BfhPoseArrays:
    return BfhPoseArrays(
        body_xyc=_channel(frame_count, "body"),
        left_hand_xyc=_channel(frame_count, "left_hand"),
        right_hand_xyc=_channel(frame_count, "right_hand"),
        face_xyc=_channel(frame_count, "face"),
        valid_frame_mask=np.ones((frame_count,), dtype=np.bool_),
    )


def _channel(frame_count: int, channel: str) -> np.ndarray:
    joints = BFH_CHANNEL_SPECS[PoseChannel(channel)].joint_count
    array = np.zeros((frame_count, joints, 3), dtype=np.float32)
    array[..., 2] = 1.0
    return array
