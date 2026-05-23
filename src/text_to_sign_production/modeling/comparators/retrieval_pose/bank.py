"""Retrieval-bank construction from modeling manifest samples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from text_to_sign_production.artifacts.store import ArtifactTopology
from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.backbones.text_encoder import TextEncoder
from text_to_sign_production.modeling.comparators.retrieval_pose.errors import (
    RetrievalPoseComparatorError,
)
from text_to_sign_production.modeling.data import ModelingManifestFamily, load_manifest_samples
from text_to_sign_production.modeling.data.bfh_schema import BfhPoseArrays


@dataclass(frozen=True, slots=True)
class RetrievalBankItem:
    sample_id: str
    split: SampleSplit
    text: str
    source_video_id: str
    source_sentence_id: str
    source_sentence_name: str
    payload_ref: str
    frame_count: int
    valid_frame_count: int
    pose: BfhPoseArrays
    embedding: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "split", SampleSplit(self.split))
        embedding = np.asarray(self.embedding, dtype=np.float32).copy()
        if embedding.ndim != 1 or embedding.shape[0] < 1:
            raise RetrievalPoseComparatorError("retrieval embeddings must be one-dimensional.")
        if not np.all(np.isfinite(embedding)):
            raise RetrievalPoseComparatorError("retrieval text embeddings contain non-finite values.")
        embedding.setflags(write=False)
        object.__setattr__(self, "embedding", embedding)
        if not isinstance(self.pose, BfhPoseArrays):
            raise RetrievalPoseComparatorError("retrieval bank item pose must be BfhPoseArrays.")


@dataclass(frozen=True, slots=True)
class RetrievalBank:
    split: SampleSplit
    items: tuple[RetrievalBankItem, ...]
    embedding_matrix: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "split", SampleSplit(self.split))
        if not self.items:
            raise RetrievalPoseComparatorError(
                f"retrieval bank for split {self.split.value!r} is empty."
            )
        matrix = np.asarray(self.embedding_matrix, dtype=np.float32).copy()
        if matrix.ndim != 2 or matrix.shape[0] != len(self.items):
            raise RetrievalPoseComparatorError("retrieval bank embedding matrix shape is invalid.")
        if not np.all(np.isfinite(matrix)):
            raise RetrievalPoseComparatorError("retrieval text embeddings contain non-finite values.")
        dims = {int(item.embedding.shape[0]) for item in self.items}
        if len(dims) != 1 or matrix.shape[1] not in dims:
            raise RetrievalPoseComparatorError("retrieval bank embedding dimensions are inconsistent.")
        matrix.setflags(write=False)
        object.__setattr__(self, "embedding_matrix", matrix)


def build_retrieval_bank(
    *,
    topology: ArtifactTopology,
    manifest_family: ModelingManifestFamily,
    split: SampleSplit | str,
    text_encoder: TextEncoder,
    max_bank_size: int | None,
) -> RetrievalBank:
    resolved_split = SampleSplit(split)
    samples = load_manifest_samples(
        topology,
        manifest_family,
        resolved_split,
        limit=max_bank_size,
    )
    if not samples:
        raise RetrievalPoseComparatorError(
            f"retrieval bank for split {resolved_split.value!r} is empty."
        )
    outputs = text_encoder.encode_batch(
        [sample.text for sample in samples],
        sentence_names=[sample.source_sentence_name for sample in samples],
    )
    items = tuple(
        RetrievalBankItem(
            sample_id=sample.sample_id,
            split=sample.split,
            text=sample.text,
            source_video_id=sample.source_video_id,
            source_sentence_id=sample.source_sentence_id,
            source_sentence_name=sample.source_sentence_name,
            payload_ref=sample.payload_ref,
            frame_count=sample.frame_count,
            valid_frame_count=int(np.count_nonzero(sample.pose.valid_frame_mask)),
            pose=sample.pose,
            embedding=output.embedding,
        )
        for sample, output in zip(samples, outputs, strict=True)
    )
    matrix = np.stack([item.embedding for item in items], axis=0)
    return RetrievalBank(split=resolved_split, items=items, embedding_matrix=matrix)


__all__ = ["RetrievalBank", "RetrievalBankItem", "build_retrieval_bank"]
