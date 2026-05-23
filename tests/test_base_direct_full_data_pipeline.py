from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

from torch.utils.data import DataLoader

from text_to_sign_production.modeling.candidates.base_direct.provider import BaseDirectProvider
from text_to_sign_production.modeling.candidates.base_direct import trainer
from text_to_sign_production.modeling.data import dataset as dataset_module


def test_base_direct_full_pipeline_is_verified_lazy_dataloader() -> None:
    capability = BaseDirectProvider().full_data_pipeline_capability

    assert capability.full_training_data_mode == "lazy_dataloader"
    assert capability.verified is True
    assert capability.is_full_safe
    assert capability.verification_evidence


def test_base_direct_full_path_delegates_to_lazy_training_loop_inputs() -> None:
    source = inspect.getsource(trainer.run_base_direct_training_stage)

    assert "train_manifest_override=paths.train_manifest" in source
    assert "val_manifest_override=paths.validation_manifest" in source
    assert "load_manifest_sample(" not in source


def test_base_direct_dataset_construction_does_not_load_payloads(monkeypatch) -> None:
    records = tuple(
        SimpleNamespace(sample_id=f"sample-{index}", sample_path=Path(f"sample-{index}.npz"))
        for index in range(8)
    )
    calls: list[str] = []

    monkeypatch.setattr(
        dataset_module,
        "read_processed_modeling_manifest",
        lambda *args, **kwargs: records,
    )

    def load_payload(record):
        calls.append(record.sample_id)
        return object()

    monkeypatch.setattr(dataset_module, "load_processed_pose_sample", load_payload)
    monkeypatch.setattr(
        dataset_module.ProcessedPoseItem,
        "from_manifest_and_sample",
        staticmethod(lambda record, sample: {"sample_id": record.sample_id, "sample": sample}),
    )

    dataset = dataset_module.ProcessedPoseDataset(Path("manifest.jsonl"), split="train")
    loader = DataLoader(dataset, batch_size=2, collate_fn=lambda batch: batch)

    assert calls == []
    first_batch = next(iter(loader))
    assert len(first_batch) == 2
    assert calls == ["sample-0", "sample-1"]
