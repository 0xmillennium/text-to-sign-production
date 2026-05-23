from __future__ import annotations

from pathlib import Path

import torch

from text_to_sign_production.modeling.data_surfaces import (
    MODEL_DATA_SURFACE_SCHEMA_VERSION,
    ModelDataSurfaceReader,
    ModelDataSurfaceWriter,
)


def test_model_data_surface_writer_reader_round_trip(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n{}\n", encoding="utf-8")
    writer = ModelDataSurfaceWriter(
        root=tmp_path / "surface",
        provider_key="learned_pose_token",
        surface_kind="pose_token_units",
        split="train",
        manifest_family="tiered:clean:included",
        source_manifest_path=manifest,
        source_manifest_sha256="a" * 64,
        provider_config_sha256="b" * 64,
        cache_key="c" * 32,
        manifest_entry_count=2,
        run_mode="debug",
        data_version="test",
        max_units_per_shard=3,
        max_source_samples_per_shard=1,
    )
    writer.append_units(
        {
            "values": torch.ones((2, 4)),
            "validity_mask": torch.ones((2, 4), dtype=torch.bool),
            "source_index": torch.zeros((2,), dtype=torch.long),
        },
        sample_count=1,
        frame_count=2,
    )
    writer.append_units(
        {
            "values": torch.zeros((1, 4)),
            "validity_mask": torch.ones((1, 4), dtype=torch.bool),
            "source_index": torch.ones((1,), dtype=torch.long),
        },
        sample_count=1,
        frame_count=1,
    )
    surface = writer.close()

    assert surface.metadata.schema_version == MODEL_DATA_SURFACE_SCHEMA_VERSION
    assert surface.metadata.loaded_sample_count == 2
    assert surface.metadata.unit_count == 3
    assert surface.metadata.shard_count == 2
    reader = ModelDataSurfaceReader(surface)
    records = reader.shard_records()
    assert [record.unit_count for record in records] == [2, 1]
    batches = list(reader.iter_batches(batch_size=2))
    assert [int(batch["values"].shape[0]) for batch in batches] == [2, 1]

