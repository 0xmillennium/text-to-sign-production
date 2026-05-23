from __future__ import annotations

from pathlib import Path

import pytest
import torch

from text_to_sign_production.modeling.data_surfaces import (
    ModelDataSurfaceError,
    ModelDataSurfaceReader,
    ModelDataSurfaceWriter,
)


def test_reader_checks_shard_sha256_and_rejects_corruption(tmp_path: Path) -> None:
    surface = _surface(tmp_path)
    shard = surface.root / "shards" / "shard_00000.pt"
    shard.write_bytes(b"corrupt")

    with pytest.raises(ModelDataSurfaceError, match="checksum mismatch"):
        list(ModelDataSurfaceReader(surface).iter_batches(batch_size=1))


def test_reader_rejects_python_object_tuple_payload(tmp_path: Path) -> None:
    surface = _surface(tmp_path)
    shard = surface.root / "shards" / "shard_00000.pt"
    torch.save({"values": (object(),), "metadata": {}}, shard)

    with pytest.raises(ModelDataSurfaceError, match="not tensor/array-heavy|could not be loaded"):
        list(ModelDataSurfaceReader(surface).iter_shards(verify_sha256=False))


def _surface(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
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
        manifest_entry_count=1,
        run_mode="debug",
        data_version="test",
    )
    writer.append_units(
        {
            "values": torch.ones((1, 2)),
            "validity_mask": torch.ones((1, 2), dtype=torch.bool),
            "source_index": torch.zeros((1,), dtype=torch.long),
        },
        sample_count=1,
        frame_count=1,
    )
    return writer.close()
