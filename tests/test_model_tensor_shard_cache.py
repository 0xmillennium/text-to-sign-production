from __future__ import annotations

from pathlib import Path

import pytest
import torch

from text_to_sign_production.modeling.data_surfaces import (
    ModelDataSurfaceError,
    ModelDataSurfaceReader,
    ModelDataSurfaceWriter,
)


def test_tensor_shard_cache_is_data_surface_layout(tmp_path: Path) -> None:
    surface = _surface(tmp_path)

    assert surface.metadata_path.name == "metadata.json"
    assert surface.manifest_path.name == "manifest.jsonl"
    assert surface.shard_root.name == "shards"
    assert list(surface.shard_root.glob("*.pt"))


def test_tensor_shard_cache_fails_fast_on_sha_mismatch(tmp_path: Path) -> None:
    surface = _surface(tmp_path)
    (surface.shard_root / "shard_00000.pt").write_bytes(b"broken")

    with pytest.raises(ModelDataSurfaceError, match="checksum mismatch"):
        list(ModelDataSurfaceReader(surface).iter_shards())


def _surface(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    writer = ModelDataSurfaceWriter(
        root=tmp_path / "runtime/model/data_surfaces/run/provider/kind/train",
        provider_key="provider",
        surface_kind="kind",
        split="train",
        manifest_family="family",
        source_manifest_path=manifest,
        source_manifest_sha256="a" * 64,
        provider_config_sha256="b" * 64,
        cache_key="c" * 32,
        manifest_entry_count=1,
        run_mode="full",
        data_version="test",
    )
    writer.append_units({"values": torch.ones((1, 1))}, sample_count=1, frame_count=1)
    return writer.close()
