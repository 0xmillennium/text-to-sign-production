"""Stable hashes and cache keys for tensor data surfaces."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections.abc import Mapping


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json_payload(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def model_data_surface_cache_key(
    *,
    provider_key: str,
    surface_kind: str,
    split: str,
    source_manifest_sha256: str,
    provider_config_sha256: str,
    run_mode: str,
    limit_samples: int | None,
    data_version: str,
) -> str:
    digest = sha256_json_payload(
        {
            "provider_key": provider_key,
            "surface_kind": surface_kind,
            "split": split,
            "source_manifest_sha256": source_manifest_sha256,
            "provider_config_sha256": provider_config_sha256,
            "run_mode": run_mode,
            "limit_samples": limit_samples,
            "data_version": data_version,
        }
    )
    return digest[:32]


__all__ = [
    "model_data_surface_cache_key",
    "sha256_file",
    "sha256_json_payload",
]
