from __future__ import annotations

from pathlib import Path

from text_to_sign_production.modeling.data_surfaces import resolve_effective_manifest_count


def test_full_manifest_count_has_exact_total(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, 5)

    assert resolve_effective_manifest_count(manifest_path=manifest, limit_samples=None) == 5


def test_limited_manifest_count_uses_minimum(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, 5)

    assert resolve_effective_manifest_count(manifest_path=manifest, limit_samples=3) == 3
    assert resolve_effective_manifest_count(manifest_path=manifest, limit_samples=9) == 5


def _manifest(tmp_path: Path, count: int) -> Path:
    path = tmp_path / "manifest.jsonl"
    path.write_text("".join("{}\n" for _ in range(count)), encoding="utf-8")
    return path

