"""Path and filesystem helpers shared by tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import text_to_sign_production.core.paths as paths_mod


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def patch_dataset_paths(monkeypatch: Any, root: Path) -> None:
    monkeypatch.setattr(paths_mod, "DEFAULT_REPO_ROOT", root.resolve())
    monkeypatch.setattr(paths_mod, "DEFAULT_DATA_ROOT", (root / "data").resolve())
