"""Workflow-owned epoch recovery persistence for Base M0 training runs.

The modeling layer emits ``TrainingEpochArtifacts`` via callback.
This module implements the workflow-side handler that:

* copies runtime-local metrics / live-log to the Drive recovery root
* copies last.pt (and best.pt when updated) as plain ``.pt`` files
* maintains a sha256 recovery manifest with per-artifact size + checksum

No subprocess, zstd, or archive logic is used during training.
Final publish/compression remains a notebook-facing post-training step.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from text_to_sign_production.core.files import sha256_file
from text_to_sign_production.core.progress import ProgressReporter

if TYPE_CHECKING:
    from text_to_sign_production.modeling.training.train import TrainingEpochArtifacts


@dataclass(frozen=True, slots=True)
class RecoveryArtifactEntry:
    """Auditable metadata for a single copied recovery artifact."""

    logical_name: str
    source_path: str
    source_size: int
    source_sha256: str
    recovery_path: str
    recovery_size: int
    recovery_sha256: str


def build_epoch_recovery_callback(
    *,
    recovery_root: Path | None,
    reporter: ProgressReporter | None = None,
) -> None | _EpochRecoveryHandler:
    """Build the on_epoch_artifacts callback for workflow recovery persistence.

    Returns ``None`` when no recovery root is configured (smoke / local modes).
    """

    if recovery_root is None:
        return None
    return _EpochRecoveryHandler(recovery_root=recovery_root, reporter=reporter)


class _EpochRecoveryHandler:
    """Callable handler that persists runtime artifacts to a recovery root."""

    __slots__ = ("_recovery_root", "_reporter")

    def __init__(
        self,
        *,
        recovery_root: Path,
        reporter: ProgressReporter | None = None,
    ) -> None:
        self._recovery_root = recovery_root.expanduser().resolve()
        self._reporter = reporter

    def __call__(self, artifacts: TrainingEpochArtifacts) -> None:
        """Persist epoch-local artifacts to the recovery root."""

        training_target = self._recovery_root / "training"
        checkpoint_target = self._recovery_root / "checkpoints"
        training_target.mkdir(parents=True, exist_ok=True)
        checkpoint_target.mkdir(parents=True, exist_ok=True)

        copied_entries: list[RecoveryArtifactEntry] = []

        # Copy metrics and live-log (always).
        copied_entries.append(
            _copy_and_record(
                logical_name="training/metrics.jsonl",
                source=artifacts.metrics_path,
                target=training_target / artifacts.metrics_path.name,
            )
        )
        copied_entries.append(
            _copy_and_record(
                logical_name="training/live.log",
                source=artifacts.live_log_path,
                target=training_target / artifacts.live_log_path.name,
            )
        )

        # Copy last.pt as plain checkpoint (always).
        copied_entries.append(
            _copy_and_record(
                logical_name="checkpoints/last.pt",
                source=artifacts.last_checkpoint_path,
                target=checkpoint_target / "last.pt",
            )
        )

        # Copy best.pt when improved.
        if artifacts.best_checkpoint_updated and artifacts.best_checkpoint_path is not None:
            copied_entries.append(
                _copy_and_record(
                    logical_name="checkpoints/best.pt",
                    source=artifacts.best_checkpoint_path,
                    target=checkpoint_target / "best.pt",
                )
            )

        # Write recovery manifest with per-artifact sha256 checksums.
        manifest = _recovery_manifest(artifacts, copied_entries=copied_entries)
        manifest_path = self._recovery_root / "recovery_manifest.json"
        _atomic_write_json(manifest_path, manifest)

        if self._reporter is not None:
            self._reporter.report(
                "recovery persistence",
                epoch=artifacts.epoch,
                target=self._recovery_root,
            )


def _copy_and_record(
    *,
    logical_name: str,
    source: Path,
    target: Path,
) -> RecoveryArtifactEntry:
    """Copy a file and build an auditable recovery entry.

    Raises on any copy or checksum failure (fail-closed).
    """

    source_size = source.stat().st_size
    source_hash = sha256_file(source)

    shutil.copyfile(source, target)

    recovery_size = target.stat().st_size
    recovery_hash = sha256_file(target)

    if source_size != recovery_size:
        raise RuntimeError(
            f"Recovery copy size mismatch for {logical_name}: "
            f"source={source_size}, recovery={recovery_size}"
        )
    if source_hash != recovery_hash:
        raise RuntimeError(
            f"Recovery copy sha256 mismatch for {logical_name}: "
            f"source={source_hash}, recovery={recovery_hash}"
        )

    return RecoveryArtifactEntry(
        logical_name=logical_name,
        source_path=str(source),
        source_size=source_size,
        source_sha256=source_hash,
        recovery_path=str(target),
        recovery_size=recovery_size,
        recovery_sha256=recovery_hash,
    )


def _recovery_manifest(
    artifacts: TrainingEpochArtifacts,
    *,
    copied_entries: list[RecoveryArtifactEntry],
) -> dict[str, object]:
    """Build a recovery manifest payload with per-artifact checksums."""

    manifest: dict[str, object] = {
        "epoch": artifacts.epoch,
        "run_mode": artifacts.run_mode,
        "best_metric_name": artifacts.best_metric_name,
        "best_metric_value": artifacts.best_metric_value,
        "best_epoch": artifacts.best_epoch,
        "copied_artifacts": [asdict(entry) for entry in copied_entries],
        "checkpoint_manifest_payload": dict(artifacts.checkpoint_manifest_payload),
    }
    return manifest


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    """Write JSON to a temporary file and atomically replace the target.

    Falls back to direct write when atomic rename is not supported.
    """

    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=".recovery_manifest_",
            suffix=".tmp",
        )
        try:
            os.write(fd, encoded.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp_path, str(path))
    except OSError:
        # Fall back to direct write if atomic replace is not possible
        # (e.g. cross-device on some mount configurations).
        with path.open("w", encoding="utf-8") as handle:
            handle.write(encoded)


__all__ = ["RecoveryArtifactEntry", "build_epoch_recovery_callback"]
