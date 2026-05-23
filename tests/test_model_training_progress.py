from __future__ import annotations

from pathlib import Path
from types import TracebackType

import pytest

from text_to_sign_production.core.progress import ProgressSession, ProgressStageSpec
from text_to_sign_production.modeling.training.events import (
    CheckpointSaved,
    EpochCompleted,
    TrainingBatchProcessed,
    ValidationBatchProcessed,
)
from text_to_sign_production.modeling.training.logging import RunStarted, TextTrainingRunLogSink
from text_to_sign_production.workflows.model.constants import MODEL_WORKFLOW_NAME
from text_to_sign_production.workflows.model.processing.training_progress import (
    ModelWorkflowTrainingProgressSink,
)


@pytest.mark.unit
def test_text_training_run_log_sink_file_only_writes_without_stdout(tmp_path: Path) -> None:
    log_path = tmp_path / "live.log"
    sink = TextTrainingRunLogSink(stream=None, log_path=log_path)

    sink.emit(RunStarted(run_mode="smoke", training_surface="broad", validation_surface="broad"))

    assert log_path.is_file()
    assert "run start" in log_path.read_text(encoding="utf-8")


@pytest.mark.unit
def test_training_progress_sink_emits_batches_and_epoch_status_but_not_checkpoints() -> None:
    recording_sink = RecordingProgressSink()
    progress = ProgressSession(workflow_id=MODEL_WORKFLOW_NAME, sink=recording_sink)
    sink = ModelWorkflowTrainingProgressSink(
        progress_session=progress,
        provider_label="base_direct",
    )

    sink.emit(
        TrainingBatchProcessed(
            epoch_index=1,
            epoch_count=1,
            batch_index=1,
            batch_count=1,
            valid_frame_count=4,
            batch_loss=0.25,
            running_loss=0.25,
        )
    )
    sink.emit(
        ValidationBatchProcessed(
            epoch_index=1,
            epoch_count=1,
            batch_index=1,
            batch_count=1,
            valid_frame_count=4,
            valid_point_count=20,
            batch_loss=0.2,
            batch_metric=0.1,
            running_loss=0.2,
        )
    )
    sink.emit(
        EpochCompleted(
            epoch_index=1,
            epoch_count=1,
            train_loss=0.25,
            validation_loss=0.2,
            validation_metric=0.1,
            elapsed_seconds=3.5,
            best_checkpoint_updated=True,
        )
    )
    sink.emit(CheckpointSaved(checkpoint_path=Path("checkpoint.pt"), role="last", epoch=1))

    assert "model.training.train_epoch.1" in recording_sink.opened_stage_ids
    assert "model.training.val_epoch.1" in recording_sink.opened_stage_ids
    assert any(message == "base_direct epoch summary" for message, _ in recording_sink.statuses)
    assert not any("checkpoint" in message for message, _ in recording_sink.statuses)


class RecordingProgressSink:
    def __init__(self) -> None:
        self.opened_stage_ids: list[str] = []
        self.advances: list[tuple[str, int, dict[str, object]]] = []
        self.statuses: list[tuple[str, dict[str, object]]] = []

    def open_task(self, spec: ProgressStageSpec, *, total: int | None = None) -> "RecordingTask":
        self.opened_stage_ids.append(spec.stage_id)
        return RecordingTask(spec, self)

    def status(self, message: str, **fields: object) -> None:
        self.statuses.append((message, dict(fields)))


class RecordingTask:
    def __init__(self, spec: ProgressStageSpec, sink: RecordingProgressSink) -> None:
        self.spec = spec
        self.sink = sink
        self.closed = False

    def __enter__(self) -> "RecordingTask":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def advance(self, count: int = 1, *, counters: dict[str, object] | None = None) -> None:
        self.sink.advances.append((self.spec.stage_id, count, counters or {}))

    def status(self, message: str, **fields: object) -> None:
        self.sink.status(message, **fields)

    def close(self) -> None:
        self.closed = True
