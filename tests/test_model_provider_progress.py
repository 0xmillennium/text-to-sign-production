from __future__ import annotations

from types import TracebackType

import pytest

from text_to_sign_production.core.progress import ProgressSession, ProgressStageSpec
from text_to_sign_production.workflows.model.constants import MODEL_WORKFLOW_NAME
from text_to_sign_production.workflows.model.processing.provider_progress import (
    ModelProviderProgress,
)


@pytest.mark.unit
def test_model_provider_progress_noops_without_session() -> None:
    progress = ModelProviderProgress(
        progress_session=None,
        provider_key="learned_pose_token",
        provider_stage_id="fit_representation",
    )

    progress.status("ignored")
    with progress.task(operation="materialize_train", label="load", unit="sample") as task:
        task.advance()


@pytest.mark.unit
def test_model_provider_progress_task_ids_are_provider_stage_scoped() -> None:
    sink = RecordingProgressSink()
    progress = ModelProviderProgress(
        progress_session=ProgressSession(workflow_id=MODEL_WORKFLOW_NAME, sink=sink),
        provider_key="learned_pose_token",
        provider_stage_id="fit_representation",
    )

    with progress.task(
        operation="materialize_train",
        label="learned_pose_token fit train sources",
        unit="sample",
        total=3,
    ) as task:
        task.advance()

    assert sink.opened_stage_ids == [
        "model.provider.learned_pose_token.fit_representation.materialize_train"
    ]
    assert sink.advances == [
        (
            "model.provider.learned_pose_token.fit_representation.materialize_train",
            1,
            {},
        )
    ]


class RecordingProgressSink:
    def __init__(self) -> None:
        self.opened_stage_ids: list[str] = []
        self.advances: list[tuple[str, int, dict[str, object]]] = []

    def open_task(self, spec: ProgressStageSpec, *, total: int | None = None) -> "RecordingTask":
        del total
        self.opened_stage_ids.append(spec.stage_id)
        return RecordingTask(spec, self)

    def status(self, message: str, **fields: object) -> None:
        del message, fields


class RecordingTask:
    def __init__(self, spec: ProgressStageSpec, sink: RecordingProgressSink) -> None:
        self.spec = spec
        self.sink = sink

    def __enter__(self) -> "RecordingTask":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exception, traceback

    def advance(self, count: int = 1, *, counters: dict[str, object] | None = None) -> None:
        self.sink.advances.append((self.spec.stage_id, count, counters or {}))

    def status(self, message: str, **fields: object) -> None:
        del message, fields

    def close(self) -> None:
        return
