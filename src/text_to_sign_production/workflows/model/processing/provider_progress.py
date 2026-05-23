"""Provider-scoped progress helpers for model stage internals."""

from __future__ import annotations

from dataclasses import dataclass

from text_to_sign_production.core.progress import ProgressSession, ProgressTaskHandle
from text_to_sign_production.core.progress.session import NoOpProgressTaskHandle
from text_to_sign_production.workflows.model.contracts import ModelWorkflowInvariantError
from text_to_sign_production.workflows.model.progress import model_progress_stage


@dataclass(frozen=True, slots=True)
class ModelProviderProgressTaskSpec:
    operation: str
    label: str
    unit: str
    total: int | None = None
    total_semantics: str | None = None
    allowed_counters: tuple[str, ...] = ()


class ModelProviderProgress:
    """Small provider-neutral adapter around the model workflow progress session."""

    def __init__(
        self,
        *,
        progress_session: ProgressSession | None,
        provider_key: str,
        provider_stage_id: str,
        run_mode: str | None = None,
    ) -> None:
        self._progress_session = progress_session
        self._provider_key = _path_token(provider_key)
        self._provider_stage_id = _path_token(provider_stage_id)
        self._run_mode = None if run_mode is None else str(run_mode)

    def status(self, message: str, **fields: object) -> None:
        if self._progress_session is None:
            return
        self._progress_session.status(
            message,
            provider_key=self._provider_key,
            provider_stage_id=self._provider_stage_id,
            **fields,
        )

    def task(
        self,
        *,
        operation: str,
        label: str,
        unit: str,
        total: int | None = None,
        total_semantics: str | None = None,
        allowed_counters: tuple[str, ...] = (),
    ) -> ProgressTaskHandle:
        if self._run_mode == "full" and total is None:
            raise ModelWorkflowInvariantError(
                "full-mode provider progress task has unknown total: "
                f"provider={self._provider_key} stage={self._provider_stage_id} "
                f"operation={operation}"
            )
        spec = self._stage_spec(
            ModelProviderProgressTaskSpec(
                operation=operation,
                label=label,
                unit=unit,
                total=total,
                total_semantics=total_semantics,
                allowed_counters=allowed_counters,
            )
        )
        if self._progress_session is None:
            return NoOpProgressTaskHandle(spec=spec)
        return self._progress_session.task(spec, total=total)

    def _stage_spec(self, spec: ModelProviderProgressTaskSpec):
        operation = _path_token(spec.operation)
        return model_progress_stage(
            stage_id=(
                f"model.provider.{self._provider_key}."
                f"{self._provider_stage_id}.{operation}"
            ),
            label=spec.label,
            unit=spec.unit,
            owner_module=__name__,
            operation_kind=operation,
            total_semantics=spec.total_semantics or f"{spec.label} {spec.unit}s",
            allowed_counters=spec.allowed_counters,
        )


def _path_token(value: str) -> str:
    return str(value).strip().replace(" ", "_")


__all__ = ["ModelProviderProgress", "ModelProviderProgressTaskSpec"]
