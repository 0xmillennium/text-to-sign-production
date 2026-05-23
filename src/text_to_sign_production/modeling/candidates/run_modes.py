"""Shared provider-consumable model run-mode policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from text_to_sign_production.modeling.candidates.configs import ModelRunMode
from text_to_sign_production.modeling.candidates.errors import ModelCandidateError


@dataclass(frozen=True, slots=True)
class ModelRunModePolicy:
    """Explicit limits and quality claim for one requested run scale."""

    mode: ModelRunMode
    limit_train_samples: int | None
    limit_validation_samples: int | None
    limit_prediction_samples: int | None
    max_epochs: int | None
    min_epochs: int | None
    early_stopping_patience: int | None
    batch_size: int | None
    num_workers: int | None
    persistent_workers: bool | None
    prefetch_factor: int | None
    quality_claim: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", ModelRunMode(self.mode))
        if not isinstance(self.quality_claim, str) or not self.quality_claim.strip():
            raise ModelCandidateError("run-mode quality_claim must be non-empty.")

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-serializable policy record."""

        payload = asdict(self)
        payload["mode"] = self.mode.value
        return payload


_RUN_MODE_POLICIES = {
    ModelRunMode.SMOKE: ModelRunModePolicy(
        mode=ModelRunMode.SMOKE,
        limit_train_samples=4,
        limit_validation_samples=2,
        limit_prediction_samples=2,
        max_epochs=1,
        min_epochs=None,
        early_stopping_patience=None,
        batch_size=1,
        num_workers=0,
        persistent_workers=False,
        prefetch_factor=None,
        quality_claim="wiring_only",
    ),
    ModelRunMode.DEBUG: ModelRunModePolicy(
        mode=ModelRunMode.DEBUG,
        limit_train_samples=64,
        limit_validation_samples=16,
        limit_prediction_samples=16,
        max_epochs=3,
        min_epochs=None,
        early_stopping_patience=None,
        batch_size=None,
        num_workers=0,
        persistent_workers=False,
        prefetch_factor=None,
        quality_claim="diagnostic_only",
    ),
    ModelRunMode.FULL: ModelRunModePolicy(
        mode=ModelRunMode.FULL,
        limit_train_samples=None,
        limit_validation_samples=None,
        limit_prediction_samples=None,
        max_epochs=None,
        min_epochs=None,
        early_stopping_patience=None,
        batch_size=None,
        num_workers=None,
        persistent_workers=None,
        prefetch_factor=None,
        quality_claim="config_governed_validation",
    ),
}


def resolve_model_run_mode_policy(mode: ModelRunMode | str) -> ModelRunModePolicy:
    """Resolve shared run limits and reporting semantics for a run mode."""

    try:
        resolved = ModelRunMode(mode)
    except (TypeError, ValueError) as exc:
        raise ModelCandidateError(f"unknown model run mode: {mode!r}") from exc
    return _RUN_MODE_POLICIES[resolved]


__all__ = ["ModelRunModePolicy", "resolve_model_run_mode_policy"]
