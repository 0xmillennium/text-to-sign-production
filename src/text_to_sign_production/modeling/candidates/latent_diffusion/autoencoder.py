"""Provider-owned temporal window autoencoder for latent_diffusion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from text_to_sign_production.modeling.training.precision import (
    ResolvedPrecisionPolicy,
    autocast_context,
    resolve_precision_policy,
)
from torch import nn

from text_to_sign_production.core.io import write_json
from text_to_sign_production.modeling.candidates.latent_diffusion.config import (
    LatentAutoencoderConfig,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.errors import (
    LatentDiffusionError,
)
from text_to_sign_production.modeling.data_surfaces import (
    ModelDataSurface,
    ModelDataSurfaceReader,
    surface_batch_count,
)


AUTOENCODER_CHECKPOINT_SCHEMA_VERSION = "t2sp-latent-window-autoencoder-checkpoint-v1"


@dataclass(frozen=True, slots=True)
class TemporalWindowAutoencoderOutput:
    latent: torch.Tensor
    reconstruction: torch.Tensor


class TemporalWindowAutoencoder(nn.Module):
    """Simple continuous MLP autoencoder over flattened standardized BFH windows."""

    def __init__(self, *, input_dim: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        for value, name in (
            (input_dim, "input_dim"),
            (latent_dim, "latent_dim"),
            (hidden_dim, "hidden_dim"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise LatentDiffusionError(f"{name} must be a positive integer.")
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.input_dim),
        )

    def forward(self, values: torch.Tensor) -> TemporalWindowAutoencoderOutput:
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise LatentDiffusionError(
                f"window autoencoder input must have shape (batch, {self.input_dim})."
            )
        latent = self.encoder(values)
        reconstruction = self.decoder(latent)
        return TemporalWindowAutoencoderOutput(latent=latent, reconstruction=reconstruction)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return self.forward(values).latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise LatentDiffusionError(
                f"window autoencoder latent must have shape (batch, {self.latent_dim})."
            )
        return self.decoder(latent)


def build_temporal_window_autoencoder(
    *,
    input_dim: int,
    config: LatentAutoencoderConfig,
) -> TemporalWindowAutoencoder:
    if not isinstance(config, LatentAutoencoderConfig):
        raise LatentDiffusionError("config must be a LatentAutoencoderConfig.")
    return TemporalWindowAutoencoder(
        input_dim=input_dim,
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
    )


def compute_window_autoencoder_losses(
    *,
    output: TemporalWindowAutoencoderOutput,
    target: torch.Tensor,
    validity_mask: torch.Tensor,
    window_size: int,
    feature_dim: int,
    velocity_loss_weight: float,
) -> dict[str, torch.Tensor]:
    if output.reconstruction.shape != target.shape:
        raise LatentDiffusionError("autoencoder reconstruction and target shapes must match.")
    if validity_mask.shape != target.shape:
        raise LatentDiffusionError("autoencoder validity_mask shape must match target.")
    mask = validity_mask.to(dtype=torch.bool)
    if not torch.any(mask):
        raise LatentDiffusionError("autoencoder validity_mask must contain valid features.")
    squared = (output.reconstruction - target).pow(2)
    reconstruction_loss = squared[mask].mean()
    if not torch.isfinite(reconstruction_loss):
        raise LatentDiffusionError("autoencoder reconstruction loss is not finite.")
    velocity_loss = target.new_tensor(0.0)
    if float(velocity_loss_weight) > 0.0:
        if target.shape[1] != int(window_size) * int(feature_dim):
            raise LatentDiffusionError(
                "autoencoder velocity loss target dimension does not match window_size * feature_dim."
            )
        reconstructed_windows = output.reconstruction.reshape(-1, window_size, feature_dim)
        target_windows = target.reshape(-1, window_size, feature_dim)
        mask_windows = mask.reshape(-1, window_size, feature_dim)
        predicted_velocity = reconstructed_windows[:, 1:] - reconstructed_windows[:, :-1]
        target_velocity = target_windows[:, 1:] - target_windows[:, :-1]
        velocity_mask = mask_windows[:, 1:] & mask_windows[:, :-1]
        if torch.any(velocity_mask):
            velocity_loss = (predicted_velocity - target_velocity).pow(2)[velocity_mask].mean()
            if not torch.isfinite(velocity_loss):
                raise LatentDiffusionError("autoencoder velocity loss is not finite.")
    total_loss = reconstruction_loss + float(velocity_loss_weight) * velocity_loss
    if not torch.isfinite(total_loss):
        raise LatentDiffusionError("autoencoder total loss is not finite.")
    return {
        "total_loss": total_loss,
        "reconstruction_loss": reconstruction_loss,
        "velocity_loss": velocity_loss,
    }


def train_autoencoder(
    *,
    model: TemporalWindowAutoencoder,
    train_values: np.ndarray,
    train_validity: np.ndarray,
    validation_values: np.ndarray | None,
    validation_validity: np.ndarray | None,
    config: LatentAutoencoderConfig,
    window_size: int,
    feature_dim: int,
    device: torch.device,
    progress=None,
    precision_policy: ResolvedPrecisionPolicy | None = None,
    num_workers: int = 0,
    runtime_trace: dict[str, object] | None = None,
) -> dict[str, float]:
    values = _array_2d(train_values, "train_values")
    validity = _mask_2d(train_validity, "train_validity", values.shape)
    if not np.any(validity):
        raise LatentDiffusionError("autoencoder train_validity contains no valid features.")
    resolved_device = torch.device(device)
    resolved_precision = precision_policy or resolve_precision_policy("auto", resolved_device)
    model.to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    final_train_loss = 0.0
    batch_count = math.ceil(values.shape[0] / config.batch_size)
    for epoch in range(config.max_epochs):
        model.train()
        batch_losses: list[float] = []
        with progress.task(
            operation=f"autoencoder_train_epoch_{epoch + 1}",
            label=f"latent_diffusion autoencoder train epoch {epoch + 1}/{config.max_epochs}",
            unit="batch",
            total=batch_count,
            total_semantics="latent autoencoder training batches in epoch",
            allowed_counters=("loss",),
        ) if progress is not None else _noop_task() as task:
            for start in range(0, values.shape[0], config.batch_size):
                stop = start + config.batch_size
                batch_values = torch.as_tensor(
                    values[start:stop],
                    dtype=torch.float32,
                    device=resolved_device,
                )
                batch_mask = torch.as_tensor(
                    validity[start:stop],
                    dtype=torch.bool,
                    device=resolved_device,
                )
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(resolved_precision):
                    output = model(batch_values)
                    losses = compute_window_autoencoder_losses(
                        output=output,
                        target=batch_values,
                        validity_mask=batch_mask,
                        window_size=window_size,
                        feature_dim=feature_dim,
                        velocity_loss_weight=config.velocity_loss_weight,
                    )
                losses["total_loss"].backward()
                optimizer.step()
                loss = float(losses["total_loss"].detach().cpu().item())
                batch_losses.append(loss)
                task.advance(1, counters={"loss": f"{loss:.6g}"})
        final_train_loss = float(np.mean(batch_losses))
    metrics = {"train_autoencoder_loss": _finite_float(final_train_loss, "train_autoencoder_loss")}
    if validation_values is not None and validation_validity is not None:
        validation = evaluate_autoencoder(
            model=model,
            values=validation_values,
            validity=validation_validity,
            config=config,
            window_size=window_size,
            feature_dim=feature_dim,
            device=resolved_device,
            progress=progress,
            precision_policy=resolved_precision,
        )
        metrics.update(validation)
    return metrics


def train_latent_autoencoder_from_surfaces(
    *,
    model: TemporalWindowAutoencoder,
    train_window_surface: ModelDataSurface | ModelDataSurfaceReader,
    validation_window_surface: ModelDataSurface | ModelDataSurfaceReader,
    config: LatentAutoencoderConfig,
    window_size: int,
    feature_dim: int,
    device: torch.device,
    progress=None,
    precision_policy: ResolvedPrecisionPolicy | None = None,
    num_workers: int = 0,
    runtime_trace: dict[str, object] | None = None,
) -> dict[str, float]:
    """Train the temporal autoencoder directly from latent window tensor surfaces."""

    if not isinstance(config, LatentAutoencoderConfig):
        raise LatentDiffusionError("config must be a LatentAutoencoderConfig.")
    train_reader = _surface_reader(train_window_surface, "train_window_surface")
    validation_reader = _surface_reader(validation_window_surface, "validation_window_surface")
    if train_reader.metadata.unit_count <= 0:
        raise LatentDiffusionError("train_window_surface must contain at least one window.")
    if validation_reader.metadata.unit_count <= 0:
        raise LatentDiffusionError("validation_window_surface must contain at least one window.")
    resolved_device = torch.device(device)
    resolved_precision = precision_policy or resolve_precision_policy("auto", resolved_device)
    model.to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    final_train_loss = 0.0
    batch_total = surface_batch_count(
        unit_count=train_reader.metadata.unit_count,
        batch_size=config.batch_size,
    )
    for epoch in range(config.max_epochs):
        model.train()
        loss_sum = 0.0
        seen_batches = 0
        with progress.task(
            operation=f"autoencoder_train_epoch_{epoch + 1}",
            label=f"latent_diffusion autoencoder train epoch {epoch + 1}/{config.max_epochs}",
            unit="batch",
            total=batch_total,
            total_semantics="latent autoencoder training batches in epoch",
            allowed_counters=("loss",),
        ) if progress is not None else _noop_task() as task:
            for batch in train_reader.iter_batches(
                batch_size=config.batch_size,
                shuffle_shards=True,
                shuffle_units=True,
                num_workers=num_workers,
                runtime_trace=runtime_trace,
            ):
                batch_values, batch_mask = _surface_window_batch(
                    batch,
                    window_size=window_size,
                    feature_dim=feature_dim,
                    device=resolved_device,
                )
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(resolved_precision):
                    output = model(batch_values)
                    losses = compute_window_autoencoder_losses(
                        output=output,
                        target=batch_values,
                        validity_mask=batch_mask,
                        window_size=window_size,
                        feature_dim=feature_dim,
                        velocity_loss_weight=config.velocity_loss_weight,
                    )
                losses["total_loss"].backward()
                optimizer.step()
                loss = float(losses["total_loss"].detach().cpu().item())
                loss_sum += loss
                seen_batches += 1
                task.advance(1, counters={"loss": f"{loss:.6g}"})
        final_train_loss = loss_sum / max(seen_batches, 1)
    metrics = {"train_autoencoder_loss": _finite_float(final_train_loss, "train_autoencoder_loss")}
    metrics.update(
        evaluate_latent_autoencoder_from_surface(
            model=model,
            window_surface=validation_reader,
            config=config,
            window_size=window_size,
            feature_dim=feature_dim,
            device=resolved_device,
            progress=progress,
            precision_policy=resolved_precision,
            num_workers=num_workers,
        )
    )
    return metrics


def evaluate_latent_autoencoder_from_surface(
    *,
    model: TemporalWindowAutoencoder,
    window_surface: ModelDataSurface | ModelDataSurfaceReader,
    config: LatentAutoencoderConfig,
    window_size: int,
    feature_dim: int,
    device: torch.device,
    progress=None,
    precision_policy: ResolvedPrecisionPolicy | None = None,
    num_workers: int = 0,
) -> dict[str, float]:
    """Evaluate the temporal autoencoder without materializing all windows."""

    reader = _surface_reader(window_surface, "window_surface")
    resolved_device = torch.device(device)
    resolved_precision = precision_policy or resolve_precision_policy("auto", resolved_device)
    loss_sum = 0.0
    seen_batches = 0
    model.eval()
    with torch.no_grad():
        with progress.task(
            operation="autoencoder_validation",
            label="latent_diffusion autoencoder validation",
            unit="batch",
            total=surface_batch_count(
                unit_count=reader.metadata.unit_count,
                batch_size=config.batch_size,
            ),
            total_semantics="latent autoencoder validation batches",
            allowed_counters=("loss",),
        ) if progress is not None else _noop_task() as task:
            for batch in reader.iter_batches(
                batch_size=config.batch_size,
                num_workers=num_workers,
            ):
                batch_values, batch_mask = _surface_window_batch(
                    batch,
                    window_size=window_size,
                    feature_dim=feature_dim,
                    device=resolved_device,
                )
                with autocast_context(resolved_precision):
                    output = model(batch_values)
                    batch_losses = compute_window_autoencoder_losses(
                        output=output,
                        target=batch_values,
                        validity_mask=batch_mask,
                        window_size=window_size,
                        feature_dim=feature_dim,
                        velocity_loss_weight=config.velocity_loss_weight,
                    )
                loss = float(batch_losses["total_loss"].detach().cpu().item())
                loss_sum += loss
                seen_batches += 1
                task.advance(1, counters={"loss": f"{loss:.6g}"})
    return {
        "validation_autoencoder_loss": _finite_float(
            loss_sum / max(seen_batches, 1),
            "validation_autoencoder_loss",
        )
    }


def evaluate_autoencoder(
    *,
    model: TemporalWindowAutoencoder,
    values: np.ndarray,
    validity: np.ndarray,
    config: LatentAutoencoderConfig,
    window_size: int,
    feature_dim: int,
    device: torch.device,
    progress=None,
    precision_policy: ResolvedPrecisionPolicy | None = None,
) -> dict[str, float]:
    data = _array_2d(values, "values")
    mask = _mask_2d(validity, "validity", data.shape)
    resolved_device = torch.device(device)
    resolved_precision = precision_policy or resolve_precision_policy("auto", resolved_device)
    losses: list[float] = []
    model.eval()
    with torch.no_grad():
        with progress.task(
            operation="autoencoder_validation",
            label="latent_diffusion autoencoder validation",
            unit="batch",
            total=math.ceil(data.shape[0] / config.batch_size),
            total_semantics="latent autoencoder validation batches",
            allowed_counters=("loss",),
        ) if progress is not None else _noop_task() as task:
            for start in range(0, data.shape[0], config.batch_size):
                stop = start + config.batch_size
                batch_values = torch.as_tensor(data[start:stop], dtype=torch.float32, device=resolved_device)
                batch_mask = torch.as_tensor(mask[start:stop], dtype=torch.bool, device=resolved_device)
                with autocast_context(resolved_precision):
                    output = model(batch_values)
                    batch_losses = compute_window_autoencoder_losses(
                        output=output,
                        target=batch_values,
                        validity_mask=batch_mask,
                        window_size=window_size,
                        feature_dim=feature_dim,
                        velocity_loss_weight=config.velocity_loss_weight,
                    )
                loss = float(batch_losses["total_loss"].detach().cpu().item())
                losses.append(loss)
                task.advance(1, counters={"loss": f"{loss:.6g}"})
    return {"validation_autoencoder_loss": _finite_float(float(np.mean(losses)), "validation_autoencoder_loss")}


def encode_window_values(
    *,
    model: TemporalWindowAutoencoder,
    values: np.ndarray,
    batch_size: int,
    device: torch.device,
    progress_task=None,
) -> np.ndarray:
    data = _array_2d(values, "values")
    resolved_device = torch.device(device)
    encoded = np.empty((data.shape[0], model.latent_dim), dtype=np.float32)
    cursor = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, data.shape[0], batch_size):
            batch = torch.as_tensor(
                data[start : start + batch_size],
                dtype=torch.float32,
                device=resolved_device,
            )
            batch_latents = model.encode(batch).detach().cpu().numpy().astype(np.float32)
            encoded[cursor : cursor + batch_latents.shape[0]] = batch_latents
            cursor += batch_latents.shape[0]
            if progress_task is not None:
                progress_task.advance(batch.shape[0])
    if not np.all(np.isfinite(encoded)):
        raise LatentDiffusionError("encoded window latents contain non-finite values.")
    return encoded


class _NoOpTask:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exception, traceback) -> None:
        return

    def advance(self, count: int = 1, *, counters=None) -> None:
        del count, counters


def _noop_task() -> _NoOpTask:
    return _NoOpTask()


def decode_window_latents(
    *,
    model: TemporalWindowAutoencoder,
    latents: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    data = _array_2d(latents, "latents")
    resolved_device = torch.device(device)
    decoded = np.empty((data.shape[0], model.input_dim), dtype=np.float32)
    cursor = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, data.shape[0], batch_size):
            batch = torch.as_tensor(
                data[start : start + batch_size],
                dtype=torch.float32,
                device=resolved_device,
            )
            batch_windows = model.decode(batch).detach().cpu().numpy().astype(np.float32)
            decoded[cursor : cursor + batch_windows.shape[0]] = batch_windows
            cursor += batch_windows.shape[0]
    if not np.all(np.isfinite(decoded)):
        raise LatentDiffusionError("decoded window values contain non-finite values.")
    return decoded


def write_autoencoder_checkpoint(
    *,
    path: Path,
    model: TemporalWindowAutoencoder,
    config: LatentAutoencoderConfig,
    input_dim: int,
    target_spec: MappingLike,
    metrics: dict[str, float],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": AUTOENCODER_CHECKPOINT_SCHEMA_VERSION,
            "architecture": config.architecture,
            "input_dim": int(input_dim),
            "latent_dim": config.latent_dim,
            "hidden_dim": config.hidden_dim,
            "config": config.to_dict(),
            "target_spec": target_spec.to_dict(),
            "state_dict": model.state_dict(),
            "metrics": metrics,
        },
        output_path,
    )


def write_autoencoder_config(path: Path, config: LatentAutoencoderConfig) -> None:
    write_json(
        Path(path),
        {
            "schema_version": "t2sp-latent-window-autoencoder-config-v1",
            **config.to_dict(),
        },
    )


def load_autoencoder_checkpoint(
    path: Path,
    *,
    config: LatentAutoencoderConfig,
    input_dim: int,
) -> TemporalWindowAutoencoder:
    try:
        checkpoint = torch.load(Path(path), map_location="cpu")
    except (OSError, RuntimeError, ValueError) as exc:
        raise LatentDiffusionError(f"autoencoder checkpoint could not be loaded: {path}: {exc}") from exc
    if not isinstance(checkpoint, dict):
        raise LatentDiffusionError("autoencoder checkpoint root must be a mapping.")
    if checkpoint.get("schema_version") != AUTOENCODER_CHECKPOINT_SCHEMA_VERSION:
        raise LatentDiffusionError("autoencoder checkpoint schema_version is unsupported.")
    if checkpoint.get("input_dim") != int(input_dim):
        raise LatentDiffusionError("autoencoder checkpoint input_dim does not match target spec.")
    if checkpoint.get("latent_dim") != config.latent_dim:
        raise LatentDiffusionError("autoencoder checkpoint latent_dim does not match config.")
    model = build_temporal_window_autoencoder(input_dim=input_dim, config=config)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def _surface_reader(
    surface: ModelDataSurface | ModelDataSurfaceReader,
    name: str,
) -> ModelDataSurfaceReader:
    if isinstance(surface, ModelDataSurfaceReader):
        return surface
    if isinstance(surface, ModelDataSurface):
        return ModelDataSurfaceReader(surface)
    raise LatentDiffusionError(f"{name} must be a ModelDataSurface or ModelDataSurfaceReader.")


def _surface_window_batch(
    batch: dict[str, object],
    *,
    window_size: int,
    feature_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    windows = batch.get("windows")
    mask = batch.get("window_mask")
    if not isinstance(windows, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise LatentDiffusionError("latent window surface batch is missing windows/window_mask tensors.")
    if windows.ndim != 3 or windows.shape[1:] != (int(window_size), int(feature_dim)):
        raise LatentDiffusionError(
            "latent window surface windows must have shape "
            f"(batch, {int(window_size)}, {int(feature_dim)})."
        )
    if mask.shape != windows.shape:
        raise LatentDiffusionError("latent window surface window_mask must match windows.")
    flattened = windows.to(device=device, dtype=torch.float32).reshape(windows.shape[0], -1)
    flattened_mask = mask.to(device=device, dtype=torch.bool).reshape(mask.shape[0], -1)
    if not torch.any(flattened_mask):
        raise LatentDiffusionError("latent window surface batch contains no valid features.")
    return flattened, flattened_mask


def _array_2d(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] <= 0:
        raise LatentDiffusionError(f"{name} must be a non-empty 2D array.")
    if not np.all(np.isfinite(array)):
        raise LatentDiffusionError(f"{name} must contain only finite values.")
    return np.array(array, dtype=np.float32, copy=True)


def _mask_2d(value: np.ndarray, name: str, shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(value, dtype=np.bool_)
    if mask.shape != shape:
        raise LatentDiffusionError(f"{name} must have shape {shape}; got {mask.shape}.")
    return np.array(mask, dtype=np.bool_, copy=True)


def _finite_float(value: float, name: str) -> float:
    if not np.isfinite(float(value)):
        raise LatentDiffusionError(f"{name} must be finite.")
    return float(value)


class MappingLike(Protocol):
    def to_dict(self) -> dict[str, object]: ...


__all__ = [
    "AUTOENCODER_CHECKPOINT_SCHEMA_VERSION",
    "TemporalWindowAutoencoder",
    "TemporalWindowAutoencoderOutput",
    "build_temporal_window_autoencoder",
    "compute_window_autoencoder_losses",
    "decode_window_latents",
    "encode_window_values",
    "evaluate_autoencoder",
    "evaluate_latent_autoencoder_from_surface",
    "load_autoencoder_checkpoint",
    "train_autoencoder",
    "train_latent_autoencoder_from_surfaces",
    "write_autoencoder_checkpoint",
    "write_autoencoder_config",
]
