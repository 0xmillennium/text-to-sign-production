"""Runtime support manifest contracts for model/test_model restore."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.core.integrity import sha256_file
from text_to_sign_production.modeling.research import ModelKey

MODEL_RUNTIME_SUPPORT_SCHEMA_VERSION = "model.runtime_support.v1"


@dataclass(frozen=True, slots=True)
class ModelRuntimeSupportArtifact:
    role: str
    kind: str
    relative_path: Path
    required_for_test_model: bool
    sha256: str
    size_bytes: int
    provider_key: str
    description: str | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.role, "role"),
            (self.kind, "kind"),
            (self.sha256, "sha256"),
            (self.provider_key, "provider_key"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"runtime support artifact {name} must be non-empty.")
        path = Path(self.relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("runtime support artifact relative_path must stay under the model run root.")
        object.__setattr__(self, "relative_path", path)
        if not isinstance(self.required_for_test_model, bool):
            raise ValueError("runtime support artifact required_for_test_model must be bool.")
        if (
            not isinstance(self.size_bytes, int)
            or isinstance(self.size_bytes, bool)
            or self.size_bytes < 0
        ):
            raise ValueError("runtime support artifact size_bytes must be non-negative.")
        if self.description is not None and (
            not isinstance(self.description, str) or not self.description.strip()
        ):
            raise ValueError("runtime support artifact description must be non-empty when provided.")

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "role": self.role,
            "kind": self.kind,
            "relative_path": self.relative_path.as_posix(),
            "required_for_test_model": self.required_for_test_model,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "provider_key": self.provider_key,
        }
        if self.description is not None:
            payload["description"] = self.description
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ModelRuntimeSupportArtifact":
        return cls(
            role=_text(payload, "role"),
            kind=_text(payload, "kind"),
            relative_path=Path(_text(payload, "relative_path")),
            required_for_test_model=_bool(payload, "required_for_test_model"),
            sha256=_text(payload, "sha256"),
            size_bytes=_int(payload, "size_bytes"),
            provider_key=_text(payload, "provider_key"),
            description=_optional_text(payload, "description"),
        )


@dataclass(frozen=True, slots=True)
class ModelRuntimeSupportManifest:
    schema_version: str
    model_key: str
    model_run_name: str
    manifest_family: str
    provider_config_kind: str
    artifacts: tuple[ModelRuntimeSupportArtifact, ...]

    def __post_init__(self) -> None:
        if self.schema_version != MODEL_RUNTIME_SUPPORT_SCHEMA_VERSION:
            raise ValueError(
                "runtime support manifest schema_version must be "
                f"{MODEL_RUNTIME_SUPPORT_SCHEMA_VERSION!r}."
            )
        for value, name in (
            (self.model_key, "model_key"),
            (self.model_run_name, "model_run_name"),
            (self.manifest_family, "manifest_family"),
            (self.provider_config_kind, "provider_config_kind"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"runtime support manifest {name} must be non-empty.")
        object.__setattr__(self, "model_key", ModelKey(self.model_key).value)
        artifacts = tuple(self.artifacts)
        if any(not isinstance(artifact, ModelRuntimeSupportArtifact) for artifact in artifacts):
            raise ValueError("runtime support manifest artifacts must be support artifact values.")
        object.__setattr__(self, "artifacts", artifacts)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model_key": self.model_key,
            "model_run_name": self.model_run_name,
            "manifest_family": self.manifest_family,
            "provider_config_kind": self.provider_config_kind,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ModelRuntimeSupportManifest":
        raw_artifacts = payload.get("artifacts")
        if not isinstance(raw_artifacts, list):
            raise ValueError("runtime support manifest artifacts must be a list.")
        artifacts: list[ModelRuntimeSupportArtifact] = []
        for item in raw_artifacts:
            if not isinstance(item, dict):
                raise ValueError("runtime support manifest artifact entries must be objects.")
            artifacts.append(ModelRuntimeSupportArtifact.from_dict(item))
        return cls(
            schema_version=_text(payload, "schema_version"),
            model_key=_text(payload, "model_key"),
            model_run_name=_text(payload, "model_run_name"),
            manifest_family=_text(payload, "manifest_family"),
            provider_config_kind=_text(payload, "provider_config_kind"),
            artifacts=tuple(artifacts),
        )


def support_artifact_from_model_run_file(
    *,
    model_run_root: Path,
    path: Path,
    role: str,
    kind: str = "model_provider_support",
    provider_key: str,
    required_for_test_model: bool = True,
    description: str | None = None,
) -> ModelRuntimeSupportArtifact:
    resolved_root = model_run_root.resolve(strict=False)
    resolved_path = Path(path).resolve(strict=False)
    try:
        relative_path = resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"support artifact is outside model run root: {path}") from exc
    if not resolved_path.is_file():
        raise FileNotFoundError(f"required runtime support artifact is missing: {path}")
    return ModelRuntimeSupportArtifact(
        role=role,
        kind=kind,
        relative_path=relative_path,
        required_for_test_model=required_for_test_model,
        sha256=sha256_file(resolved_path),
        size_bytes=resolved_path.stat().st_size,
        provider_key=provider_key,
        description=description,
    )


def write_runtime_support_manifest(
    path: Path,
    manifest: ModelRuntimeSupportManifest,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_runtime_support_manifest(path: Path) -> ModelRuntimeSupportManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("runtime support manifest root must be an object.")
    return ModelRuntimeSupportManifest.from_dict(payload)


def _text(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"runtime support field {key!r} must be non-empty text.")
    return value


def _optional_text(payload: dict[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"runtime support field {key!r} must be non-empty text when present.")
    return value


def _bool(payload: dict[str, object], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"runtime support field {key!r} must be bool.")
    return value


def _int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"runtime support field {key!r} must be a non-negative integer.")
    return value


__all__ = [
    "MODEL_RUNTIME_SUPPORT_SCHEMA_VERSION",
    "ModelRuntimeSupportArtifact",
    "ModelRuntimeSupportManifest",
    "read_runtime_support_manifest",
    "support_artifact_from_model_run_file",
    "write_runtime_support_manifest",
]
