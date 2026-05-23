"""Deterministic IO for generated-pose payloads and manifests."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, cast
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile, ZipInfo

import numpy as np

from text_to_sign_production.artifacts.store import (
    ArtifactTopology,
    validate_generated_pose_relative_path,
)
from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.artifacts.generated_pose import GeneratedPoseSample
from text_to_sign_production.modeling.artifacts.generated_pose_manifest import (
    GeneratedPoseManifestEntry,
    generated_manifest_entry_from_record,
    generated_manifest_entry_from_sample,
    generated_manifest_entry_to_record,
)
from text_to_sign_production.modeling.artifacts.run_metadata import (
    GeneratedPoseConfidencePolicy,
    GeneratedPoseGenerationMode,
    GeneratedPoseLengthPolicy,
    GeneratedPoseProducerType,
)
from text_to_sign_production.modeling.artifacts.validation import (
    GeneratedPoseArtifactError,
    validate_generated_pose_manifest_entries,
    validate_generated_pose_sample,
)
from text_to_sign_production.modeling.data.bfh_schema import BfhPoseArrays

_PAYLOAD_KEYS = frozenset(
    {
        "metadata_json",
        "valid_frame_mask",
        "body_xyc",
        "left_hand_xyc",
        "right_hand_xyc",
        "face_xyc",
    }
)


@dataclass(frozen=True, slots=True)
class GeneratedPoseSplitWriteResult:
    """Result of writing one generated-pose split."""

    split: SampleSplit
    manifest_path: Path
    entries: tuple[GeneratedPoseManifestEntry, ...]
    payload_paths: tuple[Path, ...]
    sample_count: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "split", SampleSplit(self.split))
        object.__setattr__(self, "manifest_path", Path(self.manifest_path))
        entries = tuple(self.entries)
        if any(not isinstance(entry, GeneratedPoseManifestEntry) for entry in entries):
            raise GeneratedPoseArtifactError(
                "entries must contain GeneratedPoseManifestEntry values."
            )
        payload_paths = tuple(Path(path) for path in self.payload_paths)
        sample_count = len(payload_paths) if self.sample_count is None else self.sample_count
        if not isinstance(sample_count, int) or isinstance(sample_count, bool):
            raise GeneratedPoseArtifactError("sample_count must be an integer.")
        if sample_count < 0:
            raise GeneratedPoseArtifactError("sample_count must be non-negative.")
        if sample_count != len(payload_paths):
            raise GeneratedPoseArtifactError(
                "sample_count must match the number of generated payload paths."
            )
        if len(entries) != sample_count:
            raise GeneratedPoseArtifactError(
                "sample_count must match the number of generated manifest entries."
            )
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "payload_paths", payload_paths)
        object.__setattr__(self, "sample_count", sample_count)


@dataclass(frozen=True, slots=True)
class GeneratedPoseStreamWriteResult:
    """Bounded result for streaming generated-pose writers."""

    split: SampleSplit
    manifest_path: Path
    sample_root: Path
    sample_count: int
    payload_count: int
    preview_entries: tuple[GeneratedPoseManifestEntry, ...] = ()
    preview_payload_paths: tuple[Path, ...] = ()
    preview_limit: int = 10

    def __post_init__(self) -> None:
        object.__setattr__(self, "split", SampleSplit(self.split))
        object.__setattr__(self, "manifest_path", Path(self.manifest_path))
        object.__setattr__(self, "sample_root", Path(self.sample_root))
        for name in ("sample_count", "payload_count", "preview_limit"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise GeneratedPoseArtifactError(f"{name} must be a non-negative integer.")
        entries = tuple(self.preview_entries)
        if len(entries) > self.preview_limit:
            raise GeneratedPoseArtifactError("preview_entries exceeds preview_limit.")
        if any(not isinstance(entry, GeneratedPoseManifestEntry) for entry in entries):
            raise GeneratedPoseArtifactError(
                "preview_entries must contain GeneratedPoseManifestEntry values."
            )
        paths = tuple(Path(path) for path in self.preview_payload_paths)
        if len(paths) > self.preview_limit:
            raise GeneratedPoseArtifactError("preview_payload_paths exceeds preview_limit.")
        object.__setattr__(self, "preview_entries", entries)
        object.__setattr__(self, "preview_payload_paths", paths)


@dataclass(frozen=True, slots=True)
class DiagnosticGeneratedPosePaths:
    """Explicit generated-pose layout for runtime-only diagnostic surfaces."""

    root: Path
    generated_pose_root: Path
    manifest_path: Path
    samples_root: Path
    payload_ref_root: Path


def diagnostic_generated_pose_paths(root: Path) -> DiagnosticGeneratedPosePaths:
    """Return the diagnostic generated-pose manifest/sample layout under ``root``."""

    root = Path(root)
    generated_pose_root = root / "generated_pose"
    return DiagnosticGeneratedPosePaths(
        root=root,
        generated_pose_root=generated_pose_root,
        manifest_path=generated_pose_root / "manifest.jsonl",
        samples_root=generated_pose_root / "samples",
        payload_ref_root=root,
    )


def validate_diagnostic_generated_pose_paths(paths: DiagnosticGeneratedPosePaths) -> None:
    """Validate that diagnostic generated-pose refs resolve as generated_pose/samples/..."""

    if not isinstance(paths, DiagnosticGeneratedPosePaths):
        raise GeneratedPoseArtifactError(
            "diagnostic generated-pose paths must be a DiagnosticGeneratedPosePaths value."
        )
    if paths.generated_pose_root != paths.root / "generated_pose":
        raise GeneratedPoseArtifactError(
            "explicit runtime-only generated-pose outputs must use "
            "<root>/generated_pose/manifest.jsonl and <root>/generated_pose/samples "
            "so payload refs are generated_pose/samples/..."
        )
    if paths.manifest_path.parent.name != "generated_pose":
        raise GeneratedPoseArtifactError(
            "explicit runtime-only generated-pose outputs must use "
            "<root>/generated_pose/manifest.jsonl and <root>/generated_pose/samples "
            "so payload refs are generated_pose/samples/..."
        )
    if paths.manifest_path != paths.generated_pose_root / "manifest.jsonl":
        raise GeneratedPoseArtifactError(
            "explicit runtime-only generated-pose manifest_path must be "
            "<root>/generated_pose/manifest.jsonl."
        )
    if paths.samples_root.parent != paths.manifest_path.parent:
        raise GeneratedPoseArtifactError(
            "explicit runtime-only generated-pose samples_root must share the "
            "generated_pose directory with manifest_path."
        )
    if paths.samples_root != paths.generated_pose_root / "samples":
        raise GeneratedPoseArtifactError(
            "explicit runtime-only generated-pose samples_root must be "
            "<root>/generated_pose/samples."
        )
    if paths.payload_ref_root != paths.root:
        raise GeneratedPoseArtifactError(
            "explicit runtime-only generated-pose payload_ref_root must be <root>."
        )


def write_generated_pose_payload(
    path: str | Path,
    sample: GeneratedPoseSample,
) -> None:
    """Write one GeneratedPoseSample as a deterministic compressed NPZ artifact."""

    _raise_if_issues(validate_generated_pose_sample(sample), "Invalid generated-pose sample")
    payload_path = Path(path)
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "metadata_json": np.asarray(json.dumps(sample.to_metadata_dict(), sort_keys=True)),
        "valid_frame_mask": np.asarray(sample.pose.valid_frame_mask, dtype=np.bool_),
        "body_xyc": np.asarray(sample.pose.body_xyc, dtype=np.float32),
        "left_hand_xyc": np.asarray(sample.pose.left_hand_xyc, dtype=np.float32),
        "right_hand_xyc": np.asarray(sample.pose.right_hand_xyc, dtype=np.float32),
        "face_xyc": np.asarray(sample.pose.face_xyc, dtype=np.float32),
    }
    _write_deterministic_npz(payload_path, arrays)


def load_generated_pose_payload(
    path: str | Path,
) -> GeneratedPoseSample:
    """Load one GeneratedPoseSample from a deterministic NPZ artifact."""

    payload_path = Path(path)
    try:
        with np.load(payload_path, allow_pickle=False) as loaded:
            keys = frozenset(loaded.files)
            missing = sorted(_PAYLOAD_KEYS.difference(keys))
            extra = sorted(keys.difference(_PAYLOAD_KEYS))
            if missing or extra:
                raise GeneratedPoseArtifactError(
                    f"generated-pose payload keys mismatch: missing={missing}, extra={extra}"
                )
            metadata = _metadata_from_json(str(np.asarray(loaded["metadata_json"]).item()))
            pose = BfhPoseArrays(
                body_xyc=np.asarray(loaded["body_xyc"], dtype=np.float32),
                left_hand_xyc=np.asarray(loaded["left_hand_xyc"], dtype=np.float32),
                right_hand_xyc=np.asarray(loaded["right_hand_xyc"], dtype=np.float32),
                face_xyc=np.asarray(loaded["face_xyc"], dtype=np.float32),
                valid_frame_mask=np.asarray(loaded["valid_frame_mask"], dtype=np.bool_),
            )
    except GeneratedPoseArtifactError:
        raise
    except (BadZipFile, EOFError, OSError, KeyError, TypeError, ValueError) as exc:
        raise GeneratedPoseArtifactError(
            f"generated-pose payload could not be loaded from {payload_path}: {exc}"
        ) from exc
    try:
        sample = GeneratedPoseSample(
            schema_version=_text(metadata["schema_version"], "schema_version"),
            producer_type=GeneratedPoseProducerType(_text(metadata["producer_type"], "producer_type")),
            producer_key=_text(metadata["producer_key"], "producer_key"),
            canonical_id=_text(metadata["canonical_id"], "canonical_id"),
            phase_number=_int(metadata["phase_number"], "phase_number"),
            research_role=_text(metadata["research_role"], "research_role"),
            run_name=_text(metadata["run_name"], "run_name"),
            split=SampleSplit(_text(metadata["split"], "split")),
            sample_id=_text(metadata["sample_id"], "sample_id"),
            text=_text(metadata["text"], "text"),
            source_video_id=_text(metadata["source_video_id"], "source_video_id"),
            source_sentence_id=_text(metadata["source_sentence_id"], "source_sentence_id"),
            source_sentence_name=_text(metadata["source_sentence_name"], "source_sentence_name"),
            reference_payload_ref=_text(metadata["reference_payload_ref"], "reference_payload_ref"),
            generation_index=_int(metadata["generation_index"], "generation_index"),
            num_candidates_for_sample=_int(
                metadata["num_candidates_for_sample"],
                "num_candidates_for_sample",
            ),
            generation_mode=GeneratedPoseGenerationMode(
                _text(metadata["generation_mode"], "generation_mode")
            ),
            length_policy=GeneratedPoseLengthPolicy(_text(metadata["length_policy"], "length_policy")),
            channel_policy=_text(metadata["channel_policy"], "channel_policy"),
            confidence_policy=GeneratedPoseConfidencePolicy(
                _text(metadata["confidence_policy"], "confidence_policy")
            ),
            seed=_optional_int(metadata["seed"], "seed"),
            failure_reason=_optional_text(metadata["failure_reason"], "failure_reason"),
            pose=pose,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise GeneratedPoseArtifactError(
            f"generated-pose payload metadata is invalid in {payload_path}: {exc}"
        ) from exc
    _raise_if_issues(validate_generated_pose_sample(sample), "Invalid generated-pose sample")
    return sample


def write_generated_pose_manifest_jsonl(
    path: str | Path,
    entries: Iterable[GeneratedPoseManifestEntry],
    *,
    split: SampleSplit | str,
) -> None:
    """Write generated-pose manifest entries as deterministic JSONL."""

    resolved_split = SampleSplit(split)
    materialized = tuple(entries)
    _raise_if_issues(
        validate_generated_pose_manifest_entries(materialized, expected_split=resolved_split),
        "Invalid generated-pose manifest",
    )
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for entry in materialized:
            record = generated_manifest_entry_to_record(entry)
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def read_generated_pose_manifest_jsonl(
    path: str | Path,
    *,
    expected_split: SampleSplit | str | None = None,
) -> tuple[GeneratedPoseManifestEntry, ...]:
    """Read generated-pose manifest JSONL entries."""

    manifest_path = Path(path)
    entries: list[GeneratedPoseManifestEntry] = []
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise GeneratedPoseArtifactError(
                        f"malformed generated-pose manifest JSON at line {line_number}: {exc}"
                    ) from exc
                entries.append(generated_manifest_entry_from_record(record))
    except FileNotFoundError:
        raise
    issues = validate_generated_pose_manifest_entries(entries, expected_split=expected_split)
    _raise_if_issues(issues, "Invalid generated-pose manifest")
    return tuple(entries)


def write_generated_pose_split(
    topology: ArtifactTopology,
    *,
    producer_key: str,
    run_name: str,
    split: SampleSplit | str,
    samples: Iterable[GeneratedPoseSample],
) -> GeneratedPoseSplitWriteResult:
    """Write generated-pose payloads and manifest for one split."""

    resolved_split = SampleSplit(split)
    materialized = tuple(samples)
    payload_paths: list[Path] = []
    entries: list[GeneratedPoseManifestEntry] = []
    for sample in materialized:
        _raise_if_issues(validate_generated_pose_sample(sample), "Invalid generated-pose sample")
        if sample.producer_key != producer_key or sample.run_name != run_name:
            raise GeneratedPoseArtifactError("sample producer_key/run_name does not match request.")
        if sample.split is not resolved_split:
            raise GeneratedPoseArtifactError("sample split does not match request.")
        try:
            payload_path = topology.evaluations.generated_pose_sample_file(
                producer_key,
                run_name,
                resolved_split,
                sample.sample_id,
                sample.generation_index,
            ).path
        except ValueError as exc:
            raise GeneratedPoseArtifactError(
                f"generated-pose sample path could not be resolved: {exc}"
            ) from exc
        try:
            generated_payload_ref = payload_path.relative_to(topology.repo_root).as_posix()
        except ValueError as exc:
            raise GeneratedPoseArtifactError(
                f"generated-pose payload path is not repo-relative: {payload_path}"
            ) from exc
        payload_paths.append(payload_path)
        entries.append(
            generated_manifest_entry_from_sample(
                sample,
                generated_payload_ref=generated_payload_ref,
            )
        )
    manifest_entries = tuple(entries)
    _raise_if_issues(
        validate_generated_pose_manifest_entries(manifest_entries, expected_split=resolved_split),
        "Invalid generated-pose manifest",
    )
    for sample, payload_path in zip(materialized, payload_paths, strict=True):
        write_generated_pose_payload(payload_path, sample)
    manifest_path = topology.evaluations.generated_pose_manifest(
        producer_key,
        run_name,
        resolved_split,
    ).path
    write_generated_pose_manifest_jsonl(manifest_path, manifest_entries, split=resolved_split)
    return GeneratedPoseSplitWriteResult(
        split=resolved_split,
        manifest_path=manifest_path,
        entries=manifest_entries,
        payload_paths=tuple(payload_paths),
        sample_count=len(payload_paths),
    )


def write_generated_pose_split_to_explicit_root(
    *,
    manifest_path: Path,
    samples_root: Path,
    split: SampleSplit | str,
    samples: Iterable[GeneratedPoseSample],
    payload_ref_root: Path | None = None,
) -> GeneratedPoseSplitWriteResult:
    """Write generated-pose payloads and manifest under an explicit output root."""

    resolved_split = SampleSplit(split)
    materialized = tuple(samples)
    manifest_path = Path(manifest_path)
    samples_root = Path(samples_root)
    payload_ref_root = None if payload_ref_root is None else Path(payload_ref_root)
    _validate_explicit_generated_pose_layout(
        manifest_path=manifest_path,
        samples_root=samples_root,
        payload_ref_root=payload_ref_root,
    )
    payload_paths: list[Path] = []
    entries: list[GeneratedPoseManifestEntry] = []
    for sample in materialized:
        _raise_if_issues(validate_generated_pose_sample(sample), "Invalid generated-pose sample")
        if sample.split is not resolved_split:
            raise GeneratedPoseArtifactError("sample split does not match request.")
        payload_path = samples_root / _generated_pose_sample_filename(
            sample.sample_id,
            sample.generation_index,
        )
        payload_paths.append(payload_path)
        entries.append(
            generated_manifest_entry_from_sample(
                sample,
                generated_payload_ref=_diagnostic_payload_ref(
                    payload_path,
                    manifest_path=manifest_path,
                    payload_ref_root=payload_ref_root,
                ),
            )
        )
    manifest_entries = tuple(entries)
    _raise_if_issues(
        validate_generated_pose_manifest_entries(manifest_entries, expected_split=resolved_split),
        "Invalid generated-pose manifest",
    )
    for sample, payload_path in zip(materialized, payload_paths, strict=True):
        write_generated_pose_payload(payload_path, sample)
    write_generated_pose_manifest_jsonl(manifest_path, manifest_entries, split=resolved_split)
    return GeneratedPoseSplitWriteResult(
        split=resolved_split,
        manifest_path=manifest_path,
        entries=manifest_entries,
        payload_paths=tuple(payload_paths),
        sample_count=len(payload_paths),
    )


def _diagnostic_payload_ref(
    payload_path: Path,
    *,
    manifest_path: Path,
    payload_ref_root: Path | None,
) -> str:
    root = manifest_path.parent.parent if payload_ref_root is None else payload_ref_root
    try:
        return payload_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise GeneratedPoseArtifactError(
            f"generated-pose payload path is not relative to explicit root: {payload_path}"
        ) from exc


def _validate_explicit_generated_pose_layout(
    *,
    manifest_path: Path,
    samples_root: Path,
    payload_ref_root: Path | None,
) -> None:
    root = manifest_path.parent.parent if payload_ref_root is None else payload_ref_root
    if samples_root.parent != manifest_path.parent:
        raise GeneratedPoseArtifactError(
            "explicit runtime-only generated-pose outputs must use "
            "<root>/generated_pose/manifest.jsonl and <root>/generated_pose/samples "
            "so payload refs are generated_pose/samples/..."
        )
    try:
        manifest_ref = manifest_path.relative_to(root)
        samples_ref = samples_root.relative_to(root)
    except ValueError as exc:
        raise GeneratedPoseArtifactError(
            "explicit generated-pose manifest and samples must be relative to payload_ref_root."
        ) from exc
    if (
        manifest_ref.parts == ("generated_pose", "manifest.jsonl")
        and samples_ref.parts == ("generated_pose", "samples")
    ):
        validate_diagnostic_generated_pose_paths(
            DiagnosticGeneratedPosePaths(
                root=root,
                generated_pose_root=root / "generated_pose",
                manifest_path=manifest_path,
                samples_root=samples_root,
                payload_ref_root=root,
            )
        )
        return
    manifest_issues = validate_generated_pose_relative_path(manifest_ref)
    sample_probe = samples_ref / "__probe__g0.npz"
    sample_issues = validate_generated_pose_relative_path(sample_probe)
    if not manifest_issues and not sample_issues:
        return
    raise GeneratedPoseArtifactError(
        "explicit runtime-only generated-pose outputs must use "
        "<root>/generated_pose/manifest.jsonl and <root>/generated_pose/samples "
        "so payload refs are generated_pose/samples/..."
    )


def _generated_pose_sample_filename(sample_id: str, generation_index: int) -> str:
    if not isinstance(generation_index, int) or isinstance(generation_index, bool):
        raise GeneratedPoseArtifactError("generation_index must be an integer.")
    if generation_index < 0:
        raise GeneratedPoseArtifactError("generation_index must be non-negative.")
    token = str(sample_id)
    if not token.strip() or token in {".", ".."} or "/" in token or "\\" in token or "." in token:
        raise GeneratedPoseArtifactError(
            "generated-pose sample_id must be a concrete safe path token."
        )
    return f"{token}__g{generation_index}.npz"


def _write_deterministic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    with ZipFile(path, mode="w", compression=ZIP_DEFLATED) as archive:
        for key in sorted(arrays):
            info = ZipInfo(filename=f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, _npy_bytes(np.asarray(arrays[key])))


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = BytesIO()
    np.lib.format.write_array(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _metadata_from_json(value: str) -> Mapping[str, Any]:
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise GeneratedPoseArtifactError(f"metadata_json is malformed: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise GeneratedPoseArtifactError("metadata_json must contain a JSON object.")
    if any(not isinstance(key, str) for key in loaded):
        raise GeneratedPoseArtifactError("metadata_json keys must be strings.")
    return cast(Mapping[str, Any], loaded)


def _raise_if_issues(issues: tuple[Any, ...], prefix: str) -> None:
    if issues:
        messages = "; ".join(issue.message for issue in issues)
        raise GeneratedPoseArtifactError(f"{prefix}: {messages}")


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise GeneratedPoseArtifactError(f"{field_name} must be a string.")
    return value


def _optional_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _text(value, field_name)


def _int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise GeneratedPoseArtifactError(f"{field_name} must be an integer.")
    return value


def _optional_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    return _int(value, field_name)


__all__ = [
    "DiagnosticGeneratedPosePaths",
    "GeneratedPoseSplitWriteResult",
    "GeneratedPoseStreamWriteResult",
    "diagnostic_generated_pose_paths",
    "load_generated_pose_payload",
    "read_generated_pose_manifest_jsonl",
    "validate_diagnostic_generated_pose_paths",
    "write_generated_pose_manifest_jsonl",
    "write_generated_pose_payload",
    "write_generated_pose_split",
    "write_generated_pose_split_to_explicit_root",
]
