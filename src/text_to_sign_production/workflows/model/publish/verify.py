"""Verify digest-bearing model workflow publication targets."""

from __future__ import annotations

import json
from pathlib import Path

from text_to_sign_production.core.integrity import sha256_file
from text_to_sign_production.modeling.artifacts import (
    GENERATED_POSE_SAMPLE_ARCHIVE_MANIFEST_NAME,
    GENERATED_POSE_SAMPLE_ARCHIVE_NAME,
    GENERATED_POSE_SAMPLE_ARCHIVE_SHA256_NAME,
)
from text_to_sign_production.workflows.model.contracts import (
    ModelPublishCheck,
    ModelPublishExecution,
    ModelPublishVerification,
)


def verify_model_publish(
    execution: ModelPublishExecution,
) -> ModelPublishVerification:
    """Verify each published file exists and matches its planned source digest."""

    checks: list[ModelPublishCheck] = []
    for target in execution.plan.targets:
        exists = target.target_path.is_file()
        digest_matches = exists and sha256_file(target.target_path) == target.source_sha256
        message = None
        if not exists:
            message = "published target does not exist as a file"
        elif not digest_matches:
            message = "published target digest does not match planned source digest"
        if exists and digest_matches:
            archive_message = _generated_pose_archive_consistency_message(target.target_path)
            if archive_message is not None:
                digest_matches = False
                message = archive_message
        checks.append(
            ModelPublishCheck(
                target=target,
                exists=exists,
                sha256_matches=digest_matches,
                message=message,
            )
        )
    return ModelPublishVerification(checks=tuple(checks))


def _generated_pose_archive_consistency_message(target_path: Path) -> str | None:
    if target_path.name == GENERATED_POSE_SAMPLE_ARCHIVE_SHA256_NAME:
        archive_path = target_path.parent / GENERATED_POSE_SAMPLE_ARCHIVE_NAME
        if not archive_path.is_file():
            return "generated-pose archive sha256 target cannot be verified because archive is missing"
        expected = target_path.read_text(encoding="utf-8").strip()
        observed = sha256_file(archive_path)
        if expected != observed:
            return "generated-pose archive sha256 file does not match archive content"
        return None
    if target_path.name != GENERATED_POSE_SAMPLE_ARCHIVE_MANIFEST_NAME:
        return None
    try:
        payload = json.loads(target_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return f"generated-pose archive manifest could not be read: {exc}"
    if not isinstance(payload, dict):
        return "generated-pose archive manifest root is not an object"
    generated_manifest_path = target_path.parent / "manifest.jsonl"
    archive_path = target_path.parent / GENERATED_POSE_SAMPLE_ARCHIVE_NAME
    if not generated_manifest_path.is_file():
        return "generated-pose archive manifest cannot be verified because generated manifest is missing"
    if not archive_path.is_file():
        return "generated-pose archive manifest cannot be verified because archive is missing"
    member_count = payload.get("member_count")
    entry_count = _jsonl_entry_count(generated_manifest_path)
    if member_count != entry_count:
        return (
            "generated-pose archive member_count does not match generated manifest "
            f"entry_count: member_count={member_count!r}, entry_count={entry_count}"
        )
    if payload.get("archive_sha256") != sha256_file(archive_path):
        return "generated-pose archive manifest archive_sha256 does not match archive content"
    if payload.get("generated_manifest_sha256") != sha256_file(generated_manifest_path):
        return (
            "generated-pose archive manifest generated_manifest_sha256 does not match "
            "generated manifest content"
        )
    return None


def _jsonl_entry_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


__all__ = ["verify_model_publish"]
