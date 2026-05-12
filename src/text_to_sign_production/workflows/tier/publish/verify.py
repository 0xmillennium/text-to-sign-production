from __future__ import annotations

from pathlib import Path

from text_to_sign_production.core.integrity import sha256_file
from text_to_sign_production.workflows.tier.contracts import (
    TierPublishCheck,
    TierPublishPlan,
    TierPublishTarget,
    TierPublishVerification,
)


def verify_tier_publish(
    plan: TierPublishPlan,
) -> TierPublishVerification:
    return TierPublishVerification(
        checks=tuple(_publish_check(target) for target in plan.targets),
    )


def _publish_check(target: TierPublishTarget) -> TierPublishCheck:
    source_exists = target.source_path.exists()
    target_exists = target.target_path.exists()
    observed_source_sha256 = (
        _file_sha256(target.source_path)
        if source_exists and target.source_path.is_file()
        else None
    )
    target_sha256 = (
        _file_sha256(target.target_path)
        if target_exists and target.target_path.is_file()
        else None
    )
    coherent = observed_source_sha256 == target.source_sha256
    message = None
    if not coherent:
        message = (
            "source digest no longer matches current-execution receipt: "
            f"expected {target.source_sha256}, observed {observed_source_sha256}"
        )
    digest_match = (
        target.source_sha256 is not None
        and target_sha256 is not None
        and target.source_sha256 == target_sha256
    )
    return TierPublishCheck(
        label=target.label,
        kind=target.kind,
        source_path=target.source_path,
        source_exists=source_exists,
        target_path=target.target_path,
        target_exists=target_exists,
        source_sha256=target.source_sha256,
        target_sha256=target_sha256,
        digest_match=digest_match,
        source_execution_id=target.source_execution_id,
        coherent=coherent,
        message=message,
    )


def _file_sha256(path: Path) -> str | None:
    try:
        return sha256_file(path)
    except OSError:
        return None
