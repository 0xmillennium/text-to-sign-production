"""Source-level structural gate."""

from __future__ import annotations

from text_to_sign_production.legacy_data.gates.types import GateResult, GatesConfig, GateStatus
from text_to_sign_production.legacy_data.sources.types import SourceCandidate
from text_to_sign_production.legacy_data.sources.validate import validate_candidate


def evaluate_source_gate(candidate: SourceCandidate, config: GatesConfig) -> GateResult:
    """Evaluate if the source candidate is structurally viable.

    Uses established source-level validation plus gate-owned text sanity.
    """
    issues = validate_candidate(candidate)
    reasons = [issue.code for issue in issues]

    normalized_text = " ".join(candidate.text.split())
    character_count = len(normalized_text)
    token_count = 0 if normalized_text == "" else len(normalized_text.split(" "))
    text_sanity = config.text_sanity

    if character_count < text_sanity.min_character_count:
        reasons.append(
            f"insufficient_character_count:{character_count}<{text_sanity.min_character_count}"
        )
    if token_count < text_sanity.min_token_count:
        reasons.append(f"insufficient_token_count:{token_count}<{text_sanity.min_token_count}")

    if reasons:
        return GateResult(status=GateStatus.DROPPED, reasons=tuple(reasons))

    return GateResult(status=GateStatus.PASSED)
