"""Research traceability contracts for modeling registry specifications."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.modeling.research.validation import (
    ModelingResearchSpecError,
    require_non_empty,
    require_unique_strings,
)


@dataclass(frozen=True, slots=True)
class ResearchTraceIssue:
    """A non-fatal issue found while validating research trace paths."""

    code: str
    message: str
    path: Path | None = None

    def __post_init__(self) -> None:
        require_non_empty(self.code, field_name="code")
        require_non_empty(self.message, field_name="message")
        if self.path is not None:
            object.__setattr__(self, "path", _coerce_relative_path(self.path, "path"))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable trace issue."""

        return {
            "code": self.code,
            "message": self.message,
            "path": None if self.path is None else self.path.as_posix(),
        }


@dataclass(frozen=True, slots=True)
class ResearchTrace:
    """Research-document traceability for a modeling registry artifact."""

    candidate_id: str
    candidate_name: str
    candidate_card_path: Path
    candidate_universe_path: Path | None
    scorecard_path: Path | None
    selection_decision_path: Path | None
    audit_result_path: Path
    literature_positioning_path: Path
    roadmap_path: Path
    source_corpus_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        require_non_empty(self.candidate_id, field_name="candidate_id")
        require_non_empty(self.candidate_name, field_name="candidate_name")
        for field_name in (
            "candidate_card_path",
            "audit_result_path",
            "literature_positioning_path",
            "roadmap_path",
        ):
            object.__setattr__(
                self,
                field_name,
                _coerce_relative_path(getattr(self, field_name), field_name),
            )
        for field_name in (
            "candidate_universe_path",
            "scorecard_path",
            "selection_decision_path",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _coerce_relative_path(value, field_name))
        require_unique_strings(self.source_corpus_ids, field_name="source_corpus_ids")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable traceability record."""

        return {
            "candidate_id": self.candidate_id,
            "candidate_name": self.candidate_name,
            "candidate_card_path": self.candidate_card_path.as_posix(),
            "candidate_universe_path": _optional_path_to_string(self.candidate_universe_path),
            "scorecard_path": _optional_path_to_string(self.scorecard_path),
            "selection_decision_path": _optional_path_to_string(self.selection_decision_path),
            "audit_result_path": self.audit_result_path.as_posix(),
            "literature_positioning_path": self.literature_positioning_path.as_posix(),
            "roadmap_path": self.roadmap_path.as_posix(),
            "source_corpus_ids": list(self.source_corpus_ids),
        }


def validate_trace_paths(
    trace: ResearchTrace,
    *,
    project_root: Path,
) -> tuple[ResearchTraceIssue, ...]:
    """Return non-fatal issues for trace paths that do not exist under project_root."""

    root = Path(project_root).resolve(strict=False)
    issues: list[ResearchTraceIssue] = []
    for field_name, path in _trace_paths(trace):
        resolved_path = (root / path).resolve(strict=False)
        try:
            resolved_path.relative_to(root)
        except ValueError:
            issues.append(
                ResearchTraceIssue(
                    code="trace_path_escapes_project_root",
                    message=f"{field_name} resolves outside project_root.",
                    path=path,
                )
            )
        if not resolved_path.exists():
            issues.append(
                ResearchTraceIssue(
                    code="missing_trace_path",
                    message=f"{field_name} does not exist under project_root.",
                    path=path,
                )
            )
    return tuple(issues)


def _trace_paths(trace: ResearchTrace) -> tuple[tuple[str, Path], ...]:
    paths: list[tuple[str, Path]] = [
        ("candidate_card_path", trace.candidate_card_path),
        ("audit_result_path", trace.audit_result_path),
        ("literature_positioning_path", trace.literature_positioning_path),
        ("roadmap_path", trace.roadmap_path),
    ]
    for field_name in (
        "candidate_universe_path",
        "scorecard_path",
        "selection_decision_path",
    ):
        value = getattr(trace, field_name)
        if value is not None:
            paths.append((field_name, value))
    return tuple(paths)


def _coerce_relative_path(value: Path, field_name: str) -> Path:
    path = Path(value)
    if path.is_absolute() or str(path).strip() in {"", "."} or ".." in path.parts:
        raise ModelingResearchSpecError(f"{field_name} must be a relative path.")
    return path


def _optional_path_to_string(path: Path | None) -> str | None:
    if path is None:
        return None
    return path.as_posix()


__all__ = [
    "ResearchTrace",
    "ResearchTraceIssue",
    "validate_trace_paths",
]
