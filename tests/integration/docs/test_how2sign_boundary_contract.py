"""How2Sign dataset boundary documentation contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[3]
HOW2SIGN_DOC = PROJECT_ROOT / "docs" / "how2sign.md"


def _read_repo_file(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def _read_how2sign_doc() -> str:
    return HOW2SIGN_DOC.read_text(encoding="utf-8")


def _assert_contains_all(source: str, snippets: tuple[str, ...]) -> None:
    for snippet in snippets:
        assert snippet in source


def _assert_contains_all_casefold(source: str, snippets: tuple[str, ...]) -> None:
    normalized_source = source.casefold()
    for snippet in snippets:
        assert snippet.casefold() in normalized_source


def test_required_how2sign_files_and_links_exist() -> None:
    assert HOW2SIGN_DOC.is_file()

    assert "how2sign.md" in _read_repo_file("docs/index.md")
    assert "../how2sign.md" in _read_repo_file("docs/data/index.md")
    assert "How2Sign: how2sign.md" in _read_repo_file("mkdocs.yml")


def test_how2sign_boundary_page_contains_required_policy_content() -> None:
    source = _read_how2sign_doc()

    _assert_contains_all(
        source,
        (
            "https://how2sign.github.io/",
            "B-F-H 2D Keypoints clips* (frontal view)",
            "English Translation (manually re-aligned)",
            "Creative Commons Attribution-NonCommercial 4.0",
            "research purposes only",
            "MIT code license",
            "not proof of sign-language intelligibility",
            "How2Sign-compatible",
            "CVPR",
        ),
    )
    assert "not redistributed" in source or "not redistribute" in source
    assert "no public gloss" in source.casefold()


def test_how2sign_boundary_page_names_required_artifact_and_supervision_classes() -> None:
    source = _read_how2sign_doc()

    _assert_contains_all_casefold(
        source,
        (
            ".tar.zst",
            "private Google Drive",
            "Official raw How2Sign files",
            "Project-local",
            "B-F-H 2D keypoint",
            "English translation",
            "Processed",
            "Synthetic",
            "Model checkpoints",
            "Qualitative exports",
            "English translation text",
            "Segment timing",
            "Raw RGB video",
            "Gloss annotations",
            "Human intelligibility labels",
        ),
    )


def test_how2sign_boundary_page_rejects_unsafe_active_claims() -> None:
    source = _read_how2sign_doc()
    normalized_source = source.casefold()

    forbidden_active_claims = (
        "How2Sign data is MIT licensed",
        "MIT license applies to How2Sign data",
        "publicly redistributed by this repository",
        "released public model artifact",
    )
    for claim in forbidden_active_claims:
        assert claim.casefold() not in normalized_source

    intelligibility_lines = [
        line for line in source.splitlines() if "proof of sign-language intelligibility" in line
    ]
    assert intelligibility_lines
    for line in intelligibility_lines:
        assert "not proof of sign-language intelligibility" in line
