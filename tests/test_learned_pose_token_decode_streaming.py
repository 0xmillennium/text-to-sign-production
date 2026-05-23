from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_decode_preserves_diagnostic_generated_pose_contract() -> None:
    source = (
        ROOT
        / "src/text_to_sign_production/modeling/candidates/learned_pose_token/provider.py"
    ).read_text(encoding="utf-8")

    assert "diagnostic_root=_decode_root(context) / \"decoded_pose_intermediates\" / \"val\"" in source
    assert "decoded_generated_pose_manifest_path" in source
