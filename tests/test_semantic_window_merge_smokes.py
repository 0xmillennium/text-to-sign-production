from __future__ import annotations

from pathlib import Path

import pytest
import torch

from text_to_sign_production.modeling.backbones.bfh_vectorization import (
    default_bfh_tensor_layout,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.config import (
    load_semantic_consistency_config,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.errors import (
    SemanticConsistencyError,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.io import (
    read_semantic_objective_config_json,
    write_semantic_objective_config_json,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.training import (
    compute_semantic_training_loss,
    merge_overlapping_windows_torch,
)


@pytest.mark.unit
def test_semantic_window_merge_uses_cover_all_tail_starts() -> None:
    starts = (0, 4, 6)
    windows = _source_value_windows(starts=starts, frame_count=10, window_size=4)

    merged = merge_overlapping_windows_torch(
        windows,
        frame_count=10,
        window_size=4,
        stride=4,
    )

    assert torch.equal(merged.squeeze(-1), torch.arange(10, dtype=torch.float32))


@pytest.mark.unit
def test_semantic_window_merge_second_cover_all_tail_case() -> None:
    starts = (0, 2, 4, 5)
    windows = _source_value_windows(starts=starts, frame_count=9, window_size=4)

    merged = merge_overlapping_windows_torch(
        windows,
        frame_count=9,
        window_size=4,
        stride=2,
    )

    assert torch.equal(merged.squeeze(-1), torch.arange(9, dtype=torch.float32))


@pytest.mark.unit
def test_semantic_window_merge_preserves_gradients() -> None:
    windows = _source_value_windows(starts=(0, 4, 6), frame_count=10, window_size=4)
    windows.requires_grad_(True)

    merged = merge_overlapping_windows_torch(
        windows,
        frame_count=10,
        window_size=4,
        stride=4,
    )
    merged.square().mean().backward()

    assert windows.grad is not None, (
        "semantic window merge gradient smoke failed; windows.grad was missing or non-finite."
    )
    assert torch.isfinite(windows.grad).all(), (
        "semantic window merge gradient smoke failed; windows.grad was missing or non-finite."
    )


@pytest.mark.unit
def test_semantic_window_merge_supports_source_frame_indices_and_padding() -> None:
    windows = _source_value_windows(starts=(0, 4, 6), frame_count=10, window_size=4)
    source_frame_indices = torch.tensor(
        [[0, 1, 2, 3], [4, 5, 6, 7], [6, 7, 8, 9]],
        dtype=torch.long,
    )

    merged = merge_overlapping_windows_torch(
        windows,
        frame_count=10,
        window_size=4,
        stride=4,
        source_frame_indices=source_frame_indices,
    )

    assert torch.equal(merged.squeeze(-1), torch.arange(10, dtype=torch.float32))

    padded = torch.tensor([[[10.0], [11.0], [12.0], [99.0], [99.0]]])
    padded_indices = torch.tensor([[0, 1, 2, -1, -1]], dtype=torch.long)
    padded_merged = merge_overlapping_windows_torch(
        padded,
        frame_count=3,
        window_size=5,
        stride=2,
        source_frame_indices=padded_indices,
    )

    assert torch.equal(padded_merged.squeeze(-1), torch.tensor([10.0, 11.0, 12.0]))


@pytest.mark.unit
def test_semantic_window_merge_rejects_bad_window_count_and_conflicting_sources() -> None:
    windows = torch.zeros((2, 4, 1))

    with pytest.raises(
        SemanticConsistencyError,
        match=(
            "semantic window count=2 does not match expected temporal_window_starts count=3 "
            "for frame_count=10, window_size=4, stride=4."
        ),
    ):
        merge_overlapping_windows_torch(windows, frame_count=10, window_size=4, stride=4)

    with pytest.raises(
        SemanticConsistencyError,
        match="provide either start_indices or source_frame_indices, not both.",
    ):
        merge_overlapping_windows_torch(
            torch.zeros((3, 4, 1)),
            frame_count=10,
            window_size=4,
            stride=4,
            start_indices=(0, 4, 6),
            source_frame_indices=torch.zeros((3, 4), dtype=torch.long),
        )


@pytest.mark.unit
def test_semantic_window_merge_rejects_explicit_start_outside_frame_range() -> None:
    with pytest.raises(
        SemanticConsistencyError,
        match=r"semantic start_indices contain an index outside \[0, frame_count\).",
    ):
        merge_overlapping_windows_torch(
            torch.zeros((3, 4, 1)),
            frame_count=10,
            window_size=4,
            stride=4,
            start_indices=(0, 4, 10),
        )


@pytest.mark.unit
def test_learned_pose_token_style_semantic_window_smoke_backprops_to_logits() -> None:
    config = _semantic_config()
    layout = default_bfh_tensor_layout()
    feature_dim = layout.total_feature_dim
    window_size = 4
    token_count = 3
    logits = torch.randn((1, token_count, 7), requires_grad=True)
    codebook = torch.randn((7, 5))
    decoder = torch.nn.Linear(5, window_size * feature_dim)
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)

    probabilities = torch.softmax(logits, dim=-1)
    expected_latents = probabilities @ codebook
    decoded_windows = decoder(expected_latents.reshape(-1, 5)).reshape(
        token_count,
        window_size,
        feature_dim,
    )
    pose = merge_overlapping_windows_torch(
        decoded_windows,
        frame_count=10,
        window_size=window_size,
        stride=4,
        start_indices=(0, 4, 6),
    ).unsqueeze(0)
    semantic_result = compute_semantic_training_loss(
        text_embeddings=torch.randn((1, 256)),
        pose_values=pose,
        pose_validity_mask=torch.ones_like(pose, dtype=torch.bool),
        frame_mask=torch.ones((1, 10), dtype=torch.bool),
        layout=layout,
        config=config,
    )

    semantic_result.weighted_semantic_loss.backward()

    assert torch.isfinite(semantic_result.semantic_loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert all(parameter.grad is None for parameter in decoder.parameters())


@pytest.mark.unit
def test_latent_diffusion_style_temporal_latent_semantic_smoke_backprops_to_x0() -> None:
    config = _semantic_config()
    layout = default_bfh_tensor_layout()
    feature_dim = layout.total_feature_dim
    predicted_x0 = torch.randn((3, 6), requires_grad=True)
    decoder = torch.nn.Linear(6, 4 * feature_dim)
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)

    decoded_windows = decoder(predicted_x0).reshape(3, 4, feature_dim)
    pose = merge_overlapping_windows_torch(
        decoded_windows,
        frame_count=10,
        window_size=4,
        stride=4,
        start_indices=(0, 4, 6),
    ).unsqueeze(0)
    semantic_result = compute_semantic_training_loss(
        text_embeddings=torch.randn((1, 256)),
        pose_values=pose,
        pose_validity_mask=torch.ones_like(pose, dtype=torch.bool),
        frame_mask=torch.ones((1, 10), dtype=torch.bool),
        layout=layout,
        config=config,
    )

    semantic_result.weighted_semantic_loss.backward()

    assert pose.shape == (1, 10, feature_dim)
    assert torch.isfinite(semantic_result.semantic_loss)
    assert predicted_x0.grad is not None and torch.isfinite(predicted_x0.grad).all()


@pytest.mark.unit
def test_articulator_factorized_style_semantic_smoke_backprops_to_pose_output() -> None:
    config = _semantic_config()
    layout = default_bfh_tensor_layout()
    pose_output = torch.randn((1, 10, layout.total_feature_dim), requires_grad=True)

    semantic_result = compute_semantic_training_loss(
        text_embeddings=torch.randn((1, 256)),
        pose_values=pose_output,
        pose_validity_mask=torch.ones_like(pose_output, dtype=torch.bool),
        frame_mask=torch.ones((1, 10), dtype=torch.bool),
        layout=layout,
        config=config,
    )
    semantic_result.weighted_semantic_loss.backward()

    assert torch.isfinite(semantic_result.semantic_loss)
    assert pose_output.grad is not None and torch.isfinite(pose_output.grad).all()


@pytest.mark.unit
def test_semantic_v2_config_artifact_round_trips_for_proxy_diagnostic(tmp_path: Path) -> None:
    config_path = Path("configs/modeling/objectives/semantic_consistency.yaml")
    config = load_semantic_consistency_config(config_path)
    artifact_path = tmp_path / "semantic_config.json"

    write_semantic_objective_config_json(
        artifact_path,
        config,
        config_snapshot_path=config_path,
        config_snapshot_sha256="0" * 64,
    )
    loaded = read_semantic_objective_config_json(artifact_path)

    assert loaded.training_objective.enabled is True
    assert loaded.text_embedding.proxy_only is True
    assert loaded.pose_embedding.proxy_only is True


def _source_value_windows(
    *,
    starts: tuple[int, ...],
    frame_count: int,
    window_size: int,
) -> torch.Tensor:
    values = torch.full((len(starts), window_size, 1), -1000.0)
    for window_index, start in enumerate(starts):
        for offset in range(window_size):
            source_index = start + offset
            if source_index < frame_count:
                values[window_index, offset, 0] = float(source_index)
    return values


def _semantic_config():
    return load_semantic_consistency_config(
        Path("configs/modeling/objectives/semantic_consistency.yaml")
    )
