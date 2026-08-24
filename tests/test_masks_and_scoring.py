from __future__ import annotations

import numpy as np
import pytest
import torch

from continuity_lens.config import (
    IMAGE_SIZE,
    PATCH_SIZE,
    TOTAL_FRAMES,
    VJEPA_CHECKPOINT_URL,
    VJEPA_COMMIT,
)
from continuity_lens.vjepa import ModelLoadError, future_masks, score_with_modules


class FakeEncoder(torch.nn.Module):
    def __init__(self, dimension: int = 8) -> None:
        super().__init__()
        self.dimension = dimension

    def forward(self, video: torch.Tensor, masks: torch.Tensor | None = None) -> torch.Tensor:
        spatial = (IMAGE_SIZE // PATCH_SIZE) ** 2
        tokens = TOTAL_FRAMES // 2 * spatial
        base = torch.arange(tokens, device=video.device, dtype=video.dtype).view(1, tokens, 1)
        output = base.repeat(video.shape[0], 1, self.dimension) / tokens
        if masks is not None:
            output = torch.gather(
                output,
                1,
                masks.unsqueeze(-1).expand(-1, -1, self.dimension),
            )
        return output


class FakePredictor(torch.nn.Module):
    def __init__(self, dimension: int = 8) -> None:
        super().__init__()
        self.dimension = dimension

    def forward(
        self, context: torch.Tensor, context_mask: torch.Tensor, target_mask: torch.Tensor
    ) -> torch.Tensor:
        return torch.zeros(
            context.shape[0],
            target_mask.shape[1],
            self.dimension,
            device=context.device,
            dtype=context.dtype,
        )


def test_future_masks_are_disjoint_ordered_and_complete() -> None:
    context, target = future_masks(4)
    spatial = (IMAGE_SIZE // PATCH_SIZE) ** 2
    total = TOTAL_FRAMES // 2 * spatial
    assert context.shape == (1, total - 2 * spatial)
    assert target.shape == (1, 2 * spatial)
    assert int(context.max()) < int(target.min())
    assert set(context.flatten().tolist()).isdisjoint(target.flatten().tolist())
    assert torch.equal(torch.cat([context, target], dim=1), torch.arange(total).view(1, -1))


def test_loader_contract_is_pinned_and_never_uses_localhost() -> None:
    assert len(VJEPA_COMMIT) == 40
    assert VJEPA_CHECKPOINT_URL == "https://dl.fbaipublicfiles.com/vjepa2/vitl.pt"
    assert "localhost" not in VJEPA_CHECKPOINT_URL


@pytest.mark.parametrize("horizon", [2, 4, 8])
def test_fake_modules_produce_dimensionally_valid_score(horizon: int) -> None:
    context_count = TOTAL_FRAMES - horizon
    context = np.zeros((context_count, 32, 32, 3), dtype=np.uint8)
    target = np.full((horizon, 32, 32, 3), 32, dtype=np.uint8)
    score = score_with_modules(
        context,
        target,
        context_encoder=FakeEncoder(),
        target_encoder=FakeEncoder(),
        predictor=FakePredictor(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        horizon=horizon,
    )
    assert score.prediction_error >= 0
    assert len(score.tubelet_errors) == horizon // 2
    assert score.timings_ms["model_inference"] >= 0


def test_dimension_mismatch_fails_loudly() -> None:
    context = np.zeros((12, 32, 32, 3), dtype=np.uint8)
    target = np.zeros((4, 32, 32, 3), dtype=np.uint8)
    with pytest.raises(ModelLoadError, match="incompatible"):
        score_with_modules(
            context,
            target,
            context_encoder=FakeEncoder(),
            target_encoder=FakeEncoder(),
            predictor=FakePredictor(dimension=7),
            device=torch.device("cpu"),
            dtype=torch.float32,
            horizon=4,
        )


def test_frames_with_different_source_dimensions_are_normalized_independently() -> None:
    context = [np.zeros((32, 40, 3), dtype=np.uint8) for _ in range(12)]
    target = [np.full((28, 36, 3), 32, dtype=np.uint8) for _ in range(4)]
    score = score_with_modules(
        context,
        target,
        context_encoder=FakeEncoder(),
        target_encoder=FakeEncoder(),
        predictor=FakePredictor(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        horizon=4,
    )
    assert score.prediction_error >= 0
