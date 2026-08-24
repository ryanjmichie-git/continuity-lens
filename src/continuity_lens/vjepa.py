from __future__ import annotations

import copy
import gc
import platform
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as functional

from continuity_lens.config import (
    IMAGE_SIZE,
    PATCH_SIZE,
    TOTAL_FRAMES,
    TUBELET_SIZE,
    VJEPA_CHECKPOINT_URL,
    VJEPA_COMMIT,
    VJEPA_MODEL_ENTRYPOINT,
    VJEPA_REPOSITORY,
    default_cache_dir,
)
from continuity_lens.features import compute_cheap_features
from continuity_lens.schemas import ScoreBundle
from continuity_lens.utils import download_resumable, file_sha256, write_json


class ModelLoadError(RuntimeError):
    """A targeted model-loading error with enough context to diagnose the failure."""


class TransitionScorer(Protocol):
    def score(
        self,
        context_frames: Sequence[np.ndarray],
        target_frames: Sequence[np.ndarray],
    ) -> ScoreBundle: ...


@dataclass
class VJEPABundle:
    context_encoder: torch.nn.Module
    target_encoder: torch.nn.Module
    predictor: torch.nn.Module
    device: torch.device
    dtype: torch.dtype
    horizon: int
    manifest: dict[str, object]

    def score(
        self,
        context_frames: Sequence[np.ndarray],
        target_frames: Sequence[np.ndarray],
    ) -> ScoreBundle:
        score = score_with_modules(
            context_frames,
            target_frames,
            context_encoder=self.context_encoder,
            target_encoder=self.target_encoder,
            predictor=self.predictor,
            device=self.device,
            dtype=self.dtype,
            horizon=self.horizon,
        )
        score.timings_ms["cold_model_load"] = float(self.manifest["model_load_ms"])
        return score


class MockTransitionScorer:
    """Deterministic CPU scorer for CI and end-to-end interface tests."""

    def __init__(self, horizon: int) -> None:
        self.horizon = horizon

    def score(
        self,
        context_frames: Sequence[np.ndarray],
        target_frames: Sequence[np.ndarray],
    ) -> ScoreBundle:
        cheap = compute_cheap_features(context_frames, target_frames)
        prediction_error = (
            0.75 * cheap.histogram_distance
            + 0.75 * cheap.ssim_distance
            + 0.05 * min(cheap.flow_discontinuity, 10.0)
        )
        encoder_distance = 0.65 * cheap.histogram_distance + 0.35 * cheap.ssim_distance
        tubelet_count = max(1, self.horizon // TUBELET_SIZE)
        return ScoreBundle(
            prediction_error=float(prediction_error),
            encoder_distance=float(encoder_distance),
            histogram_distance=cheap.histogram_distance,
            ssim_distance=cheap.ssim_distance,
            flow_discontinuity=cheap.flow_discontinuity,
            tubelet_errors=[float(prediction_error)] * tubelet_count,
            timings_ms={"cheap_features": cheap.elapsed_ms, "model_inference": 0.0},
        )


def future_masks(
    horizon: int,
    *,
    total_frames: int = TOTAL_FRAMES,
    image_size: int = IMAGE_SIZE,
    patch_size: int = PATCH_SIZE,
    tubelet_size: int = TUBELET_SIZE,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    if horizon <= 0 or horizon >= total_frames or horizon % tubelet_size:
        raise ValueError("Horizon must be positive, shorter than the clip, and tubelet-aligned.")
    if total_frames % tubelet_size:
        raise ValueError("Total frames must be tubelet-aligned.")
    spatial_tokens = (image_size // patch_size) ** 2
    temporal_tokens = total_frames // tubelet_size
    target_temporal = horizon // tubelet_size
    context_end = (temporal_tokens - target_temporal) * spatial_tokens
    total_tokens = temporal_tokens * spatial_tokens
    context = torch.arange(context_end, dtype=torch.long, device=device).unsqueeze(0)
    target = torch.arange(context_end, total_tokens, dtype=torch.long, device=device).unsqueeze(0)
    return context, target


def _prepare_frame(frame: np.ndarray) -> torch.Tensor:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected an RGB frame, received shape {frame.shape}.")
    tensor = torch.from_numpy(np.array(frame, dtype=np.uint8, copy=True)).permute(2, 0, 1)
    tensor = tensor.float().unsqueeze(0) / 255.0
    _, _, height, width = tensor.shape
    short_side = int(IMAGE_SIZE * 256 / 224)
    if height <= width:
        resized_height = short_side
        resized_width = round(width * short_side / height)
    else:
        resized_width = short_side
        resized_height = round(height * short_side / width)
    tensor = functional.interpolate(
        tensor,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )
    top = (resized_height - IMAGE_SIZE) // 2
    left = (resized_width - IMAGE_SIZE) // 2
    return tensor[0, :, top : top + IMAGE_SIZE, left : left + IMAGE_SIZE]


def _prepare_video(
    frames: Sequence[np.ndarray],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if len(frames) != TOTAL_FRAMES:
        raise ValueError(f"Expected {TOTAL_FRAMES} frames, received {len(frames)}.")
    # DAVIS includes a few sequences whose source frames change dimensions. Normalize each
    # frame independently before stacking so a boundary never inherits padding or distortion
    # from the neighboring frame's geometry.
    tensor = torch.stack([_prepare_frame(frame) for frame in frames])
    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.permute(1, 0, 2, 3).unsqueeze(0).to(device=device, dtype=dtype)


def _select_tokens(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
    return torch.gather(tokens, dim=1, index=expanded)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def score_with_modules(
    context_frames: Sequence[np.ndarray],
    target_frames: Sequence[np.ndarray],
    *,
    context_encoder: torch.nn.Module,
    target_encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    horizon: int,
) -> ScoreBundle:
    context_count = TOTAL_FRAMES - horizon
    if len(context_frames) != context_count or len(target_frames) != horizon:
        raise ValueError(
            f"Expected {context_count} context and {horizon} target frames; "
            f"received {len(context_frames)} and {len(target_frames)}."
    )
    cheap = compute_cheap_features(context_frames, target_frames)
    video = _prepare_video([*context_frames, *target_frames], device, dtype)
    context_mask, target_mask = future_masks(horizon, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    _sync(device)
    started = time.perf_counter()
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=device.type == "cuda",
        ),
    ):
        target_all = target_encoder(video)
        target_tokens = _select_tokens(target_all, target_mask)
        target_tokens = functional.layer_norm(target_tokens, (target_tokens.shape[-1],))
        context_tokens = context_encoder(video, context_mask)
        predicted_tokens = predictor(context_tokens, context_mask, target_mask)
        if isinstance(predicted_tokens, tuple):
            predicted_tokens = predicted_tokens[0]
    _sync(device)
    inference_ms = (time.perf_counter() - started) * 1000.0
    peak_vram_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
    )
    if predicted_tokens.shape != target_tokens.shape:
        raise ModelLoadError(
            "Predictor and target representations are incompatible: "
            f"{tuple(predicted_tokens.shape)} vs {tuple(target_tokens.shape)}."
        )

    absolute_error = torch.abs(predicted_tokens.float() - target_tokens.float())
    prediction_error = float(absolute_error.mean().cpu())
    spatial_tokens = (IMAGE_SIZE // PATCH_SIZE) ** 2
    tubelet_errors = (
        absolute_error.reshape(horizon // TUBELET_SIZE, spatial_tokens, -1)
        .mean(dim=(1, 2))
        .cpu()
        .tolist()
    )
    context_boundary = _select_tokens(
        target_all,
        context_mask[:, -spatial_tokens:],
    ).mean(dim=1)
    target_boundary = _select_tokens(
        target_all,
        target_mask[:, :spatial_tokens],
    ).mean(dim=1)
    encoder_distance = float(
        (1.0 - functional.cosine_similarity(context_boundary, target_boundary)).mean().float().cpu()
    )
    return ScoreBundle(
        prediction_error=prediction_error,
        encoder_distance=encoder_distance,
        histogram_distance=cheap.histogram_distance,
        ssim_distance=cheap.ssim_distance,
        flow_discontinuity=cheap.flow_discontinuity,
        tubelet_errors=[float(value) for value in tubelet_errors],
        timings_ms={
            "cheap_features": cheap.elapsed_ms,
            "model_inference": inference_ms,
            "peak_vram_mb": float(peak_vram_mb),
        },
    )


def _clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key.replace("module.", "").replace("backbone.", ""): value
        for key, value in state_dict.items()
    }


def _load_checked(module: torch.nn.Module, state: dict[str, torch.Tensor], label: str) -> None:
    incompatible = module.load_state_dict(_clean_state_dict(state), strict=False)
    allowed_fragment = ("pos_embed",)
    bad_missing = [
        key
        for key in incompatible.missing_keys
        if not any(fragment in key for fragment in allowed_fragment)
    ]
    bad_unexpected = [
        key for key in incompatible.unexpected_keys if not any(x in key for x in allowed_fragment)
    ]
    if bad_missing or bad_unexpected:
        raise ModelLoadError(
            f"Unexpected {label} checkpoint mismatch. Missing={bad_missing[:8]}, "
            f"unexpected={bad_unexpected[:8]}."
        )


def _download_progress(written: int, total: int | None) -> None:
    if total:
        print(f"\rDownloading V-JEPA checkpoint: {100 * written / total:5.1f}%", end="", flush=True)


def load_vjepa(
    *,
    horizon: int,
    cache_dir: Path | None = None,
    device: str | None = None,
) -> VJEPABundle:
    load_started = time.perf_counter()
    cache_dir = (cache_dir or default_cache_dir()).resolve()
    checkpoint_path = cache_dir / "checkpoints" / "vitl.pt"
    if not checkpoint_path.exists():
        try:
            download_resumable(VJEPA_CHECKPOINT_URL, checkpoint_path, progress=_download_progress)
            print()
        except Exception as exc:
            raise ModelLoadError(
                "V-JEPA checkpoint download failed. This project bypasses the upstream "
                "localhost:8300 Hub value and expects the explicit official vitl.pt URL. "
                f"Partial downloads remain under {checkpoint_path.parent}."
            ) from exc

    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if selected_device.type == "cuda" else torch.float32
    try:
        loaded = torch.hub.load(
            f"{VJEPA_REPOSITORY}:{VJEPA_COMMIT}",
            VJEPA_MODEL_ENTRYPOINT,
            pretrained=False,
            num_frames=TOTAL_FRAMES,
            trust_repo=True,
            skip_validation=True,
        )
        context_encoder, predictor = loaded
        target_encoder = copy.deepcopy(context_encoder)
    except Exception as exc:
        raise ModelLoadError(
            f"Could not instantiate V-JEPA from {VJEPA_REPOSITORY}@{VJEPA_COMMIT}."
        ) from exc

    try:
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
        except (TypeError, RuntimeError):
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        required = {"encoder", "target_encoder", "predictor"}
        if missing := required - set(checkpoint):
            raise ModelLoadError(f"Checkpoint is missing required states: {sorted(missing)}")
        _load_checked(context_encoder, checkpoint["encoder"], "context encoder")
        _load_checked(target_encoder, checkpoint["target_encoder"], "target encoder")
        _load_checked(predictor, checkpoint["predictor"], "predictor")
    except ModelLoadError:
        raise
    except Exception as exc:
        raise ModelLoadError(f"Could not load the checkpoint at {checkpoint_path}.") from exc
    finally:
        if "checkpoint" in locals():
            del checkpoint
        gc.collect()

    for module in (context_encoder, target_encoder, predictor):
        module.eval().requires_grad_(False).to(device=selected_device, dtype=dtype)

    manifest: dict[str, object] = {
        "repository": VJEPA_REPOSITORY,
        "commit": VJEPA_COMMIT,
        "entrypoint": VJEPA_MODEL_ENTRYPOINT,
        "checkpoint_url": VJEPA_CHECKPOINT_URL,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "total_frames": TOTAL_FRAMES,
        "horizon": horizon,
        "device": str(selected_device),
        "dtype": str(dtype),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": (
            torch.cuda.get_device_name(selected_device)
            if selected_device.type == "cuda"
            else None
        ),
        "gpu_arches": torch.cuda.get_arch_list() if selected_device.type == "cuda" else [],
        "model_load_ms": (time.perf_counter() - load_started) * 1000.0,
    }
    write_json(cache_dir / "model_manifest.json", manifest)
    return VJEPABundle(
        context_encoder=context_encoder,
        target_encoder=target_encoder,
        predictor=predictor,
        device=selected_device,
        dtype=dtype,
        horizon=horizon,
        manifest=manifest,
    )
