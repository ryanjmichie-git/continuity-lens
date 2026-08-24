from __future__ import annotations

from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

from continuity_lens.config import TOTAL_FRAMES, VIDEO_SAMPLE_FPS
from continuity_lens.dataset import write_records
from continuity_lens.schemas import TransitionRecord


def _frame(position: tuple[int, int] | None, *, side: int = 256) -> Image.Image:
    image = Image.new("RGB", (side, side), (24, 28, 39))
    if position is not None:
        draw = ImageDraw.Draw(image)
        x, y = position
        draw.ellipse((x - 16, y - 16, x + 16, y + 16), fill=(45, 212, 191))
    return image


def _position(
    frame_index: int,
    *,
    context_count: int,
    condition: str,
    offset: int,
) -> tuple[int, int] | None:
    x = 32 + 8 * frame_index
    y = 96 + offset
    if frame_index < context_count or condition == "continuous":
        return x, y
    target_index = frame_index - context_count
    if condition == "teleportation":
        return x + 72, y - 40
    if condition == "velocity_change":
        return 32 + 8 * context_count - 20 * target_index, y
    if condition == "object_disappearance":
        return None
    raise ValueError(f"Unknown diagnostic condition: {condition}")


def _write_video(path: Path, frames: list[Image.Image]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=round(VIDEO_SAMPLE_FPS))
        stream.width = frames[0].width
        stream.height = frames[0].height
        stream.pix_fmt = "yuv420p"
        for image in frames:
            frame = av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def generate_diagnostics(root: Path, *, horizon: int = 4) -> list[TransitionRecord]:
    """Create controlled diagnostics; these are never part of the primary benchmark."""
    if horizon <= 0 or horizon >= TOTAL_FRAMES:
        raise ValueError("Invalid diagnostic horizon.")
    root = root.resolve()
    records: list[TransitionRecord] = []
    context_count = TOTAL_FRAMES - horizon
    conditions = ("continuous", "teleportation", "velocity_change", "object_disappearance")
    for seed in range(3):
        offset = seed * 4
        for condition in conditions:
            case_id = f"diagnostic-{seed}-{condition}"
            case_root = root / case_id
            case_root.mkdir(parents=True, exist_ok=True)
            paths: list[str] = []
            for frame_index in range(TOTAL_FRAMES):
                position = _position(
                    frame_index,
                    context_count=context_count,
                    condition=condition,
                    offset=offset,
                )
                path = case_root / f"{frame_index:05d}.png"
                _frame(position).save(path)
                paths.append(str(path))
            records.append(
                TransitionRecord(
                    case_id=case_id,
                    source_id=f"synthetic-{seed}",
                    split="diagnostic",
                    discontinuous=int(condition != "continuous"),
                    corruption=condition,
                    lane="diagnostic",
                    horizon=horizon,
                    context_paths=tuple(paths[:context_count]),
                    target_paths=tuple(paths[context_count:]),
                )
            )
    write_records(root / "diagnostic_cases.jsonl", records)
    return records


def generate_demo_clips(root: Path, *, horizon: int) -> list[tuple[Path, Path]]:
    """Generate six self-owned clip pairs for qualitative app demonstrations."""
    if horizon <= 0 or horizon >= TOTAL_FRAMES:
        raise ValueError("Invalid demo horizon.")
    context_count = TOTAL_FRAMES - horizon
    cases = (
        (0, "continuous"),
        (0, "teleportation"),
        (0, "velocity_change"),
        (0, "object_disappearance"),
        (1, "continuous"),
        (1, "teleportation"),
    )
    pairs: list[tuple[Path, Path]] = []
    for seed, condition in cases:
        frames = [
            _frame(
                _position(
                    frame_index,
                    context_count=context_count,
                    condition=condition,
                    offset=seed * 8,
                )
            )
            for frame_index in range(TOTAL_FRAMES)
        ]
        case_root = root.resolve() / f"{seed}-{condition}"
        context_path = case_root / "context.mp4"
        target_path = case_root / "target.mp4"
        _write_video(context_path, frames[:context_count])
        _write_video(target_path, frames[context_count:])
        pairs.append((context_path, target_path))
    return pairs
