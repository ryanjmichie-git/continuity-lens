from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image

from continuity_lens.config import VIDEO_SAMPLE_FPS


class VideoInputError(ValueError):
    """Raised when a video cannot supply the requested unique frame window."""


def decode_video_frames(
    path: str | Path,
    *,
    sample_fps: float = VIDEO_SAMPLE_FPS,
    max_decoded_frames: int = 12_000,
) -> list[np.ndarray]:
    """Decode RGB frames at a fixed rate without manufacturing padded frames."""
    path = Path(path)
    if not path.exists():
        raise VideoInputError(f"Video does not exist: {path}")
    sampled: list[np.ndarray] = []
    seen_pts: set[int] = set()
    next_time = 0.0
    decoded = 0
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise VideoInputError(f"No video stream found in {path.name}")
            stream = container.streams.video[0]
            fallback_fps = float(stream.average_rate) if stream.average_rate else sample_fps
            for frame in container.decode(stream):
                decoded += 1
                if decoded > max_decoded_frames:
                    raise VideoInputError(
                        f"{path.name} exceeds the {max_decoded_frames}-frame safety limit."
                    )
                pts_key = int(frame.pts) if frame.pts is not None else decoded
                if pts_key in seen_pts:
                    continue
                seen_pts.add(pts_key)
                timestamp = (
                    float(frame.time)
                    if frame.time is not None
                    else (decoded - 1) / fallback_fps
                )
                if timestamp + 1e-9 < next_time:
                    continue
                sampled.append(frame.to_ndarray(format="rgb24"))
                next_time += 1.0 / sample_fps
    except (av.error.FFmpegError, OSError) as exc:
        raise VideoInputError(f"Could not decode {path.name}: {exc}") from exc
    if not sampled:
        raise VideoInputError(f"No decodable frames found in {path.name}")
    return sampled


def sample_transition(
    clip_a: str | Path,
    clip_b: str | Path,
    *,
    context_count: int,
    target_count: int,
    sample_fps: float = VIDEO_SAMPLE_FPS,
) -> tuple[np.ndarray, np.ndarray]:
    context_frames = decode_video_frames(clip_a, sample_fps=sample_fps)
    target_frames = decode_video_frames(clip_b, sample_fps=sample_fps)
    if len(context_frames) < context_count:
        raise VideoInputError(
            f"Clip A supplies {len(context_frames)} sampled frames; {context_count} are required."
        )
    if len(target_frames) < target_count:
        raise VideoInputError(
            f"Clip B supplies {len(target_frames)} sampled frames; {target_count} are required."
        )
    return np.stack(context_frames[-context_count:]), np.stack(target_frames[:target_count])


def load_image_frames(paths: tuple[str, ...] | list[str]) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise VideoInputError(f"Frame does not exist: {path}")
        with Image.open(path) as image:
            frames.append(np.asarray(image.convert("RGB")))
    if not frames:
        raise VideoInputError("A transition must contain at least one frame.")
    return frames


def boundary_strip(
    context_frames: Sequence[np.ndarray],
    target_frames: Sequence[np.ndarray],
    *,
    side: int = 144,
    count_each: int = 4,
) -> np.ndarray:
    left = list(context_frames[-min(count_each, len(context_frames)) :])
    right = list(target_frames[: min(count_each, len(target_frames))])
    cells = [
        cv2.resize(frame, (side, side), interpolation=cv2.INTER_AREA)
        for frame in left + right
    ]
    if not cells:
        raise VideoInputError("Cannot render an empty boundary strip.")
    separator = np.full((side, 4, 3), (13, 148, 136), dtype=np.uint8)
    split_at = len(left)
    parts: list[np.ndarray] = []
    for index, cell in enumerate(cells):
        if index == split_at:
            parts.append(separator)
        parts.append(cell)
    return np.concatenate(parts, axis=1)
