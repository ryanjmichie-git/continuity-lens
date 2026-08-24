from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import pytest

from continuity_lens.synthetic import generate_demo_clips
from continuity_lens.video import VideoInputError, boundary_strip, sample_transition


def _write_video(path: Path, colors: list[int], fps: int = 4) -> None:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = 48
        stream.height = 32
        stream.pix_fmt = "yuv420p"
        for color in colors:
            pixels = np.full((32, 48, 3), color, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_video_sampling_uses_real_decoded_frames(tmp_path: Path) -> None:
    clip_a = tmp_path / "a.mp4"
    clip_b = tmp_path / "b.mp4"
    _write_video(clip_a, list(range(16)))
    _write_video(clip_b, list(range(20, 28)))
    context, target = sample_transition(clip_a, clip_b, context_count=12, target_count=4)
    assert context.shape == (12, 32, 48, 3)
    assert target.shape == (4, 32, 48, 3)
    strip = boundary_strip(context, target)
    assert strip.shape[0] == 144
    assert strip.shape[2] == 3


def test_short_video_is_rejected_instead_of_padded(tmp_path: Path) -> None:
    clip = tmp_path / "short.mp4"
    _write_video(clip, [1, 2, 3])
    with pytest.raises(VideoInputError, match="required"):
        sample_transition(clip, clip, context_count=12, target_count=4)


def test_generated_demo_pairs_round_trip_without_padding(tmp_path: Path) -> None:
    pairs = generate_demo_clips(tmp_path / "demo", horizon=4)
    assert len(pairs) == 6
    context, target = sample_transition(
        *pairs[0],
        context_count=12,
        target_count=4,
    )
    assert context.shape[0] == 12
    assert target.shape[0] == 4
