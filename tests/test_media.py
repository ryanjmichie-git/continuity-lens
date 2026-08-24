from pathlib import Path

import av


def test_captioned_demo_is_two_minutes_and_browser_compatible() -> None:
    path = Path("docs/assets/continuity-lens-demo.mp4")
    assert path.exists()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        duration = float(stream.duration * stream.time_base)
        assert 119.5 <= duration <= 120.5
        assert stream.codec_context.name == "h264"
        assert (stream.width, stream.height) == (1280, 720)
