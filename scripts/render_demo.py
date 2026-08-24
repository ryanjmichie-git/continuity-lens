from __future__ import annotations

import argparse
import textwrap
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

WIDTH = 1280
HEIGHT = 720
FPS = 10
BACKGROUND = "#0f172a"
PANEL = "#172033"
TEXT = "#f8fafc"
MUTED = "#b8c4d6"
TEAL = "#2dd4bf"
ORANGE = "#f59e0b"
PURPLE = "#8b5cf6"


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    title: str
    body: str
    asset: str | None = None
    crop: tuple[float, float] | None = None
    accent: str = TEAL


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "seguisb.ttf" if bold else "segoeui.ttf"
    path = Path("C:/Windows/Fonts") / name
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    position: tuple[int, int],
    *,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: str,
    width: int,
    spacing: int = 8,
) -> int:
    average_character = max(7, int(getattr(font, "size", 24) * 0.52))
    lines: list[str] = []
    for paragraph in text.splitlines():
        if not paragraph:
            lines.append("")
            continue
        lines.extend(textwrap.wrap(paragraph, width=max(12, width // average_character)))
    x, y = position
    draw.multiline_text((x, y), "\n".join(lines), font=font, fill=fill, spacing=spacing)
    line_height = int(getattr(font, "size", 24) * 1.25) + spacing
    return y + line_height * len(lines)


def _fit_asset(path: Path, size: tuple[int, int], crop: tuple[float, float] | None) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    if crop:
        top_fraction, bottom_fraction = crop
        top = int(image.height * top_fraction)
        bottom = int(image.height * bottom_fraction)
        image = image.crop((0, top, image.width, max(top + 1, bottom)))
    return ImageOps.contain(image, size, method=Image.Resampling.LANCZOS)


def _render_slide(segment: Segment, root: Path) -> Image.Image:
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((45, 38, WIDTH - 45, HEIGHT - 38), radius=26, fill=PANEL)
    draw.rounded_rectangle((45, 38, 59, HEIGHT - 38), radius=7, fill=segment.accent)
    title_font = _font(42, bold=True)
    body_font = _font(25)
    label_font = _font(17, bold=True)
    draw.text((86, 66), segment.title, font=title_font, fill=TEXT)
    draw.text((WIDTH - 275, 77), "CONTINUITY LENS", font=label_font, fill=segment.accent)
    if segment.asset:
        asset_box = (735, 135, 1195, 625)
        asset = _fit_asset(
            root / segment.asset,
            (asset_box[2] - asset_box[0], asset_box[3] - asset_box[1]),
            segment.crop,
        )
        x = asset_box[0] + (asset_box[2] - asset_box[0] - asset.width) // 2
        y = asset_box[1] + (asset_box[3] - asset_box[1] - asset.height) // 2
        draw.rounded_rectangle(asset_box, radius=18, fill="#ffffff")
        canvas.paste(asset, (x, y))
        body_width = 590
    else:
        body_width = 1040
    _draw_wrapped(
        draw,
        segment.body,
        (88, 155),
        font=body_font,
        fill=MUTED,
        width=body_width,
        spacing=12,
    )
    draw.text(
        (88, HEIGHT - 79),
        "Evidence level is shown explicitly; no human validation claim.",
        font=_font(17),
        fill="#8291a8",
    )
    return canvas


def render(output: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    segments = (
        Segment(
            0,
            9,
            "Continuity Lens",
            "Does masked-future latent prediction identify implausible video transitions better "
            "than inexpensive appearance and optical-flow signals?\n\n"
            "A research-to-product decision, not a forced model win.",
            accent=TEAL,
        ),
        Segment(
            9,
            24,
            "Method before outcome",
            "• V-JEPA 2 ViT-L context encoder, EMA target, and predictor\n"
            "• 60 DAVIS development groups; 30 held out\n"
            "• 2/4/8-frame horizon selection on development only\n"
            "• One frozen test pass and 5,000 source-grouped bootstrap samples\n"
            "• Cheap ensemble: HSV, SSIM, and Farneback-flow consistency",
            accent=PURPLE,
        ),
        Segment(
            24,
            44,
            "Inspectable local workflow",
            "Choose a labeled generated pair or upload two clips.\n\n"
            "The app shows the frozen 4+2 boundary evidence and keeps media local. Generated "
            "examples are qualitative diagnostics—not performance evidence.",
            asset="docs/assets/app-result.png",
            crop=(0.0, 0.50),
            accent=TEAL,
        ),
        Segment(
            44,
            64,
            "Signals, limitations, and cost",
            "Cheap and hybrid scores are shown separately. Each raw component states its direction "
            "and meaning. Cold load, decode, feature, model, and warm-pipeline latency are "
            "visible.\n\n"
            "No accept, edit, or regenerate instruction is produced.",
            asset="docs/assets/app-result.png",
            crop=(0.43, 1.0),
            accent=TEAL,
        ),
        Segment(
            64,
            82,
            "Held-out result: do not ship V-JEPA here",
            "Predictor AUPRC: 0.778\nCheap-only AUPRC: 0.975\nHybrid AUPRC: 0.969\n\n"
            "Predictor minus cheap ΔAUPRC = −0.197.\n"
            "Grouped 95% interval: [−0.229, −0.143].",
            asset="artifacts/test/figures/primary_auprc.png",
            accent=ORANGE,
        ),
        Segment(
            82,
            98,
            "Failure analysis strengthened the decision",
            "High-motion natural boundaries create large latent errors, while some temporal skips "
            "remain low.\n\nIn 12 controlled geometry cases, prediction error moved in the "
            "expected "
            "direction only 5 of 9 times; the frozen threshold detected none of the anomalies.",
            asset="artifacts/diagnostic/prediction_error_by_condition.png",
            accent=ORANGE,
        ),
        Segment(
            98,
            112,
            "Product decision",
            "Do not add V-JEPA latency and complexity to this QA path from current evidence.\n\n"
            "Keep the inexpensive ensemble as the control. Preserve the app as an inspectable "
            "research artifact. Report the negative result and its scope honestly.",
            accent=ORANGE,
        ),
        Segment(
            112,
            120,
            "Next experiment",
            "Collect creator-authored failures and intentional-cut controls. Measure incremental "
            "recall at a fixed review budget. Test localized token aggregation or a continuity "
            "head "
            "only where cheap signals fail.\n\nMethodological integrity is the portfolio result.",
            accent=PURPLE,
        ),
    )
    slides = [_render_slide(segment, root) for segment in segments]
    output.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(output), mode="w", options={"movflags": "+faststart"}) as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width = WIDTH
        stream.height = HEIGHT
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "23", "preset": "medium"}
        total_frames = int(segments[-1].end * FPS)
        for frame_index in range(total_frames):
            timestamp = frame_index / FPS
            segment_index = next(
                index
                for index, segment in enumerate(segments)
                if segment.start <= timestamp < segment.end
            )
            segment = segments[segment_index]
            image = slides[segment_index]
            fade_duration = 0.45
            if segment_index and timestamp - segment.start < fade_duration:
                alpha = (timestamp - segment.start) / fade_duration
                image = Image.blend(slides[segment_index - 1], image, alpha)
            video_frame = av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the captioned Continuity Lens walkthrough."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets/continuity-lens-demo.mp4"),
    )
    args = parser.parse_args()
    render(args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
