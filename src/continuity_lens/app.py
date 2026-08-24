from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd

from continuity_lens.config import (
    DEFAULT_HORIZON,
    TOTAL_FRAMES,
    default_artifacts_dir,
    default_data_dir,
)
from continuity_lens.schemas import LinearCalibrator, ScoreBundle
from continuity_lens.utils import read_json
from continuity_lens.video import VideoInputError, boundary_strip, sample_transition
from continuity_lens.vjepa import MockTransitionScorer, TransitionScorer, load_vjepa

LIMITATIONS = """
### What this score does—and does not mean

- It measures agreement with a masked-latent prediction task; it is not a probability of
  physical correctness.
- V-JEPA was not trained specifically as a cut-quality detector, and a future-only mask is
  a distribution shift.
- Scores are calibrated only to the documented DAVIS corruptions. Camera cuts, animation,
  and stylized footage may behave differently.
- Treat the result as triage evidence for a human reviewer, not an automatic rejection decision.
"""


class RuntimeAnalyzer:
    def __init__(self, *, artifacts_root: Path, mock_model: bool = False) -> None:
        self.artifacts_root = artifacts_root.resolve()
        self.mock_model = mock_model
        frozen_path = self.artifacts_root / "dev" / "frozen_spec.json"
        self.spec: dict[str, Any] | None = read_json(frozen_path) if frozen_path.exists() else None
        self.horizon = int(self.spec["selected_horizon"]) if self.spec else DEFAULT_HORIZON
        self.hybrid = (
            LinearCalibrator.from_dict(self.spec["hybrid_calibrator"]) if self.spec else None
        )
        self.threshold = float(self.spec["hybrid_threshold"]) if self.spec else None
        self._scorer: TransitionScorer | None = None

    @property
    def scorer(self) -> TransitionScorer:
        if self._scorer is None:
            self._scorer = (
                MockTransitionScorer(self.horizon)
                if self.mock_model
                else load_vjepa(horizon=self.horizon)
            )
        return self._scorer

    def risk(self, score: ScoreBundle) -> float | None:
        return self.hybrid.predict_probability(score.to_dict()) if self.hybrid else None

    def analyze(
        self, clip_a: str, clip_b: str
    ) -> tuple[str, Any, pd.DataFrame, Any, dict[str, float]]:
        if not clip_a or not clip_b:
            raise gr.Error("Upload both Clip A and Clip B.")
        context_count = TOTAL_FRAMES - self.horizon
        try:
            context, target = sample_transition(
                clip_a,
                clip_b,
                context_count=context_count,
                target_count=self.horizon,
            )
        except VideoInputError as exc:
            raise gr.Error(str(exc)) from exc
        score = self.scorer.score(context, target)
        risk = self.risk(score)
        if risk is None:
            headline = (
                "## Not calibrated yet\n\n"
                "Raw research signals are available below. Run the development benchmark before "
                "presenting a 0–100 risk score."
            )
        else:
            risk_percent = 100.0 * risk
            disposition = (
                "review this transition"
                if risk >= float(self.threshold)
                else "lower review priority"
            )
            headline = (
                f"## {risk_percent:.0f}/100 experimental hybrid risk\n\n"
                f"Suggested action: **{disposition}**. This is not a probability of "
                "physical correctness."
            )
        methods = pd.DataFrame(
            [
                ("Masked-future prediction error", score.prediction_error),
                ("Encoder boundary distance", score.encoder_distance),
                ("HSV histogram distance", score.histogram_distance),
                ("SSIM distance", score.ssim_distance),
                ("Flow discontinuity", score.flow_discontinuity),
            ],
            columns=["Signal", "Value"],
        )
        figure, axis = plt.subplots(figsize=(6.4, 3.0))
        axis.plot(range(1, len(score.tubelet_errors) + 1), score.tubelet_errors, marker="o")
        axis.set_xlabel("Predicted target tubelet")
        axis.set_ylabel("Mean latent L1 error")
        axis.set_title("Where prediction error rises after the boundary")
        axis.grid(alpha=0.2)
        figure.tight_layout()
        return headline, boundary_strip(context, target), methods, figure, score.timings_ms


@lru_cache(maxsize=4)
def _runtime(artifacts_root: str, mock_model: bool) -> RuntimeAnalyzer:
    return RuntimeAnalyzer(artifacts_root=Path(artifacts_root), mock_model=mock_model)


def create_app(
    *,
    artifacts_root: Path | None = None,
    mock_model: bool = False,
) -> gr.Blocks:
    artifacts_root = (artifacts_root or default_artifacts_dir()).resolve()
    analyzer = _runtime(str(artifacts_root), mock_model)
    demo_root = default_data_dir() / "demo"
    demo_examples = [
        [str(context), str(context.with_name("target.mp4"))]
        for context in sorted(demo_root.glob("*/context.mp4"))
        if context.with_name("target.mp4").exists()
    ]
    metrics_path = artifacts_root / "test" / "metrics.json"
    metrics = read_json(metrics_path) if metrics_path.exists() else None
    with gr.Blocks(title="Continuity Lens") as demo:
        gr.Markdown(
            "# Continuity Lens\n"
            "Compare the end of Clip A with the beginning of Clip B using masked-future "
            "V-JEPA prediction and inexpensive visual baselines."
        )
        if metrics:
            gr.Markdown(
                "**Held-out product decision:** the cheap-only ensemble outperformed the "
                f"predictor ({metrics['auprc']['cheap']:.3f} vs "
                f"{metrics['auprc']['predictor']:.3f} AUPRC). The hybrid score below is an "
                "inspectable research output, not a recommended production model."
            )
        with gr.Row():
            clip_a = gr.Video(label="Clip A — context", sources=["upload"])
            clip_b = gr.Video(label="Clip B — target", sources=["upload"])
        if demo_examples:
            gr.Examples(
                examples=demo_examples,
                inputs=[clip_a, clip_b],
                label="Generated diagnostic examples (qualitative only)",
            )
        analyze_button = gr.Button("Analyze transition", variant="primary")
        headline = gr.Markdown()
        boundary = gr.Image(label="Boundary strip: A tail → B head", type="numpy")
        with gr.Row():
            scores = gr.Dataframe(label="Signals", interactive=False)
            curve = gr.Plot(label="Target-tubelet error")
        timings = gr.JSON(label="Component latency (milliseconds)")
        gr.Markdown(LIMITATIONS)
        analyze_button.click(
            fn=analyzer.analyze,
            inputs=[clip_a, clip_b],
            outputs=[headline, boundary, scores, curve, timings],
        )
    return demo
