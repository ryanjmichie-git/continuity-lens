from __future__ import annotations

import time
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
- The held-out benchmark did not justify using V-JEPA for an automated decision. Treat these
  values as inspectable research signals, not accept, edit, or regenerate instructions.
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
        self.cheap = (
            LinearCalibrator.from_dict(self.spec["cheap_calibrator"]) if self.spec else None
        )
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

    def cheap_risk(self, score: ScoreBundle) -> float | None:
        return self.cheap.predict_probability(score.to_dict()) if self.cheap else None

    def analyze(
        self, clip_a: str, clip_b: str
    ) -> tuple[str, Any, pd.DataFrame, Any, dict[str, float]]:
        if not clip_a or not clip_b:
            raise gr.Error("Upload both Clip A and Clip B.")
        total_started = time.perf_counter()
        context_count = TOTAL_FRAMES - self.horizon
        decode_started = time.perf_counter()
        try:
            context, target = sample_transition(
                clip_a,
                clip_b,
                context_count=context_count,
                target_count=self.horizon,
            )
        except VideoInputError as exc:
            raise gr.Error(str(exc)) from exc
        decode_ms = (time.perf_counter() - decode_started) * 1000.0
        model_was_cold = self._scorer is None
        score = self.scorer.score(context, target)
        hybrid_risk = self.risk(score)
        cheap_risk = self.cheap_risk(score)
        if hybrid_risk is None or cheap_risk is None:
            headline = (
                "## Not calibrated yet\n\n"
                "Raw research signals are available below. Run the development benchmark before "
                "presenting a 0–100 risk score."
            )
        else:
            headline = (
                f"## Cheap-only benchmark score: {100.0 * cheap_risk:.0f}/100\n\n"
                f"**Experimental hybrid score:** {100.0 * hybrid_risk:.0f}/100. "
                "These values are calibrated to constructed DAVIS corruptions, not to physical "
                "correctness or a production decision."
            )
        methods = pd.DataFrame(
            [
                (
                    "Masked-future prediction error",
                    score.prediction_error,
                    "Higher = less expected",
                    "L1 error between predicted and observed target latents",
                ),
                (
                    "Encoder boundary distance",
                    score.encoder_distance,
                    "Higher = less similar",
                    "Cosine distance between boundary representations",
                ),
                (
                    "HSV histogram distance",
                    score.histogram_distance,
                    "Higher = larger color shift",
                    "Appearance change across boundary frame pairs",
                ),
                (
                    "SSIM distance",
                    score.ssim_distance,
                    "Higher = larger structure shift",
                    "One minus structural similarity",
                ),
                (
                    "Flow discontinuity",
                    score.flow_discontinuity,
                    "Higher = unusual motion change",
                    "Boundary flow change relative to neighboring pairs",
                ),
            ],
            columns=["Signal", "Value", "Direction", "Interpretation"],
        )
        figure, axis = plt.subplots(figsize=(6.4, 3.0))
        axis.plot(range(1, len(score.tubelet_errors) + 1), score.tubelet_errors, marker="o")
        axis.set_xlabel("Predicted target tubelet")
        axis.set_ylabel("Mean latent L1 error")
        axis.set_title("Where prediction error rises after the boundary")
        axis.grid(alpha=0.2)
        figure.tight_layout()
        timings = dict(score.timings_ms)
        cold_model_load = timings.pop("cold_model_load", None)
        timings["cheap_features_including_flow"] = timings.pop("cheap_features", 0.0)
        timings["decode"] = decode_ms
        request_total = (time.perf_counter() - total_started) * 1000.0
        timings["request_total"] = request_total
        if model_was_cold and cold_model_load is not None:
            timings["cold_model_load"] = cold_model_load
            timings["warm_pipeline_estimate"] = max(0.0, request_total - cold_model_load)
        else:
            timings["warm_pipeline_measured"] = request_total
        return headline, boundary_strip(context, target), methods, figure, timings


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
    demo_examples = {
        context.parent.name: [str(context), str(context.with_name("target.mp4"))]
        for context in sorted(demo_root.glob("*/context.mp4"))
        if context.with_name("target.mp4").exists()
    }
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
            choices = [
                (case_id.replace("-", " — ").replace("_", " ").title(), case_id)
                for case_id in demo_examples
            ]
            with gr.Row():
                example_choice = gr.Dropdown(
                    choices=choices,
                    value=next(iter(demo_examples)),
                    label="Generated diagnostic example (qualitative only)",
                )
                load_example = gr.Button("Load generated example")

            def select_example(case_id: str) -> tuple[str, str]:
                context_path, target_path = demo_examples[case_id]
                return context_path, target_path

            load_example.click(
                fn=select_example,
                inputs=example_choice,
                outputs=[clip_a, clip_b],
            )
        analyze_button = gr.Button("Analyze transition", variant="primary")
        headline = gr.Markdown()
        boundary = gr.Image(
            label=(
                f"Boundary evidence: last 4 context frames → {analyzer.horizon} frozen target "
                "frames"
            ),
            type="numpy",
        )
        scores = gr.Dataframe(
            label="Signals",
            interactive=False,
            wrap=True,
            column_widths=[220, 110, 210, 360],
        )
        curve = gr.Plot(
            label=f"Target-tubelet error ({analyzer.horizon // 2} frozen tubelet)"
        )
        timings = gr.JSON(label="Component latency (milliseconds)")
        gr.Markdown(LIMITATIONS)
        analyze_button.click(
            fn=analyzer.analyze,
            inputs=[clip_a, clip_b],
            outputs=[headline, boundary, scores, curve, timings],
        )
    return demo
