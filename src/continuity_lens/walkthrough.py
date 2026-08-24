from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from continuity_lens.config import (
    DEFAULT_HORIZON,
    TOTAL_FRAMES,
    default_artifacts_dir,
    default_data_dir,
)
from continuity_lens.schemas import LinearCalibrator
from continuity_lens.utils import read_json, write_json
from continuity_lens.video import sample_transition
from continuity_lens.vjepa import MockTransitionScorer, TransitionScorer, load_vjepa


def _calibrators(
    artifacts_root: Path,
) -> tuple[int, LinearCalibrator | None, LinearCalibrator | None, float | None]:
    frozen_path = artifacts_root / "dev" / "frozen_spec.json"
    if not frozen_path.exists():
        return DEFAULT_HORIZON, None, None, None
    spec = read_json(frozen_path)
    return (
        int(spec["selected_horizon"]),
        LinearCalibrator.from_dict(spec["cheap_calibrator"]),
        LinearCalibrator.from_dict(spec["hybrid_calibrator"]),
        float(spec["hybrid_threshold"]),
    )


def run_demo_walkthrough(
    *,
    data_root: Path | None = None,
    artifacts_root: Path | None = None,
    mock_model: bool = False,
) -> dict[str, Any]:
    """Run every generated example and record system behavior, never simulated user evidence."""
    data_root = (data_root or default_data_dir()).resolve()
    artifacts_root = (artifacts_root or default_artifacts_dir()).resolve()
    output_dir = artifacts_root / "usability"
    output_dir.mkdir(parents=True, exist_ok=True)
    horizon, cheap, hybrid, threshold = _calibrators(artifacts_root)
    demo_root = data_root / "demo"
    pairs = [
        (context.parent.name, context, context.with_name("target.mp4"))
        for context in sorted(demo_root.glob("*/context.mp4"))
        if context.with_name("target.mp4").exists()
    ]
    if len(pairs) != 6:
        raise ValueError(
            f"Expected six generated demo pairs under {demo_root}; found {len(pairs)}. "
            "Run `continuity-lens data demos` first."
        )

    model_load_started = time.perf_counter()
    scorer: TransitionScorer = (
        MockTransitionScorer(horizon) if mock_model else load_vjepa(horizon=horizon)
    )
    model_load_ms = (time.perf_counter() - model_load_started) * 1000.0
    rows: list[dict[str, Any]] = []
    for case_id, context_path, target_path in pairs:
        total_started = time.perf_counter()
        decode_started = time.perf_counter()
        context, target = sample_transition(
            context_path,
            target_path,
            context_count=TOTAL_FRAMES - horizon,
            target_count=horizon,
        )
        decode_ms = (time.perf_counter() - decode_started) * 1000.0
        score = scorer.score(context, target)
        values = score.to_dict()
        condition = case_id.split("-", maxsplit=1)[1]
        expected_discontinuous = int(condition != "continuous")
        cheap_risk = cheap.predict_probability(values) if cheap else None
        hybrid_risk = hybrid.predict_probability(values) if hybrid else None
        flagged = (
            int(hybrid_risk >= threshold)
            if hybrid_risk is not None and threshold is not None
            else None
        )
        rows.append(
            {
                "case_id": case_id,
                "condition": condition,
                "expected_discontinuous": expected_discontinuous,
                "cheap_risk": cheap_risk,
                "experimental_hybrid_risk": hybrid_risk,
                "flagged_at_frozen_threshold": flagged,
                "prediction_error": score.prediction_error,
                "encoder_distance": score.encoder_distance,
                "histogram_distance": score.histogram_distance,
                "ssim_distance": score.ssim_distance,
                "flow_discontinuity": score.flow_discontinuity,
                "decode_ms": decode_ms,
                "cheap_features_ms": score.timings_ms.get("cheap_features", 0.0),
                "model_inference_ms": score.timings_ms.get("model_inference", 0.0),
                "total_warm_ms": (time.perf_counter() - total_started) * 1000.0,
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "system_walkthrough.csv", index=False)
    summary: dict[str, Any] = {
        "evidence_level": "EMPIRICAL — system behavior; not human usability evidence",
        "model_mode": "mock" if mock_model else "V-JEPA 2 ViT-L",
        "horizon": horizon,
        "case_count": len(frame),
        "known_good_cases": int((frame["expected_discontinuous"] == 0).sum()),
        "known_bad_cases": int((frame["expected_discontinuous"] == 1).sum()),
        "model_load_ms": model_load_ms,
        "mean_decode_ms": float(frame["decode_ms"].mean()),
        "mean_cheap_features_ms": float(frame["cheap_features_ms"].mean()),
        "mean_model_inference_ms": float(frame["model_inference_ms"].mean()),
        "mean_total_warm_ms": float(frame["total_warm_ms"].mean()),
        "hybrid_flags": (
            int(frame["flagged_at_frozen_threshold"].sum())
            if frame["flagged_at_frozen_threshold"].notna().all()
            else None
        ),
        "known_bad_flag_recall": (
            float(
                frame.loc[
                    frame["expected_discontinuous"] == 1,
                    "flagged_at_frozen_threshold",
                ].mean()
            )
            if frame["flagged_at_frozen_threshold"].notna().all()
            else None
        ),
        "known_good_false_alert_rate": (
            float(
                frame.loc[
                    frame["expected_discontinuous"] == 0,
                    "flagged_at_frozen_threshold",
                ].mean()
            )
            if frame["flagged_at_frozen_threshold"].notna().all()
            else None
        ),
        "limitations": [
            "Generated geometry is a diagnostic lane, not natural-video evidence.",
            "System timings are measured; human task times are not observed.",
            "Archetype interpretations are hypotheses, not participant statements.",
        ],
    }
    write_json(output_dir / "system_walkthrough.json", summary)
    return {"summary": summary, "rows": rows, "output_dir": str(output_dir)}
