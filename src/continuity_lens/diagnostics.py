from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from continuity_lens.config import DEFAULT_HORIZON, default_artifacts_dir, default_data_dir
from continuity_lens.dataset import read_records
from continuity_lens.evaluation import score_records
from continuity_lens.schemas import LinearCalibrator
from continuity_lens.utils import read_json, write_json
from continuity_lens.vjepa import MockTransitionScorer, TransitionScorer, load_vjepa


def _setup(
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


def _directional_hits(frame: pd.DataFrame, column: str) -> dict[str, int]:
    hits = 0
    comparisons = 0
    for _, group in frame.groupby("source_id"):
        continuous = group.loc[group["corruption"] == "continuous", column]
        if continuous.empty:
            continue
        baseline = float(continuous.iloc[0])
        anomalous = group.loc[group["corruption"] != "continuous", column]
        hits += int((anomalous > baseline).sum())
        comparisons += len(anomalous)
    return {"hits": hits, "comparisons": comparisons}


def run_synthetic_diagnostics(
    *,
    data_root: Path | None = None,
    artifacts_root: Path | None = None,
    mock_model: bool = False,
) -> dict[str, Any]:
    """Score controlled geometry cases as diagnostics, never as headline evaluation."""
    data_root = (data_root or default_data_dir()).resolve()
    artifacts_root = (artifacts_root or default_artifacts_dir()).resolve()
    output_dir = artifacts_root / "diagnostic"
    output_dir.mkdir(parents=True, exist_ok=True)
    horizon, cheap, hybrid, threshold = _setup(artifacts_root)
    records_path = data_root / "diagnostics" / "diagnostic_cases.jsonl"
    if not records_path.exists():
        raise ValueError("Run `continuity-lens data diagnostics` before diagnostic scoring.")
    records = read_records(records_path)
    if len(records) != 12 or {record.horizon for record in records} != {horizon}:
        raise ValueError(
            f"Expected 12 diagnostics at the frozen horizon={horizon}. "
            f"Regenerate them with `continuity-lens data diagnostics --horizon {horizon}`."
        )
    scorer: TransitionScorer = (
        MockTransitionScorer(horizon) if mock_model else load_vjepa(horizon=horizon)
    )
    frame = score_records(records, scorer)
    if cheap and hybrid:
        frame["cheap_risk"] = [cheap.predict_probability(row) for row in frame.to_dict("records")]
        frame["experimental_hybrid_risk"] = [
            hybrid.predict_probability(row) for row in frame.to_dict("records")
        ]
        frame["flagged_at_frozen_threshold"] = (
            frame["experimental_hybrid_risk"] >= float(threshold)
        ).astype(int)
    frame.to_csv(output_dir / "predictions.csv", index=False)

    condition_means = (
        frame.groupby("corruption")[
            [
                "prediction_error",
                "encoder_distance",
                "histogram_distance",
                "ssim_distance",
                "flow_discontinuity",
            ]
        ]
        .mean()
        .round(6)
        .to_dict(orient="index")
    )
    summary: dict[str, Any] = {
        "evidence_level": "EMPIRICAL — controlled synthetic diagnostic; not headline evidence",
        "model_mode": "mock" if mock_model else "V-JEPA 2 ViT-L",
        "horizon": horizon,
        "case_count": len(frame),
        "condition_means": condition_means,
        "prediction_error_directional": _directional_hits(frame, "prediction_error"),
        "encoder_distance_directional": _directional_hits(frame, "encoder_distance"),
        "claim_boundary": (
            "These results reveal behavior on simple generated geometry only and cannot support "
            "claims about real or generative video."
        ),
    }
    if "flagged_at_frozen_threshold" in frame:
        anomalous = frame["discontinuous"] == 1
        summary["frozen_threshold_anomaly_recall"] = float(
            frame.loc[anomalous, "flagged_at_frozen_threshold"].mean()
        )
        summary["frozen_threshold_false_alert_rate"] = float(
            frame.loc[~anomalous, "flagged_at_frozen_threshold"].mean()
        )
        summary["cheap_risk_directional"] = _directional_hits(frame, "cheap_risk")
        summary["hybrid_risk_directional"] = _directional_hits(
            frame, "experimental_hybrid_risk"
        )
    write_json(output_dir / "summary.json", summary)

    order = ["continuous", "teleportation", "velocity_change", "object_disappearance"]
    means = frame.groupby("corruption")["prediction_error"].mean().reindex(order)
    figure, axis = plt.subplots(figsize=(8.0, 4.4))
    means.plot.bar(ax=axis, color=["#64748b", "#7c3aed", "#0f766e", "#f59e0b"])
    axis.set_title("Synthetic diagnostic: mean masked-future prediction error")
    axis.set_xlabel("Controlled condition")
    axis.set_ylabel("Mean latent L1 error")
    axis.tick_params(axis="x", rotation=15)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "prediction_error_by_condition.png", dpi=160)
    plt.close(figure)
    return {"summary": summary, "output_dir": str(output_dir)}
