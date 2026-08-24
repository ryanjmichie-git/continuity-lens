from __future__ import annotations

import csv
import json
import subprocess
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from continuity_lens.config import (
    BOOTSTRAP_RESAMPLES,
    CANDIDATE_HORIZONS,
    CHEAP_FEATURE_NAMES,
    DEFAULT_HORIZON,
    SEED,
    default_artifacts_dir,
)
from continuity_lens.dataset import (
    build_transition_records,
    find_davis_root,
    validate_split_isolation,
    write_records,
)
from continuity_lens.schemas import LinearCalibrator, TransitionRecord
from continuity_lens.utils import canonical_hash, hash_tree, read_json, write_json
from continuity_lens.video import load_image_frames
from continuity_lens.vjepa import MockTransitionScorer, TransitionScorer, VJEPABundle, load_vjepa


class FrozenEvaluationError(RuntimeError):
    """Raised when held-out evaluation would violate the frozen protocol."""


def score_records(records: Iterable[TransitionRecord], scorer: TransitionScorer) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        context = load_image_frames(record.context_paths)
        target = load_image_frames(record.target_paths)
        score = scorer.score(context, target)
        rows.append(
            {
                "case_id": record.case_id,
                "source_id": record.source_id,
                "target_source_id": record.target_source_id,
                "split": record.split,
                "discontinuous": record.discontinuous,
                "corruption": record.corruption,
                "lane": record.lane,
                "horizon": record.horizon,
                "prediction_error": score.prediction_error,
                "encoder_distance": score.encoder_distance,
                "histogram_distance": score.histogram_distance,
                "ssim_distance": score.ssim_distance,
                "flow_discontinuity": score.flow_discontinuity,
                "tubelet_errors": json.dumps(score.tubelet_errors),
                "cheap_ms": score.timings_ms.get("cheap_features", 0.0),
                "model_ms": score.timings_ms.get("model_inference", 0.0),
                "peak_vram_mb": score.timings_ms.get("peak_vram_mb", 0.0),
            }
        )
        if index % 10 == 0:
            print(f"Scored {index} transitions", flush=True)
    return pd.DataFrame(rows)


def _primary(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["lane"] == "primary"].reset_index(drop=True)


def _safe_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


def _safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def fit_calibrator(frame: pd.DataFrame, feature_names: tuple[str, ...]) -> LinearCalibrator:
    matrix = frame.loc[:, feature_names].to_numpy(dtype=np.float64)
    labels = frame["discontinuous"].to_numpy(dtype=np.int64)
    scaler = StandardScaler().fit(matrix)
    standardized = scaler.transform(matrix)
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1_000,
        random_state=SEED,
        solver="lbfgs",
    ).fit(standardized, labels)
    return LinearCalibrator(
        feature_names=feature_names,
        mean=tuple(float(value) for value in scaler.mean_),
        scale=tuple(float(value) for value in scaler.scale_),
        coefficients=tuple(float(value) for value in model.coef_[0]),
        intercept=float(model.intercept_[0]),
    )


def apply_calibrator(frame: pd.DataFrame, calibrator: LinearCalibrator) -> np.ndarray:
    return np.array(
        [calibrator.predict_probability(row) for row in frame.to_dict(orient="records")],
        dtype=np.float64,
    )


def select_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    best: tuple[float, float, float] | None = None
    selected = 0.5
    for threshold in candidates:
        predicted = probabilities >= threshold
        true_positive = int(np.sum((predicted == 1) & (labels == 1)))
        false_positive = int(np.sum((predicted == 1) & (labels == 0)))
        false_negative = int(np.sum((predicted == 0) & (labels == 1)))
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        candidate = (f1, recall, -float(threshold))
        if best is None or candidate > best:
            best = candidate
            selected = float(threshold)
    return selected


def _metrics(frame: pd.DataFrame, threshold: float) -> dict[str, Any]:
    primary = _primary(frame)
    labels = primary["discontinuous"].to_numpy(dtype=np.int64)
    scores = {
        "predictor": primary["prediction_error"].to_numpy(dtype=np.float64),
        "encoder": primary["encoder_distance"].to_numpy(dtype=np.float64),
        "cheap": primary["cheap_probability"].to_numpy(dtype=np.float64),
        "hybrid": primary["hybrid_probability"].to_numpy(dtype=np.float64),
    }
    output: dict[str, Any] = {
        "rows": int(len(primary)),
        "groups": int(primary["source_id"].nunique()),
        "prevalence": float(labels.mean()),
        "auprc": {name: _safe_auprc(labels, value) for name, value in scores.items()},
        "auroc": {name: _safe_auroc(labels, value) for name, value in scores.items()},
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, scores["hybrid"] >= threshold)
        ),
        "hybrid_threshold": threshold,
        "mean_latency_ms": {
            "cheap": float(primary["cheap_ms"].mean()),
            "model": float(primary["model_ms"].mean()),
        },
        "peak_vram_mb": float(primary["peak_vram_mb"].max()),
        "recall_by_corruption": {},
    }
    for corruption, group in primary.loc[primary["discontinuous"] == 1].groupby("corruption"):
        output["recall_by_corruption"][corruption] = float(
            np.mean(group["hybrid_probability"].to_numpy() >= threshold)
        )
    return output


def grouped_bootstrap(
    frame: pd.DataFrame,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = SEED,
) -> pd.DataFrame:
    primary = _primary(frame)
    groups = np.array(sorted(primary["source_id"].unique()))
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    by_group = {group: primary.loc[primary["source_id"] == group] for group in groups}
    for replicate in range(resamples):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        sample = pd.concat([by_group[group] for group in sampled], ignore_index=True)
        labels = sample["discontinuous"].to_numpy(dtype=np.int64)
        predictor = _safe_auprc(labels, sample["prediction_error"].to_numpy())
        cheap = _safe_auprc(labels, sample["cheap_probability"].to_numpy())
        hybrid = _safe_auprc(labels, sample["hybrid_probability"].to_numpy())
        rows.append(
            {
                "replicate": replicate,
                "predictor_auprc": predictor,
                "cheap_auprc": cheap,
                "hybrid_auprc": hybrid,
                "delta_predictor_minus_cheap": predictor - cheap,
                "delta_hybrid_minus_cheap": hybrid - cheap,
            }
        )
    return pd.DataFrame(rows)


def _interval(values: pd.Series) -> dict[str, float]:
    return {
        "low": float(values.quantile(0.025)),
        "median": float(values.quantile(0.5)),
        "high": float(values.quantile(0.975)),
    }


def _interpret_delta(interval: dict[str, float]) -> str:
    if interval["low"] > 0:
        return "predictor_adds_evidence"
    if interval["high"] < 0:
        return "cheap_signals_outperform"
    return "inconclusive_at_this_sample_size"


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "uncommitted"


def _append_experiment(
    *, hypothesis: str, config_hash: str, metric: str, result: str, decision: str, notes: str
) -> None:
    path = Path("experiments.tsv")
    if not path.exists():
        path.write_text(
            "timestamp_utc\tcommit\thypothesis\tconfig_hash\tmetric\tresult\tdecision\tnotes\n",
            encoding="utf-8",
        )
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                datetime.now(UTC).isoformat(),
                _git_commit(),
                hypothesis,
                config_hash,
                metric,
                result,
                decision,
                notes,
            ]
        )


def _scorer(mock_model: bool, horizon: int) -> TransitionScorer:
    return MockTransitionScorer(horizon) if mock_model else load_vjepa(horizon=horizon)


def _set_horizon(scorer: TransitionScorer, horizon: int) -> TransitionScorer:
    if isinstance(scorer, (MockTransitionScorer, VJEPABundle)):
        scorer.horizon = horizon
    if isinstance(scorer, VJEPABundle):
        scorer.manifest["horizon"] = horizon
    return scorer


def _manifest_for_scorer(scorer: TransitionScorer, mock_model: bool) -> dict[str, Any]:
    if mock_model:
        return {"mode": "mock", "intended_for": "tests_only"}
    if isinstance(scorer, VJEPABundle):
        public_keys = {
            "repository",
            "commit",
            "entrypoint",
            "checkpoint_url",
            "checkpoint_bytes",
            "checkpoint_sha256",
            "total_frames",
            "horizon",
            "device",
            "dtype",
            "python",
            "torch",
            "cuda",
            "gpu",
            "gpu_arches",
            "model_load_ms",
        }
        return {key: value for key, value in scorer.manifest.items() if key in public_keys}
    raise TypeError("Unknown scorer type")


def _write_figures(frame: pd.DataFrame, metrics: dict[str, Any], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    names = ["predictor", "encoder", "cheap", "hybrid"]
    values = [metrics["auprc"][name] for name in names]
    figure, axis = plt.subplots(figsize=(7.2, 4.0))
    bars = axis.bar(names, values, color=["#7c3aed", "#64748b", "#f59e0b", "#0d9488"])
    axis.bar_label(bars, fmt="%.3f")
    axis.set_ylim(0, 1)
    axis.set_ylabel("AUPRC (discontinuity is positive)")
    axis.set_title("Held-out transition-discontinuity performance")
    figure.tight_layout()
    figure.savefig(figures / "primary_auprc.png", dpi=180)
    plt.close(figure)

    primary = _primary(frame)
    figure, axis = plt.subplots(figsize=(7.2, 4.0))
    order = ["continuous", "temporal_skip", "block_reorder"]
    data = [
        primary.loc[primary["corruption"] == corruption, "prediction_error"].to_numpy()
        for corruption in order
    ]
    axis.boxplot(data, tick_labels=order, showfliers=False)
    axis.set_ylabel("Masked-future latent prediction error")
    axis.set_title("Prediction error by transition type")
    figure.tight_layout()
    figure.savefig(figures / "prediction_error_by_corruption.png", dpi=180)
    plt.close(figure)


def run_development(
    *,
    data_root: Path,
    artifacts_root: Path | None = None,
    mock_model: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    artifacts_root = (artifacts_root or default_artifacts_dir()).resolve()
    output_dir = artifacts_root / "dev"
    frozen_path = output_dir / "frozen_spec.json"
    if frozen_path.exists() and not force:
        raise FrozenEvaluationError(
            f"A frozen protocol already exists at {frozen_path}. "
            "Use --force only before a held-out run."
        )
    if (artifacts_root / "test" / "metrics.json").exists():
        raise FrozenEvaluationError(
            "Held-out results already exist; create a new protocol version."
        )
    davis_root = find_davis_root(data_root.resolve())
    data_manifest_path = data_root.resolve() / "manifests" / "davis.json"
    if not data_manifest_path.exists():
        raise FrozenEvaluationError("DAVIS manifest is missing; run data preparation first.")

    scorer = _scorer(mock_model, DEFAULT_HORIZON)
    horizon_results: dict[int, float] = {}
    for horizon in CANDIDATE_HORIZONS:
        records = build_transition_records(
            davis_root,
            split="dev",
            horizon=horizon,
            anchors_per_source=1,
            include_secondary=False,
        )
        frame = score_records(records, _set_horizon(scorer, horizon))
        primary = _primary(frame)
        horizon_results[horizon] = _safe_auprc(
            primary["discontinuous"].to_numpy(),
            primary["prediction_error"].to_numpy(),
        )
    best_value = max(horizon_results.values())
    tied = [horizon for horizon, value in horizon_results.items() if np.isclose(value, best_value)]
    selected_horizon = DEFAULT_HORIZON if DEFAULT_HORIZON in tied else min(tied)

    records = build_transition_records(
        davis_root,
        split="dev",
        horizon=selected_horizon,
        anchors_per_source=3,
        include_secondary=True,
    )
    test_records = build_transition_records(
        davis_root,
        split="test",
        horizon=selected_horizon,
        anchors_per_source=3,
        include_secondary=True,
    )
    validate_split_isolation(records, test_records)
    write_records(output_dir / "dev_cases.jsonl", records)
    frame = score_records(records, _set_horizon(scorer, selected_horizon))
    primary = _primary(frame)
    cheap = fit_calibrator(primary, CHEAP_FEATURE_NAMES)
    hybrid_names = (*CHEAP_FEATURE_NAMES, "prediction_error")
    hybrid = fit_calibrator(primary, hybrid_names)
    frame["cheap_probability"] = apply_calibrator(frame, cheap)
    frame["hybrid_probability"] = apply_calibrator(frame, hybrid)
    threshold = select_f1_threshold(
        primary["discontinuous"].to_numpy(),
        apply_calibrator(primary, hybrid),
    )
    metrics = _metrics(frame, threshold)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "predictions.csv", index=False)
    write_json(output_dir / "metrics.json", metrics)

    package_root = Path(__file__).resolve().parent
    protocol: dict[str, Any] = {
        "protocol_version": 1,
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "candidate_horizons": list(CANDIDATE_HORIZONS),
        "horizon_development_auprc": {str(k): v for k, v in horizon_results.items()},
        "selected_horizon": selected_horizon,
        "selection_tie_break": DEFAULT_HORIZON,
        "anchors_per_source": 3,
        "primary_corruptions": ["continuous", "temporal_skip", "block_reorder"],
        "secondary_corruption": "cross_video_splice",
        "cheap_feature_names": list(CHEAP_FEATURE_NAMES),
        "cheap_calibrator": cheap.to_dict(),
        "hybrid_calibrator": hybrid.to_dict(),
        "hybrid_threshold": threshold,
        "expected_test_groups": len({record.source_id for record in test_records}),
        "expected_test_rows": len(test_records),
        "data_manifest_hash": canonical_hash(read_json(data_manifest_path)),
        "source_tree_sha256": hash_tree(package_root),
        "model_manifest": _manifest_for_scorer(scorer, mock_model),
        "held_out_policy": "one pass; no overwrite; full protocol version required after a bug fix",
    }
    protocol["protocol_hash"] = canonical_hash(protocol)
    write_json(frozen_path, protocol)
    if not mock_model:
        _append_experiment(
            hypothesis="Select a short future horizon using development groups only",
            config_hash=protocol["protocol_hash"],
            metric="development AUPRC",
            result=json.dumps(horizon_results, sort_keys=True),
            decision=f"keep horizon={selected_horizon}",
            notes="Held-out DAVIS val was not scored.",
        )
    return {"metrics": metrics, "frozen_spec": protocol, "output_dir": str(output_dir)}


def _verify_frozen_spec(
    spec: dict[str, Any], data_manifest_path: Path, *, mock_model: bool
) -> None:
    claimed_hash = spec.get("protocol_hash")
    without_hash = {key: value for key, value in spec.items() if key != "protocol_hash"}
    if claimed_hash != canonical_hash(without_hash):
        raise FrozenEvaluationError("Frozen protocol hash is invalid.")
    if spec["data_manifest_hash"] != canonical_hash(read_json(data_manifest_path)):
        raise FrozenEvaluationError("DAVIS manifest changed after the protocol freeze.")
    current_source = hash_tree(Path(__file__).resolve().parent)
    if spec["source_tree_sha256"] != current_source:
        raise FrozenEvaluationError(
            "Scoring source changed after the protocol freeze. "
            "Re-run development under a new protocol."
        )
    spec_mock = spec.get("model_manifest", {}).get("mode") == "mock"
    if spec_mock != mock_model:
        raise FrozenEvaluationError(
            "Frozen model mode does not match the requested test model mode."
        )


def run_frozen_test(
    *,
    data_root: Path,
    artifacts_root: Path | None = None,
    mock_model: bool = False,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    artifacts_root = (artifacts_root or default_artifacts_dir()).resolve()
    output_dir = artifacts_root / "test"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FrozenEvaluationError(
            f"Held-out output already exists at {output_dir}; it will not be overwritten."
        )
    frozen_path = artifacts_root / "dev" / "frozen_spec.json"
    if not frozen_path.exists():
        raise FrozenEvaluationError("Run the development benchmark and freeze the protocol first.")
    spec = read_json(frozen_path)
    data_manifest_path = data_root.resolve() / "manifests" / "davis.json"
    _verify_frozen_spec(spec, data_manifest_path, mock_model=mock_model)

    horizon = int(spec["selected_horizon"])
    davis_root = find_davis_root(data_root.resolve())
    records = build_transition_records(
        davis_root,
        split="test",
        horizon=horizon,
        anchors_per_source=int(spec["anchors_per_source"]),
        include_secondary=True,
    )
    if len(records) != spec["expected_test_rows"]:
        raise FrozenEvaluationError(
            f"Expected {spec['expected_test_rows']} test rows, generated {len(records)}."
        )
    if len({record.source_id for record in records}) != spec["expected_test_groups"]:
        raise FrozenEvaluationError("Held-out source-group count changed.")
    write_records(output_dir / "test_cases.jsonl", records)
    scorer = _scorer(mock_model, horizon)
    frame = score_records(records, scorer)
    cheap = LinearCalibrator.from_dict(spec["cheap_calibrator"])
    hybrid = LinearCalibrator.from_dict(spec["hybrid_calibrator"])
    frame["cheap_probability"] = apply_calibrator(frame, cheap)
    frame["hybrid_probability"] = apply_calibrator(frame, hybrid)
    metrics = _metrics(frame, float(spec["hybrid_threshold"]))
    bootstrap = grouped_bootstrap(frame, resamples=bootstrap_resamples)
    primary_interval = _interval(bootstrap["delta_predictor_minus_cheap"])
    hybrid_interval = _interval(bootstrap["delta_hybrid_minus_cheap"])
    metrics["bootstrap"] = {
        "resamples": bootstrap_resamples,
        "delta_predictor_minus_cheap": primary_interval,
        "delta_hybrid_minus_cheap": hybrid_interval,
    }
    metrics["reporting_outcome"] = _interpret_delta(primary_interval)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "predictions.csv", index=False)
    bootstrap.to_csv(output_dir / "bootstrap.csv", index=False)
    write_json(output_dir / "metrics.json", metrics)
    _write_figures(frame, metrics, output_dir)
    if not mock_model:
        _append_experiment(
            hypothesis="Masked-future prediction error adds held-out value over cheap signals",
            config_hash=spec["protocol_hash"],
            metric="delta AUPRC with grouped 95% bootstrap interval",
            result=json.dumps(primary_interval, sort_keys=True),
            decision=metrics["reporting_outcome"],
            notes="Frozen DAVIS val result; no post-test tuning.",
        )
    return {"metrics": metrics, "output_dir": str(output_dir)}
