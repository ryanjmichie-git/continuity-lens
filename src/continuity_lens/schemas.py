from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TransitionRecord:
    case_id: str
    source_id: str
    split: str
    discontinuous: int
    corruption: str
    lane: str
    horizon: int
    context_paths: tuple[str, ...]
    target_paths: tuple[str, ...]
    target_source_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TransitionRecord:
        value = dict(value)
        value["context_paths"] = tuple(value["context_paths"])
        value["target_paths"] = tuple(value["target_paths"])
        return cls(**value)

    @property
    def all_paths(self) -> tuple[Path, ...]:
        return tuple(Path(p) for p in (*self.context_paths, *self.target_paths))


@dataclass
class ScoreBundle:
    prediction_error: float
    encoder_distance: float
    histogram_distance: float
    ssim_distance: float
    flow_discontinuity: float
    tubelet_errors: list[float] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LinearCalibrator:
    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float

    def predict_probability(self, row: dict[str, float]) -> float:
        import math

        z = self.intercept
        for name, mean, scale, coefficient in zip(
            self.feature_names,
            self.mean,
            self.scale,
            self.coefficients,
            strict=True,
        ):
            safe_scale = scale if abs(scale) > 1e-12 else 1.0
            z += coefficient * ((float(row[name]) - mean) / safe_scale)
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        exp_z = math.exp(z)
        return exp_z / (1.0 + exp_z)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> LinearCalibrator:
        return cls(
            feature_names=tuple(value["feature_names"]),
            mean=tuple(value["mean"]),
            scale=tuple(value["scale"]),
            coefficients=tuple(value["coefficients"]),
            intercept=float(value["intercept"]),
        )
