from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from skimage.metrics import structural_similarity


@dataclass(frozen=True)
class CheapFeatureResult:
    histogram_distance: float
    ssim_distance: float
    flow_discontinuity: float
    elapsed_ms: float

    def as_dict(self) -> dict[str, float]:
        return {
            "histogram_distance": self.histogram_distance,
            "ssim_distance": self.ssim_distance,
            "flow_discontinuity": self.flow_discontinuity,
        }


def _resize(frame: np.ndarray, side: int = 192) -> np.ndarray:
    return cv2.resize(frame, (side, side), interpolation=cv2.INTER_AREA)


def _hsv_histogram(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    return cv2.normalize(histogram, histogram).flatten()


def _flow_magnitude(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        gray_a,
        gray_b,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.median(magnitude))


def compute_cheap_features(
    context_frames: Sequence[np.ndarray],
    target_frames: Sequence[np.ndarray],
    *,
    boundary_pairs: int = 4,
) -> CheapFeatureResult:
    if len(context_frames) == 0 or len(target_frames) == 0:
        raise ValueError("Cheap features require context and target frames.")
    started = time.perf_counter()
    pair_count = min(boundary_pairs, len(context_frames), len(target_frames))
    context = [_resize(frame) for frame in context_frames[-pair_count:]]
    target = [_resize(frame) for frame in target_frames[:pair_count]]

    hist_distances = [
        cv2.compareHist(_hsv_histogram(left), _hsv_histogram(right), cv2.HISTCMP_BHATTACHARYYA)
        for left, right in zip(context, target, strict=True)
    ]
    ssim_distances = []
    for left, right in zip(context, target, strict=True):
        left_gray = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
        ssim_distances.append(1.0 - structural_similarity(left_gray, right_gray, data_range=255))

    sequence = context + target
    magnitudes = [
        _flow_magnitude(a, b) for a, b in zip(sequence, sequence[1:], strict=False)
    ]
    boundary_index = len(context) - 1
    boundary_magnitude = magnitudes[boundary_index]
    neighboring = [value for index, value in enumerate(magnitudes) if index != boundary_index]
    baseline = float(np.median(neighboring)) if neighboring else boundary_magnitude
    flow_discontinuity = abs(boundary_magnitude - baseline) / (baseline + 1e-6)

    return CheapFeatureResult(
        histogram_distance=float(np.mean(hist_distances)),
        ssim_distance=float(np.mean(ssim_distances)),
        flow_discontinuity=float(flow_discontinuity),
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
    )
