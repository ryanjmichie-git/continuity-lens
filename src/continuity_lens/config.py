from __future__ import annotations

import os
from pathlib import Path

VJEPA_REPOSITORY = "facebookresearch/vjepa2"
VJEPA_COMMIT = "204698b45b3712590f06245fbfba32d3be539812"
VJEPA_MODEL_ENTRYPOINT = "vjepa2_vit_large"
VJEPA_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/vjepa2/vitl.pt"

DAVIS_URL = (
    "https://data.vision.ee.ethz.ch/csergi/share/davis/"
    "DAVIS-2017-trainval-480p.zip"
)

TOTAL_FRAMES = 16
CANDIDATE_HORIZONS = (2, 4, 8)
DEFAULT_HORIZON = 4
TUBELET_SIZE = 2
IMAGE_SIZE = 256
PATCH_SIZE = 16
VIDEO_SAMPLE_FPS = 4.0
SEED = 239
BOOTSTRAP_RESAMPLES = 5_000
PRIMARY_CORRUPTIONS = ("continuous", "temporal_skip", "block_reorder")
CHEAP_FEATURE_NAMES = (
    "histogram_distance",
    "ssim_distance",
    "flow_discontinuity",
)


def default_cache_dir() -> Path:
    override = os.environ.get("CONTINUITY_LENS_CACHE")
    if override:
        return Path(override).expanduser().resolve()
    root = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME")
    if root:
        return Path(root) / "continuity-lens"
    return Path.home() / ".cache" / "continuity-lens"


def default_data_dir() -> Path:
    return Path(os.environ.get("CONTINUITY_LENS_DATA", "data")).resolve()


def default_artifacts_dir() -> Path:
    return Path(os.environ.get("CONTINUITY_LENS_ARTIFACTS", "artifacts")).resolve()
