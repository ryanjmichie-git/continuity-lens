from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from continuity_lens.utils import write_json


@pytest.fixture
def tiny_davis(tmp_path: Path) -> Path:
    data_root = tmp_path / "data"
    davis = data_root / "raw" / "DAVIS"
    image_root = davis / "JPEGImages" / "480p"
    split_root = davis / "ImageSets" / "2017"
    split_root.mkdir(parents=True)
    dev_names = ["dev-a", "dev-b", "dev-c"]
    test_names = ["test-a", "test-b"]
    split_root.joinpath("train.txt").write_text("\n".join(dev_names) + "\n", encoding="utf-8")
    split_root.joinpath("val.txt").write_text("\n".join(test_names) + "\n", encoding="utf-8")
    for sequence_index, name in enumerate([*dev_names, *test_names]):
        sequence_root = image_root / name
        sequence_root.mkdir(parents=True)
        for frame_index in range(40):
            image = np.zeros((48, 64, 3), dtype=np.uint8)
            image[..., 0] = 30 + sequence_index * 25
            image[..., 1] = (frame_index * 5) % 255
            x = min(60, 4 + frame_index)
            image[16:28, max(0, x - 4) : x, 2] = 255
            Image.fromarray(image).save(sequence_root / f"{frame_index:05d}.jpg")
    manifest = {
        "dataset": "tiny-DAVIS-fixture",
        "dev_sources": dev_names,
        "test_sources": test_names,
    }
    write_json(data_root / "manifests" / "davis.json", manifest)
    return data_root
