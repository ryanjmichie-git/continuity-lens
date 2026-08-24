from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from continuity_lens.config import DAVIS_URL, SEED, TOTAL_FRAMES
from continuity_lens.schemas import TransitionRecord
from continuity_lens.utils import download_resumable, file_sha256, safe_extract_zip, write_json


class DatasetError(RuntimeError):
    """Raised when DAVIS is absent or violates the benchmark contract."""


def _progress(written: int, total: int | None) -> None:
    if total:
        percentage = 100.0 * written / total
        print(f"\rDownloading DAVIS: {percentage:5.1f}%", end="", flush=True)
    else:
        print(f"\rDownloading DAVIS: {written / (1024**2):.1f} MiB", end="", flush=True)


def find_davis_root(data_root: Path) -> Path:
    candidates = (
        data_root / "raw" / "DAVIS",
        data_root / "DAVIS",
        data_root,
    )
    for candidate in candidates:
        if (candidate / "JPEGImages" / "480p").is_dir() and (
            candidate / "ImageSets" / "2017"
        ).is_dir():
            return candidate.resolve()
    raise DatasetError(
        f"DAVIS 2017 was not found beneath {data_root}. Run `continuity-lens data prepare`."
    )


def _read_split(davis_root: Path, split: str) -> list[str]:
    mapped = {"dev": "train", "test": "val"}.get(split, split)
    path = davis_root / "ImageSets" / "2017" / f"{mapped}.txt"
    if not path.exists():
        raise DatasetError(f"Missing official DAVIS split file: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def split_sequences(davis_root: Path, split: str) -> dict[str, tuple[Path, ...]]:
    sequences: dict[str, tuple[Path, ...]] = {}
    for name in _read_split(davis_root, split):
        frames = tuple(sorted((davis_root / "JPEGImages" / "480p" / name).glob("*.jpg")))
        if not frames:
            raise DatasetError(f"Sequence {name!r} has no frames.")
        sequences[name] = frames
    return sequences


def prepare_davis(data_root: Path, *, download: bool = True) -> dict[str, object]:
    data_root = data_root.resolve()
    try:
        davis_root = find_davis_root(data_root)
    except DatasetError:
        if not download:
            raise
        archive = data_root / "downloads" / "DAVIS-2017-trainval-480p.zip"
        if not archive.exists():
            download_resumable(DAVIS_URL, archive, progress=_progress)
            print()
        safe_extract_zip(archive, data_root / "raw")
        davis_root = find_davis_root(data_root)

    dev = split_sequences(davis_root, "dev")
    test = split_sequences(davis_root, "test")
    overlap = set(dev) & set(test)
    if overlap:
        raise DatasetError(f"DAVIS source leakage across train/val: {sorted(overlap)}")
    if len(dev) != 60 or len(test) != 30:
        raise DatasetError(
            f"Expected official DAVIS 2017 split sizes 60/30; found {len(dev)}/{len(test)}."
        )

    archive = data_root / "downloads" / "DAVIS-2017-trainval-480p.zip"
    manifest: dict[str, object] = {
        "dataset": "DAVIS-2017-trainval-480p",
        "source_url": DAVIS_URL,
        "davis_root": str(davis_root),
        "archive_sha256": file_sha256(archive) if archive.exists() else None,
        "dev_sources": sorted(dev),
        "test_sources": sorted(test),
        "dev_count": len(dev),
        "test_count": len(test),
    }
    write_json(data_root / "manifests" / "davis.json", manifest)
    return manifest


def _representative_color(frames: tuple[Path, ...]) -> np.ndarray:
    with Image.open(frames[len(frames) // 2]) as image:
        rgb = np.asarray(image.convert("RGB").resize((32, 32)), dtype=np.float32)
    return np.mean(rgb, axis=(0, 1)) / 255.0


def _cross_matches(sequences: dict[str, tuple[Path, ...]]) -> dict[str, str]:
    colors = {name: _representative_color(frames) for name, frames in sequences.items()}
    matches: dict[str, str] = {}
    for name, color in colors.items():
        candidates = sorted(other for other in colors if other != name)
        matches[name] = min(
            candidates,
            key=lambda other: float(np.linalg.norm(color - colors[other])),
        )
    return matches


def _anchors(frame_count: int, context_count: int, horizon: int, count: int) -> list[int]:
    skip = max(horizon, 2)
    first = context_count
    last = frame_count - skip - horizon
    if last < first:
        return []
    raw = np.linspace(first, last, num=count)
    return sorted({int(round(value)) for value in raw})


def _reorder(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    if len(paths) == 2:
        return tuple(reversed(paths))
    blocks = [paths[index : index + 2] for index in range(0, len(paths), 2)]
    return tuple(path for block in reversed(blocks) for path in block)


def build_transition_records(
    davis_root: Path,
    *,
    split: str,
    horizon: int,
    anchors_per_source: int = 3,
    include_secondary: bool = True,
) -> list[TransitionRecord]:
    if horizon <= 0 or horizon >= TOTAL_FRAMES or horizon % 2:
        raise ValueError("Horizon must be a positive, even value below TOTAL_FRAMES.")
    sequences = split_sequences(davis_root, split)
    cross_matches = _cross_matches(sequences) if include_secondary else {}
    context_count = TOTAL_FRAMES - horizon
    records: list[TransitionRecord] = []
    for source_name in sorted(sequences):
        frames = sequences[source_name]
        anchors = _anchors(len(frames), context_count, horizon, anchors_per_source)
        if not anchors:
            raise DatasetError(
                f"Sequence {source_name!r} is too short for {TOTAL_FRAMES} frames "
                "and skip corruption."
            )
        for ordinal, anchor in enumerate(anchors):
            context = frames[anchor - context_count : anchor]
            natural_target = frames[anchor : anchor + horizon]
            skip_start = anchor + max(horizon, 2)
            skipped_target = frames[skip_start : skip_start + horizon]
            prefix = f"{split}-{source_name}-{ordinal:02d}-h{horizon}"

            base = dict(
                source_id=source_name,
                split=split,
                horizon=horizon,
                context_paths=tuple(str(path.resolve()) for path in context),
            )
            records.extend(
                [
                    TransitionRecord(
                        case_id=f"{prefix}-continuous",
                        discontinuous=0,
                        corruption="continuous",
                        lane="primary",
                        target_paths=tuple(str(path.resolve()) for path in natural_target),
                        **base,
                    ),
                    TransitionRecord(
                        case_id=f"{prefix}-skip",
                        discontinuous=1,
                        corruption="temporal_skip",
                        lane="primary",
                        target_paths=tuple(str(path.resolve()) for path in skipped_target),
                        **base,
                    ),
                    TransitionRecord(
                        case_id=f"{prefix}-reorder",
                        discontinuous=1,
                        corruption="block_reorder",
                        lane="primary",
                        target_paths=tuple(
                            str(path.resolve()) for path in _reorder(natural_target)
                        ),
                        **base,
                    ),
                ]
            )
            if include_secondary:
                target_source = cross_matches[source_name]
                other_frames = sequences[target_source]
                relative = anchor / max(1, len(frames) - 1)
                other_start = min(
                    max(0, int(round(relative * len(other_frames)))),
                    len(other_frames) - horizon,
                )
                other_target = other_frames[other_start : other_start + horizon]
                records.append(
                    TransitionRecord(
                        case_id=f"{prefix}-cross",
                        discontinuous=1,
                        corruption="cross_video_splice",
                        lane="secondary",
                        target_paths=tuple(str(path.resolve()) for path in other_target),
                        target_source_id=target_source,
                        **base,
                    )
                )
    return records


def write_records(path: Path, records: list[TransitionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")


def read_records(path: Path) -> list[TransitionRecord]:
    with path.open(encoding="utf-8") as handle:
        return [TransitionRecord.from_dict(json.loads(line)) for line in handle if line.strip()]


def validate_split_isolation(dev: list[TransitionRecord], test: list[TransitionRecord]) -> None:
    dev_sources = {record.source_id for record in dev}
    test_sources = {record.source_id for record in test}
    if overlap := dev_sources & test_sources:
        raise DatasetError(f"Source leakage across development/test: {sorted(overlap)}")


def deterministic_seed() -> int:
    return SEED
