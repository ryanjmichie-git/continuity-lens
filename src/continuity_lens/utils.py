from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any


class DownloadError(RuntimeError):
    """Raised when an external artifact cannot be downloaded safely."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_tree(root: Path, patterns: Iterable[str] = ("*.py",)) -> str:
    digest = hashlib.sha256()
    paths = sorted({path for pattern in patterns for path in root.rglob(pattern)})
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def download_resumable(
    url: str,
    destination: Path,
    *,
    progress: Callable[[int, int | None], None] | None = None,
    chunk_size: int = 8 * 1024 * 1024,
) -> Path:
    """Download to a partial file and resume when the server supports ranges."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    start = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "continuity-lens/0.1"}
    if start:
        headers["Range"] = f"bytes={start}-"

    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=60)  # noqa: S310
    except (urllib.error.URLError, TimeoutError) as exc:
        raise DownloadError(
            f"Could not download {url}. Existing partial file is preserved at {partial}."
        ) from exc

    status = getattr(response, "status", 200)
    if start and status != 206:
        start = 0
        partial.unlink(missing_ok=True)
    content_length = response.headers.get("Content-Length")
    total = (int(content_length) + start) if content_length else None
    mode = "ab" if start else "wb"
    written = start
    try:
        with response, partial.open(mode) as handle:
            while chunk := response.read(chunk_size):
                handle.write(chunk)
                written += len(chunk)
                if progress:
                    progress(written, total)
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        raise DownloadError(
            f"Download interrupted. Resume data is preserved at {partial}."
        ) from exc

    if total is not None and written != total:
        raise DownloadError(f"Expected {total} bytes from {url}, received {written}.")
    partial.replace(destination)
    return destination


def safe_extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if destination_root != target and destination_root not in target.parents:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        bundle.extractall(destination)


def copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temp)
    os.replace(temp, destination)
