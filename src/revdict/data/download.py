from __future__ import annotations

import hashlib
import gzip
import json
import os
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_RETRIES = 3
USER_AGENT = "revdict-index-builder/0.1"


class DatasetDownloadError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance_path(path: Path) -> Path:
    path = Path(path)
    return path.with_name(path.name + ".source.json")


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _provenance(
    path: Path,
    url: str,
    *,
    sha256: str,
    headers=None,
    reused: bool,
    validation_id: str,
) -> dict:
    return {
        "url": url,
        "bytes": path.stat().st_size,
        "sha256": sha256,
        "etag": headers.get("ETag") if headers is not None else None,
        "last_modified": headers.get("Last-Modified") if headers is not None else None,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "reused_cache": reused,
        "validation_id": validation_id,
    }


def _verified_cached_file(
    path: Path,
    url: str,
    validator: Callable[[Path], None],
    validation_id: str,
) -> dict:
    digest = sha256_file(path)
    metadata_path = provenance_path(path)
    existing = None
    if metadata_path.is_file():
        try:
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = None
    if existing is not None:
        if existing.get("url") != url:
            raise DatasetDownloadError(
                f"Cached dataset came from {existing.get('url')!r}, not the configured {url!r}"
            )
        if existing.get("bytes") != path.stat().st_size or existing.get("sha256") != digest:
            raise DatasetDownloadError(f"Cached dataset checksum does not match its provenance: {path}")
    if existing is None or existing.get("validation_id") != validation_id:
        validator(path)
    info = _provenance(
        path,
        url,
        sha256=digest,
        reused=True,
        validation_id=validation_id,
    )
    _write_json_atomic(metadata_path, info)
    return info


def download_cached_file(
    url: str,
    destination: Path,
    *,
    validator: Callable[[Path], None],
    refresh: bool = False,
    retries: int = DEFAULT_RETRIES,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    validation_id: str = "v1",
) -> dict:
    """Return a verified cached file, or atomically replace it after download.

    A failed refresh never alters the known-good destination. The exact
    ``.download.part`` path is the only data file this function removes.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and not refresh:
        try:
            return _verified_cached_file(destination, url, validator, validation_id)
        except (DatasetDownloadError, OSError, ValueError):
            # A corrupt, mismatched, or structurally invalid cache is repaired
            # by downloading to a separate path; it remains untouched unless
            # the replacement has fully validated.
            pass

    part = destination.with_name(destination.name + ".download.part")
    if part.exists() or part.is_symlink():
        part.unlink()

    last_error = None
    headers = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                headers = response.headers
                with part.open("xb") as output:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
            validator(part)
            digest = sha256_file(part)
            os.replace(part, destination)
            info = _provenance(
                destination,
                url,
                sha256=digest,
                headers=headers,
                reused=False,
                validation_id=validation_id,
            )
            _write_json_atomic(provenance_path(destination), info)
            return info
        except Exception as error:
            last_error = error
            if part.exists() or part.is_symlink():
                part.unlink()
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 4))

    raise DatasetDownloadError(
        f"Failed to download {url} after {retries} attempts: {last_error}"
    ) from last_error


def validate_nonempty_file(path: Path) -> None:
    if not Path(path).is_file() or Path(path).stat().st_size == 0:
        raise ValueError(f"Dataset is empty or missing: {path}")


def validate_gzip_header(path: Path) -> None:
    validate_nonempty_file(path)
    with Path(path).open("rb") as f:
        if f.read(2) != b"\x1f\x8b":
            raise ValueError(f"Dataset is not a gzip stream: {path}")


def validate_gzip_stream(path: Path) -> None:
    validate_gzip_header(path)
    with gzip.open(path, "rb") as f:
        for _chunk in iter(lambda: f.read(1024 * 1024), b""):
            pass
