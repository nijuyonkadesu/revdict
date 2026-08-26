"""Versioned, validated index bundles with atomic publication.

The historical index layout placed all artifacts directly in ``INDEX_DIR``.
That layout remains readable so an upgrade does not invalidate an existing
multi-gigabyte index.  Every new build is written under ``INDEX_DIR/versions``
and becomes visible through a single atomic ``INDEX_DIR/current`` symlink.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

INDEX_FORMAT = "revdict-index"
SCHEMA_VERSION = 2
MANIFEST_FILENAME = "manifest.json"
CURRENT_LINK_NAME = "current"
VERSIONS_DIR_NAME = "versions"
LEGACY_REQUIRED_ARTIFACTS = (
    "embeddings.npy",
    "metadata.jsonl",
    "word_index.json",
    "literary_frequency.json",
)
REQUIRED_ARTIFACTS = (*LEGACY_REQUIRED_ARTIFACTS, "record_embeddings.npy")
_STAGING_FILES = (*REQUIRED_ARTIFACTS, MANIFEST_FILENAME)


class IndexValidationError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    """Persist directory-entry changes on platforms that support it."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def resolve_active_index_dir(index_root: Path) -> Path:
    """Resolve one immutable bundle, or the legacy root when no pointer exists."""
    root = Path(index_root)
    current = root / CURRENT_LINK_NAME
    if not current.is_symlink():
        if current.exists():
            raise IndexValidationError(f"Index pointer is not a symlink: {current}")
        return root

    resolved = current.resolve(strict=True)
    versions = (root / VERSIONS_DIR_NAME).resolve()
    if not resolved.is_relative_to(versions):
        raise IndexValidationError(f"Index pointer escapes the versions directory: {current}")
    if not resolved.is_dir():
        raise IndexValidationError(f"Index pointer does not target a directory: {current}")
    return resolved


def index_layout_exists(index_root: Path) -> bool:
    try:
        active = resolve_active_index_dir(index_root)
    except (IndexValidationError, OSError):
        return False
    required = REQUIRED_ARTIFACTS if (active / MANIFEST_FILENAME).is_file() else LEGACY_REQUIRED_ARTIFACTS
    return all((active / filename).is_file() for filename in required)


def create_staging_index(index_root: Path) -> Path:
    root = Path(index_root)
    versions = root / VERSIONS_DIR_NAME
    versions.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=".staging-", dir=versions))


def discard_staging_index(index_root: Path, staging_dir: Path) -> None:
    """Remove only a staging directory created directly under this index root."""
    root = Path(index_root)
    staging = Path(staging_dir)
    versions = (root / VERSIONS_DIR_NAME).resolve()
    if staging.parent.resolve() != versions or not staging.name.startswith(".staging-"):
        raise ValueError(f"Refusing to remove unexpected staging path: {staging}")
    if not staging.exists():
        return
    if staging.is_symlink() or not staging.is_dir():
        raise RuntimeError(f"Refusing to remove a non-directory staging path: {staging}")
    unknown = [path.name for path in staging.iterdir() if path.name not in _STAGING_FILES]
    if unknown:
        raise RuntimeError(f"Refusing to remove staging directory with unknown files: {unknown}")
    for filename in _STAGING_FILES:
        path = staging / filename
        if path.is_file() or path.is_symlink():
            path.unlink()
    staging.rmdir()


def build_manifest(
    index_dir: Path,
    *,
    record_count: int,
    definition_count: int,
    embeddings: np.ndarray,
    record_embedding_indices: np.ndarray,
    datasets: dict | None = None,
    models: dict | None = None,
    statistics: dict | None = None,
    build_id: str | None = None,
) -> dict:
    index_dir = Path(index_dir)
    artifacts = {}
    for filename in REQUIRED_ARTIFACTS:
        path = index_dir / filename
        artifacts[filename] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    return {
        "format": INDEX_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "build_id": build_id or uuid.uuid4().hex,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "record_count": record_count,
        "definition_count": definition_count,
        "embeddings": {
            "rows": int(embeddings.shape[0]),
            "dimensions": int(embeddings.shape[1]),
            "dtype": str(embeddings.dtype),
        },
        "record_embeddings": {
            "rows": int(record_embedding_indices.shape[0]),
            "dtype": str(record_embedding_indices.dtype),
        },
        "artifacts": artifacts,
        "datasets": datasets or {},
        "models": models or {},
        "statistics": statistics or {},
    }


def write_manifest(index_dir: Path, manifest: dict) -> None:
    path = Path(index_dir) / MANIFEST_FILENAME
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def load_manifest(index_dir: Path, *, required: bool = False) -> dict | None:
    path = Path(index_dir) / MANIFEST_FILENAME
    if not path.is_file():
        if required:
            raise IndexValidationError(f"Index manifest is missing: {path}")
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IndexValidationError(f"Could not read index manifest {path}: {error}") from error
    if manifest.get("format") != INDEX_FORMAT:
        raise IndexValidationError(f"Unsupported index format in {path}")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise IndexValidationError(
            f"Unsupported index schema {manifest.get('schema_version')!r}; expected {SCHEMA_VERSION}"
        )
    return manifest


def validate_index_directory(index_dir: Path, *, verify_hashes: bool = False) -> dict:
    """Validate a newly built bundle before it can become current."""
    index_dir = Path(index_dir)
    manifest = load_manifest(index_dir, required=True)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise IndexValidationError("Index manifest has no artifact table")

    for filename in REQUIRED_ARTIFACTS:
        path = index_dir / filename
        expected = artifacts.get(filename)
        if not path.is_file() or not isinstance(expected, dict):
            raise IndexValidationError(f"Index artifact is missing: {path}")
        if path.stat().st_size != expected.get("bytes"):
            raise IndexValidationError(f"Index artifact size mismatch: {path}")
        if verify_hashes and _sha256(path) != expected.get("sha256"):
            raise IndexValidationError(f"Index artifact checksum mismatch: {path}")

    embeddings = np.load(index_dir / "embeddings.npy", mmap_mode="r")
    shape = manifest.get("embeddings", {})
    expected_shape = (shape.get("rows"), shape.get("dimensions"))
    if embeddings.ndim != 2 or embeddings.shape != expected_shape:
        raise IndexValidationError(
            f"Embedding shape {embeddings.shape} does not match manifest {expected_shape}"
        )
    if str(embeddings.dtype) != shape.get("dtype"):
        raise IndexValidationError("Embedding dtype does not match manifest")
    if embeddings.shape[0] != manifest.get("definition_count"):
        raise IndexValidationError("Embedding row count does not match definition count")

    record_embeddings = np.load(index_dir / "record_embeddings.npy", mmap_mode="r")
    record_shape = manifest.get("record_embeddings", {})
    if record_embeddings.ndim != 1 or record_embeddings.shape[0] != manifest.get("record_count"):
        raise IndexValidationError("Record-to-embedding mapping does not match record count")
    if record_embeddings.shape[0] != record_shape.get("rows"):
        raise IndexValidationError("Record-to-embedding mapping shape does not match manifest")
    if str(record_embeddings.dtype) != record_shape.get("dtype"):
        raise IndexValidationError("Record-to-embedding mapping dtype does not match manifest")
    if record_embeddings.size and (
        int(record_embeddings.min()) < 0 or int(record_embeddings.max()) >= embeddings.shape[0]
    ):
        raise IndexValidationError("Record-to-embedding mapping contains an invalid vector reference")

    metadata_rows = 0
    metadata_headwords: list[str] = []
    with (index_dir / "metadata.jsonl").open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise IndexValidationError(
                    f"Invalid metadata JSON on line {line_number}: {error}"
                ) from error
            metadata_rows += 1
            try:
                metadata_headwords.append(record["headword"].casefold())
            except (KeyError, AttributeError) as error:
                raise IndexValidationError(
                    f"Metadata line {line_number} has no valid headword"
                ) from error
    if metadata_rows != manifest.get("record_count"):
        raise IndexValidationError(
            f"Metadata has {metadata_rows} rows; expected {manifest.get('record_count')}"
        )

    try:
        word_index = json.loads((index_dir / "word_index.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise IndexValidationError(f"Invalid word index JSON: {error}") from error
    referenced: set[int] = set()
    reference_count = 0
    for headword, indices in word_index.items():
        if not isinstance(headword, str) or not isinstance(indices, list):
            raise IndexValidationError("Word index has an invalid entry")
        for row in indices:
            if not isinstance(row, int) or not 0 <= row < metadata_rows:
                raise IndexValidationError(f"Word index contains invalid row reference: {row!r}")
            if metadata_headwords[row] != headword.casefold():
                raise IndexValidationError(
                    f"Word index key {headword!r} does not match metadata row {row}"
                )
            referenced.add(row)
            reference_count += 1
    if reference_count != metadata_rows:
        raise IndexValidationError(
            f"Word index has {reference_count} references for {metadata_rows} metadata rows"
        )
    if len(referenced) != metadata_rows:
        raise IndexValidationError(
            f"Word index covers {len(referenced)} of {metadata_rows} metadata rows"
        )
    try:
        frequencies = json.loads(
            (index_dir / "literary_frequency.json").read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as error:
        raise IndexValidationError(f"Invalid literary frequency JSON: {error}") from error
    if not isinstance(frequencies, dict) or any(
        not isinstance(word, str) or not isinstance(score, (int, float))
        for word, score in frequencies.items()
    ):
        raise IndexValidationError("Literary frequency artifact has an invalid entry")
    return manifest


def validate_loaded_index(
    index_dir: Path,
    embeddings: np.ndarray,
    metadata: list[dict],
    word_index: dict[str, list[int]],
    record_embedding_indices: np.ndarray | None = None,
) -> None:
    """Cheap startup validation, including compatibility with legacy indexes."""
    rows = len(metadata)
    expected_embedding_rows = rows if record_embedding_indices is None else None
    if embeddings.ndim != 2 or (
        expected_embedding_rows is not None and embeddings.shape[0] != expected_embedding_rows
    ):
        raise IndexValidationError(
            f"Index row mismatch: embeddings={embeddings.shape}, metadata={rows}"
        )
    if record_embedding_indices is not None:
        if record_embedding_indices.ndim != 1 or record_embedding_indices.shape[0] != rows:
            raise IndexValidationError("Record-to-embedding mapping does not match metadata")
        if record_embedding_indices.size and (
            int(record_embedding_indices.min()) < 0
            or int(record_embedding_indices.max()) >= embeddings.shape[0]
        ):
            raise IndexValidationError("Record-to-embedding mapping has an invalid reference")
    referenced: set[int] = set()
    reference_count = 0
    for headword, indices in word_index.items():
        if not isinstance(indices, list):
            raise IndexValidationError("Word index entry is not a list")
        for row in indices:
            if not isinstance(row, int) or not 0 <= row < rows:
                raise IndexValidationError(f"Word index contains invalid row reference: {row!r}")
            if metadata[row].get("headword", "").casefold() != headword.casefold():
                raise IndexValidationError(
                    f"Word index key {headword!r} does not match metadata row {row}"
                )
            referenced.add(row)
            reference_count += 1
    if reference_count != rows:
        raise IndexValidationError(f"Word index has {reference_count} references for {rows} rows")
    if len(referenced) != rows:
        raise IndexValidationError(f"Word index covers {len(referenced)} of {rows} metadata rows")

    manifest = load_manifest(index_dir)
    if manifest is not None and manifest.get("record_count") != rows:
        raise IndexValidationError("Loaded record count does not match manifest")


def publish_staged_index(index_root: Path, staging_dir: Path, manifest: dict) -> Path:
    """Make a validated staging bundle current without touching the old bundle."""
    root = Path(index_root)
    staging = Path(staging_dir)
    versions = root / VERSIONS_DIR_NAME
    resolved_versions = versions.resolve()
    if staging.parent.resolve() != resolved_versions or not staging.name.startswith(".staging-"):
        raise ValueError(f"Refusing to publish unexpected staging path: {staging}")
    if staging.is_symlink() or not staging.is_dir():
        raise ValueError(f"Refusing to publish a non-directory staging path: {staging}")

    build_id = manifest["build_id"]
    if (
        not isinstance(build_id, str)
        or len(build_id) != 32
        or any(c not in "0123456789abcdef" for c in build_id)
    ):
        raise ValueError(f"Unsafe build identifier: {build_id!r}")
    final_dir = versions / build_id
    if final_dir.exists():
        raise FileExistsError(f"Index build already exists: {final_dir}")

    staging.rename(final_dir)
    _fsync_directory(versions)

    temporary_link = root / f".current-{uuid.uuid4().hex}"
    relative_target = Path(VERSIONS_DIR_NAME) / build_id
    try:
        os.symlink(relative_target, temporary_link)
        os.replace(temporary_link, root / CURRENT_LINK_NAME)
    except Exception:
        if temporary_link.is_symlink() or temporary_link.exists():
            temporary_link.unlink()
        # Restore the staging name so the caller's narrowly-scoped cleanup can
        # remove only files it created. The old current pointer was untouched.
        if final_dir.is_dir() and not staging.exists():
            final_dir.rename(staging)
        raise
    _fsync_directory(root)
    return final_dir
