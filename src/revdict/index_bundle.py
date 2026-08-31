"""Versioned, validated index bundles with atomic publication.

The historical index layout placed all artifacts directly in ``INDEX_DIR``.
Daemon startup upgrades that layout into an immutable schema-v3 bundle before
runtime search opens it. Every new build is written under ``INDEX_DIR/versions``
and becomes visible through a single atomic ``INDEX_DIR/current`` symlink.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
import fcntl
from datetime import datetime, timezone
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from revdict.compact_index import (
    COMPACT_ARTIFACTS, CORE_COMPACT_ARTIFACTS, HEADWORD_FREQUENCIES,
    METADATA_OFFSETS, ROW_HEADWORDS,
    CompactFacets, validate_compact_artifacts, write_compact_artifacts,
)
from revdict.structural_index import (
    STRUCTURAL_ARTIFACTS,
    validate_structural_artifacts,
    write_structural_artifacts,
)

INDEX_FORMAT = "revdict-index"
SCHEMA_VERSION = 3
MANIFEST_FILENAME = "manifest.json"
CURRENT_LINK_NAME = "current"
VERSIONS_DIR_NAME = "versions"
LEGACY_REQUIRED_ARTIFACTS = (
    "embeddings.npy",
    "metadata.jsonl",
    "word_index.json",
    "literary_frequency.json",
)
BASE_REQUIRED_ARTIFACTS = (*LEGACY_REQUIRED_ARTIFACTS, "record_embeddings.npy")
REQUIRED_ARTIFACTS = (*BASE_REQUIRED_ARTIFACTS, *COMPACT_ARTIFACTS)
COMPACT_WORD_INDEX_ARTIFACTS = CORE_COMPACT_ARTIFACTS
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
    # A published v3 bundle from an earlier revdict release can be a valid
    # optimization source even when it predates newly derived artifacts.  Let
    # the daemon acquire the build lock and augment that bundle atomically;
    # rejecting it here would incorrectly tell the user to rebuild embeddings.
    required = (
        (*BASE_REQUIRED_ARTIFACTS, *CORE_COMPACT_ARTIFACTS)
        if index_schema_version(active) == SCHEMA_VERSION
        else LEGACY_REQUIRED_ARTIFACTS
    )
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
    compact_present = [filename for filename in COMPACT_ARTIFACTS if (index_dir / filename).is_file()]
    if not compact_present:
        word_index = json.loads((index_dir / "word_index.json").read_text(encoding="utf-8"))
        frequencies = json.loads((index_dir / "literary_frequency.json").read_text(encoding="utf-8"))
        write_compact_artifacts(index_dir, word_index, frequencies)
        compact_present = list(COMPACT_ARTIFACTS)
    if len(compact_present) != len(COMPACT_ARTIFACTS):
        raise ValueError("Schema-v3 compact index is incomplete")
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


def read_manifest_unchecked(index_dir: Path) -> dict | None:
    """Read format/version metadata for upgrade detection only."""
    path = Path(index_dir) / MANIFEST_FILENAME
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IndexValidationError(f"Could not read index manifest {path}: {error}") from error
    if manifest.get("format") != INDEX_FORMAT:
        raise IndexValidationError(f"Unsupported index format in {path}")
    return manifest


def index_schema_version(index_dir: Path) -> int | None:
    manifest = read_manifest_unchecked(index_dir)
    return None if manifest is None else manifest.get("schema_version")


def index_optimization_required(index_dir: Path) -> bool:
    manifest = read_manifest_unchecked(index_dir)
    if manifest is None or manifest.get("schema_version") != SCHEMA_VERSION:
        return True
    artifacts = manifest.get("artifacts")
    return not isinstance(artifacts, dict) or any(
        not (Path(index_dir) / filename).is_file() or filename not in artifacts
        for filename in REQUIRED_ARTIFACTS
    )


def _validate_compact_word_index(index_dir: Path, word_index: dict[str, list[int]]) -> None:
    try:
        word_offsets = np.load(index_dir / "word_offsets.npy", mmap_mode="r")
        row_offsets = np.load(index_dir / "word_row_offsets.npy", mmap_mode="r")
        rows = np.load(index_dir / "word_rows.npy", mmap_mode="r")
        word_bytes = (index_dir / "word_bytes.bin").read_bytes()
    except (OSError, ValueError) as error:
        raise IndexValidationError(f"Could not read compact word index: {error}") from error

    items = sorted(word_index.items())
    encoded_words = [word.encode("utf-8") for word, _indices in items]
    expected_word_offsets = np.empty(len(items) + 1, dtype="uint64")
    expected_row_offsets = np.empty(len(items) + 1, dtype="uint64")
    expected_word_offsets[0] = 0
    expected_row_offsets[0] = 0
    for index, (encoded, (_word, indices)) in enumerate(
        zip(encoded_words, items), start=1
    ):
        expected_word_offsets[index] = expected_word_offsets[index - 1] + len(encoded)
        expected_row_offsets[index] = expected_row_offsets[index - 1] + len(indices)
    expected_rows = np.fromiter(
        (row for _word, indices in items for row in indices),
        dtype="uint32",
        count=int(expected_row_offsets[-1]),
    )

    try:
        if (
            word_offsets.dtype != np.dtype("uint64")
            or row_offsets.dtype != np.dtype("uint64")
            or rows.dtype != np.dtype("uint32")
            or word_bytes != b"".join(encoded_words)
            or not np.array_equal(word_offsets, expected_word_offsets)
            or not np.array_equal(row_offsets, expected_row_offsets)
            or not np.array_equal(rows, expected_rows)
        ):
            raise IndexValidationError("Published compact word index disagrees with word_index.json")
    finally:
        for values in (word_offsets, row_offsets, rows):
            mapping = getattr(values, "_mmap", None)
            if mapping is not None:
                mapping.close()


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
    try:
        shape = manifest.get("embeddings", {})
        expected_shape = (shape.get("rows"), shape.get("dimensions"))
        if embeddings.ndim != 2 or embeddings.shape != expected_shape:
            raise IndexValidationError(
                f"Embedding shape {embeddings.shape} does not match manifest {expected_shape}"
            )
        if str(embeddings.dtype) != shape.get("dtype"):
            raise IndexValidationError("Embedding dtype does not match manifest")
        if embeddings.dtype != np.dtype("float32"):
            raise IndexValidationError("Schema-v3 embeddings must use float32")
        if embeddings.shape[0] != manifest.get("definition_count"):
            raise IndexValidationError("Embedding row count does not match definition count")
        embedding_rows = embeddings.shape[0]
    finally:
        embeddings._mmap.close()

    record_embeddings = np.load(index_dir / "record_embeddings.npy", mmap_mode="r")
    try:
        record_shape = manifest.get("record_embeddings", {})
        if record_embeddings.ndim != 1 or record_embeddings.shape[0] != manifest.get("record_count"):
            raise IndexValidationError("Record-to-embedding mapping does not match record count")
        if record_embeddings.shape[0] != record_shape.get("rows"):
            raise IndexValidationError("Record-to-embedding mapping shape does not match manifest")
        if str(record_embeddings.dtype) != record_shape.get("dtype"):
            raise IndexValidationError("Record-to-embedding mapping dtype does not match manifest")
        if record_embeddings.size and (
            int(record_embeddings.min()) < 0 or int(record_embeddings.max()) >= embedding_rows
        ):
            raise IndexValidationError("Record-to-embedding mapping contains an invalid vector reference")
    finally:
        record_embeddings._mmap.close()

    metadata_rows = 0
    metadata_headwords: list[str] = []
    metadata_offsets = []
    compact_facets = CompactFacets.open(index_dir)
    try:
        with (index_dir / "metadata.jsonl").open("rb") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                metadata_offsets.append(f.tell() - len(line))
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise IndexValidationError(
                        f"Invalid metadata JSON on line {line_number}: {error}"
                    ) from error
                metadata_rows += 1
                if not compact_facets.matches_record(metadata_rows - 1, record):
                    raise IndexValidationError(
                        f"Compact facets disagree with metadata row {metadata_rows - 1}"
                    )
                try:
                    metadata_headwords.append(record["headword"].casefold())
                except (KeyError, AttributeError) as error:
                    raise IndexValidationError(
                        f"Metadata line {line_number} has no valid headword"
                    ) from error
            metadata_offsets.append(f.tell())
    finally:
        compact_facets.close()
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
    _validate_compact_word_index(index_dir, word_index)
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
    compact_offsets = np.load(index_dir / METADATA_OFFSETS, mmap_mode="r")
    compact_row_headwords = np.load(index_dir / ROW_HEADWORDS, mmap_mode="r")
    compact_frequencies = np.load(index_dir / HEADWORD_FREQUENCIES, mmap_mode="r")
    try:
        items = sorted(word_index.items())
        expected_row_headwords = np.empty(metadata_rows, dtype="uint32")
        expected_frequencies = np.full(len(items), np.nan, dtype="float64")
        for word_id, (word, rows) in enumerate(items):
            expected_row_headwords[np.asarray(rows, dtype="uint32")] = word_id
            if word in frequencies:
                expected_frequencies[word_id] = frequencies[word]
        if not np.array_equal(compact_offsets, np.asarray(metadata_offsets, dtype="uint64")):
            raise IndexValidationError("Compact metadata offsets disagree with metadata.jsonl")
        if not np.array_equal(compact_row_headwords, expected_row_headwords):
            raise IndexValidationError("Compact row headwords disagree with word_index.json")
        if not np.array_equal(compact_frequencies, expected_frequencies, equal_nan=True):
            raise IndexValidationError("Compact frequencies disagree with literary_frequency.json")
    finally:
        for values in (compact_offsets, compact_row_headwords, compact_frequencies):
            mapping = getattr(values, "_mmap", None)
            if mapping is not None:
                mapping.close()
    try:
        validate_compact_artifacts(index_dir, metadata_rows)
        validate_structural_artifacts(index_dir, [word for word, _rows in items])
    except (OSError, ValueError) as error:
        raise IndexValidationError(f"Invalid schema-v3 compact index: {error}") from error
    return manifest


def validate_loaded_index(
    index_dir: Path,
    embeddings: np.ndarray,
    metadata: Sequence[dict],
    word_index: dict[str, list[int]],
    record_embedding_indices: np.ndarray | None = None,
) -> None:
    """Cheap startup validation, including compatibility with legacy indexes."""
    rows = len(metadata)
    manifest = load_manifest(index_dir)
    expected_embedding_rows = rows if record_embedding_indices is None else None
    if embeddings.ndim != 2 or (
        expected_embedding_rows is not None and embeddings.shape[0] != expected_embedding_rows
    ):
        raise IndexValidationError(
            f"Index row mismatch: embeddings={embeddings.shape}, metadata={rows}"
        )
    if embeddings.dtype != np.dtype("float32"):
        raise IndexValidationError("Schema-v3 embeddings must use float32")
    if record_embedding_indices is not None:
        if record_embedding_indices.ndim != 1 or record_embedding_indices.shape[0] != rows:
            raise IndexValidationError("Record-to-embedding mapping does not match metadata")
        if record_embedding_indices.size and (
            int(record_embedding_indices.min()) < 0
            or int(record_embedding_indices.max()) >= embeddings.shape[0]
        ):
            raise IndexValidationError("Record-to-embedding mapping has an invalid reference")
    referenced = bytearray(rows)
    referenced_rows = 0
    reference_count = 0
    for headword, indices in word_index.items():
        if not isinstance(indices, list):
            raise IndexValidationError("Word index entry is not a list")
        for row in indices:
            if not isinstance(row, int) or not 0 <= row < rows:
                raise IndexValidationError(f"Word index contains invalid row reference: {row!r}")
            # Versioned bundles received the expensive JSON/headword/checksum
            # validation before atomic publication. Legacy indexes have no
            # such guarantee, so retain the complete cross-check for them.
            if manifest is None and metadata[row].get("headword", "").casefold() != headword.casefold():
                raise IndexValidationError(
                    f"Word index key {headword!r} does not match metadata row {row}"
                )
            if not referenced[row]:
                referenced[row] = 1
                referenced_rows += 1
            reference_count += 1
    if reference_count != rows:
        raise IndexValidationError(f"Word index has {reference_count} references for {rows} rows")
    if referenced_rows != rows:
        raise IndexValidationError(
            f"Word index covers {referenced_rows} of {rows} metadata rows"
        )

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


def _copy_or_link(source: Path, destination: Path, *, immutable: bool) -> None:
    """Reuse immutable version artifacts; never link a mutable legacy file."""
    if immutable:
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def _augment_schema_v3_locked(index_root: Path, source: Path, prior_manifest: dict) -> Path:
    """Atomically add newly required compact artifacts to an older v3 bundle."""
    artifacts = prior_manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise IndexValidationError("Schema-v3 source manifest has no artifact table")
    reusable = (*BASE_REQUIRED_ARTIFACTS, *CORE_COMPACT_ARTIFACTS)
    for filename in reusable:
        path = source / filename
        expected = artifacts.get(filename)
        if not path.is_file() or not isinstance(expected, dict):
            raise IndexValidationError(f"Schema-v3 optimization source is missing: {path}")
        if path.stat().st_size != expected.get("bytes") or _sha256(path) != expected.get("sha256"):
            raise IndexValidationError(f"Schema-v3 optimization source checksum failed: {path}")

    staging = create_staging_index(index_root)
    immutable = source.parent.resolve() == (Path(index_root) / VERSIONS_DIR_NAME).resolve()
    try:
        for filename in reusable:
            _copy_or_link(source / filename, staging / filename, immutable=immutable)
        try:
            word_index = json.loads((staging / "word_index.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise IndexValidationError(f"Schema-v3 word index is invalid: {error}") from error
        words = sorted(word_index)
        write_structural_artifacts(staging, words)

        embeddings = record_embeddings = None
        try:
            embeddings = np.load(staging / "embeddings.npy", mmap_mode="r")
            record_embeddings = np.load(staging / "record_embeddings.npy", mmap_mode="r")
            manifest = build_manifest(
                staging,
                record_count=int(prior_manifest["record_count"]),
                definition_count=int(prior_manifest["definition_count"]),
                embeddings=embeddings,
                record_embedding_indices=record_embeddings,
                datasets=prior_manifest.get("datasets"),
                models=prior_manifest.get("models"),
                statistics=prior_manifest.get("statistics"),
            )
            write_manifest(staging, manifest)
            validate_index_directory(staging, verify_hashes=True)
            return publish_staged_index(index_root, staging, manifest)
        finally:
            for values in (embeddings, record_embeddings):
                mapping = getattr(values, "_mmap", None)
                if mapping is not None:
                    mapping.close()
    except Exception:
        discard_staging_index(index_root, staging)
        raise


def _upgrade_locked(index_root: Path) -> Path:
    source = resolve_active_index_dir(index_root)
    if index_schema_version(source) == SCHEMA_VERSION:
        manifest = load_manifest(source, required=True)
        artifacts = manifest.get("artifacts", {})
        complete = all(
            (source / filename).is_file() and filename in artifacts
            for filename in STRUCTURAL_ARTIFACTS
        )
        if not complete:
            return _augment_schema_v3_locked(index_root, source, manifest)
        try:
            validate_compact_artifacts(source, int(manifest["record_count"]))
        except (KeyError, OSError, ValueError) as error:
            raise IndexValidationError(f"Invalid schema-v3 compact index: {error}") from error
        return source

    prior_manifest = read_manifest_unchecked(source)
    prior_version = None if prior_manifest is None else prior_manifest.get("schema_version")
    if prior_version not in (None, 1, 2):
        raise IndexValidationError(
            f"Index optimization cannot upgrade schema {prior_version!r} to {SCHEMA_VERSION}"
        )
    for filename in LEGACY_REQUIRED_ARTIFACTS:
        if not (source / filename).is_file():
            raise IndexValidationError(f"Index optimization source is missing: {source / filename}")

    staging = create_staging_index(index_root)
    immutable = source.parent.resolve() == (Path(index_root) / VERSIONS_DIR_NAME).resolve()
    try:
        for filename in LEGACY_REQUIRED_ARTIFACTS:
            _copy_or_link(source / filename, staging / filename, immutable=immutable)

        mapping_source = source / "record_embeddings.npy"
        embeddings = np.load(staging / "embeddings.npy", mmap_mode="r")
        if mapping_source.is_file():
            _copy_or_link(mapping_source, staging / "record_embeddings.npy", immutable=immutable)
            record_embeddings = np.load(staging / "record_embeddings.npy", mmap_mode="r")
        else:
            # Legacy indexes stored one embedding per record.
            record_embeddings = np.arange(embeddings.shape[0], dtype="int32")
            np.save(staging / "record_embeddings.npy", record_embeddings)

        try:
            word_index = json.loads((staging / "word_index.json").read_text(encoding="utf-8"))
            frequencies = json.loads(
                (staging / "literary_frequency.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise IndexValidationError(f"Index optimization source JSON is invalid: {error}") from error
        write_compact_artifacts(staging, word_index, frequencies)

        record_count = sum(len(rows) for rows in word_index.values())
        manifest = build_manifest(
            staging,
            record_count=record_count,
            definition_count=int(embeddings.shape[0]),
            embeddings=embeddings,
            record_embedding_indices=record_embeddings,
            datasets=(prior_manifest or {}).get("datasets"),
            models=(prior_manifest or {}).get("models"),
            statistics=(prior_manifest or {}).get("statistics"),
        )
        write_manifest(staging, manifest)
        validate_index_directory(staging, verify_hashes=True)
        return publish_staged_index(index_root, staging, manifest)
    except Exception:
        discard_staging_index(index_root, staging)
        raise
    finally:
        # np.memmap descriptors are deterministic, including on failure.
        for values_name in ("embeddings", "record_embeddings"):
            values = locals().get(values_name)
            mapping = getattr(values, "_mmap", None)
            if mapping is not None:
                mapping.close()


def ensure_schema_v3(index_root: Path) -> Path:
    """Atomically optimize v2/legacy indexes, waiting on the live lock owner."""
    root = Path(index_root)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "build.lock"
    with lock_path.open("a+") as lock_file:
        # Blocking flock intentionally has no elapsed-time cutoff. The kernel
        # releases it when the actual owner exits.
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            return _upgrade_locked(root)
        except Exception as error:
            raise RuntimeError(f"Index optimization failed; previous index remains current: {error}") from error
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
