import json
import mmap
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import overload

import numpy as np

from revdict.compact_index import (
    COMPACT_ARTIFACTS, HEADWORD_FREQUENCIES, METADATA_OFFSETS, ROW_HEADWORDS,
    WORD_BYTES, WORD_OFFSETS, WORD_ROW_OFFSETS, WORD_ROWS, MappedBytes,
    _close_array, write_compact_artifacts,
)
from revdict.index_bundle import resolve_active_index_dir
from revdict.paths import INDEX_DIR


COMPACT_WORD_INDEX_ARTIFACTS = (WORD_BYTES, WORD_OFFSETS, WORD_ROW_OFFSETS, WORD_ROWS)


class MetadataStore(Sequence[dict]):
    """Bounded random-access decoding over persisted JSONL byte offsets."""

    def __init__(
        self,
        path: Path,
        offsets_path: Path | None = None,
        *,
        cache_size: int = 4096,
        allow_source_offset_derivation: bool = False,
    ) -> None:
        self._path = Path(path)
        self._file = self._path.open("rb")
        self._mapping = None
        self._offsets = None
        self._cache: OrderedDict[int, dict] = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.RLock()
        self._closed = False
        try:
            offsets_path = offsets_path or self._path.parent / METADATA_OFFSETS
            if not Path(offsets_path).is_file():
                if not allow_source_offset_derivation:
                    raise RuntimeError("Index optimization is required: metadata offsets are missing")
                # Unversioned build/source fixtures have no published runtime
                # contract. Derive offsets without decoding records; search
                # rejects unversioned indexes before calling this loader.
                offsets = [0]
                with self._path.open("rb") as source:
                    while line := source.readline():
                        if line.strip():
                            offsets.append(source.tell())
                self._offsets = np.asarray(offsets, dtype="uint64")
            else:
                self._offsets = np.load(offsets_path, mmap_mode="r")
            size = self._path.stat().st_size
            if (
                self._offsets.dtype != np.dtype("uint64") or self._offsets.ndim != 1
                or not len(self._offsets) or int(self._offsets[0]) != 0
                or int(self._offsets[-1]) != size
                or np.any(self._offsets[1:] < self._offsets[:-1])
            ):
                raise RuntimeError("Index optimization failed validation: invalid metadata offsets")
            self._mapping = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ) if size else None
        except Exception:
            self.close()
            raise

    def __len__(self) -> int:
        return len(self._offsets) - 1

    @overload
    def __getitem__(self, index: int) -> dict: ...
    @overload
    def __getitem__(self, index: slice) -> list[dict]: ...

    def __getitem__(self, index: int | slice) -> dict | list[dict]:
        if isinstance(index, slice):
            return [self[row] for row in range(*index.indices(len(self)))]
        if not isinstance(index, int):
            raise TypeError(f"metadata indices must be integers or slices, not {type(index).__name__}")
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("metadata index out of range")
        with self._lock:
            if self._closed:
                raise ValueError("metadata store is closed")
            cached = self._cache.get(index)
            if cached is not None:
                self._cache.move_to_end(index)
                return cached
            assert self._mapping is not None
            start, end = int(self._offsets[index]), int(self._offsets[index + 1])
            record = json.loads(self._mapping[start:end])
            self._cache[index] = record
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
            return record

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._cache.clear()
            if self._mapping is not None:
                self._mapping.close()
                self._mapping = None
            _close_array(self._offsets)
            self._offsets = None
            self._file.close()
            self._closed = True

    def __enter__(self): return self
    def __exit__(self, *_args) -> None: self.close()
    def __del__(self) -> None:
        try: self.close()
        except Exception: pass


class CompactWordIndex(Mapping[str, list[int]]):
    """Sorted UTF-8 lexicon, CSR rows, and headword-aligned frequency."""

    def __init__(self, index_dir: Path) -> None:
        index_dir = Path(index_dir)
        self._closed = False
        self._word_bytes = MappedBytes(index_dir / WORD_BYTES)
        self._word_offsets = self._row_offsets = self._rows = None
        self.row_headwords = self.frequencies = None
        try:
            self._word_offsets = np.load(index_dir / WORD_OFFSETS, mmap_mode="r")
            self._row_offsets = np.load(index_dir / WORD_ROW_OFFSETS, mmap_mode="r")
            self._rows = np.load(index_dir / WORD_ROWS, mmap_mode="r")
            self.row_headwords = np.load(index_dir / ROW_HEADWORDS, mmap_mode="r")
            self.frequencies = np.load(index_dir / HEADWORD_FREQUENCIES, mmap_mode="r")
            self._validate_layout()
        except Exception:
            self.close()
            raise

    def _validate_layout(self) -> None:
        if (
            self._word_offsets.dtype != np.dtype("uint64")
            or self._row_offsets.dtype != np.dtype("uint64")
            or self._rows.dtype != np.dtype("uint32")
            or self.row_headwords.dtype != np.dtype("uint32")
            or self.frequencies.dtype != np.dtype("float64")
            or self._word_offsets.ndim != 1 or self._row_offsets.ndim != 1
            or self._rows.ndim != 1 or self.row_headwords.ndim != 1
            or self.frequencies.ndim != 1
            or len(self._word_offsets) != len(self._row_offsets)
            or len(self._word_offsets) != len(self.frequencies) + 1
            or int(self._word_offsets[0]) != 0
            or int(self._word_offsets[-1]) != len(self._word_bytes)
            or int(self._row_offsets[0]) != 0
            or int(self._row_offsets[-1]) != len(self._rows)
            or len(self.row_headwords) != len(self._rows)
            or np.any(self._word_offsets[1:] < self._word_offsets[:-1])
            or np.any(self._row_offsets[1:] < self._row_offsets[:-1])
            or (len(self.row_headwords) and int(self.row_headwords.max()) >= len(self))
        ):
            raise RuntimeError("Index optimization failed validation: invalid compact lexicon")

    def __len__(self) -> int: return len(self._word_offsets) - 1

    def word_at(self, index: int) -> str:
        return self._word_bytes.slice(int(self._word_offsets[index]), int(self._word_offsets[index + 1])).decode("utf-8")

    def word_id(self, word: str) -> int | None:
        low, high = 0, len(self)
        while low < high:
            middle = (low + high) // 2
            if self.word_at(middle) < word: low = middle + 1
            else: high = middle
        return low if low < len(self) and self.word_at(low) == word else None

    def rows_for_id(self, word_id: int) -> np.ndarray:
        start, end = int(self._row_offsets[word_id]), int(self._row_offsets[word_id + 1])
        return self._rows[start:end]

    def rows_for_id_range(self, start_word_id: int, end_word_id: int) -> np.ndarray:
        """Return all record rows for a contiguous sorted headword range."""
        start = int(self._row_offsets[start_word_id])
        end = int(self._row_offsets[end_word_id])
        return self._rows[start:end]

    def first_row_for_id(self, word_id: int) -> int:
        return int(self._rows[int(self._row_offsets[word_id])])

    def frequency_by_id(self, word_id: int, default=None):
        value = float(self.frequencies[word_id])
        return default if np.isnan(value) else value

    def frequency(self, word: str, default=None):
        word_id = self.word_id(word)
        return default if word_id is None else self.frequency_by_id(word_id, default)

    def headword_for_row(self, row: int) -> str:
        return self.word_at(int(self.row_headwords[row]))

    def __iter__(self):
        for index in range(len(self)): yield self.word_at(index)

    def __getitem__(self, word: str) -> list[int]:
        word_id = self.word_id(word)
        if word_id is None: raise KeyError(word)
        return self.rows_for_id(word_id).tolist()

    def items(self):
        for word_id in range(len(self)):
            yield self.word_at(word_id), self.rows_for_id(word_id).tolist()

    def close(self) -> None:
        if self._closed: return
        for attribute in ("_word_offsets", "_row_offsets", "_rows", "row_headwords", "frequencies"):
            _close_array(getattr(self, attribute, None)); setattr(self, attribute, None)
        self._word_bytes.close(); self._closed = True

    def __del__(self) -> None:
        try: self.close()
        except Exception: pass


class CompactFrequency(Mapping[str, float]):
    def __init__(self, words: CompactWordIndex) -> None: self.words = words
    def __len__(self) -> int: return int(np.count_nonzero(~np.isnan(self.words.frequencies)))
    def __iter__(self):
        for word_id in range(len(self.words)):
            if not np.isnan(self.words.frequencies[word_id]): yield self.words.word_at(word_id)
    def __getitem__(self, word: str) -> float:
        value = self.words.frequency(word)
        if value is None: raise KeyError(word)
        return value
    def get(self, word: str, default=None): return self.words.frequency(word, default)


def write_compact_word_index(index_dir: Path, word_index: Mapping[str, Sequence[int]]) -> None:
    """Write the lexicon subset for low-level callers and migration tests."""
    index_dir = Path(index_dir)
    words = sorted(word_index)
    encoded = [word.encode("utf-8") for word in words]
    word_offsets = np.zeros(len(words) + 1, dtype="uint64")
    row_offsets = np.zeros(len(words) + 1, dtype="uint64")
    for index, (value, word) in enumerate(zip(encoded, words), start=1):
        word_offsets[index] = word_offsets[index - 1] + len(value)
        row_offsets[index] = row_offsets[index - 1] + len(word_index[word])
    rows = np.empty(int(row_offsets[-1]), dtype="uint32")
    row_headwords = np.empty(len(rows), dtype="uint32")
    for word_id, word in enumerate(words):
        start, end = int(row_offsets[word_id]), int(row_offsets[word_id + 1])
        values = word_index[word]
        rows[start:end] = values
        row_headwords[np.asarray(values, dtype="uint32")] = word_id
    (index_dir / WORD_BYTES).write_bytes(b"".join(encoded))
    np.save(index_dir / WORD_OFFSETS, word_offsets)
    np.save(index_dir / WORD_ROW_OFFSETS, row_offsets)
    np.save(index_dir / WORD_ROWS, rows)
    np.save(index_dir / ROW_HEADWORDS, row_headwords)
    np.save(index_dir / HEADWORD_FREQUENCIES, np.full(len(words), np.nan, dtype="float64"))


def load_word_index(index_dir: Path = INDEX_DIR) -> CompactWordIndex:
    active = resolve_active_index_dir(Path(index_dir))
    required = (*COMPACT_WORD_INDEX_ARTIFACTS, ROW_HEADWORDS, HEADWORD_FREQUENCIES)
    missing = [name for name in required if not (active / name).is_file()]
    if missing:
        if not (active / "manifest.json").exists():
            # Authoritative JSON remains available to unversioned build/source
            # tooling. Search requires a schema-v3 manifest before loading.
            return json.loads((active / "word_index.json").read_text(encoding="utf-8"))
        raise RuntimeError("Index optimization is required before search; missing compact artifacts: " + ", ".join(missing))
    return CompactWordIndex(active)


def load_metadata(index_dir: Path = INDEX_DIR) -> MetadataStore:
    active = resolve_active_index_dir(Path(index_dir))
    return MetadataStore(
        active / "metadata.jsonl",
        active / METADATA_OFFSETS,
        allow_source_offset_derivation=not (active / "manifest.json").exists(),
    )


def lookup_exact(word: str, word_index: Mapping[str, list[int]], metadata: Sequence[dict]) -> dict | None:
    indices = word_index.get(word.lower())
    if not indices: return None
    senses = []
    for index in indices:
        record = metadata[index]
        senses.append({
            "pos": record["pos"], "definition": record["definition"],
            "examples": record["examples"], "source": record["source"],
            "sources": record.get("sources") or [record["source"]],
            "sentiwordnet": record.get("sentiwordnet"), "emolex": record.get("emolex"),
            "synonyms": record.get("synonyms"), "synset": record.get("synset"),
            "antonyms": record.get("antonyms") or [], "topics": record.get("topics") or [],
            "wiktionary_sense_ids": record.get("wiktionary_sense_ids") or [],
            "wikidata_ids": record.get("wikidata_ids") or [],
        })
    return {"headword": word, "senses": senses}
