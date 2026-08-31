"""Schema-v3 compact index artifacts.

The JSON files in an index bundle are recovery/build inputs.  This module is
the runtime view: all numeric data is opened read-only with ``numpy.memmap``
and text vocabularies are decoded from compact byte tables only on demand.
"""

from __future__ import annotations

import json
import mmap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rapidfuzz import process as rapidfuzz_process
from rapidfuzz.distance import Levenshtein

from revdict import category as category_module
from revdict.phonetics import SOUNDS_LIKE_THRESHOLD
from revdict.structural_index import (
    PHONEME_LENGTH_BYTES,
    PHONEME_LENGTH_OFFSETS,
    PHONEME_LENGTH_SEQUENCE_IDS,
    STRUCTURAL_ARTIFACTS,
    CompactStructuralIndex,
    write_structural_artifacts,
)


METADATA_OFFSETS = "metadata_offsets.npy"
WORD_BYTES = "word_bytes.bin"
WORD_OFFSETS = "word_offsets.npy"
WORD_ROW_OFFSETS = "word_row_offsets.npy"
WORD_ROWS = "word_rows.npy"
ROW_HEADWORDS = "row_headwords.npy"
HEADWORD_FREQUENCIES = "headword_frequencies.npy"
FACETS = "facets.npy"

VOCAB_NAMES = ("vowel", "rhyme", "meter")
TEXT_VOCAB_ARTIFACTS = tuple(
    filename
    for name in VOCAB_NAMES
    for filename in (f"{name}_bytes.bin", f"{name}_offsets.npy")
)
PHONEME_ARTIFACTS = (
    "phoneme_token_bytes.bin",
    "phoneme_token_offsets.npy",
    "phoneme_sequence_bytes.bin",
    "phoneme_sequence_offsets.npy",
)

CORE_COMPACT_ARTIFACTS = (
    METADATA_OFFSETS,
    WORD_BYTES,
    WORD_OFFSETS,
    WORD_ROW_OFFSETS,
    WORD_ROWS,
    ROW_HEADWORDS,
    HEADWORD_FREQUENCIES,
    FACETS,
    *TEXT_VOCAB_ARTIFACTS,
    *PHONEME_ARTIFACTS,
)
COMPACT_ARTIFACTS = (*CORE_COMPACT_ARTIFACTS, *STRUCTURAL_ARTIFACTS)

FACET_DTYPE = np.dtype(
    [
        ("flags", "<u2"),
        ("syllables", "<i2"),
        ("vowel", "<u2"),
        ("rhyme", "<u4"),
        ("meter", "<u2"),
        ("phonemes", "<u4"),
    ]
)

FLAG_NOUN = 1 << 0
FLAG_ADJECTIVE = 1 << 1
FLAG_VERB = 1 << 2
FLAG_ADVERB = 1 << 3
FLAG_IDIOM_SLANG = 1 << 4
FLAG_OLD = 1 << 5
CATEGORY_FLAGS = {
    "noun": FLAG_NOUN,
    "adjective": FLAG_ADJECTIVE,
    "verb": FLAG_VERB,
    "adverb": FLAG_ADVERB,
    "idiom_slang": FLAG_IDIOM_SLANG,
    "old": FLAG_OLD,
}

MISSING_U16 = np.iinfo(np.uint16).max
MISSING_U32 = np.iinfo(np.uint32).max


def _close_array(values: np.ndarray | None) -> None:
    mapping = getattr(values, "_mmap", None)
    if mapping is not None:
        mapping.close()


def _write_text_vocab(index_dir: Path, name: str, values: Sequence[str]) -> None:
    encoded = [value.encode("utf-8") for value in values]
    offsets = np.empty(len(encoded) + 1, dtype="uint64")
    offsets[0] = 0
    for index, value in enumerate(encoded, start=1):
        offsets[index] = offsets[index - 1] + len(value)
    (index_dir / f"{name}_bytes.bin").write_bytes(b"".join(encoded))
    np.save(index_dir / f"{name}_offsets.npy", offsets)


def _facet_flags(record: dict) -> int:
    flags = 0
    pos = record.get("pos")
    if pos in CATEGORY_FLAGS:
        flags |= CATEGORY_FLAGS[pos]
    if category_module.matches_category(record, "idiom_slang"):
        flags |= FLAG_IDIOM_SLANG
    if category_module.matches_category(record, "old"):
        flags |= FLAG_OLD
    return flags


def _normal_phonemes(phonemes: Sequence[str]) -> tuple[str, ...]:
    return tuple(value.rstrip("012") for value in phonemes)


def write_compact_artifacts(
    index_dir: Path,
    word_index: Mapping[str, Sequence[int]],
    literary_frequency: Mapping[str, float],
) -> None:
    """Derive all schema-v3 artifacts with one streaming metadata pass."""
    index_dir = Path(index_dir)
    words = sorted(word_index)
    encoded_words = [word.encode("utf-8") for word in words]
    word_offsets = np.empty(len(words) + 1, dtype="uint64")
    row_offsets = np.empty(len(words) + 1, dtype="uint64")
    word_offsets[0] = 0
    row_offsets[0] = 0
    for index, (encoded, word) in enumerate(zip(encoded_words, words), start=1):
        word_offsets[index] = word_offsets[index - 1] + len(encoded)
        row_offsets[index] = row_offsets[index - 1] + len(word_index[word])

    row_count = int(row_offsets[-1])
    rows = np.empty(row_count, dtype="uint32")
    row_headwords = np.full(row_count, MISSING_U32, dtype="uint32")
    for word_id, word in enumerate(words):
        start, end = int(row_offsets[word_id]), int(row_offsets[word_id + 1])
        word_rows = word_index[word]
        rows[start:end] = word_rows
        row_headwords[np.asarray(word_rows, dtype="uint32")] = word_id
    if row_count and np.any(row_headwords == MISSING_U32):
        raise ValueError("word index does not cover every metadata row")

    (index_dir / WORD_BYTES).write_bytes(b"".join(encoded_words))
    np.save(index_dir / WORD_OFFSETS, word_offsets)
    np.save(index_dir / WORD_ROW_OFFSETS, row_offsets)
    np.save(index_dir / WORD_ROWS, rows)
    np.save(index_dir / ROW_HEADWORDS, row_headwords)
    frequencies = np.full(len(words), np.nan, dtype="float64")
    for word_id, word in enumerate(words):
        value = literary_frequency.get(word)
        if value is not None:
            frequencies[word_id] = float(value)
    np.save(index_dir / HEADWORD_FREQUENCIES, frequencies)

    facets = np.zeros(row_count, dtype=FACET_DTYPE)
    facets["syllables"] = -1
    facets["vowel"] = MISSING_U16
    facets["rhyme"] = MISSING_U32
    facets["meter"] = MISSING_U16
    facets["phonemes"] = MISSING_U32
    vocab_maps: dict[str, dict[str, int]] = {name: {} for name in VOCAB_NAMES}
    vocab_values: dict[str, list[str]] = {name: [] for name in VOCAB_NAMES}
    phoneme_sequences: dict[tuple[str, ...], int] = {}
    sequence_values: list[tuple[str, ...]] = []
    metadata_offsets = np.empty(row_count + 1, dtype="uint64")
    metadata_offsets[0] = 0

    metadata_path = index_dir / "metadata.jsonl"
    with metadata_path.open("rb") as source:
        row = 0
        while True:
            line = source.readline()
            if not line:
                break
            if not line.strip():
                continue
            if row >= row_count:
                raise ValueError("metadata contains more rows than word index")
            record = json.loads(line)
            metadata_offsets[row] = source.tell() - len(line)
            facets["flags"][row] = _facet_flags(record)
            phonetics = record.get("phonetics")
            if phonetics:
                facets["syllables"][row] = int(phonetics["syllable_count"])
                for field, name in (("primary_vowel", "vowel"), ("rhyme_key", "rhyme"), ("meter", "meter")):
                    value = phonetics.get(field)
                    if value is None:
                        continue
                    value_id = vocab_maps[name].get(value)
                    if value_id is None:
                        value_id = len(vocab_values[name])
                        vocab_maps[name][value] = value_id
                        vocab_values[name].append(value)
                    facets[name][row] = value_id
                sequence = _normal_phonemes(phonetics.get("phonemes") or [])
                if sequence:
                    sequence_id = phoneme_sequences.get(sequence)
                    if sequence_id is None:
                        sequence_id = len(sequence_values)
                        phoneme_sequences[sequence] = sequence_id
                        sequence_values.append(sequence)
                    facets["phonemes"][row] = sequence_id
            row += 1
        if row != row_count:
            raise ValueError(f"metadata has {row} rows; word index has {row_count}")
        metadata_offsets[row_count] = source.tell()
    np.save(index_dir / METADATA_OFFSETS, metadata_offsets)
    np.save(index_dir / FACETS, facets)
    for name in VOCAB_NAMES:
        _write_text_vocab(index_dir, name, vocab_values[name])

    tokens = sorted({token for sequence in sequence_values for token in sequence})
    if len(tokens) > 255:
        raise ValueError("phoneme vocabulary exceeds compact byte encoding")
    _write_text_vocab(index_dir, "phoneme_token", tokens)
    token_ids = {token: index + 1 for index, token in enumerate(tokens)}
    sequence_offsets = np.empty(len(sequence_values) + 1, dtype="uint64")
    sequence_offsets[0] = 0
    encoded_sequences = []
    for index, sequence in enumerate(sequence_values, start=1):
        encoded = bytes(token_ids[token] for token in sequence)
        encoded_sequences.append(encoded)
        sequence_offsets[index] = sequence_offsets[index - 1] + len(encoded)
    (index_dir / "phoneme_sequence_bytes.bin").write_bytes(b"".join(encoded_sequences))
    np.save(index_dir / "phoneme_sequence_offsets.npy", sequence_offsets)
    write_structural_artifacts(index_dir, words)


class MappedBytes:
    def __init__(self, path: Path) -> None:
        self.file = Path(path).open("rb")
        try:
            self.mapping = (
                mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
                if self.file.seek(0, 2)
                else None
            )
        except Exception:
            self.file.close()
            raise

    def __len__(self) -> int:
        return 0 if self.mapping is None else len(self.mapping)

    def slice(self, start: int, end: int) -> bytes:
        if self.mapping is None:
            return b""
        return self.mapping[start:end]

    def close(self) -> None:
        if self.mapping is not None:
            self.mapping.close()
            self.mapping = None
        self.file.close()


class TextVocab(Sequence[str]):
    def __init__(self, index_dir: Path, name: str) -> None:
        self.bytes = MappedBytes(index_dir / f"{name}_bytes.bin")
        self.offsets = None
        try:
            self.offsets = np.load(index_dir / f"{name}_offsets.npy", mmap_mode="r")
            if self.offsets.ndim != 1 or not len(self.offsets) or int(self.offsets[0]) != 0 or int(self.offsets[-1]) != len(self.bytes):
                raise ValueError(f"compact {name} vocabulary has an invalid layout")
        except Exception:
            self.close()
            raise

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.bytes.slice(int(self.offsets[index]), int(self.offsets[index + 1])).decode("utf-8")

    def find(self, value: str) -> int | None:
        # Facet vocabularies are deliberately small; avoid a duplicate dict.
        for index in range(len(self)):
            if self[index] == value:
                return index
        return None

    def close(self) -> None:
        _close_array(getattr(self, "offsets", None))
        self.offsets = None
        self.bytes.close()


@dataclass
class CompactFacets:
    values: np.ndarray
    vowels: TextVocab
    rhymes: TextVocab
    meters: TextVocab
    phoneme_bytes: MappedBytes
    phoneme_offsets: np.ndarray
    token_vocab: TextVocab
    phoneme_length_bytes: MappedBytes
    phoneme_length_offsets: np.ndarray
    phoneme_length_sequence_ids: np.ndarray
    token_ids: dict[str, int] | None = None

    @classmethod
    def open(cls, index_dir: Path) -> "CompactFacets":
        opened = []
        try:
            values = np.load(index_dir / FACETS, mmap_mode="r")
            opened.append(values)
            vowels = TextVocab(index_dir, "vowel")
            rhymes = TextVocab(index_dir, "rhyme")
            meters = TextVocab(index_dir, "meter")
            token_vocab = TextVocab(index_dir, "phoneme_token")
            phoneme_bytes = MappedBytes(index_dir / "phoneme_sequence_bytes.bin")
            phoneme_offsets = np.load(index_dir / "phoneme_sequence_offsets.npy", mmap_mode="r")
            opened.append(phoneme_offsets)
            phoneme_length_bytes = MappedBytes(index_dir / PHONEME_LENGTH_BYTES)
            phoneme_length_offsets = np.load(
                index_dir / PHONEME_LENGTH_OFFSETS, mmap_mode="r"
            )
            phoneme_length_sequence_ids = np.load(
                index_dir / PHONEME_LENGTH_SEQUENCE_IDS, mmap_mode="r"
            )
            opened.extend((phoneme_length_offsets, phoneme_length_sequence_ids))
            result = cls(
                values,
                vowels,
                rhymes,
                meters,
                phoneme_bytes,
                phoneme_offsets,
                token_vocab,
                phoneme_length_bytes,
                phoneme_length_offsets,
                phoneme_length_sequence_ids,
            )
            result.validate()
            return result
        except Exception:
            for value in opened:
                _close_array(value)
            for name in (
                "vowels", "rhymes", "meters", "token_vocab", "phoneme_bytes",
                "phoneme_length_bytes",
            ):
                value = locals().get(name)
                if value is not None:
                    value.close()
            raise

    def validate(self) -> None:
        if self.values.ndim != 1 or self.values.dtype != FACET_DTYPE:
            raise ValueError("compact facet array has an invalid layout")
        if self.phoneme_offsets.ndim != 1 or not len(self.phoneme_offsets) or int(self.phoneme_offsets[0]) != 0 or int(self.phoneme_offsets[-1]) != len(self.phoneme_bytes):
            raise ValueError("compact phoneme sequences have an invalid layout")
        sequence_count = len(self.phoneme_offsets) - 1
        if (
            self.phoneme_length_offsets.dtype != np.dtype("uint64")
            or self.phoneme_length_offsets.ndim != 1
            or len(self.phoneme_length_offsets) < 2
            or int(self.phoneme_length_offsets[0]) != 0
            or int(self.phoneme_length_offsets[-1]) != sequence_count
            or np.any(self.phoneme_length_offsets[1:] < self.phoneme_length_offsets[:-1])
            or self.phoneme_length_sequence_ids.dtype != np.dtype("uint32")
            or self.phoneme_length_sequence_ids.shape != (sequence_count,)
            or not np.array_equal(
                np.sort(self.phoneme_length_sequence_ids),
                np.arange(sequence_count, dtype="uint32"),
            )
        ):
            raise ValueError("length-grouped phoneme index has an invalid layout")
        lengths = np.diff(self.phoneme_offsets)
        expected_bytes = 0
        for length in range(len(self.phoneme_length_offsets) - 1):
            start = int(self.phoneme_length_offsets[length])
            end = int(self.phoneme_length_offsets[length + 1])
            sequence_ids = self.phoneme_length_sequence_ids[start:end]
            if len(sequence_ids) and np.any(lengths[sequence_ids] != length):
                raise ValueError("length-grouped phoneme index disagrees with sequences")
            expected_bytes += (end - start) * length
        if expected_bytes != len(self.phoneme_length_bytes):
            raise ValueError("length-grouped phoneme byte table has an invalid size")
        for field, limit, missing in (
            ("vowel", len(self.vowels), MISSING_U16),
            ("rhyme", len(self.rhymes), MISSING_U32),
            ("meter", len(self.meters), MISSING_U16),
            ("phonemes", len(self.phoneme_offsets) - 1, MISSING_U32),
        ):
            values = self.values[field]
            if np.any((values != missing) & (values >= limit)):
                raise ValueError(f"compact facet {field} contains an invalid reference")

    def _phoneme_sequence(self, sequence_id: int) -> bytes:
        return self.phoneme_bytes.slice(
            int(self.phoneme_offsets[sequence_id]), int(self.phoneme_offsets[sequence_id + 1])
        )

    def _encode_target(self, target: Sequence[str]) -> bytes | None:
        if self.token_ids is None:
            self.token_ids = {
                self.token_vocab[index]: index + 1 for index in range(len(self.token_vocab))
            }
        try:
            return bytes(self.token_ids[token.rstrip("012")] for token in target)
        except KeyError:
            return None

    def sounds_like_ids(self, target: Sequence[str]) -> np.ndarray:
        encoded = self._encode_target(target)
        if encoded is None:
            return np.empty(0, dtype="uint32")
        target_length = len(encoded)
        max_stored_length = len(self.phoneme_length_offsets) - 2
        minimum_length = max(1, int(np.ceil(target_length * (1 - SOUNDS_LIKE_THRESHOLD))))
        maximum_length = min(
            max_stored_length,
            int(np.floor(target_length / (1 - SOUNDS_LIKE_THRESHOLD))),
        )
        choices: list[bytes] = []
        choice_ids: list[np.ndarray] = []
        byte_start = sum(
            (
                int(self.phoneme_length_offsets[length + 1])
                - int(self.phoneme_length_offsets[length])
            )
            * length
            for length in range(minimum_length)
        )
        for length in range(minimum_length, maximum_length + 1):
            id_start = int(self.phoneme_length_offsets[length])
            id_end = int(self.phoneme_length_offsets[length + 1])
            byte_end = byte_start + (id_end - id_start) * length
            group = self.phoneme_length_bytes.slice(byte_start, byte_end)
            choices.extend(
                group[offset : offset + length]
                for offset in range(0, len(group), length)
            )
            choice_ids.append(self.phoneme_length_sequence_ids[id_start:id_end])
            byte_start = byte_end
        if not choices:
            return np.empty(0, dtype="uint32")
        sequence_ids = np.concatenate(choice_ids)
        accepted = []
        coarse = rapidfuzz_process.extract(
            encoded,
            choices,
            scorer=Levenshtein.normalized_distance,
            score_cutoff=SOUNDS_LIKE_THRESHOLD,
            limit=None,
        )
        for candidate, _normalized, sequence_id in coarse:
            # Pin the inclusive boundary to the original integer definition;
            # this avoids floating point disagreement at exactly 0.34.
            distance = Levenshtein.distance(candidate, encoded)
            if distance / max(len(candidate), len(encoded), 1) <= SOUNDS_LIKE_THRESHOLD:
                accepted.append(int(sequence_ids[sequence_id]))
        return np.asarray(accepted, dtype="uint32")

    def matches_record(self, row: int, record: dict) -> bool:
        """Exact build-time equivalence check against authoritative JSON."""
        value = self.values[row]
        if int(value["flags"]) != _facet_flags(record):
            return False
        phonetics = record.get("phonetics")
        if not phonetics:
            return (
                int(value["syllables"]) == -1
                and int(value["vowel"]) == MISSING_U16
                and int(value["rhyme"]) == MISSING_U32
                and int(value["meter"]) == MISSING_U16
                and int(value["phonemes"]) == MISSING_U32
            )
        if int(value["syllables"]) != int(phonetics["syllable_count"]):
            return False
        for field, vocab, source_field, missing in (
            ("vowel", self.vowels, "primary_vowel", MISSING_U16),
            ("rhyme", self.rhymes, "rhyme_key", MISSING_U32),
            ("meter", self.meters, "meter", MISSING_U16),
        ):
            expected = phonetics.get(source_field)
            actual_id = int(value[field])
            if expected is None:
                if actual_id != missing:
                    return False
            elif actual_id == missing or vocab[actual_id] != expected:
                return False
        sequence = _normal_phonemes(phonetics.get("phonemes") or [])
        sequence_id = int(value["phonemes"])
        if not sequence:
            return sequence_id == MISSING_U32
        encoded = self._encode_target(sequence)
        return (
            sequence_id != MISSING_U32
            and encoded is not None
            and self._phoneme_sequence(sequence_id) == encoded
        )

    def matching_rows(
        self,
        candidate_rows: Sequence[int] | np.ndarray | None,
        *,
        category: str | None,
        syllables: int | None,
        primary_vowel: str | None,
        rhyme_key: str | None,
        sounds_like_phonemes: Sequence[str] | None,
        meter: str | None,
    ) -> np.ndarray | None:
        has_filter = bool(category and category != "all") or syllables is not None or any(
            (primary_vowel, rhyme_key, sounds_like_phonemes, meter)
        )
        if not has_filter:
            return None if candidate_rows is None else np.asarray(candidate_rows, dtype="int64")
        rows = np.arange(len(self.values), dtype="int64") if candidate_rows is None else np.asarray(candidate_rows, dtype="int64")
        selected = self.values[rows]
        mask = np.ones(len(rows), dtype=bool)
        if category and category != "all":
            mask &= (selected["flags"] & CATEGORY_FLAGS[category]) != 0
        if syllables is not None:
            mask &= selected["syllables"] == syllables
        for requested, vocab, field in (
            (primary_vowel.upper() if primary_vowel else None, self.vowels, "vowel"),
            (rhyme_key, self.rhymes, "rhyme"),
            (meter, self.meters, "meter"),
        ):
            if requested:
                value_id = vocab.find(requested)
                if value_id is None:
                    return np.empty(0, dtype="int64")
                mask &= selected[field] == value_id
        if sounds_like_phonemes:
            accepted = self.sounds_like_ids(sounds_like_phonemes)
            if not len(accepted):
                return np.empty(0, dtype="int64")
            mask &= np.isin(selected["phonemes"], accepted)
        return rows[mask]

    def close(self) -> None:
        self.token_ids = None
        _close_array(self.values)
        _close_array(self.phoneme_offsets)
        _close_array(self.phoneme_length_offsets)
        _close_array(self.phoneme_length_sequence_ids)
        self.vowels.close()
        self.rhymes.close()
        self.meters.close()
        self.token_vocab.close()
        self.phoneme_bytes.close()
        self.phoneme_length_bytes.close()


def validate_compact_artifacts(index_dir: Path, record_count: int) -> None:
    """Validate mapped layouts and all row/reference bounds without JSON scans."""
    index_dir = Path(index_dir)
    missing = [name for name in COMPACT_ARTIFACTS if not (index_dir / name).is_file()]
    if missing:
        raise ValueError(f"schema-v3 compact artifacts are missing: {', '.join(missing)}")
    offsets = np.load(index_dir / METADATA_OFFSETS, mmap_mode="r")
    row_headwords = np.load(index_dir / ROW_HEADWORDS, mmap_mode="r")
    frequencies = np.load(index_dir / HEADWORD_FREQUENCIES, mmap_mode="r")
    facets = structural = None
    try:
        facets = CompactFacets.open(index_dir)
        structural = CompactStructuralIndex.open(index_dir, len(frequencies))
        metadata_size = (index_dir / "metadata.jsonl").stat().st_size
        if offsets.dtype != np.dtype("uint64") or offsets.shape != (record_count + 1,) or int(offsets[0]) != 0 or int(offsets[-1]) != metadata_size or np.any(offsets[1:] < offsets[:-1]):
            raise ValueError("metadata offsets have an invalid layout")
        if row_headwords.dtype != np.dtype("uint32") or row_headwords.shape != (record_count,):
            raise ValueError("row-to-headword mapping has an invalid layout")
        if frequencies.dtype != np.dtype("float64") or frequencies.ndim != 1:
            raise ValueError("headword frequencies have an invalid layout")
        if len(facets.values) != record_count:
            raise ValueError("facet row count does not match metadata")
        if row_headwords.size and int(row_headwords.max()) >= len(frequencies):
            raise ValueError("row-to-headword mapping has an invalid reference")
    finally:
        _close_array(offsets)
        _close_array(row_headwords)
        _close_array(frequencies)
        if facets is not None:
            facets.close()
        if structural is not None:
            structural.close()
