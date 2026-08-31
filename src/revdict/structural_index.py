"""Compact, exact candidate indexes for structural headword queries."""

from __future__ import annotations

import hashlib
import mmap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HEADWORD_FEATURES = "headword_features.npy"
REVERSE_WORD_BYTES = "reverse_word_bytes.bin"
REVERSE_WORD_OFFSETS = "reverse_word_offsets.npy"
REVERSE_WORD_IDS = "reverse_word_ids.npy"
ANAGRAM_HASHES = "anagram_hashes.npy"
ANAGRAM_WORD_IDS = "anagram_word_ids.npy"
PHONEME_LENGTH_BYTES = "phoneme_length_bytes.bin"
PHONEME_LENGTH_OFFSETS = "phoneme_length_offsets.npy"
PHONEME_LENGTH_SEQUENCE_IDS = "phoneme_length_sequence_ids.npy"

POSTING_INDEX_NAMES = ("phrase_token", "acronym")
POSTING_ARTIFACTS = tuple(
    filename
    for name in POSTING_INDEX_NAMES
    for filename in (
        f"{name}_bytes.bin",
        f"{name}_offsets.npy",
        f"{name}_posting_offsets.npy",
        f"{name}_word_ids.npy",
    )
)

STRUCTURAL_ARTIFACTS = (
    HEADWORD_FEATURES,
    REVERSE_WORD_BYTES,
    REVERSE_WORD_OFFSETS,
    REVERSE_WORD_IDS,
    ANAGRAM_HASHES,
    ANAGRAM_WORD_IDS,
    PHONEME_LENGTH_BYTES,
    PHONEME_LENGTH_OFFSETS,
    PHONEME_LENGTH_SEQUENCE_IDS,
    *POSTING_ARTIFACTS,
)

FEATURE_DTYPE = np.dtype(
    [
        ("length", "<u2"),
        ("letter_mask", "<u4"),
        ("has_other", "u1"),
    ]
)

EXPAND_SKIP_WORDS = frozenset({"and", "of", "the", "for", "a", "an", "&"})
MAX_STORED_LENGTH = np.iinfo(np.uint16).max


def ascii_letter_mask(text: str) -> int:
    mask = 0
    for character in text.lower():
        codepoint = ord(character)
        if ord("a") <= codepoint <= ord("z"):
            mask |= 1 << (codepoint - ord("a"))
    return mask


def _has_other_characters(text: str) -> bool:
    return any(not ("a" <= character <= "z") for character in text.lower())


def anagram_hash(text: str) -> int:
    signature = "".join(sorted(text.lower())).encode("utf-8")
    digest = hashlib.blake2b(signature, digest_size=8).digest()
    return int.from_bytes(digest, "little")


def acronym_for_word(word: str) -> str | None:
    tokens = [token for token in word.split() if token.lower() not in EXPAND_SKIP_WORDS]
    if len(tokens) < 2:
        return None
    return "".join(token[0] for token in tokens if token).lower()


def _write_text_table(path: Path, offsets_path: Path, values: Sequence[str]) -> None:
    encoded = [value.encode("utf-8") for value in values]
    offsets = np.empty(len(encoded) + 1, dtype="uint64")
    offsets[0] = 0
    for index, value in enumerate(encoded, start=1):
        offsets[index] = offsets[index - 1] + len(value)
    path.write_bytes(b"".join(encoded))
    np.save(offsets_path, offsets)


def _write_postings(index_dir: Path, name: str, postings: Mapping[str, Sequence[int]]) -> None:
    keys = sorted(postings)
    _write_text_table(
        index_dir / f"{name}_bytes.bin",
        index_dir / f"{name}_offsets.npy",
        keys,
    )
    offsets = np.empty(len(keys) + 1, dtype="uint64")
    offsets[0] = 0
    for index, key in enumerate(keys, start=1):
        offsets[index] = offsets[index - 1] + len(postings[key])
    word_ids = np.fromiter(
        (word_id for key in keys for word_id in postings[key]),
        dtype="uint32",
        count=int(offsets[-1]),
    )
    np.save(index_dir / f"{name}_posting_offsets.npy", offsets)
    np.save(index_dir / f"{name}_word_ids.npy", word_ids)


def _write_phoneme_length_artifacts(index_dir: Path) -> None:
    """Group deduplicated phoneme sequences by their encoded length.

    Normalized Levenshtein cannot accept a candidate whose length difference
    alone exceeds the cutoff.  Keeping each viable length in one contiguous
    region lets a fresh sounds-like query construct only that conservative
    candidate set instead of materializing the entire phoneme vocabulary.
    """
    sequence_offsets = np.load(index_dir / "phoneme_sequence_offsets.npy", mmap_mode="r")
    try:
        lengths = np.diff(sequence_offsets)
        order = np.argsort(lengths, kind="stable")
        max_length = int(lengths.max()) if len(lengths) else 0
        counts = np.bincount(lengths.astype("int64"), minlength=max_length + 1)
        length_offsets = np.empty(max_length + 2, dtype="uint64")
        length_offsets[0] = 0
        np.cumsum(counts, out=length_offsets[1:])
        source = (index_dir / "phoneme_sequence_bytes.bin").read_bytes()
        grouped = b"".join(
            source[int(sequence_offsets[sequence_id]) : int(sequence_offsets[sequence_id + 1])]
            for sequence_id in order
        )
        (index_dir / PHONEME_LENGTH_BYTES).write_bytes(grouped)
        np.save(index_dir / PHONEME_LENGTH_OFFSETS, length_offsets)
        np.save(
            index_dir / PHONEME_LENGTH_SEQUENCE_IDS,
            np.asarray(order, dtype="uint32"),
        )
    finally:
        _close_array(sequence_offsets)


def write_structural_artifacts(index_dir: Path, words: Sequence[str]) -> None:
    """Build headword-side planner artifacts without touching metadata JSON."""
    index_dir = Path(index_dir)
    features = np.zeros(len(words), dtype=FEATURE_DTYPE)
    reverse_items: list[tuple[str, int]] = []
    anagrams: list[tuple[int, int]] = []
    phrase_postings: dict[str, list[int]] = {}
    acronym_postings: dict[str, list[int]] = {}

    for word_id, word in enumerate(words):
        features["length"][word_id] = min(len(word), MAX_STORED_LENGTH)
        features["letter_mask"][word_id] = ascii_letter_mask(word)
        features["has_other"][word_id] = _has_other_characters(word)
        reverse_items.append((word[::-1], word_id))
        anagrams.append((anagram_hash(word), word_id))

        tokens = word.split()
        if len(tokens) >= 2:
            for token in set(tokens):
                phrase_postings.setdefault(token.lower(), []).append(word_id)
        acronym = acronym_for_word(word)
        if acronym:
            acronym_postings.setdefault(acronym, []).append(word_id)

    np.save(index_dir / HEADWORD_FEATURES, features)

    reverse_items.sort()
    _write_text_table(
        index_dir / REVERSE_WORD_BYTES,
        index_dir / REVERSE_WORD_OFFSETS,
        [value for value, _word_id in reverse_items],
    )
    np.save(
        index_dir / REVERSE_WORD_IDS,
        np.asarray([word_id for _value, word_id in reverse_items], dtype="uint32"),
    )

    anagrams.sort()
    np.save(
        index_dir / ANAGRAM_HASHES,
        np.asarray([value for value, _word_id in anagrams], dtype="uint64"),
    )
    np.save(
        index_dir / ANAGRAM_WORD_IDS,
        np.asarray([word_id for _value, word_id in anagrams], dtype="uint32"),
    )
    _write_postings(index_dir, "phrase_token", phrase_postings)
    _write_postings(index_dir, "acronym", acronym_postings)
    _write_phoneme_length_artifacts(index_dir)


def _close_array(values: np.ndarray | None) -> None:
    mapping = getattr(values, "_mmap", None)
    if mapping is not None:
        mapping.close()


class _MappedBytes:
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
        return b"" if self.mapping is None else self.mapping[start:end]

    def close(self) -> None:
        if self.mapping is not None:
            self.mapping.close()
            self.mapping = None
        self.file.close()


class _TextTable(Sequence[str]):
    def __init__(self, bytes_path: Path, offsets_path: Path) -> None:
        self.bytes = _MappedBytes(bytes_path)
        self.offsets = None
        try:
            self.offsets = np.load(offsets_path, mmap_mode="r")
            if (
                self.offsets.dtype != np.dtype("uint64")
                or self.offsets.ndim != 1
                or not len(self.offsets)
                or int(self.offsets[0]) != 0
                or int(self.offsets[-1]) != len(self.bytes)
                or np.any(self.offsets[1:] < self.offsets[:-1])
            ):
                raise ValueError("compact structural text table has an invalid layout")
        except Exception:
            self.close()
            raise

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.bytes.slice(
            int(self.offsets[index]), int(self.offsets[index + 1])
        ).decode("utf-8")

    def lower_bound(self, value: str) -> int:
        low, high = 0, len(self)
        while low < high:
            middle = (low + high) // 2
            if self[middle] < value:
                low = middle + 1
            else:
                high = middle
        return low

    def find(self, value: str) -> int | None:
        index = self.lower_bound(value)
        return index if index < len(self) and self[index] == value else None

    def close(self) -> None:
        _close_array(getattr(self, "offsets", None))
        self.offsets = None
        self.bytes.close()


@dataclass
class _PostingIndex:
    keys: _TextTable
    offsets: np.ndarray
    word_ids: np.ndarray

    @classmethod
    def open(cls, index_dir: Path, name: str) -> "_PostingIndex":
        keys = _TextTable(
            index_dir / f"{name}_bytes.bin",
            index_dir / f"{name}_offsets.npy",
        )
        offsets = word_ids = None
        try:
            offsets = np.load(index_dir / f"{name}_posting_offsets.npy", mmap_mode="r")
            word_ids = np.load(index_dir / f"{name}_word_ids.npy", mmap_mode="r")
            if (
                offsets.dtype != np.dtype("uint64")
                or offsets.shape != (len(keys) + 1,)
                or word_ids.dtype != np.dtype("uint32")
                or word_ids.ndim != 1
                or int(offsets[0]) != 0
                or int(offsets[-1]) != len(word_ids)
                or np.any(offsets[1:] < offsets[:-1])
            ):
                raise ValueError(f"compact {name} postings have an invalid layout")
            return cls(keys, offsets, word_ids)
        except Exception:
            keys.close()
            _close_array(offsets)
            _close_array(word_ids)
            raise

    def lookup(self, key: str) -> np.ndarray:
        key_id = self.keys.find(key)
        if key_id is None:
            return np.empty(0, dtype="uint32")
        start, end = int(self.offsets[key_id]), int(self.offsets[key_id + 1])
        return self.word_ids[start:end]

    def close(self) -> None:
        self.keys.close()
        _close_array(self.offsets)
        _close_array(self.word_ids)


@dataclass
class CompactStructuralIndex:
    features: np.ndarray
    reverse_words: _TextTable
    reverse_word_ids: np.ndarray
    anagram_hashes: np.ndarray
    anagram_word_ids: np.ndarray
    phrase_tokens: _PostingIndex
    acronyms: _PostingIndex

    @classmethod
    def open(cls, index_dir: Path, word_count: int) -> "CompactStructuralIndex":
        index_dir = Path(index_dir)
        opened = []
        try:
            features = np.load(index_dir / HEADWORD_FEATURES, mmap_mode="r")
            reverse_words = _TextTable(
                index_dir / REVERSE_WORD_BYTES,
                index_dir / REVERSE_WORD_OFFSETS,
            )
            reverse_word_ids = np.load(index_dir / REVERSE_WORD_IDS, mmap_mode="r")
            anagram_hashes = np.load(index_dir / ANAGRAM_HASHES, mmap_mode="r")
            anagram_word_ids = np.load(index_dir / ANAGRAM_WORD_IDS, mmap_mode="r")
            opened.extend((features, reverse_word_ids, anagram_hashes, anagram_word_ids))
            phrase_tokens = _PostingIndex.open(index_dir, "phrase_token")
            acronyms = _PostingIndex.open(index_dir, "acronym")
            result = cls(
                features,
                reverse_words,
                reverse_word_ids,
                anagram_hashes,
                anagram_word_ids,
                phrase_tokens,
                acronyms,
            )
            result.validate(word_count)
            return result
        except Exception:
            for values in opened:
                _close_array(values)
            for name in ("reverse_words", "phrase_tokens", "acronyms"):
                value = locals().get(name)
                if value is not None:
                    value.close()
            raise

    def validate(self, word_count: int) -> None:
        if self.features.dtype != FEATURE_DTYPE or self.features.shape != (word_count,):
            raise ValueError("headword feature array has an invalid layout")
        if (
            len(self.reverse_words) != word_count
            or self.reverse_word_ids.dtype != np.dtype("uint32")
            or self.reverse_word_ids.shape != (word_count,)
            or self.anagram_hashes.dtype != np.dtype("uint64")
            or self.anagram_hashes.shape != (word_count,)
            or self.anagram_word_ids.dtype != np.dtype("uint32")
            or self.anagram_word_ids.shape != (word_count,)
            or (word_count and int(self.reverse_word_ids.max()) >= word_count)
            or (word_count and int(self.anagram_word_ids.max()) >= word_count)
            or np.any(self.anagram_hashes[1:] < self.anagram_hashes[:-1])
        ):
            raise ValueError("compact structural index has an invalid layout")
        expected_ids = np.arange(word_count, dtype="uint32")
        if not np.array_equal(np.sort(self.reverse_word_ids), expected_ids):
            raise ValueError("reverse headword index is not a word-ID permutation")
        if not np.array_equal(np.sort(self.anagram_word_ids), expected_ids):
            raise ValueError("anagram index is not a word-ID permutation")
        for postings in (self.phrase_tokens, self.acronyms):
            if len(postings.word_ids) and int(postings.word_ids.max()) >= word_count:
                raise ValueError("compact structural postings contain an invalid word ID")

    def ids_with_length(self, length: int) -> np.ndarray:
        if not 0 <= length <= MAX_STORED_LENGTH:
            return np.empty(0, dtype="uint32")
        return np.flatnonzero(self.features["length"] == length).astype("uint32")

    def ids_with_letters(self, required_mask: int) -> np.ndarray:
        values = self.features["letter_mask"]
        return np.flatnonzero((values & required_mask) == required_mask).astype("uint32")

    def ids_without_letters(self, excluded_mask: int) -> np.ndarray:
        values = self.features["letter_mask"]
        return np.flatnonzero((values & excluded_mask) == 0).astype("uint32")

    def ids_restricted_to_letters(self, allowed_mask: int) -> np.ndarray:
        values = self.features
        forbidden_mask = np.uint32(((1 << 26) - 1) ^ allowed_mask)
        mask = ((values["letter_mask"] & forbidden_mask) == 0) & (values["has_other"] == 0)
        return np.flatnonzero(mask).astype("uint32")

    def suffix_ids(self, suffix: str) -> np.ndarray:
        reversed_prefix = suffix[::-1]
        start = self.reverse_words.lower_bound(reversed_prefix)
        successor = _lexical_successor(reversed_prefix)
        end = len(self.reverse_words) if successor is None else self.reverse_words.lower_bound(successor)
        return np.sort(np.asarray(self.reverse_word_ids[start:end], dtype="uint32"))

    def anagram_ids(self, signature: str) -> np.ndarray:
        value = np.uint64(anagram_hash(signature))
        start = int(np.searchsorted(self.anagram_hashes, value, side="left"))
        end = int(np.searchsorted(self.anagram_hashes, value, side="right"))
        return np.sort(np.asarray(self.anagram_word_ids[start:end], dtype="uint32"))

    def phrase_ids(self, token: str) -> np.ndarray:
        return np.asarray(self.phrase_tokens.lookup(token.lower()), dtype="uint32")

    def acronym_ids(self, acronym: str) -> np.ndarray:
        return np.asarray(self.acronyms.lookup(acronym.lower()), dtype="uint32")

    def close(self) -> None:
        for values in (
            self.features,
            self.reverse_word_ids,
            self.anagram_hashes,
            self.anagram_word_ids,
        ):
            _close_array(values)
        self.reverse_words.close()
        self.phrase_tokens.close()
        self.acronyms.close()


def _lexical_successor(prefix: str) -> str | None:
    for index in range(len(prefix) - 1, -1, -1):
        codepoint = ord(prefix[index])
        if codepoint < 0x10FFFF:
            return prefix[:index] + chr(codepoint + 1)
    return None


def validate_structural_artifacts(index_dir: Path, words: Sequence[str]) -> None:
    """Validate layouts and exact headword equivalence before publication."""
    index = CompactStructuralIndex.open(index_dir, len(words))
    try:
        for word_id, word in enumerate(words):
            feature = index.features[word_id]
            if (
                int(feature["length"]) != min(len(word), MAX_STORED_LENGTH)
                or int(feature["letter_mask"]) != ascii_letter_mask(word)
                or bool(feature["has_other"]) != _has_other_characters(word)
            ):
                raise ValueError(f"headword features disagree at word ID {word_id}")
        previous_reverse = None
        for position in range(len(words)):
            reverse_word_id = int(index.reverse_word_ids[position])
            reverse_word = index.reverse_words[position]
            if reverse_word != words[reverse_word_id][::-1]:
                raise ValueError("reverse headword index disagrees with the lexicon")
            if previous_reverse is not None and reverse_word < previous_reverse:
                raise ValueError("reverse headword index is not sorted")
            previous_reverse = reverse_word
            anagram_word_id = int(index.anagram_word_ids[position])
            if int(index.anagram_hashes[position]) != anagram_hash(words[anagram_word_id]):
                raise ValueError("anagram index disagrees with the lexicon")

        expected_phrase: dict[str, list[int]] = {}
        expected_acronyms: dict[str, list[int]] = {}
        for word_id, word in enumerate(words):
            tokens = word.split()
            if len(tokens) >= 2:
                for token in set(tokens):
                    expected_phrase.setdefault(token.lower(), []).append(word_id)
            acronym = acronym_for_word(word)
            if acronym:
                expected_acronyms.setdefault(acronym, []).append(word_id)
        for postings, expected in (
            (index.phrase_tokens, expected_phrase),
            (index.acronyms, expected_acronyms),
        ):
            if len(postings.keys) != len(expected):
                raise ValueError("structural posting vocabulary has an invalid size")
            for key, expected_ids in expected.items():
                if not np.array_equal(
                    postings.lookup(key), np.asarray(expected_ids, dtype="uint32")
                ):
                    raise ValueError(f"structural postings disagree for {key!r}")
    finally:
        index.close()
