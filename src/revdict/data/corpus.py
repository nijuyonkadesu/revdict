import unicodedata
from itertools import chain
from collections.abc import Iterable


def _normalize_headword(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _normalize_definition(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    # Formatting punctuation is not a semantic distinction between two
    # dictionary sources. Turning it into spacing catches trivial variants
    # without applying fuzzy/paraphrase matching that could merge real senses.
    normalized = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in normalized
    )
    return " ".join(normalized.split())


def _identity(record: dict) -> tuple[str, str, str]:
    return (
        _normalize_headword(record["headword"]),
        record["pos"].casefold(),
        _normalize_definition(record["definition"]),
    )


def _unique_values(*collections) -> list:
    values = []
    seen = set()
    for collection in collections:
        for value in collection or []:
            marker = value if isinstance(value, (str, int, float, tuple)) else repr(value)
            if marker in seen:
                continue
            seen.add(marker)
            values.append(value)
    return values


def _sources(record: dict) -> list[str]:
    return [value for value in _unique_values(record.get("sources"), [record.get("source")]) if value]


def _merge_duplicate(primary: dict, additional: dict) -> None:
    primary["sources"] = _unique_values(_sources(primary), _sources(additional))
    for field in (
        "examples",
        "synonyms",
        "antonyms",
        "tags",
        "topics",
        "wiktionary_sense_ids",
        "wikidata_ids",
    ):
        combined = _unique_values(primary.get(field), additional.get(field))
        if combined or field in primary or field in additional:
            primary[field] = combined
    for field in ("sentiwordnet", "synset", "etymology_number"):
        if primary.get(field) is None and additional.get(field) is not None:
            primary[field] = additional[field]


def merge_records(
    wordnet_records: Iterable[dict], wiktionary_records: Iterable[dict]
) -> list[dict]:
    """Merge exact semantic identities while retaining complementary fields.

    POS is part of the identity, every source is deduplicated (including
    duplicates within WordNet), and the first source remains the display
    authority while provenance and useful fields from later sources are kept.
    """
    merged: list[dict] = []
    positions: dict[tuple[str, str, str], int] = {}
    for original in chain(wordnet_records, wiktionary_records):
        record = dict(original)
        record["sources"] = _sources(record)
        key = _identity(record)
        position = positions.get(key)
        if position is None:
            positions[key] = len(merged)
            merged.append(record)
        else:
            _merge_duplicate(merged[position], record)
    return merged
