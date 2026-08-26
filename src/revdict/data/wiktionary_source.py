import gzip
import json
from pathlib import Path
from typing import Iterable, Iterator

from revdict.data.download import download_cached_file, validate_gzip_header

RAW_WIKTEXTRACT_URL = "https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz"

_POS_NORMALIZATION = {"adj": "adjective", "adv": "adverb"}


def _normalize_pos(pos: str) -> str:
    return _POS_NORMALIZATION.get(pos, pos)


def _combine_glosses(glosses: list[str]) -> str:
    """Wiktionary's `glosses` field is often a hierarchy, not a flat list --
    e.g. ["Unconstrained.", "Not imprisoned or enslaved."] (broad category,
    then the actual specific meaning). Taking only glosses[0] grabs the
    vague category and throws away the specific part -- and since many
    different senses of a word share the same broad first-level gloss (all
    8 senses of "free" start with "Unconstrained."), doing so also made
    corpus.py's definition-based dedup collapse genuinely different senses
    into one. Joining the full hierarchy keeps each sense's text distinct
    and preserves the specific meaning."""
    if len(glosses) == 1:
        return glosses[0]
    return "; ".join(gloss.rstrip(".") for gloss in glosses) + "."


def _relation_words(items: list[dict] | None) -> list[str]:
    words = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        word = item.get("word")
        if isinstance(word, str) and word.strip():
            words.append(word.strip())
    return list(dict.fromkeys(words))


def _string_values(items: list | None) -> list[str]:
    return list(
        dict.fromkeys(
            item.strip()
            for item in items or []
            if isinstance(item, str) and item.strip()
        )
    )


def iter_filtered_entries(lines: Iterable[str]) -> Iterator[dict]:
    for line in lines:
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        if entry.get("lang_code") != "en":
            continue
        word = entry.get("word")
        pos = entry.get("pos")
        if (
            not isinstance(word, str)
            or not word.strip()
            or not isinstance(pos, str)
            or not pos.strip()
        ):
            continue
        word = word.strip()
        pos = pos.strip()
        for sense in entry.get("senses", []):
            if not isinstance(sense, dict):
                continue
            tags = sense.get("tags") or []
            if "form-of" in tags or "form_of" in sense or "alt-of" in tags:
                continue
            glosses = _string_values(sense.get("glosses"))
            if not glosses:
                continue
            examples = [
                example["text"].strip()
                for example in sense.get("examples", [])
                if isinstance(example, dict)
                and isinstance(example.get("text"), str)
                and example["text"].strip()
            ]
            yield {
                "headword": word,
                "pos": _normalize_pos(pos),
                "definition": _combine_glosses(glosses),
                "examples": examples,
                "source": "wiktionary",
                "tags": _string_values(tags),
                "synonyms": _relation_words(sense.get("synonyms")),
                "antonyms": _relation_words(sense.get("antonyms")),
                "topics": _string_values(sense.get("topics")),
                "wiktionary_sense_ids": _string_values(sense.get("senseid")),
                "wikidata_ids": _string_values(sense.get("wikidata")),
                "etymology_number": entry.get("etymology_number"),
            }


def parse_filtered_entries(lines: Iterable[str]) -> list[dict]:
    return list(iter_filtered_entries(lines))


def stream_filtered_entries_from_gzip(path: str) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        yield from iter_filtered_entries(f)


def _validate_wiktextract_dump(path: Path) -> None:
    validate_gzip_header(path)
    parsed = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid Wiktextract JSON on line {line_number}: {error}") from error
            if not isinstance(entry, dict):
                raise ValueError(f"Invalid Wiktextract entry on line {line_number}")
            parsed += 1
    if parsed == 0:
        raise ValueError(f"Wiktextract dump contains no entries: {path}")


def download_raw_wiktextract(dest_path: str, refresh: bool = False) -> dict:
    return download_cached_file(
        RAW_WIKTEXTRACT_URL,
        Path(dest_path),
        validator=_validate_wiktextract_dump,
        refresh=refresh,
        validation_id="wiktextract-jsonl-v1",
    )
