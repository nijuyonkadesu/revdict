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
    return list(dict.fromkeys(item["word"] for item in items or [] if item.get("word")))


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
        if not word or not pos:
            continue
        for sense in entry.get("senses", []):
            tags = sense.get("tags") or []
            if "form-of" in tags or "form_of" in sense or "alt-of" in tags:
                continue
            glosses = sense.get("glosses") or []
            if not glosses:
                continue
            examples = [
                example.get("text", "")
                for example in sense.get("examples", [])
                if example.get("text")
            ]
            yield {
                "headword": word,
                "pos": _normalize_pos(pos),
                "definition": _combine_glosses(glosses),
                "examples": examples,
                "source": "wiktionary",
                "tags": tags,
                "synonyms": _relation_words(sense.get("synonyms")),
                "antonyms": _relation_words(sense.get("antonyms")),
                "topics": list(dict.fromkeys(sense.get("topics") or [])),
                "wiktionary_sense_ids": list(dict.fromkeys(sense.get("senseid") or [])),
                "wikidata_ids": list(dict.fromkeys(sense.get("wikidata") or [])),
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
