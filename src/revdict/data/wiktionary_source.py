import gzip
import json
import urllib.request
from pathlib import Path
from typing import Iterable, Iterator

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
            }


def parse_filtered_entries(lines: Iterable[str]) -> list[dict]:
    return list(iter_filtered_entries(lines))


def stream_filtered_entries_from_gzip(path: str) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        yield from iter_filtered_entries(f)


def download_raw_wiktextract(dest_path: str) -> None:
    dest = Path(dest_path)
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(RAW_WIKTEXTRACT_URL, tmp)
    tmp.rename(dest)
