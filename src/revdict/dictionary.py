import json
from pathlib import Path

from revdict.index_bundle import resolve_active_index_dir
from revdict.paths import INDEX_DIR


def load_word_index(index_dir: Path = INDEX_DIR) -> dict[str, list[int]]:
    path = resolve_active_index_dir(Path(index_dir)) / "word_index.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_metadata(index_dir: Path = INDEX_DIR) -> list[dict]:
    path = resolve_active_index_dir(Path(index_dir)) / "metadata.jsonl"
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def lookup_exact(word: str, word_index: dict[str, list[int]], metadata: list[dict]) -> dict | None:
    indices = word_index.get(word.lower())
    if not indices:
        return None
    senses = [
        {
            "pos": metadata[i]["pos"],
            "definition": metadata[i]["definition"],
            "examples": metadata[i]["examples"],
            "source": metadata[i]["source"],
            "sources": metadata[i].get("sources") or [metadata[i]["source"]],
            "sentiwordnet": metadata[i].get("sentiwordnet"),
            "emolex": metadata[i].get("emolex"),
            "synonyms": metadata[i].get("synonyms"),
            "synset": metadata[i].get("synset"),
            "antonyms": metadata[i].get("antonyms") or [],
            "topics": metadata[i].get("topics") or [],
            "wiktionary_sense_ids": metadata[i].get("wiktionary_sense_ids") or [],
            "wikidata_ids": metadata[i].get("wikidata_ids") or [],
        }
        for i in indices
    ]
    return {"headword": word, "senses": senses}
