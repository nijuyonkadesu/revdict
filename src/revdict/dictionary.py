import json
from pathlib import Path

from revdict.paths import INDEX_DIR


def load_word_index(index_dir: Path = INDEX_DIR) -> dict[str, list[int]]:
    path = Path(index_dir) / "word_index.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_metadata(index_dir: Path = INDEX_DIR) -> list[dict]:
    path = Path(index_dir) / "metadata.jsonl"
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
        }
        for i in indices
    ]
    return {"headword": word, "senses": senses}
