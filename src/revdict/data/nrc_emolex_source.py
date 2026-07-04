import importlib.resources
import json


def load_emolex() -> dict[str, frozenset[str]]:
    data_path = importlib.resources.files("nrclex.data").joinpath("nrc_en.json")
    raw = json.loads(data_path.read_text(encoding="utf-8"))
    return {word: frozenset(labels) for word, labels in raw.items()}


def lookup_emolex(word: str, emolex: dict[str, frozenset[str]]) -> frozenset[str] | None:
    return emolex.get(word.lower())
