import importlib.resources
import importlib.metadata
import hashlib
import json

from nltk.corpus import wordnet as wn

_WORDNET_POS = {
    "noun": wn.NOUN,
    "verb": wn.VERB,
    "adjective": wn.ADJ,
    "adverb": wn.ADV,
}


def load_emolex() -> dict[str, frozenset[str]]:
    data_path = importlib.resources.files("nrclex.data").joinpath("nrc_en.json")
    raw = json.loads(data_path.read_text(encoding="utf-8"))
    return {word: frozenset(labels) for word, labels in raw.items()}


def emolex_provenance() -> dict:
    data_path = importlib.resources.files("nrclex.data").joinpath("nrc_en.json")
    raw = data_path.read_bytes()
    return {
        "provider": "nrclex",
        "package_version": importlib.metadata.version("nrclex"),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def lookup_emolex(
    word: str,
    emolex: dict[str, frozenset[str]],
    pos: str | None = None,
) -> frozenset[str] | None:
    normalized = word.casefold()
    direct = emolex.get(normalized)
    if direct is not None or " " in normalized or "-" in normalized:
        return direct
    wordnet_pos = _WORDNET_POS.get(pos) if pos is not None else None
    lemma = wn.morphy(normalized, wordnet_pos)
    return emolex.get(lemma) if lemma and lemma != normalized else None
