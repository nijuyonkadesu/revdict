# src/revdict/search.py
import numpy as np

from revdict import dictionary
from revdict.models.embedder import Embedder
from revdict.models.emotion import EmotionClassifier, tag_emotion
from revdict.models.reranker import Reranker
from revdict.paths import INDEX_DIR

_state: dict = {}


def cosine_top_k(query_vec: np.ndarray, matrix: np.ndarray, k: int) -> list[tuple[int, float]]:
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    matrix_norms = np.linalg.norm(matrix, axis=1) + 1e-12
    scores = (matrix @ query_norm) / matrix_norms
    k = min(k, len(scores))
    top_indices = np.argpartition(-scores, k - 1)[:k]
    top_indices = top_indices[np.argsort(-scores[top_indices])]
    return [(int(i), float(scores[i])) for i in top_indices]


def dedupe_by_headword(
    scored_rows: list[tuple[int, float]], metadata: list[dict]
) -> list[tuple[int, float]]:
    best: dict[str, tuple[int, float]] = {}
    for index, score in scored_rows:
        key = metadata[index]["headword"].lower()
        if key not in best or score > best[key][1]:
            best[key] = (index, score)
    return sorted(best.values(), key=lambda pair: -pair[1])


def relative_relevance(scores: list[float]) -> list[int]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [50] * len(scores)
    return [round(100 * (score - lo) / (hi - lo)) for score in scores]


def _load_state() -> dict:
    if not _state:
        _state["embeddings"] = np.load(INDEX_DIR / "embeddings.npy")
        _state["metadata"] = dictionary.load_metadata(INDEX_DIR)
        _state["word_index"] = dictionary.load_word_index(INDEX_DIR)
        _state["embedder"] = Embedder()
        _state["reranker"] = Reranker()
        _state["classifier"] = None
    return _state


def _get_classifier(state: dict) -> EmotionClassifier:
    if state["classifier"] is None:
        state["classifier"] = EmotionClassifier()
    return state["classifier"]


def tag_exact_match_senses(exact_match_raw: dict | None, classifier_factory) -> dict | None:
    """Tags each sense of an exact-match lookup (dictionary.lookup_exact's raw
    output) with the same label/polarity shape candidates use, so the
    exact-match headword gets the emotion badge too, not just candidates.

    SentiWordNet is per-synset, so tagging happens per-sense (a word like
    "happy" can have senses with different definitions/emotions).
    """
    if exact_match_raw is None:
        return None

    tagged_senses = []
    for sense in exact_match_raw["senses"]:
        record = {
            "source": sense.get("source"),
            "definition": sense.get("definition"),
            "sentiwordnet": sense.get("sentiwordnet"),
            "emolex": frozenset(sense["emolex"]) if sense.get("emolex") else None,
        }
        emotion = tag_emotion(record, classifier_factory=classifier_factory)
        tagged_senses.append(
            {
                "pos": sense["pos"],
                "definition": sense["definition"],
                "examples": sense["examples"],
                "source": sense["source"],
                "synonyms": sense.get("synonyms"),
                "label": emotion["label"],
                "polarity": emotion["polarity"],
            }
        )
    return {"headword": exact_match_raw["headword"], "senses": tagged_senses}


def search(query: str, top_n: int = 10) -> dict:
    state = _load_state()
    metadata = state["metadata"]

    query_vec = state["embedder"].encode_query(query)
    retrieved = cosine_top_k(query_vec, state["embeddings"], k=75)
    definitions = [metadata[index]["definition"] for index, _ in retrieved]
    rerank_scores = state["reranker"].score(query, definitions)
    scored = [(retrieved[i][0], rerank_scores[i]) for i in range(len(retrieved))]

    deduped = dedupe_by_headword(scored, metadata)[:top_n]
    relevances = relative_relevance([score for _, score in deduped])

    candidates = []
    for (row_index, _), relevance in zip(deduped, relevances):
        record = dict(metadata[row_index])
        if record.get("emolex"):
            record["emolex"] = frozenset(record["emolex"])
        emotion = tag_emotion(record, classifier_factory=lambda: _get_classifier(state))
        candidates.append(
            {
                "headword": record["headword"],
                "pos": record["pos"],
                "definition": record["definition"],
                "examples": record["examples"],
                "label": emotion["label"],
                "polarity": emotion["polarity"],
                "relevance": relevance,
            }
        )

    exact_match_raw = dictionary.lookup_exact(query.strip(), state["word_index"], metadata)
    exact_match = tag_exact_match_senses(
        exact_match_raw, classifier_factory=lambda: _get_classifier(state)
    )
    return {"exact_match": exact_match, "candidates": candidates}
