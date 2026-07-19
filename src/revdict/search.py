import json
import math

import numpy as np

from revdict import dictionary
from revdict.models import stress
from revdict.models.embedder import Embedder
from revdict.models.emotion import EmotionClassifier, tag_emotion
from revdict.models.reranker import Reranker
from revdict.paths import INDEX_DIR

_state: dict = {}


def cosine_top_k(
    query_vec: np.ndarray,
    matrix: np.ndarray,
    k: int,
    matrix_norms: np.ndarray | None = None,
) -> list[tuple[int, float]]:
    """`matrix_norms` should be `_load_state()`'s precomputed
    `embedding_norms` in production -- computing it fresh here is a ~0.9s
    cost over the full ~800K-row embedding matrix (measured), which used to
    be paid on every single query. Recomputed on the fly only when omitted,
    e.g. in tests with small matrices."""
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    if matrix_norms is None:
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


def exclude_headword(
    scored_rows: list[tuple[int, float]], metadata: list[dict], headword: str | None
) -> list[tuple[int, float]]:
    """Drops any row whose headword (case-insensitively) matches `headword`.
    Used to keep the exact-match word from also showing up redundantly in
    the candidate list. A no-op when `headword` is None/falsy."""
    if not headword:
        return scored_rows
    excluded = headword.lower()
    return [
        (index, score)
        for index, score in scored_rows
        if metadata[index]["headword"].lower() != excluded
    ]


def relative_relevance(scores: list[float]) -> list[int]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [50] * len(scores)
    return [round(100 * (score - lo) / (hi - lo)) for score in scores]


def _stable_sigmoid(x: float) -> float:
    """A numerically-stable logistic sigmoid: never calls math.exp on a
    positive argument, so it can't overflow regardless of how extreme the
    input is."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def absolute_relevance(scores: list[float]) -> list[int]:
    """Maps each raw cross-encoder score independently -- NOT relative to the
    rest of the candidate pool -- to a 0-100 display value, so a
    gibberish/low-confidence query reads as genuinely low-confidence even
    when comparing the "best of a bad bunch" within one query's result set.
    (relative_relevance's pure min-max normalization always stretches the
    top candidate to 100% and the bottom to 0%, regardless of whether every
    candidate is actually a garbage match -- this is the real spec gap Fix 5
    closes: querying "asdkjfhqwoeiruty" must not show a 100% top result.)

    Calibrated against real ms-marco-MiniLM-L-6-v2 scores observed live
    against this corpus (see final-review-fixes-report.md for the full
    investigation): excellent gloss matches and common-word matches land
    roughly in +3 to +8.5, while gibberish/non-matches land roughly in -6 to
    -11. The cross-encoder's raw output is itself a relevance logit (trained
    with a sigmoid-based loss), so an un-scaled sigmoid is both the
    theoretically appropriate transform and, empirically, places the real
    observed good/bad boundaries at sensible points without needing any
    extra scale or offset tuning.
    """
    return [round(100 * _stable_sigmoid(score)) for score in scores]


def combine_score(
    raw_score: float, headword: str, literary_frequency: dict[str, float]
) -> float:
    """Adds a real, measured "how common is this word in modern published
    fiction" signal to the raw reranker score -- this is what actually
    separates common, natural-sounding synonyms (e.g. "glad") from obscure
    or dialectal ones (e.g. "wealful", "vogie") when both restate the query
    word in their definition equally, which the reranker score alone can't
    do (verified: overlap count is identical across good and bad candidates
    in the real "happy" investigation, so it was never the discriminator).

    literary_frequency is keyed by lowercased headword and holds a
    zipf-scale score (log10 of matches per billion words in the Google
    Books Ngram "English Fiction" corpus, 2010-2019) -- see
    literary_frequency_source.compute_literary_frequencies. A missing entry
    means one of two different things, handled differently:

    - The headword is hyphenated or multi-word: the Ngram corpus's
      tokenizer doesn't represent these at all (confirmed: even "well-known"
      has zero raw occurrences), so a missing entry here is inconclusive,
      not evidence of rarity. The raw score is left unadjusted.
    - The headword is a single token: a confirmed zero-attestation result
      across ten years of published fiction is a real signal, treated the
      same as an explicit 0.0 frequency.
    """
    freq = literary_frequency.get(headword.lower())
    if freq is None:
        if "-" in headword or " " in headword:
            return raw_score
        freq = 0.0
    return raw_score + freq


def _load_literary_frequency() -> dict[str, float]:
    path = INDEX_DIR / "literary_frequency.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _load_state() -> dict:
    if not _state:
        _state["embeddings"] = np.load(INDEX_DIR / "embeddings.npy")
        _state["embedding_norms"] = np.linalg.norm(_state["embeddings"], axis=1) + 1e-12
        _state["metadata"] = dictionary.load_metadata(INDEX_DIR)
        _state["word_index"] = dictionary.load_word_index(INDEX_DIR)
        _state["embedder"] = Embedder()
        _state["reranker"] = Reranker()
        _state["literary_frequency"] = _load_literary_frequency()
        _state["classifier"] = None
    return _state


def get_classifier(state: dict) -> EmotionClassifier:
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
                "stress": stress.mark(exact_match_raw["headword"], sense["pos"]),
            }
        )
    return {"headword": exact_match_raw["headword"], "senses": tagged_senses}


def search(query: str, top_n: int = 10) -> dict:
    state = _load_state()
    metadata = state["metadata"]

    # The retrieval pool must stay bigger than top_n even after dedup and
    # exact-match exclusion shrink it, so a larger -n still has enough real
    # candidates to draw from instead of silently returning fewer than asked.
    retrieval_pool_size = max(75, top_n * 3)

    query_vec = state["embedder"].encode_query(query)
    retrieved = cosine_top_k(
        query_vec, state["embeddings"], k=retrieval_pool_size, matrix_norms=state["embedding_norms"]
    )
    definitions = [metadata[index]["definition"] for index, _ in retrieved]
    rerank_scores = state["reranker"].score(query, definitions)
    literary_frequency = state["literary_frequency"]
    scored = []
    for i in range(len(retrieved)):
        row_index = retrieved[i][0]
        headword = metadata[row_index]["headword"]
        adjusted = combine_score(rerank_scores[i], headword, literary_frequency)
        scored.append((row_index, adjusted))

    exact_match_raw = dictionary.lookup_exact(query.strip(), state["word_index"], metadata)
    exact_headword = exact_match_raw["headword"] if exact_match_raw is not None else None

    deduped = dedupe_by_headword(scored, metadata)
    deduped = exclude_headword(deduped, metadata, exact_headword)[:top_n]
    # absolute_relevance (not relative_relevance) drives the displayed
    # confidence: it reflects genuine absolute match quality, so a
    # low-confidence/gibberish query reads as visibly low across the board
    # instead of always showing a 0-100 spread regardless of match quality.
    relevances = absolute_relevance([score for _, score in deduped])

    candidates = []
    for (row_index, _), relevance in zip(deduped, relevances):
        record = dict(metadata[row_index])
        if record.get("emolex"):
            record["emolex"] = frozenset(record["emolex"])
        emotion = tag_emotion(record, classifier_factory=lambda: get_classifier(state))
        candidates.append(
            {
                "headword": record["headword"],
                "pos": record["pos"],
                "definition": record["definition"],
                "examples": record["examples"],
                "label": emotion["label"],
                "polarity": emotion["polarity"],
                "relevance": relevance,
                "stress": stress.mark(record["headword"], record["pos"]),
                "synonyms": record.get("synonyms"),
            }
        )

    exact_match = tag_exact_match_senses(
        exact_match_raw, classifier_factory=lambda: get_classifier(state)
    )
    return {"exact_match": exact_match, "candidates": candidates}
