import math
from contextvars import ContextVar

import numpy as np

from revdict import category as category_module
from revdict import dictionary
from revdict import phonetics
from revdict import query_syntax
from revdict import sort
from revdict import structural_search
from revdict.compact_index import CompactFacets
from revdict.progress import ProgressReporter
from revdict.models import phonetics as phonetics_models
from revdict.models import stress
from revdict.models.embedder import MODEL_NAME as EMBEDDING_MODEL_NAME
from revdict.models.embedder import MODEL_REVISION as EMBEDDING_MODEL_REVISION
from revdict.models.embedder import Embedder
from revdict.models.emotion import EmotionClassifier, format_emotion_label, tag_emotion
from revdict.models.reranker import Reranker
from revdict.index_bundle import (
    SCHEMA_VERSION, load_manifest, read_manifest_unchecked,
    resolve_active_index_dir, validate_loaded_index,
)
from revdict.paths import INDEX_DIR

_state: dict = {}
RERANK_CANDIDATE_POOL = 600
_load_progress: ContextVar[ProgressReporter | None] = ContextVar("revdict_load_progress", default=None)


def _load_detail(message: str) -> None:
    reporter = _load_progress.get()
    if reporter is not None:
        reporter.detail("ready", message)


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


def cosine_top_k_mapped(
    query_vec: np.ndarray,
    unique_matrix: np.ndarray,
    record_embedding_indices: np.ndarray,
    k: int,
    *,
    matrix_norms: np.ndarray | None = None,
    record_indices: list[int] | np.ndarray | None = None,
) -> list[tuple[int, float]]:
    """Rank record rows while computing each unique definition score once."""
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    if matrix_norms is None:
        matrix_norms = np.linalg.norm(unique_matrix, axis=1) + 1e-12
    definition_scores = (unique_matrix @ query_norm) / matrix_norms
    if record_indices is None:
        selected_records = np.arange(len(record_embedding_indices), dtype=np.int64)
    else:
        selected_records = np.asarray(record_indices, dtype=np.int64)
    if selected_records.size == 0:
        return []
    scores = definition_scores[record_embedding_indices[selected_records]]
    k = min(k, len(scores))
    top_local = np.argpartition(-scores, k - 1)[:k]
    top_local = top_local[np.argsort(-scores[top_local])]
    return [
        (int(selected_records[local]), float(scores[local]))
        for local in top_local
    ]


def rerank_retrieved(
    query: str,
    retrieved: list[tuple[int, float]],
    metadata,
    reranker: Reranker,
    record_embedding_indices: np.ndarray | None = None,
) -> list[float]:
    """Score every retrieved record while evaluating each definition once.

    The compact record-to-embedding mapping is also the authoritative mapping
    from sense rows to unique definition text. Multiple records in the
    600-record retrieval pool often share one definition, and a cross-encoder
    receives exactly the same pair for each of them. Score one representative
    and expand its result back to every original record position so the pool,
    frequency adjustment, headword deduplication, and tie behavior all retain
    their existing inputs.
    """
    if not retrieved:
        return []

    if record_embedding_indices is None:
        definitions = [metadata[row]["definition"] for row, _ in retrieved]
        return reranker.score(query, definitions)

    unique_definitions: list[str] = []
    unique_by_embedding: dict[int, int] = {}
    record_to_unique: list[int] = []
    for row, _ in retrieved:
        embedding_id = int(record_embedding_indices[row])
        unique_index = unique_by_embedding.get(embedding_id)
        if unique_index is None:
            unique_index = len(unique_definitions)
            unique_by_embedding[embedding_id] = unique_index
            unique_definitions.append(metadata[row]["definition"])
        record_to_unique.append(unique_index)

    unique_scores = reranker.score(query, unique_definitions)
    if len(unique_scores) != len(unique_definitions):
        raise RuntimeError(
            "Reranker returned a different number of scores than definitions"
        )
    return [unique_scores[index] for index in record_to_unique]


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


def filter_by_category(
    scored_rows: list[tuple[int, float]], metadata: list[dict], category: str | None
) -> list[tuple[int, float]]:
    """Filters BEFORE any top_n truncation happens -- applying this after
    truncation would silently return fewer than top_n results whenever
    non-matching rows occupied slots that got cut, even though more real
    matches existed further down the ranked list."""
    if not category or category == "all":
        return scored_rows
    return [
        (index, score)
        for index, score in scored_rows
        if category_module.matches_category(metadata[index], category)
    ]


def filter_by_phonetics(
    scored_rows: list[tuple[int, float]],
    metadata: list[dict],
    syllables: int | None,
    primary_vowel: str | None,
    rhyme_key: str | None,
    sounds_like_phonemes: list[str] | None,
    meter: str | None,
) -> list[tuple[int, float]]:
    """Same before-top_n-truncation contract as filter_by_category -- see
    that function's docstring. All 5 filters AND together; each is
    individually a no-op when its argument is falsy/None.

    syllables is checked with `is None` rather than folded into the same
    any([...]) truthiness check as the other 4 -- 0 is a real, meaningful
    filter value for syllable count (no real word has 0 syllables, so
    syllables=0 should exclude everything), but Python's any() treats 0 as
    falsy, which would otherwise make this guard silently skip filtering
    whenever syllables was exactly 0. primary_vowel/rhyme_key/meter are
    strings (where "" is never a real value) and sounds_like_phonemes is a
    list (where [] is never a real value), so truthiness is correct for
    those 4.
    """
    if syllables is None and not any([primary_vowel, rhyme_key, sounds_like_phonemes, meter]):
        return scored_rows
    return [
        (index, score)
        for index, score in scored_rows
        if phonetics.matches_syllable_count(metadata[index], syllables)
        and phonetics.matches_primary_vowel(metadata[index], primary_vowel)
        and phonetics.matches_rhyme(metadata[index], rhyme_key)
        and phonetics.matches_sounds_like(metadata[index], sounds_like_phonemes)
        and phonetics.matches_meter(metadata[index], meter)
    ]


def matching_filter_row_indices(
    metadata: list[dict] | CompactFacets,
    candidate_rows: list[int] | None,
    *,
    category: str | None,
    syllables: int | None,
    primary_vowel: str | None,
    rhyme_key: str | None,
    sounds_like_phonemes: list[str] | None,
    meter: str | None,
) -> list[int] | None:
    has_category = bool(category and category != "all")
    has_phonetics = syllables is not None or any(
        [primary_vowel, rhyme_key, sounds_like_phonemes, meter]
    )
    if not has_category and not has_phonetics:
        return candidate_rows
    if isinstance(metadata, CompactFacets):
        matched = metadata.matching_rows(
            candidate_rows,
            category=category,
            syllables=syllables,
            primary_vowel=primary_vowel,
            rhyme_key=rhyme_key,
            sounds_like_phonemes=sounds_like_phonemes,
            meter=meter,
        )
        return None if matched is None else matched.tolist()
    rows = range(len(metadata)) if candidate_rows is None else candidate_rows
    matched = []
    for row in rows:
        record = metadata[row]
        if has_category and not category_module.matches_category(record, category):
            continue
        if has_phonetics and not (
            phonetics.matches_syllable_count(record, syllables)
            and phonetics.matches_primary_vowel(record, primary_vowel)
            and phonetics.matches_rhyme(record, rhyme_key)
            and phonetics.matches_sounds_like(record, sounds_like_phonemes)
            and phonetics.matches_meter(record, meter)
        ):
            continue
        matched.append(row)
    return matched


def resolve_phonetic_target(word: str, flag_name: str) -> dict:
    """Resolves an arbitrary user-typed word (the target of --rhymes-with
    or --sounds-like) into its phonetic data at QUERY time -- this is the
    one place phonetic filtering still depends on stressmark being
    available live, since the target is unprecomputable. Raises
    ValueError (never returns None) so a missing/outdated stressmark, or
    an unresolvable target word, surfaces as a clear error message instead
    of silently behaving like "nothing matches"."""
    if not phonetics_models.is_available():
        raise ValueError(
            f"--{flag_name} requires the stressmark library (>= 0.2.0) to be installed and importable."
        )
    resolved = phonetics_models.resolve(word, "noun")
    if resolved is None:
        raise ValueError(f"Could not resolve a pronunciation for --{flag_name} target {word!r}.")
    return resolved


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
    """Adds a real, measured "how common was this word in 2010s published
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


def _load_literary_frequency(index_dir=None, word_index=None):
    """Compatibility seam backed only by compact frequency artifacts."""
    index_dir = resolve_active_index_dir(INDEX_DIR) if index_dir is None else index_dir
    words = word_index if word_index is not None else dictionary.load_word_index(index_dir)
    return dictionary.CompactFrequency(words)


def _load_state() -> dict:
    if not _state:
        # Resolve the atomic pointer exactly once. A concurrent rebuild may
        # publish a newer bundle, but this process finishes loading one
        # immutable, internally consistent version.
        index_dir = resolve_active_index_dir(INDEX_DIR)
        raw_manifest = read_manifest_unchecked(index_dir)
        manifest = raw_manifest if raw_manifest and raw_manifest.get("schema_version") == SCHEMA_VERSION else None
        if manifest is None:
            version = None if raw_manifest is None else raw_manifest.get("schema_version")
            raise RuntimeError(
                f"Index optimization is required (found schema {version!r}, expected {SCHEMA_VERSION}); "
                "start the daemon to perform the atomic upgrade"
            )
        loading = {}
        try:
            _load_detail("Loading embedding index")
            loading["embeddings"] = np.load(index_dir / "embeddings.npy", mmap_mode="r")
            loading["record_embedding_indices"] = np.load(
                index_dir / "record_embeddings.npy", mmap_mode="r"
            )
            indexed_model = manifest.get("models", {}).get("embedding", {})
            expected_model = {"name": EMBEDDING_MODEL_NAME, "revision": EMBEDDING_MODEL_REVISION}
            if indexed_model != expected_model:
                raise RuntimeError(
                    "Index embedding model does not match this revdict version; "
                    "run `revdict build-index` to rebuild it"
                )
            _load_detail("Calculating embedding norms")
            loading["embedding_norms"] = np.linalg.norm(loading["embeddings"], axis=1) + 1e-12
            _load_detail("Loading dictionary metadata")
            loading["metadata"] = dictionary.load_metadata(index_dir)
            _load_detail("Loading word index")
            loading["word_index"] = dictionary.load_word_index(index_dir)
            if isinstance(loading["word_index"], dictionary.CompactWordIndex):
                loading["facets"] = CompactFacets.open(index_dir)
                loading["headwords"] = list(loading["word_index"])
            validate_loaded_index(
                index_dir,
                loading["embeddings"],
                loading["metadata"],
                loading["word_index"],
                loading["record_embedding_indices"],
            )
            _load_detail("Starting embedding model")
            loading["embedder"] = Embedder()
            _load_detail("Starting reranker")
            loading["reranker"] = Reranker()
            _load_detail("Loading frequency data")
            if isinstance(loading["word_index"], dictionary.CompactWordIndex):
                loading["literary_frequency"] = dictionary.CompactFrequency(
                    loading["word_index"]
                )
            else:
                loading["literary_frequency"] = _load_literary_frequency(index_dir)
            loading["index_dir"] = index_dir
            loading["manifest"] = manifest
            loading["classifier"] = None
        except Exception:
            _close_state_values(loading)
            raise
        _state.update(loading)
    else:
        _load_detail("Using warm search state")
    return _state


def _close_state_values(state: dict) -> None:
    for name in ("facets", "metadata", "word_index"):
        value = state.get(name)
        close = getattr(value, "close", None)
        if close is not None:
            close()
    for name in ("embeddings", "record_embedding_indices"):
        mapping = getattr(state.get(name), "_mmap", None)
        if mapping is not None:
            mapping.close()


def close_state() -> None:
    """Release every mapping and descriptor owned by the warm runtime."""
    _close_state_values(_state)
    _state.clear()


def get_classifier(state: dict) -> EmotionClassifier:
    if state["classifier"] is None:
        state["classifier"] = EmotionClassifier()
    return state["classifier"]


def preload_emotions(records: list[dict], state: dict) -> None:
    if not records:
        return
    classifier = get_classifier(state)
    classify_many = getattr(classifier, "classify_many", None)
    if classify_many is not None:
        classify_many([record["definition"] for record in records])


def build_candidate(record: dict, relevance: int, state: dict) -> dict:
    record = dict(record)
    if record.get("emolex"):
        record["emolex"] = frozenset(record["emolex"])
    emotion = tag_emotion(record, classifier_factory=lambda: get_classifier(state))
    return {
        "headword": record["headword"],
        "pos": record["pos"],
        "definition": record["definition"],
        "examples": record["examples"],
        "label": emotion["label"],
        "emotion_labels": emotion["labels"],
        "emotion_display": format_emotion_label(emotion["label"], emotion["labels"]),
        "polarity": emotion["polarity"],
        "emotion_source": emotion["emotion_source"],
        "category_source": emotion["category_source"],
        "polarity_source": emotion["polarity_source"],
        "emotion_confidence": emotion["confidence"],
        "emotion_status": emotion["status"],
        "emotion_evidence": emotion["evidence"],
        "emolex_labels": emotion["emolex_labels"],
        "relevance": relevance,
        "stress": stress.mark(record["headword"], record["pos"], preserve_color=True),
        "synonyms": record.get("synonyms"),
        "tags": record.get("tags") or [],
        "phonetics": record.get("phonetics"),
    }


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
            "headword": exact_match_raw["headword"],
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
                "emotion_labels": emotion["labels"],
                "emotion_display": format_emotion_label(emotion["label"], emotion["labels"]),
                "polarity": emotion["polarity"],
                "emotion_source": emotion["emotion_source"],
                "category_source": emotion["category_source"],
                "polarity_source": emotion["polarity_source"],
                "emotion_confidence": emotion["confidence"],
                "emotion_status": emotion["status"],
                "emotion_evidence": emotion["evidence"],
                "emolex_labels": emotion["emolex_labels"],
                "stress": stress.mark(exact_match_raw["headword"], sense["pos"], preserve_color=True),
            }
        )
    return {"headword": exact_match_raw["headword"], "senses": tagged_senses}


def search(
    query: str,
    top_n: int = 10,
    sort_mode: str | None = None,
    category: str | None = None,
    syllables: int | None = None,
    primary_vowel: str | None = None,
    rhymes_with: str | None = None,
    sounds_like: str | None = None,
    meter: str | None = None,
    progress: ProgressReporter | None = None,
) -> dict:
    progress = progress or ProgressReporter()
    progress.active("ready", "Preparing search state")
    token = _load_progress.set(progress)
    try:
        state = _load_state()
    finally:
        _load_progress.reset(token)
    progress.completed("ready")

    # Validated eagerly, independent of whether any row survives to reach
    # filter_by_category below -- an unrecognized category must always
    # raise, even when the candidate pool happens to be empty (e.g. the
    # only row is excluded as the exact match), so this can't rely on
    # matches_category being reached by the per-row filter.
    progress.active("validate")
    if category and category not in category_module.CATEGORIES:
        raise ValueError(f"Unknown category: {category!r}")
    progress.completed("validate")

    # Resolved eagerly too, for the same reason as the category guard above
    # -- --rhymes-with/--sounds-like's target word must resolve (or raise a
    # clear error) regardless of parsed.mode or how the candidate pool ends
    # up shaped, not only when a row happens to reach filter_by_phonetics.
    progress.active("phonetics")
    rhyme_key = None
    if rhymes_with:
        rhyme_key = resolve_phonetic_target(rhymes_with, "rhymes-with")["rhyme_key"]

    sounds_like_phonemes = None
    if sounds_like:
        sounds_like_phonemes = resolve_phonetic_target(sounds_like, "sounds-like")["phonemes"]
    progress.completed("phonetics")

    progress.active("parse")
    parsed = query_syntax.parse_query(query)
    progress.completed("parse")
    if parsed.mode in ("structural", "expand", "phrase_contains"):
        result = structural_search.run_structural(
            parsed,
            state,
            top_n,
            category=category,
            syllables=syllables,
            primary_vowel=primary_vowel,
            rhyme_key=rhyme_key,
            sounds_like_phonemes=sounds_like_phonemes,
            meter=meter,
            progress=progress,
        )
        result["candidates"] = sort.apply_sort(
            result["candidates"], sort_mode, state["literary_frequency"]
        )
        progress.active("finalize")
        progress.completed("finalize")
        return result

    progress.active("scope")
    metadata = state["metadata"]
    # The retrieval pool must stay bigger than top_n even after dedup and
    # exact-match exclusion shrink it, so a larger -n still has enough real
    # candidates to draw from instead of silently returning fewer than asked.
    retrieval_pool_size = max(RERANK_CANDIDATE_POOL, top_n * 3)

    restrict_row_indices = None
    suppress_exact_match = False
    if parsed.mode == "combined":
        restrict_row_indices = structural_search.matching_row_indices(
            parsed,
            state["word_index"],
            state.get("headwords"),
        )
        suppress_exact_match = True

    # Category and phonetic constraints narrow the cosine search domain, not
    # merely the first 75 retrieved rows. Otherwise a valid match outside that
    # arbitrary pool can never reach the UI.
    restrict_row_indices = matching_filter_row_indices(
        state.get("facets", metadata),
        restrict_row_indices,
        category=category,
        syllables=syllables,
        primary_vowel=primary_vowel,
        rhyme_key=rhyme_key,
        sounds_like_phonemes=sounds_like_phonemes,
        meter=meter,
    )

    meaning_query = parsed.meaning_text if parsed.meaning_text is not None else query
    progress.completed("scope")

    # Every non-empty semantic scope receives exhaustive cosine scores. When
    # filters leave fewer than the pool floor, all eligible rows are reranked
    # and cosine order remains the stable tie-break for equal reranker scores.
    progress.active("retrieve")
    if restrict_row_indices is not None and not restrict_row_indices:
        retrieved = []
    elif restrict_row_indices is not None:
        query_vec = state["embedder"].encode_query(meaning_query)
        if state.get("record_embedding_indices") is not None:
            retrieved = cosine_top_k_mapped(
                query_vec,
                state["embeddings"],
                state["record_embedding_indices"],
                retrieval_pool_size,
                matrix_norms=state["embedding_norms"],
                record_indices=restrict_row_indices,
            )
        else:
            subset_matrix = state["embeddings"][restrict_row_indices]
            subset_norms = state["embedding_norms"][restrict_row_indices]
            local_top = cosine_top_k(
                query_vec, subset_matrix, k=retrieval_pool_size, matrix_norms=subset_norms
            )
            retrieved = [
                (restrict_row_indices[local_index], score) for local_index, score in local_top
            ]
    else:
        query_vec = state["embedder"].encode_query(meaning_query)
        if state.get("record_embedding_indices") is not None:
            retrieved = cosine_top_k_mapped(
                query_vec,
                state["embeddings"],
                state["record_embedding_indices"],
                retrieval_pool_size,
                matrix_norms=state["embedding_norms"],
            )
        else:
            retrieved = cosine_top_k(
                query_vec,
                state["embeddings"],
                k=retrieval_pool_size,
                matrix_norms=state["embedding_norms"],
            )
    progress.completed("retrieve")

    # A structural filter that matches zero headwords (e.g. an anagram with
    # no real solutions) reaches here with an empty `retrieved`
    # -- skip the reranker call entirely rather than relying on
    # CrossEncoder.predict's undocumented behavior on an empty batch.
    if retrieved:
        rerank_scores = rerank_retrieved(
            meaning_query,
            retrieved,
            metadata,
            state["reranker"],
            state.get("record_embedding_indices"),
        )
    else:
        rerank_scores = []
    literary_frequency = state["literary_frequency"]
    scored = []
    for i in range(len(retrieved)):
        row_index = retrieved[i][0]
        headword = metadata[row_index]["headword"]
        adjusted = combine_score(rerank_scores[i], headword, literary_frequency)
        scored.append((row_index, adjusted))

    progress.active("filter")
    exact_match_raw = None
    if not suppress_exact_match:
        exact_match_raw = dictionary.lookup_exact(meaning_query.strip(), state["word_index"], metadata)
    exact_headword = exact_match_raw["headword"] if exact_match_raw is not None else None

    deduped = dedupe_by_headword(scored, metadata)
    deduped = exclude_headword(deduped, metadata, exact_headword)
    # category/phonetics never filter the exact-match panel above -- they
    # narrow the candidate list only, so a query like "run" --category
    # noun still shows the verb sense of "run" in the exact-match block.
    deduped = filter_by_category(deduped, metadata, category)
    deduped = filter_by_phonetics(
        deduped, metadata, syllables, primary_vowel, rhyme_key, sounds_like_phonemes, meter
    )[:top_n]
    progress.completed("filter")
    # absolute_relevance (not relative_relevance) drives the displayed
    # confidence: it reflects genuine absolute match quality, so a
    # low-confidence/gibberish query reads as visibly low across the board
    # instead of always showing a 0-100 spread regardless of match quality.
    relevances = absolute_relevance([score for _, score in deduped])

    progress.active("enrich")
    emotion_records = [metadata[row_index] for row_index, _ in deduped]
    if exact_match_raw is not None:
        emotion_records.extend(exact_match_raw["senses"])
    preload_emotions(emotion_records, state)
    candidates = [
        build_candidate(metadata[row_index], relevance, state)
        for (row_index, _), relevance in zip(deduped, relevances)
    ]
    candidates = sort.apply_sort(candidates, sort_mode, literary_frequency)

    exact_match = tag_exact_match_senses(
        exact_match_raw, classifier_factory=lambda: get_classifier(state)
    )
    progress.completed("enrich")
    progress.active("finalize")
    result = {"exact_match": exact_match, "candidates": candidates}
    progress.completed("finalize")
    return result
