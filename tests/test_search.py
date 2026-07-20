# tests/test_search.py
import numpy as np
import pytest

from revdict import search as search_mod
from revdict.search import (
    absolute_relevance,
    combine_score,
    cosine_top_k,
    dedupe_by_headword,
    exclude_headword,
    relative_relevance,
    tag_exact_match_senses,
)


def test_cosine_top_k_ranks_the_most_similar_vector_first():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]], dtype="float32")
    query = np.array([1.0, 0.0], dtype="float32")

    results = cosine_top_k(query, matrix, k=2)

    assert results[0][0] == 0
    assert results[1][0] == 2


def test_cosine_top_k_gives_the_same_ranking_with_precomputed_matrix_norms():
    """The production path (search()) passes precomputed matrix_norms to
    avoid re-deriving them from the full ~800K-row embedding matrix on
    every query (measured ~0.9s wasted per call before this was cached at
    daemon startup) -- must produce identical results to the fresh-compute
    fallback used when matrix_norms is omitted, e.g. in the test above."""
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]], dtype="float32")
    query = np.array([1.0, 0.0], dtype="float32")
    precomputed_norms = np.linalg.norm(matrix, axis=1) + 1e-12

    results = cosine_top_k(query, matrix, k=2, matrix_norms=precomputed_norms)

    assert results[0][0] == 0
    assert results[1][0] == 2


def test_dedupe_by_headword_keeps_the_best_scoring_sense_per_word_case_insensitively():
    metadata = [{"headword": "Happy"}, {"headword": "happy"}, {"headword": "joyful"}]
    scored = [(0, 0.5), (1, 0.9), (2, 0.7)]

    result = dedupe_by_headword(scored, metadata)

    assert result == [(1, 0.9), (2, 0.7)]


def test_exclude_headword_drops_matching_entries_case_insensitively():
    """Fix 4: the exact-match headword must not also reappear in the
    candidate list (e.g. querying "happy" pinning "happy" as the exact match
    and then showing it again as one of the related-word candidates)."""
    metadata = [{"headword": "Happy"}, {"headword": "joyful"}, {"headword": "glad"}]
    scored = [(0, 0.9), (1, 0.7), (2, 0.5)]

    result = exclude_headword(scored, metadata, "happy")

    assert result == [(1, 0.7), (2, 0.5)]


def test_exclude_headword_is_a_noop_when_headword_is_none_or_absent():
    metadata = [{"headword": "Happy"}, {"headword": "joyful"}]
    scored = [(0, 0.9), (1, 0.7)]

    assert exclude_headword(scored, metadata, None) == scored
    assert exclude_headword(scored, metadata, "nonexistent") == scored


def test_relative_relevance_min_max_scales_and_handles_equal_scores():
    assert relative_relevance([0.2, 0.6, 1.0]) == [0, 50, 100]
    assert relative_relevance([0.5, 0.5]) == [50, 50]
    assert relative_relevance([]) == []


def test_absolute_relevance_maps_high_confidence_scores_to_high_percentages():
    """Real raw ms-marco-MiniLM-L-6-v2 cross-encoder scores observed against
    the live index for an excellent query match ("feeling of intense
    annoyance" -> harassment/torment, near-perfect gloss matches)."""
    scores = [8.5017, 4.7611, 4.2976, 3.6611, 3.4583]

    result = absolute_relevance(scores)

    assert all(value >= 95 for value in result[:2])
    assert all(value > 50 for value in result)


def test_absolute_relevance_maps_all_low_gibberish_scores_to_all_low_percentages():
    """The case relative_relevance structurally cannot handle: a gibberish
    query's best candidate is still the "best of a bad bunch" and would read
    100% under pure min-max normalization within the returned set. Real raw
    scores observed for the gibberish query "asdkjfhqwoeiruty"."""
    scores = [-6.3086, -10.3728, -10.4662, -10.4662, -10.6007, -10.8346]

    result = absolute_relevance(scores)

    assert all(value <= 5 for value in result)
    # Contrast with relative_relevance, which shows a 0-100 spread regardless
    # of how bad every candidate in the set actually is -- this is exactly
    # the spec gap Fix 5 closes.
    assert relative_relevance(scores)[0] == 100


def test_absolute_relevance_maps_a_neutral_zero_score_to_fifty_percent():
    assert absolute_relevance([0.0]) == [50]


def test_absolute_relevance_handles_empty_list():
    assert absolute_relevance([]) == []


def test_absolute_relevance_does_not_overflow_on_extreme_scores():
    result = absolute_relevance([-1000.0, 1000.0])
    assert result == [0, 100]


class _FakeClassifier:
    def __init__(self, result):
        self._result = result
        self.calls = 0

    def classify(self, text):
        self.calls += 1
        return self._result


def test_tag_exact_match_senses_returns_none_for_no_exact_match():
    assert tag_exact_match_senses(None, classifier_factory=lambda: None) is None


def test_tag_exact_match_senses_tags_each_sense_and_preserves_display_fields():
    """The exact-match headline feature: every sense of the exact match must
    carry its own label/polarity, tagged the same way candidates are, plus
    keep pos/definition/examples/source/synonyms intact for downstream
    rendering (cli/picker)."""
    classifier = _FakeClassifier(("anger", "negative"))
    exact_match_raw = {
        "headword": "happy",
        "senses": [
            {
                "pos": "adjective",
                "definition": "feeling great pleasure",
                "examples": ["a happy child"],
                "source": "wordnet",
                "sentiwordnet": {"pos": 0.8, "neg": 0.0, "obj": 0.2},
                "emolex": ["joy"],
                "synonyms": ["glad", "content"],
            },
            {
                "pos": "adjective",
                "definition": "willing to do something",
                "examples": [],
                "source": "wiktionary",
                "sentiwordnet": None,
                "emolex": None,
                "synonyms": None,
            },
        ],
    }

    result = tag_exact_match_senses(exact_match_raw, classifier_factory=lambda: classifier)

    assert result["headword"] == "happy"
    first, second = result["senses"]

    # First sense: EmoLex already supplies a specific category ("joy"), so
    # the classifier fallback must not fire for it.
    assert first["label"] == "joy"
    assert first["polarity"] == "positive"
    assert first["definition"] == "feeling great pleasure"
    assert first["examples"] == ["a happy child"]
    assert first["source"] == "wordnet"
    assert first["synonyms"] == ["glad", "content"]

    # Second sense has no EmoLex category, so the classifier fallback fires.
    assert second["label"] == "anger"
    assert second["polarity"] == "negative"
    assert second["synonyms"] is None
    assert classifier.calls == 1


def test_search_candidates_and_exact_match_senses_include_a_stress_key(monkeypatch):
    """search() must always include a "stress" key on every candidate and
    every exact-match sense -- None when stressmark isn't installed/fails,
    a string when it succeeds -- so callers never need a .get() with a
    default; the key is always present."""
    import revdict.search as search_mod

    monkeypatch.setattr(search_mod.stress, "mark", lambda word, pos: f"STRESS[{word}/{pos}]")

    exact_match_raw = {
        "headword": "happy",
        "senses": [
            {
                "pos": "adjective",
                "definition": "feeling pleasure",
                "examples": [],
                "source": "wordnet",
                "sentiwordnet": None,
                "emolex": None,
                "synonyms": None,
            }
        ],
    }

    # classifier_factory=None (not a lambda) -- the fixture's record has no
    # emolex/sentiwordnet data, so tag_emotion needs to know no classifier
    # is available at all; passing a callable here (even one returning
    # None) would make tag_emotion call it and then crash calling
    # .classify() on the None it got back.
    tagged = search_mod.tag_exact_match_senses(exact_match_raw, classifier_factory=None)

    assert tagged["senses"][0]["stress"] == "STRESS[happy/adjective]"


def test_combine_score_adds_the_literary_frequency_for_a_single_token_headword():
    result = combine_score(5.424, "glad", {"glad": 5.327811529454499})

    assert result == 5.424 + 5.327811529454499


def test_combine_score_treats_a_missing_single_token_headword_as_zero_frequency():
    # Confirmed zero attestation across ten years of published fiction for a
    # non-hyphenated word is a real signal, not a data gap -- same as an
    # explicit 0.0 entry.
    result = combine_score(6.157, "wealful", {})

    assert result == 6.157


def test_combine_score_leaves_a_missing_hyphenated_headword_unadjusted():
    # The Ngram corpus's tokenizer doesn't represent hyphenated compounds at
    # all (confirmed: even "well-known" has zero raw occurrences), so a
    # missing entry here is inconclusive, not evidence of rarity.
    result = combine_score(7.280, "twinkly-eyed", {})

    assert result == 7.280


def test_combine_score_leaves_a_missing_multi_word_headword_unadjusted():
    result = combine_score(4.0, "lone wolf", {})

    assert result == 4.0


def test_combine_score_is_case_insensitive_against_the_frequency_table():
    result = combine_score(1.0, "Glad", {"glad": 2.0})

    assert result == 3.0


def test_combine_score_real_happy_candidate_set_ranks_glad_first():
    """Regression guard pinning the real investigation finding: raw reranker
    score alone (or any overlap-based discount) can't separate "glad" from
    obscure candidates like "wealful"/"vogie", because they restate "happy"
    in their definition equally often. The real, measured literary-frequency
    signal is what actually discriminates them."""
    literary_frequency = {
        "vogie": 0.1145316536674059,
        "happies": 1.3606607793308423,
        "wealful": 0.0,
        "glad": 5.327811529454499,
        "joyful": 3.9033162819996567,
        "cheerful": 4.473047472843254,
    }
    candidates = [
        ("twinkly-eyed", 7.280),
        ("vogie", 7.003),
        ("happies", 6.637),
        ("good-humored", 6.468),
        ("wealful", 6.157),
        ("glad", 5.424),
    ]

    combined = [
        (combine_score(raw, headword, literary_frequency), headword)
        for headword, raw in candidates
    ]
    combined.sort(reverse=True)
    order = [headword for _, headword in combined]

    assert order[0] == "glad"


def _fake_state():
    # emolex=["joy"] (a specific category, not None) so tag_emotion's
    # classifier fallback never fires and these tests never construct a
    # real EmotionClassifier -- see the identical note in
    # tests/test_structural_search.py's _build_state().
    metadata = [
        {
            "headword": "bluebird",
            "pos": "noun",
            "definition": "an American songbird",
            "examples": [],
            "source": "wordnet",
            "sentiwordnet": None,
            "emolex": ["joy"],
            "synonyms": None,
        },
    ]
    return {
        "metadata": metadata,
        "word_index": {"bluebird": [0]},
        "literary_frequency": {"bluebird": 1.0},
        "classifier": None,
    }


def test_search_dispatches_structural_queries_to_run_structural_and_skips_embedding(monkeypatch):
    """'blue*' must never touch the embedder/reranker at all -- asserting
    _load_state's embedder/reranker slots are never accessed proves the
    dispatch genuinely bypasses the semantic pipeline rather than just
    happening to produce the same answer."""
    state = _fake_state()
    monkeypatch.setattr(search_mod, "_load_state", lambda: state)

    result = search_mod.search("blue*", top_n=10)

    assert result["exact_match"] is None
    assert [c["headword"] for c in result["candidates"]] == ["bluebird"]


def test_search_still_handles_a_plain_meaning_query_via_the_existing_path(monkeypatch):
    """Backward compatibility: a query with no special characters at all
    must still reach the existing embed/rerank/exact-match code path,
    proven here by confirming the embedder is actually invoked."""
    state = _fake_state()
    calls = []

    class FakeEmbedder:
        def encode_query(self, query):
            calls.append(query)
            import numpy as np

            return np.array([1.0], dtype="float32")

    class FakeReranker:
        def score(self, query, definitions):
            return [1.0 for _ in definitions]

    state["embedder"] = FakeEmbedder()
    state["reranker"] = FakeReranker()
    import numpy as np

    state["embeddings"] = np.array([[1.0]], dtype="float32")
    state["embedding_norms"] = np.array([1.0])
    monkeypatch.setattr(search_mod, "_load_state", lambda: state)

    search_mod.search("bluebird", top_n=10)

    assert calls == ["bluebird"]


def test_search_combined_mode_restricts_candidates_to_the_pattern_match(monkeypatch):
    """'blue*:snow' must only ever surface headwords matching 'blue*',
    even though the fake reranker below would happily score every row
    equally -- proving the structural filter actually narrows the pool
    before reranking, not just after."""
    # emolex=["joy"] on both records, not None -- see _fake_state()'s note
    # above on why this is required to avoid constructing a real
    # EmotionClassifier inside this test.
    metadata = [
        {
            "headword": "bluebird", "pos": "noun", "definition": "a songbird",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None,
        },
        {
            "headword": "redbird", "pos": "noun", "definition": "a songbird",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None,
        },
    ]
    state = {
        "metadata": metadata,
        "word_index": {"bluebird": [0], "redbird": [1]},
        "literary_frequency": {},
        "classifier": None,
    }

    import numpy as np

    class FakeEmbedder:
        def encode_query(self, query):
            return np.array([1.0, 0.0], dtype="float32")

    class FakeReranker:
        def score(self, query, definitions):
            return [1.0 for _ in definitions]

    state["embedder"] = FakeEmbedder()
    state["reranker"] = FakeReranker()
    state["embeddings"] = np.array([[1.0, 0.0], [1.0, 0.0]], dtype="float32")
    state["embedding_norms"] = np.array([1.0, 1.0])
    monkeypatch.setattr(search_mod, "_load_state", lambda: state)

    result = search_mod.search("blue*:snow", top_n=10)

    assert [c["headword"] for c in result["candidates"]] == ["bluebird"]
    assert result["exact_match"] is None


def test_search_combined_mode_with_no_structural_matches_returns_no_candidates(monkeypatch):
    """A structural clause that matches nothing (e.g. an anagram with no
    real solutions) must return an empty result, not crash -- this is the
    regression test for the empty-definitions guard around the reranker
    call in search()'s combined-mode branch. Deliberately uses the bare
    _fake_state() fixture with no embedder/reranker/embeddings configured:
    zero structural matches means len(restrict_row_indices) == 0 <=
    retrieval_pool_size, so the code must take the direct small-match path
    and never touch those fields at all -- if it did, this test would fail
    with a KeyError instead of the assertions below, which is exactly the
    proof this guard is load-bearing."""
    state = _fake_state()
    monkeypatch.setattr(search_mod, "_load_state", lambda: state)

    result = search_mod.search("//zzzzqx:snow", top_n=10)

    assert result["candidates"] == []
    assert result["exact_match"] is None


def test_search_combined_mode_large_match_count_uses_restricted_cosine_retrieval_with_correct_index_remap(monkeypatch):
    """When a structural clause matches MORE than retrieval_pool_size rows,
    search() must take the cosine-retrieval elif branch, which subsets
    state["embeddings"]/state["embedding_norms"] by restrict_row_indices
    and then must remap cosine_top_k's LOCAL indices (positions within the
    subset) back to GLOBAL row indices via
    restrict_row_indices[local_index] -- an off-by-one or missing-remap bug
    here would silently return the WRONG headword's data instead of
    crashing, so this test pins one exact headword's definition rather
    than just checking a count.

    Design: 160 metadata rows, alternating "blueword{i}" at EVEN global
    indices (0, 2, 4, ..., 158) and "redword{i}" at ODD global indices (1,
    3, ..., 159) -- all 80 "blueword*" headwords match the "blue*"
    structural pattern (comfortably over top_n=1's retrieval_pool_size of
    75), none of the "redword*" headwords do. Every matched row's
    embedding is [0.1, 0.99] (low cosine similarity to the query vector
    [1.0, 0.0]) EXCEPT "blueword42" (global row 84), which gets
    [1.0, 0.0] (perfect similarity) -- guaranteed rank #1 within the
    matched subset with no ties. If the remap used `local_index` directly
    instead of `restrict_row_indices[local_index]` (an off-by-one/
    no-remap bug), local index 42 would resolve to global row 42, which is
    even and therefore "blueword21" -- a real, distinctly-defined headword,
    not a crash -- caught directly by the definition assertion below.
    """
    import numpy as np

    metadata = []
    word_index = {}
    embeddings = np.zeros((160, 2), dtype="float32")
    for i in range(80):
        blue_row = len(metadata)
        metadata.append(
            {
                "headword": f"blueword{i}",
                "pos": "noun",
                "definition": f"definition of blueword{i}",
                "examples": [],
                "source": "wordnet",
                "sentiwordnet": None,
                "emolex": ["joy"],
                "synonyms": None,
            }
        )
        word_index[f"blueword{i}"] = [blue_row]
        embeddings[blue_row] = [1.0, 0.0] if i == 42 else [0.1, 0.99]

        red_row = len(metadata)
        metadata.append(
            {
                "headword": f"redword{i}",
                "pos": "noun",
                "definition": f"definition of redword{i}",
                "examples": [],
                "source": "wordnet",
                "sentiwordnet": None,
                "emolex": ["joy"],
                "synonyms": None,
            }
        )
        word_index[f"redword{i}"] = [red_row]
        embeddings[red_row] = [0.1, 0.99]

    state = {
        "metadata": metadata,
        "word_index": word_index,
        "literary_frequency": {},
        "classifier": None,
    }

    class FakeEmbedder:
        def encode_query(self, query):
            return np.array([1.0, 0.0], dtype="float32")

    class FakeReranker:
        def score(self, query, definitions):
            return [1.0 for _ in definitions]

    state["embedder"] = FakeEmbedder()
    state["reranker"] = FakeReranker()
    state["embeddings"] = embeddings
    state["embedding_norms"] = np.linalg.norm(embeddings, axis=1)
    monkeypatch.setattr(search_mod, "_load_state", lambda: state)

    result = search_mod.search("blue*:snow", top_n=1)

    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["headword"] == "blueword42"
    assert result["candidates"][0]["definition"] == "definition of blueword42"
    assert result["exact_match"] is None


def test_search_sort_mode_defaults_to_none_and_preserves_relevance_order(monkeypatch):
    """Backward compatibility: calling search() exactly as before (no
    sort_mode argument at all) must produce the same order as today --
    proven here by NOT passing sort_mode and confirming the plain
    meaning-mode candidate order matches what the reranker/combine_score
    pipeline alone would produce (unsorted by anything sort.py adds)."""
    metadata = [
        {
            "headword": "aardvark", "pos": "noun", "definition": "def a",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None,
        },
        {
            "headword": "zebra", "pos": "noun", "definition": "def z",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None,
        },
    ]
    state = {
        "metadata": metadata,
        "word_index": {"aardvark": [0], "zebra": [1]},
        "literary_frequency": {},
        "classifier": None,
    }

    class FakeEmbedder:
        def encode_query(self, query):
            return np.array([1.0, 0.0], dtype="float32")

    class FakeReranker:
        def score(self, query, definitions):
            # "zebra"'s definition scores higher than "aardvark"'s, so the
            # un-sorted relevance order is [zebra, aardvark] -- the REVERSE
            # of alphabetical, so this test can't accidentally pass just
            # because alpha-sort happens to match the default order.
            return [1.0 if "def z" in d else 0.5 for d in definitions]

    state["embedder"] = FakeEmbedder()
    state["reranker"] = FakeReranker()
    state["embeddings"] = np.array([[1.0, 0.0], [1.0, 0.0]], dtype="float32")
    state["embedding_norms"] = np.array([1.0, 1.0])
    monkeypatch.setattr(search_mod, "_load_state", lambda: state)

    result = search_mod.search("some query")

    assert [c["headword"] for c in result["candidates"]] == ["zebra", "aardvark"]


def test_search_alpha_sort_mode_reorders_meaning_mode_candidates(monkeypatch):
    """Same fixture as the default-order test above, but with
    sort_mode="alpha" -- must flip the order to alphabetical, proving
    sort_mode is actually threaded through the meaning-mode return path."""
    metadata = [
        {
            "headword": "aardvark", "pos": "noun", "definition": "def a",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None,
        },
        {
            "headword": "zebra", "pos": "noun", "definition": "def z",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None,
        },
    ]
    state = {
        "metadata": metadata,
        "word_index": {"aardvark": [0], "zebra": [1]},
        "literary_frequency": {},
        "classifier": None,
    }

    class FakeEmbedder:
        def encode_query(self, query):
            return np.array([1.0, 0.0], dtype="float32")

    class FakeReranker:
        def score(self, query, definitions):
            return [1.0 if "def z" in d else 0.5 for d in definitions]

    state["embedder"] = FakeEmbedder()
    state["reranker"] = FakeReranker()
    state["embeddings"] = np.array([[1.0, 0.0], [1.0, 0.0]], dtype="float32")
    state["embedding_norms"] = np.array([1.0, 1.0])
    monkeypatch.setattr(search_mod, "_load_state", lambda: state)

    result = search_mod.search("some query", sort_mode="alpha")

    assert [c["headword"] for c in result["candidates"]] == ["aardvark", "zebra"]


def test_search_sort_mode_applies_to_structural_mode_results_too(monkeypatch):
    """Structural-mode results default to frequency-descending order
    (structural_search._score_and_sort) -- sort_mode="alpha" must override
    that default too, proving the dispatch branch's sort is wired, not
    just the meaning-mode branch's."""
    metadata = [
        {
            "headword": "bluebird", "pos": "noun", "definition": "a songbird",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None,
        },
        {
            "headword": "blueprint", "pos": "noun", "definition": "a drawing",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None,
        },
    ]
    state = {
        "metadata": metadata,
        "word_index": {"bluebird": [0], "blueprint": [1]},
        "literary_frequency": {"bluebird": 1.0, "blueprint": 3.0},
        "classifier": None,
    }
    monkeypatch.setattr(search_mod, "_load_state", lambda: state)

    default_result = search_mod.search("blue*")
    alpha_result = search_mod.search("blue*", sort_mode="alpha")

    # Default: frequency-descending (blueprint=3.0 > bluebird=1.0).
    assert [c["headword"] for c in default_result["candidates"]] == ["blueprint", "bluebird"]
    # sort_mode="alpha" overrides that to alphabetical.
    assert [c["headword"] for c in alpha_result["candidates"]] == ["bluebird", "blueprint"]


def test_search_category_none_returns_candidates_of_every_part_of_speech(monkeypatch):
    """category=None (the default) must not filter anything -- proven with
    a fixture containing multiple different POS values, all of which must
    still appear."""
    metadata = [
        {
            "headword": "bluebird", "pos": "noun", "definition": "a songbird",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
        },
        {
            "headword": "blue", "pos": "adjective", "definition": "the color of the sky",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
        },
    ]
    state = {
        "metadata": metadata,
        "word_index": {"bluebird": [0], "blue": [1]},
        "literary_frequency": {},
        "classifier": None,
    }
    import numpy as np

    class FakeEmbedder:
        def encode_query(self, query):
            return np.array([1.0, 0.0], dtype="float32")

    class FakeReranker:
        def score(self, query, definitions):
            return [1.0 for _ in definitions]

    state["embedder"] = FakeEmbedder()
    state["reranker"] = FakeReranker()
    state["embeddings"] = np.array([[1.0, 0.0], [1.0, 0.0]], dtype="float32")
    state["embedding_norms"] = np.array([1.0, 1.0])
    monkeypatch.setattr(search_mod, "_load_state", lambda: state)

    result = search_mod.search("sky color", top_n=10)

    headwords = {c["headword"] for c in result["candidates"]}
    assert headwords == {"bluebird", "blue"}


def test_search_category_filters_meaning_mode_candidates_to_the_matching_pos(monkeypatch):
    metadata = [
        {
            "headword": "bluebird", "pos": "noun", "definition": "a songbird",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
        },
        {
            "headword": "blue", "pos": "adjective", "definition": "the color of the sky",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
        },
    ]
    state = {
        "metadata": metadata,
        "word_index": {"bluebird": [0], "blue": [1]},
        "literary_frequency": {},
        "classifier": None,
    }
    import numpy as np

    class FakeEmbedder:
        def encode_query(self, query):
            return np.array([1.0, 0.0], dtype="float32")

    class FakeReranker:
        def score(self, query, definitions):
            return [1.0 for _ in definitions]

    state["embedder"] = FakeEmbedder()
    state["reranker"] = FakeReranker()
    state["embeddings"] = np.array([[1.0, 0.0], [1.0, 0.0]], dtype="float32")
    state["embedding_norms"] = np.array([1.0, 1.0])
    monkeypatch.setattr(search_mod, "_load_state", lambda: state)

    result = search_mod.search("sky color", top_n=10, category="noun")

    assert [c["headword"] for c in result["candidates"]] == ["bluebird"]


def test_search_category_filters_before_top_n_truncation_so_real_matches_are_not_dropped(monkeypatch):
    """The category filter must apply to the FULL scored candidate pool
    before slicing to top_n, not after -- otherwise a high-scoring
    non-matching row occupying a top_n slot would silently squeeze out a
    real match ranked just below it. Fixture: the single highest-scoring
    candidate is an adjective (excluded by category='noun'); two lower-
    scoring nouns follow. Asking for top_n=2 nouns must return BOTH of
    them, not just one."""
    metadata = [
        {
            "headword": "bluely", "pos": "adjective", "definition": "a common adjective sense",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
        },
        {
            "headword": "blueness", "pos": "noun", "definition": "a rare noun sense one",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
        },
        {
            "headword": "bluebell", "pos": "noun", "definition": "a rare noun sense two",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
        },
    ]
    state = {
        "metadata": metadata,
        "word_index": {"bluely": [0], "blueness": [1], "bluebell": [2]},
        "literary_frequency": {},
        "classifier": None,
    }
    import numpy as np

    class FakeEmbedder:
        def encode_query(self, query):
            return np.array([1.0, 0.0], dtype="float32")

    class FakeReranker:
        def score(self, query, definitions):
            score_by_definition = {
                "a common adjective sense": 5.0,
                "a rare noun sense one": 3.0,
                "a rare noun sense two": 2.0,
            }
            return [score_by_definition[d] for d in definitions]

    state["embedder"] = FakeEmbedder()
    state["reranker"] = FakeReranker()
    state["embeddings"] = np.array([[1.0, 0.0]] * 3, dtype="float32")
    state["embedding_norms"] = np.array([1.0, 1.0, 1.0])
    monkeypatch.setattr(search_mod, "_load_state", lambda: state)

    result = search_mod.search("blue things", top_n=2, category="noun")

    assert {c["headword"] for c in result["candidates"]} == {"blueness", "bluebell"}


def test_search_category_does_not_filter_the_exact_match_panel(monkeypatch):
    """category narrows the candidate list only -- the exact-match block
    always shows the typed word's own senses regardless of category, since
    the user explicitly typed that exact word."""
    metadata = [
        {
            "headword": "run", "pos": "verb", "definition": "to move fast on foot",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
        },
    ]
    state = {
        "metadata": metadata,
        "word_index": {"run": [0]},
        "literary_frequency": {},
        "classifier": None,
    }
    import numpy as np

    class FakeEmbedder:
        def encode_query(self, query):
            return np.array([1.0], dtype="float32")

    class FakeReranker:
        def score(self, query, definitions):
            return [1.0 for _ in definitions]

    state["embedder"] = FakeEmbedder()
    state["reranker"] = FakeReranker()
    state["embeddings"] = np.array([[1.0]], dtype="float32")
    state["embedding_norms"] = np.array([1.0])
    monkeypatch.setattr(search_mod, "_load_state", lambda: state)

    result = search_mod.search("run", top_n=10, category="noun")

    assert result["exact_match"]["headword"] == "run"


def test_search_category_applies_to_combined_mode_too(monkeypatch):
    """Combined mode ('blue*:snow') restricts the retrieval pool by
    structural pattern first; category must further narrow the SAME final
    candidate list, not be bypassed by the combined-mode code path."""
    metadata = [
        {
            "headword": "bluebird", "pos": "noun", "definition": "a songbird",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
        },
        {
            "headword": "bluely", "pos": "adjective", "definition": "in a blue manner",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
        },
    ]
    state = {
        "metadata": metadata,
        "word_index": {"bluebird": [0], "bluely": [1]},
        "literary_frequency": {},
        "classifier": None,
    }
    import numpy as np

    class FakeEmbedder:
        def encode_query(self, query):
            return np.array([1.0, 0.0], dtype="float32")

    class FakeReranker:
        def score(self, query, definitions):
            return [1.0 for _ in definitions]

    state["embedder"] = FakeEmbedder()
    state["reranker"] = FakeReranker()
    state["embeddings"] = np.array([[1.0, 0.0], [1.0, 0.0]], dtype="float32")
    state["embedding_norms"] = np.array([1.0, 1.0])
    monkeypatch.setattr(search_mod, "_load_state", lambda: state)

    result = search_mod.search("blue*:snow", top_n=10, category="noun")

    assert [c["headword"] for c in result["candidates"]] == ["bluebird"]


def test_search_unknown_category_raises_value_error(monkeypatch):
    metadata = [
        {
            "headword": "bluebird", "pos": "noun", "definition": "a songbird",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
        },
    ]
    state = {
        "metadata": metadata,
        "word_index": {"bluebird": [0]},
        "literary_frequency": {},
        "classifier": None,
    }
    import numpy as np

    class FakeEmbedder:
        def encode_query(self, query):
            return np.array([1.0], dtype="float32")

    class FakeReranker:
        def score(self, query, definitions):
            return [1.0 for _ in definitions]

    state["embedder"] = FakeEmbedder()
    state["reranker"] = FakeReranker()
    state["embeddings"] = np.array([[1.0]], dtype="float32")
    state["embedding_norms"] = np.array([1.0])
    monkeypatch.setattr(search_mod, "_load_state", lambda: state)

    with pytest.raises(ValueError, match="Unknown category"):
        search_mod.search("bluebird", top_n=10, category="verb_phrase")


def test_search_category_filters_structural_mode_candidates_too(monkeypatch):
    metadata = [
        {
            "headword": "bluebird", "pos": "noun", "definition": "an American songbird",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
        },
        {
            "headword": "bluely", "pos": "adverb", "definition": "in a blue manner",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
        },
    ]
    state = {
        "metadata": metadata,
        "word_index": {"bluebird": [0], "bluely": [1]},
        "literary_frequency": {},
        "classifier": None,
    }
    monkeypatch.setattr(search_mod, "_load_state", lambda: state)

    result = search_mod.search("blue*", top_n=10, category="noun")

    assert [c["headword"] for c in result["candidates"]] == ["bluebird"]


def test_search_syllables_filters_structural_mode_candidates_too(monkeypatch):
    metadata = [
        {
            "headword": "bluebird", "pos": "noun", "definition": "an American songbird",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
            "phonetics": {"syllable_count": 2, "primary_vowel": "UW", "rhyme_key": "Y", "meter": "/x", "phonemes": []},
        },
        {
            "headword": "bluely", "pos": "adverb", "definition": "in a blue manner",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
            "phonetics": {"syllable_count": 3, "primary_vowel": "UW", "rhyme_key": "Z", "meter": "/xx", "phonemes": []},
        },
    ]
    state = {
        "metadata": metadata,
        "word_index": {"bluebird": [0], "bluely": [1]},
        "literary_frequency": {},
        "classifier": None,
    }
    monkeypatch.setattr(search_mod, "_load_state", lambda: state)

    result = search_mod.search("blue*", top_n=10, syllables=2)

    assert [c["headword"] for c in result["candidates"]] == ["bluebird"]


def _phonetics_dict(syllable_count, primary_vowel, rhyme_key, meter, phonemes):
    return {
        "syllable_count": syllable_count,
        "primary_vowel": primary_vowel,
        "rhyme_key": rhyme_key,
        "meter": meter,
        "phonemes": phonemes,
    }


def test_search_phonetic_filters_none_is_a_noop(monkeypatch):
    metadata = [
        {
            "headword": "bluebird", "pos": "noun", "definition": "a songbird",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
            "phonetics": _phonetics_dict(2, "UW", "UW B ER D", "/x", ["B", "L", "UW1", "B", "ER0", "D"]),
        },
        {
            "headword": "blue", "pos": "adjective", "definition": "the color of the sky",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
            "phonetics": _phonetics_dict(1, "UW", "UW", "/", ["B", "L", "UW1"]),
        },
    ]
    state = {
        "metadata": metadata,
        "word_index": {"bluebird": [0], "blue": [1]},
        "literary_frequency": {},
        "classifier": None,
    }
    import numpy as np

    class FakeEmbedder:
        def encode_query(self, query):
            return np.array([1.0, 0.0], dtype="float32")

    class FakeReranker:
        def score(self, query, definitions):
            return [1.0 for _ in definitions]

    state["embedder"] = FakeEmbedder()
    state["reranker"] = FakeReranker()
    state["embeddings"] = np.array([[1.0, 0.0], [1.0, 0.0]], dtype="float32")
    state["embedding_norms"] = np.array([1.0, 1.0])
    monkeypatch.setattr(search_mod, "_load_state", lambda: state)

    result = search_mod.search("sky color", top_n=10)

    assert {c["headword"] for c in result["candidates"]} == {"bluebird", "blue"}


def test_search_syllables_filter_restricts_candidates(monkeypatch):
    metadata = [
        {
            "headword": "bluebird", "pos": "noun", "definition": "a songbird",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
            "phonetics": _phonetics_dict(2, "UW", "UW B ER D", "/x", ["B", "L", "UW1", "B", "ER0", "D"]),
        },
        {
            "headword": "blue", "pos": "adjective", "definition": "the color of the sky",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
            "phonetics": _phonetics_dict(1, "UW", "UW", "/", ["B", "L", "UW1"]),
        },
    ]
    state = {
        "metadata": metadata,
        "word_index": {"bluebird": [0], "blue": [1]},
        "literary_frequency": {},
        "classifier": None,
    }
    import numpy as np

    class FakeEmbedder:
        def encode_query(self, query):
            return np.array([1.0, 0.0], dtype="float32")

    class FakeReranker:
        def score(self, query, definitions):
            return [1.0 for _ in definitions]

    state["embedder"] = FakeEmbedder()
    state["reranker"] = FakeReranker()
    state["embeddings"] = np.array([[1.0, 0.0], [1.0, 0.0]], dtype="float32")
    state["embedding_norms"] = np.array([1.0, 1.0])
    monkeypatch.setattr(search_mod, "_load_state", lambda: state)

    result = search_mod.search("sky color", top_n=10, syllables=1)

    assert [c["headword"] for c in result["candidates"]] == ["blue"]


def test_search_phonetic_filters_apply_before_top_n_truncation(monkeypatch):
    """Same ordering guarantee as category: the highest-scoring candidate
    fails the syllables filter, two lower-scoring ones pass -- top_n=2
    must return both, not just one, proving the filter runs before
    truncation, not after."""
    metadata = [
        {
            "headword": "bluely", "pos": "adverb", "definition": "a common sense",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
            "phonetics": _phonetics_dict(2, "UW", "UW L IY", "/x", ["B", "L", "UW1", "L", "IY0"]),
        },
        {
            "headword": "blueness", "pos": "noun", "definition": "a rare noun sense one",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
            "phonetics": _phonetics_dict(1, "UW", "UW N AH S", "/", ["B", "L", "UW1", "N", "AH0", "S"]),
        },
        {
            "headword": "bluebell", "pos": "noun", "definition": "a rare noun sense two",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
            "phonetics": _phonetics_dict(1, "UW", "UW B EH L", "/", ["B", "L", "UW1", "B", "EH1", "L"]),
        },
    ]
    state = {
        "metadata": metadata,
        "word_index": {"bluely": [0], "blueness": [1], "bluebell": [2]},
        "literary_frequency": {},
        "classifier": None,
    }
    import numpy as np

    class FakeEmbedder:
        def encode_query(self, query):
            return np.array([1.0, 0.0], dtype="float32")

    class FakeReranker:
        def score(self, query, definitions):
            score_by_definition = {"a common sense": 5.0, "a rare noun sense one": 3.0, "a rare noun sense two": 2.0}
            return [score_by_definition[d] for d in definitions]

    state["embedder"] = FakeEmbedder()
    state["reranker"] = FakeReranker()
    state["embeddings"] = np.array([[1.0, 0.0]] * 3, dtype="float32")
    state["embedding_norms"] = np.array([1.0, 1.0, 1.0])
    monkeypatch.setattr(search_mod, "_load_state", lambda: state)

    result = search_mod.search("blue things", top_n=2, syllables=1)

    assert {c["headword"] for c in result["candidates"]} == {"blueness", "bluebell"}


def test_search_meter_filter_restricts_candidates(monkeypatch):
    metadata = [
        {
            "headword": "happy", "pos": "adjective", "definition": "feeling joy",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
            "phonetics": _phonetics_dict(2, "AE", "AE P IY", "/x", ["HH", "AE1", "P", "IY0"]),
        },
        {
            "headword": "glad", "pos": "adjective", "definition": "feeling joy too",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
            "phonetics": _phonetics_dict(1, "AE", "AE D", "/", ["G", "L", "AE1", "D"]),
        },
    ]
    state = {
        "metadata": metadata,
        "word_index": {"happy": [0], "glad": [1]},
        "literary_frequency": {},
        "classifier": None,
    }
    import numpy as np

    class FakeEmbedder:
        def encode_query(self, query):
            return np.array([1.0, 0.0], dtype="float32")

    class FakeReranker:
        def score(self, query, definitions):
            return [1.0 for _ in definitions]

    state["embedder"] = FakeEmbedder()
    state["reranker"] = FakeReranker()
    state["embeddings"] = np.array([[1.0, 0.0], [1.0, 0.0]], dtype="float32")
    state["embedding_norms"] = np.array([1.0, 1.0])
    monkeypatch.setattr(search_mod, "_load_state", lambda: state)

    result = search_mod.search("feeling joy", top_n=10, meter="/x")

    assert [c["headword"] for c in result["candidates"]] == ["happy"]


def test_search_rhymes_with_resolves_the_target_and_filters(monkeypatch):
    from revdict.models import phonetics as phonetics_models

    if not phonetics_models.is_available():
        pytest.skip("requires stressmark to be installed")

    metadata = [
        {
            "headword": "cat", "pos": "noun", "definition": "a small carnivore",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
            "phonetics": _phonetics_dict(1, "AE", "AE T", "/", ["K", "AE1", "T"]),
        },
        {
            "headword": "dog", "pos": "noun", "definition": "a small carnivore too",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
            "phonetics": _phonetics_dict(1, "AO", "AO G", "/", ["D", "AO1", "G"]),
        },
    ]
    state = {
        "metadata": metadata,
        "word_index": {"cat": [0], "dog": [1]},
        "literary_frequency": {},
        "classifier": None,
    }
    import numpy as np

    class FakeEmbedder:
        def encode_query(self, query):
            return np.array([1.0, 0.0], dtype="float32")

    class FakeReranker:
        def score(self, query, definitions):
            return [1.0 for _ in definitions]

    state["embedder"] = FakeEmbedder()
    state["reranker"] = FakeReranker()
    state["embeddings"] = np.array([[1.0, 0.0], [1.0, 0.0]], dtype="float32")
    state["embedding_norms"] = np.array([1.0, 1.0])
    monkeypatch.setattr(search_mod, "_load_state", lambda: state)

    result = search_mod.search("small carnivore", top_n=10, rhymes_with="hat")

    assert [c["headword"] for c in result["candidates"]] == ["cat"]


def test_search_rhymes_with_raises_when_stressmark_is_unavailable(monkeypatch):
    state = {
        "metadata": [],
        "word_index": {},
        "literary_frequency": {},
        "classifier": None,
    }
    monkeypatch.setattr(search_mod, "_load_state", lambda: state)
    monkeypatch.setattr(search_mod.phonetics_models, "is_available", lambda: False)

    with pytest.raises(ValueError, match="stressmark"):
        search_mod.search("anything", top_n=10, rhymes_with="hat")


def test_search_category_and_phonetics_filters_combine(monkeypatch):
    """Filters from different phases must AND together, not override each
    other -- category (Phase 3) and syllables (Phase 4) both apply."""
    metadata = [
        {
            "headword": "bluebird", "pos": "noun", "definition": "a songbird",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
            "phonetics": _phonetics_dict(2, "UW", "UW B ER D", "/x", ["B", "L", "UW1", "B", "ER0", "D"]),
        },
        {
            "headword": "blue", "pos": "adjective", "definition": "the color of the sky",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
            "phonetics": _phonetics_dict(1, "UW", "UW", "/", ["B", "L", "UW1"]),
        },
    ]
    state = {
        "metadata": metadata,
        "word_index": {"bluebird": [0], "blue": [1]},
        "literary_frequency": {},
        "classifier": None,
    }
    import numpy as np

    class FakeEmbedder:
        def encode_query(self, query):
            return np.array([1.0, 0.0], dtype="float32")

    class FakeReranker:
        def score(self, query, definitions):
            return [1.0 for _ in definitions]

    state["embedder"] = FakeEmbedder()
    state["reranker"] = FakeReranker()
    state["embeddings"] = np.array([[1.0, 0.0], [1.0, 0.0]], dtype="float32")
    state["embedding_norms"] = np.array([1.0, 1.0])
    monkeypatch.setattr(search_mod, "_load_state", lambda: state)

    result = search_mod.search("sky color", top_n=10, category="noun", syllables=2)

    assert [c["headword"] for c in result["candidates"]] == ["bluebird"]
