# tests/test_search.py
import numpy as np

from revdict.search import (
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
