# tests/test_search.py
import numpy as np

from revdict.search import cosine_top_k, dedupe_by_headword, relative_relevance


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


def test_relative_relevance_min_max_scales_and_handles_equal_scores():
    assert relative_relevance([0.2, 0.6, 1.0]) == [0, 50, 100]
    assert relative_relevance([0.5, 0.5]) == [50, 50]
    assert relative_relevance([]) == []
