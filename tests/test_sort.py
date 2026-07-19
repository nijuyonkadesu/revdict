import pytest

from revdict.sort import SORT_MODES, apply_sort


def _candidates(*headwords):
    return [{"headword": hw} for hw in headwords]


def test_sort_modes_contains_exactly_the_seven_documented_modes():
    assert SORT_MODES == (
        "relevance",
        "alpha",
        "alpha_desc",
        "shortest",
        "longest",
        "most_common",
        "least_common",
    )


def test_none_sort_mode_returns_candidates_in_their_original_order():
    candidates = _candidates("zebra", "apple", "mango")

    assert apply_sort(candidates, None, {}) == candidates


def test_relevance_sort_mode_returns_candidates_in_their_original_order():
    candidates = _candidates("zebra", "apple", "mango")

    assert apply_sort(candidates, "relevance", {}) == candidates


def test_alpha_sorts_case_insensitively_ascending():
    candidates = _candidates("Zebra", "apple", "Mango")

    result = apply_sort(candidates, "alpha", {})

    assert [c["headword"] for c in result] == ["apple", "Mango", "Zebra"]


def test_alpha_desc_sorts_case_insensitively_descending():
    candidates = _candidates("apple", "Zebra", "mango")

    result = apply_sort(candidates, "alpha_desc", {})

    assert [c["headword"] for c in result] == ["Zebra", "mango", "apple"]


def test_shortest_sorts_by_length_ascending_with_alphabetical_tiebreak():
    candidates = _candidates("bb", "aaa", "z", "aa")

    result = apply_sort(candidates, "shortest", {})

    assert [c["headword"] for c in result] == ["z", "aa", "bb", "aaa"]


def test_longest_sorts_by_length_descending_with_alphabetical_tiebreak():
    candidates = _candidates("bb", "aaa", "z", "aa")

    result = apply_sort(candidates, "longest", {})

    assert [c["headword"] for c in result] == ["aaa", "aa", "bb", "z"]


def test_most_common_sorts_by_literary_frequency_descending():
    candidates = _candidates("rare", "common", "medium")
    literary_frequency = {"common": 5.0, "medium": 2.0, "rare": 0.1}

    result = apply_sort(candidates, "most_common", literary_frequency)

    assert [c["headword"] for c in result] == ["common", "medium", "rare"]


def test_least_common_sorts_by_literary_frequency_ascending():
    candidates = _candidates("rare", "common", "medium")
    literary_frequency = {"common": 5.0, "medium": 2.0, "rare": 0.1}

    result = apply_sort(candidates, "least_common", literary_frequency)

    assert [c["headword"] for c in result] == ["rare", "medium", "common"]


def test_most_common_treats_a_missing_frequency_entry_as_zero():
    candidates = _candidates("known", "unknown")
    literary_frequency = {"known": 3.0}

    result = apply_sort(candidates, "most_common", literary_frequency)

    assert [c["headword"] for c in result] == ["known", "unknown"]


def test_frequency_lookup_is_case_insensitive():
    candidates = _candidates("Common")
    literary_frequency = {"common": 5.0}

    result = apply_sort(candidates, "most_common", literary_frequency)

    assert result == candidates


def test_unknown_sort_mode_raises_value_error():
    with pytest.raises(ValueError, match="nonsense"):
        apply_sort(_candidates("a"), "nonsense", {})
