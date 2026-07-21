import pytest

from revdict.sort import SORT_MODES, apply_sort


def _candidates(*headwords):
    return [{"headword": hw} for hw in headwords]


def _candidate(headword, tags=None, phonetics=None):
    return {"headword": headword, "tags": tags or [], "phonetics": phonetics}


def test_sort_modes_contains_exactly_the_eleven_documented_modes():
    assert SORT_MODES == (
        "relevance",
        "alpha",
        "alpha_desc",
        "shortest",
        "longest",
        "most_common",
        "least_common",
        "most_formal",
        "oldest",
        "most_modern",
        "most_lyrical",
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
    candidates = _candidates("Common", "other")
    literary_frequency = {"common": 5.0, "other": 1.0}

    result = apply_sort(candidates, "most_common", literary_frequency)

    assert [c["headword"] for c in result] == ["Common", "other"]


def test_unknown_sort_mode_raises_value_error():
    with pytest.raises(ValueError, match="nonsense"):
        apply_sort(_candidates("a"), "nonsense", {})


def test_most_formal_ranks_formal_tagged_first():
    candidates = [
        _candidate("khazi", tags=["slang"]),
        _candidate("lavatory", tags=["formal"]),
        _candidate("toilet"),
    ]

    result = apply_sort(candidates, "most_formal", {})

    assert [c["headword"] for c in result] == ["lavatory", "toilet", "khazi"]


@pytest.mark.parametrize("tag", ["slang", "vulgar", "colloquial", "idiomatic", "informal"])
def test_most_formal_treats_every_informal_register_tag_as_informal(tag):
    """"informal" is included deliberately even though category.py's
    idiom_slang CATEGORY grouping excludes it (category.py:9-13) -- a sort
    axis and a category filter are allowed to define "informal"
    differently; this test locks in that most_formal's definition covers
    all 5 tags, not just the 4 category.py happens to use."""
    candidates = [_candidate("plain"), _candidate("marked", tags=[tag])]

    result = apply_sort(candidates, "most_formal", {})

    assert [c["headword"] for c in result] == ["plain", "marked"]


def test_most_formal_treats_archaic_and_dated_tags_as_neutral_not_informal():
    """archaic/dated/obsolete/historical belong to the oldest/most_modern
    axis, not the formal/informal axis -- a purely archaic-tagged sense
    must tie with an untagged sense here, not get demoted like slang."""
    candidates = [
        _candidate("zebra", tags=["archaic"]),
        _candidate("apple"),
    ]

    result = apply_sort(candidates, "most_formal", {})

    assert [c["headword"] for c in result] == ["zebra", "apple"]


def test_most_formal_preserves_relevance_order_within_a_tie():
    candidates = [_candidate("zebra"), _candidate("apple"), _candidate("mango")]

    result = apply_sort(candidates, "most_formal", {})

    assert [c["headword"] for c in result] == ["zebra", "apple", "mango"]


@pytest.mark.parametrize("tag", ["archaic", "dated", "obsolete", "historical"])
def test_oldest_ranks_any_old_register_tag_first(tag):
    candidates = [_candidate("plain"), _candidate("marked", tags=[tag])]

    result = apply_sort(candidates, "oldest", {})

    assert [c["headword"] for c in result] == ["marked", "plain"]


def test_oldest_preserves_relevance_order_among_untagged_candidates():
    candidates = [_candidate("zebra"), _candidate("apple"), _candidate("mango")]

    result = apply_sort(candidates, "oldest", {})

    assert [c["headword"] for c in result] == ["zebra", "apple", "mango"]


def test_most_modern_is_the_exact_reverse_of_oldest():
    candidates = [
        _candidate("plain"),
        _candidate("marked", tags=["archaic"]),
        _candidate("other"),
    ]

    oldest_order = [c["headword"] for c in apply_sort(candidates, "oldest", {})]
    modern_order = [c["headword"] for c in apply_sort(candidates, "most_modern", {})]

    assert oldest_order == ["marked", "plain", "other"]
    assert modern_order == ["plain", "other", "marked"]


def test_most_lyrical_ranks_lower_average_consonant_cluster_length_first():
    """Real measurement (see this plan's Global Constraints): "moon"
    (phonemes M-UW1-N) has consonant clusters [1, 1] -> average 1.0;
    "strengths" (phonemes S-T-R-EH1-NG-K-TH-S) has clusters [3, 4] ->
    average 3.5. moon is the more lyrical (lower-cluster) word and must
    rank first."""
    candidates = [
        _candidate("strengths", phonetics={
            "phonemes": ["S", "T", "R", "EH1", "NG", "K", "TH", "S"],
        }),
        _candidate("moon", phonetics={"phonemes": ["M", "UW1", "N"]}),
    ]

    result = apply_sort(candidates, "most_lyrical", {})

    assert [c["headword"] for c in result] == ["moon", "strengths"]


def test_most_lyrical_treats_missing_phonetics_as_the_least_lyrical():
    """Mirrors most_common/least_common's convention of defaulting a
    missing signal to the worst-case value rather than dropping the
    candidate or giving it an arbitrary rank."""
    candidates = [
        _candidate("strengths", phonetics={
            "phonemes": ["S", "T", "R", "EH1", "NG", "K", "TH", "S"],
        }),
        _candidate("unresolved", phonetics=None),
    ]

    result = apply_sort(candidates, "most_lyrical", {})

    assert [c["headword"] for c in result] == ["strengths", "unresolved"]


def test_most_lyrical_treats_a_word_with_no_consonants_as_maximally_lyrical():
    candidates = [
        _candidate("strengths", phonetics={
            "phonemes": ["S", "T", "R", "EH1", "NG", "K", "TH", "S"],
        }),
        _candidate("aye", phonetics={"phonemes": ["AY1"]}),
    ]

    result = apply_sort(candidates, "most_lyrical", {})

    assert [c["headword"] for c in result] == ["aye", "strengths"]


def test_most_lyrical_preserves_relevance_order_within_a_tie():
    """Real phonemes (confirmed via stressmark): "flow" is F-L-OW1 and
    "glow" is G-L-OW1 -- both have exactly one cluster of length 2 (the
    two consonants before the vowel) and nothing after, so both average
    2.0. A genuine tie, so relevance order (input order) must be
    preserved."""
    candidates = [
        _candidate("flow", phonetics={"phonemes": ["F", "L", "OW1"]}),
        _candidate("glow", phonetics={"phonemes": ["G", "L", "OW1"]}),
    ]

    result = apply_sort(candidates, "most_lyrical", {})

    assert [c["headword"] for c in result] == ["flow", "glow"]
