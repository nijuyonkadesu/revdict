from revdict.query_syntax import ParsedQuery
from revdict.structural_search import matching_headwords


def test_structural_mode_returns_headwords_matching_the_compiled_clauses():
    parsed = ParsedQuery(mode="structural", pattern_clauses=["blue*"])
    word_index = {"bluebird": [0], "blueprint": [1], "redbird": [2]}

    result = matching_headwords(parsed, word_index)

    assert set(result) == {"bluebird", "blueprint"}


def test_expand_mode_matches_multiword_headwords_by_initials():
    """'expand:nasa' -> phrases that spell out n.a.s.a. (TODO.md line 15).
    Real acronym expansion skips small function words (and/of/the/...)
    rather than taking every token's initial literally -- verified by hand:
    "national aeronautics and space administration" only reduces to n-a-s-a
    once "and" is skipped (naively it's n-a-a-s-a)."""
    parsed = ParsedQuery(mode="expand", expand_target="nasa")
    word_index = {
        "national aeronautics and space administration": [0],
        "national association of state agencies": [1],
        "bluebird": [2],
    }

    result = matching_headwords(parsed, word_index)

    assert set(result) == {
        "national aeronautics and space administration",
        "national association of state agencies",
    }


def test_expand_mode_skips_single_word_headwords():
    parsed = ParsedQuery(mode="expand", expand_target="n")
    word_index = {"nice": [0]}

    assert matching_headwords(parsed, word_index) == []


def test_phrase_contains_mode_matches_whole_word_tokens_only():
    """'**winter**' -> phrases that contain the word winter (TODO.md line 14) --
    must match the whole token 'winter', not any headword whose letters
    happen to contain that substring across a word boundary."""
    parsed = ParsedQuery(mode="phrase_contains", phrase_word="winter")
    word_index = {
        "winter sport": [0],
        "harsh winter": [1],
        "wintertime": [2],  # single word containing the substring -- must NOT match
        "midwinter storm": [3],  # 'midwinter' is one token, not 'winter' -- must NOT match
    }

    result = matching_headwords(parsed, word_index)

    assert set(result) == {"winter sport", "harsh winter"}


from revdict.structural_search import run_structural


def _build_state():
    # emolex carries a specific category ("joy", not just a bare sentiment
    # flag) for both fixture records so tag_emotion's classifier fallback
    # never fires -- see emotion.py's _emolex_has_specific_category. Without
    # this, run_structural's classifier_factory would actually construct a
    # real EmotionClassifier (downloads/loads a transformers pipeline),
    # exactly as test_search.py's own existing fixtures are careful to avoid
    # (see its test_tag_exact_match_senses_tags_each_sense_... first-sense
    # comment and _FakeClassifier usage).
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
        {
            "headword": "blueprint",
            "pos": "noun",
            "definition": "a technical drawing",
            "examples": [],
            "source": "wordnet",
            "sentiwordnet": None,
            "emolex": ["joy"],
            "synonyms": None,
        },
    ]
    word_index = {"bluebird": [0], "blueprint": [1]}
    literary_frequency = {"bluebird": 1.5, "blueprint": 3.2}
    return {
        "metadata": metadata,
        "word_index": word_index,
        "literary_frequency": literary_frequency,
        "classifier": None,
    }


def test_run_structural_returns_no_exact_match():
    """Structural search matches a set of words, not one pinned headword --
    exact_match is always None for these modes, distinguishing them from a
    dictionary lookup."""
    parsed = ParsedQuery(mode="structural", pattern_clauses=["blue*"])
    state = _build_state()

    result = run_structural(parsed, state, top_n=10)

    assert result["exact_match"] is None


def test_run_structural_builds_full_candidate_records_for_every_match():
    parsed = ParsedQuery(mode="structural", pattern_clauses=["blue*"])
    state = _build_state()

    result = run_structural(parsed, state, top_n=10)

    headwords = {candidate["headword"] for candidate in result["candidates"]}
    assert headwords == {"bluebird", "blueprint"}
    for candidate in result["candidates"]:
        assert set(candidate.keys()) == {
            "headword", "pos", "definition", "examples",
            "label", "polarity", "relevance", "stress", "synonyms",
        }


def test_run_structural_ranks_more_frequent_words_first():
    """No embedding-based relevance score exists for structural matches, so
    matches are ranked by literary_frequency (blueprint=3.2 > bluebird=1.5)
    -- reuses the same signal combine_score already applies as a nudge in
    the semantic path, just exposed directly here as the primary order."""
    parsed = ParsedQuery(mode="structural", pattern_clauses=["blue*"])
    state = _build_state()

    result = run_structural(parsed, state, top_n=10)

    assert [c["headword"] for c in result["candidates"]] == ["blueprint", "bluebird"]


def test_run_structural_respects_top_n():
    parsed = ParsedQuery(mode="structural", pattern_clauses=["blue*"])
    state = _build_state()

    result = run_structural(parsed, state, top_n=1)

    assert len(result["candidates"]) == 1
