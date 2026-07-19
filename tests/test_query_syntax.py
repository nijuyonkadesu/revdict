from revdict.query_syntax import ParsedQuery, parse_query


def test_parsed_query_defaults_have_empty_pattern_clauses_and_none_fields():
    parsed = ParsedQuery(mode="meaning")

    assert parsed.pattern_clauses == []
    assert parsed.meaning_text is None
    assert parsed.expand_target is None
    assert parsed.phrase_word is None


def test_plain_word_with_no_special_characters_parses_as_meaning_mode():
    """Backward compatibility: revdict's existing default behavior for a
    plain query like "bluebird" or a full descriptive phrase must be
    completely unaffected by the new query DSL."""
    parsed = parse_query("bluebird")

    assert parsed.mode == "meaning"
    assert parsed.meaning_text == "bluebird"


def test_plain_meaning_query_is_stripped_of_surrounding_whitespace():
    parsed = parse_query("  a feeling of intense joy  ")

    assert parsed.mode == "meaning"
    assert parsed.meaning_text == "a feeling of intense joy"


def test_colon_prefix_with_empty_pattern_part_is_meaning_mode():
    """':snow' -> list words related to snow (TODO.md line 12) -- same
    result as typing 'snow' directly; the colon here is just the
    degenerate case of the pattern:meaning separator with an empty
    pattern part."""
    parsed = parse_query(":snow")

    assert parsed.mode == "meaning"
    assert parsed.meaning_text == "snow"


def test_colon_prefix_meaning_text_can_contain_spaces():
    """':winter sport' -> related to the concept winter sport (TODO.md line 13)."""
    parsed = parse_query(":winter sport")

    assert parsed.mode == "meaning"
    assert parsed.meaning_text == "winter sport"
