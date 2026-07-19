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


def test_expand_prefix_parses_the_target_letters_lowercased():
    """'expand:nasa' -> phrases that spell out n.a.s.a. (TODO.md line 15)."""
    parsed = parse_query("expand:NASA")

    assert parsed.mode == "expand"
    assert parsed.expand_target == "nasa"


def test_double_star_wrapped_word_parses_as_phrase_contains():
    """'**winter**' -> phrases that contain the word winter (TODO.md line 14)."""
    parsed = parse_query("**winter**")

    assert parsed.mode == "phrase_contains"
    assert parsed.phrase_word == "winter"


def test_prefix_wildcard_parses_as_a_single_structural_clause():
    """'blue*' -> list words that start with blue (TODO.md line 6)."""
    parsed = parse_query("blue*")

    assert parsed.mode == "structural"
    assert parsed.pattern_clauses == ["blue*"]


def test_suffix_wildcard_parses_as_a_single_structural_clause():
    """'*bird' -> ...that end with bird (TODO.md line 7)."""
    parsed = parse_query("*bird")

    assert parsed.mode == "structural"
    assert parsed.pattern_clauses == ["*bird"]


def test_letter_position_wildcard_parses_as_a_single_structural_clause():
    """'bl????rd' -> start with bl, end with rd, 4 letters between (TODO.md line 8)."""
    parsed = parse_query("bl????rd")

    assert parsed.mode == "structural"
    assert parsed.pattern_clauses == ["bl????rd"]


def test_double_slash_contains_letters_parses_as_a_single_structural_clause():
    """'//fuljyo' -> have the letters "fuljyo" (TODO.md line 9)."""
    parsed = parse_query("//fuljyo")

    assert parsed.mode == "structural"
    assert parsed.pattern_clauses == ["//fuljyo"]


def test_comma_separated_clauses_split_into_multiple_pattern_clauses():
    """'?????,*y*' -> 5 letters AND contains a y (TODO.md line 10)."""
    parsed = parse_query("?????,*y*")

    assert parsed.mode == "structural"
    assert parsed.pattern_clauses == ["?????", "*y*"]


def test_disallow_letters_clause_parses_as_structural():
    parsed = parse_query("-abcd")

    assert parsed.mode == "structural"
    assert parsed.pattern_clauses == ["-abcd"]


def test_restrict_letters_clause_parses_as_structural():
    parsed = parse_query("+abcd")

    assert parsed.mode == "structural"
    assert parsed.pattern_clauses == ["+abcd"]


def test_pattern_colon_meaning_parses_as_combined_mode():
    """'bl*:snow' -> start with bl and have a meaning related to snow (TODO.md line 11)."""
    parsed = parse_query("bl*:snow")

    assert parsed.mode == "combined"
    assert parsed.pattern_clauses == ["bl*"]
    assert parsed.meaning_text == "snow"


def test_comma_clauses_combined_with_meaning():
    parsed = parse_query("?????,*y*:winter sport")

    assert parsed.mode == "combined"
    assert parsed.pattern_clauses == ["?????", "*y*"]
    assert parsed.meaning_text == "winter sport"
