from revdict.pattern_matcher import compile_pattern


def test_prefix_wildcard_matches_words_starting_with_the_literal_prefix():
    matches = compile_pattern("blue*")

    assert matches("bluebird") is True
    assert matches("blueprint") is True
    assert matches("skyblue") is False


def test_suffix_wildcard_matches_words_ending_with_the_literal_suffix():
    matches = compile_pattern("*bird")

    assert matches("bluebird") is True
    assert matches("mockingbird") is True
    assert matches("birdcage") is False


def test_middle_wildcard_matches_words_containing_the_literal_substring():
    matches = compile_pattern("*y*")

    assert matches("happy") is True
    assert matches("yellow") is True
    assert matches("gold") is False


def test_single_letter_wildcard_matches_exact_length_with_fixed_head_and_tail():
    """'bl????rd' -> start with bl, end with rd, 4 letters between (TODO.md line 8)."""
    matches = compile_pattern("bl????rd")

    assert matches("blizzard") is True  # b-l-i-z-z-a-rd: 8 letters, bl + 4 + rd
    assert matches("blackbird") is False  # wrong length
    assert matches("blrd") is False  # too short


def test_all_question_marks_matches_pure_length():
    """'?????' -> 5-letter words (TODO.md line 10)."""
    matches = compile_pattern("?????")

    assert matches("happy") is True
    assert matches("sad") is False
    assert matches("gloomy") is False


def test_consonant_wildcard_matches_only_consonants_and_y_counts_as_consonant():
    matches = compile_pattern("#at")

    assert matches("bat") is True
    assert matches("yat") is True
    assert matches("eat") is False


def test_vowel_wildcard_matches_only_vowels():
    matches = compile_pattern("c@t")

    assert matches("cat") is True
    assert matches("cot") is True
    assert matches("cyt") is False


def test_wildcard_matching_is_case_insensitive():
    matches = compile_pattern("Blue*")

    assert matches("BLUEBIRD") is True
    assert matches("BlueJay") is True


def test_literal_characters_in_a_clause_are_matched_exactly():
    matches = compile_pattern("cat")

    assert matches("cat") is True
    assert matches("cats") is False
    assert matches("scat") is False
