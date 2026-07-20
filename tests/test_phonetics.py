import pytest

from revdict.phonetics import (
    SOUNDS_LIKE_THRESHOLD,
    matches_meter,
    matches_primary_vowel,
    matches_rhyme,
    matches_sounds_like,
    matches_syllable_count,
)

_CAT = {"phonetics": {"syllable_count": 1, "primary_vowel": "AE", "rhyme_key": "AE T", "meter": "/", "phonemes": ["K", "AE1", "T"]}}
_BAT = {"phonetics": {"syllable_count": 1, "primary_vowel": "AE", "rhyme_key": "AE T", "meter": "/", "phonemes": ["B", "AE1", "T"]}}
_DOG = {"phonetics": {"syllable_count": 1, "primary_vowel": "AO", "rhyme_key": "AO G", "meter": "/", "phonemes": ["D", "AO1", "G"]}}
_ELEPHANT = {"phonetics": {"syllable_count": 3, "primary_vowel": "EH", "rhyme_key": "EH L AH F AH N T", "meter": "/xx", "phonemes": ["EH1", "L", "AH0", "F", "AH0", "N", "T"]}}
_NO_PHONETICS = {"phonetics": None}


def test_matches_syllable_count_none_is_a_noop():
    assert matches_syllable_count(_CAT, None) is True
    assert matches_syllable_count(_NO_PHONETICS, None) is True


def test_matches_syllable_count_exact_match_only():
    assert matches_syllable_count(_CAT, 1) is True
    assert matches_syllable_count(_ELEPHANT, 1) is False
    assert matches_syllable_count(_ELEPHANT, 3) is True


def test_matches_syllable_count_false_when_phonetics_is_none():
    assert matches_syllable_count(_NO_PHONETICS, 1) is False


def test_matches_primary_vowel_none_or_empty_is_a_noop():
    assert matches_primary_vowel(_CAT, None) is True
    assert matches_primary_vowel(_CAT, "") is True


def test_matches_primary_vowel_case_insensitive_exact_match():
    assert matches_primary_vowel(_CAT, "AE") is True
    assert matches_primary_vowel(_CAT, "ae") is True
    assert matches_primary_vowel(_DOG, "AE") is False


def test_matches_primary_vowel_false_when_phonetics_is_none():
    assert matches_primary_vowel(_NO_PHONETICS, "AE") is False


def test_matches_rhyme_none_or_empty_is_a_noop():
    assert matches_rhyme(_CAT, None) is True
    assert matches_rhyme(_CAT, "") is True


def test_matches_rhyme_exact_key_match():
    assert matches_rhyme(_CAT, "AE T") is True
    assert matches_rhyme(_BAT, "AE T") is True
    assert matches_rhyme(_DOG, "AE T") is False


def test_matches_meter_none_or_empty_is_a_noop():
    assert matches_meter(_ELEPHANT, None) is True
    assert matches_meter(_ELEPHANT, "") is True


def test_matches_meter_exact_pattern_match():
    assert matches_meter(_ELEPHANT, "/xx") is True
    assert matches_meter(_ELEPHANT, "/x") is False
    assert matches_meter(_CAT, "/") is True


def test_matches_sounds_like_none_or_empty_target_is_a_noop():
    assert matches_sounds_like(_CAT, None) is True
    assert matches_sounds_like(_CAT, []) is True


def test_matches_sounds_like_exact_homophone_matches():
    # "cat" against its own phonemes -- distance 0, must match regardless
    # of threshold value.
    assert matches_sounds_like(_CAT, ["K", "AE1", "T"]) is True


def test_matches_sounds_like_one_phoneme_substitution_matches():
    # cat vs bat: real measured normalized distance 0.33, which is <=
    # SOUNDS_LIKE_THRESHOLD (0.34) -- pinned in the plan's Global
    # Constraints as an intentional match.
    assert matches_sounds_like(_CAT, ["B", "AE1", "T"]) is True


def test_matches_sounds_like_unrelated_word_does_not_match():
    # cat vs elephant: real measured normalized distance 0.86, far above
    # threshold.
    assert matches_sounds_like(_CAT, ["EH1", "L", "AH0", "F", "AH0", "N", "T"]) is False


def test_matches_sounds_like_false_when_phonetics_is_none():
    assert matches_sounds_like(_NO_PHONETICS, ["K", "AE1", "T"]) is False


def test_sounds_like_threshold_is_the_pinned_value():
    assert SOUNDS_LIKE_THRESHOLD == 0.34


def test_matching_predicates_actually_consume_real_resolve_output():
    """Closes the one seam no other test in this plan covers: every other
    test here hand-writes its own "phonetics" dict, and Task 2's own tests
    check resolve()'s output shape in isolation -- nothing asserts the
    dict resolve() PRODUCES is the dict these predicates ACTUALLY READ. If
    the two tasks silently disagreed on a key name, every test in both
    tasks would still pass individually, and the real integrated behavior
    would be "every phonetic filter matches nothing" -- a silent failure
    that would only surface after a real multi-hour reindex. This test
    uses Task 2's real resolve() (skipped if stressmark isn't installed --
    same guard as this plan's other real-stressmark tests), builds a
    metadata list from its real output, and runs the real predicates
    end-to-end."""
    import pytest as _pytest

    from revdict.models import phonetics as phonetics_models

    if not phonetics_models.is_available():
        _pytest.skip("requires stressmark to be installed")

    cat = {"phonetics": phonetics_models.resolve("cat", "noun")}
    dog = {"phonetics": phonetics_models.resolve("dog", "noun")}
    hat = {"phonetics": phonetics_models.resolve("hat", "noun")}

    assert matches_rhyme(cat, hat["phonetics"]["rhyme_key"]) is True
    assert matches_rhyme(dog, hat["phonetics"]["rhyme_key"]) is False
    assert matches_syllable_count(cat, 1) is True
    assert matches_sounds_like(cat, hat["phonetics"]["phonemes"]) is True
    assert matches_sounds_like(cat, dog["phonetics"]["phonemes"]) is False
