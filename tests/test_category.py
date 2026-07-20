import pytest

from revdict.category import CATEGORIES, matches_category


def test_categories_lists_all_seven_values_in_a_stable_order():
    assert CATEGORIES == ("all", "noun", "adjective", "verb", "adverb", "idiom_slang", "old")


def test_matches_category_all_accepts_everything_including_a_bare_record():
    assert matches_category({}, "all") is True
    assert matches_category({"pos": "noun", "tags": ["archaic"]}, "all") is True


def test_matches_category_none_is_treated_the_same_as_all():
    assert matches_category({"pos": "noun"}, None) is True


@pytest.mark.parametrize("pos", ["noun", "adjective", "verb", "adverb"])
def test_matches_category_pos_buckets_require_an_exact_pos_match(pos):
    assert matches_category({"pos": pos, "tags": []}, pos) is True
    other_pos = "noun" if pos != "noun" else "verb"
    assert matches_category({"pos": other_pos, "tags": []}, pos) is False


def test_matches_category_idiom_slang_matches_via_tags():
    assert matches_category({"pos": "noun", "tags": ["slang"]}, "idiom_slang") is True
    assert matches_category({"pos": "noun", "tags": ["colloquial", "rare"]}, "idiom_slang") is True
    assert matches_category({"pos": "noun", "tags": ["formal"]}, "idiom_slang") is False


def test_matches_category_idiom_slang_matches_via_phrase_or_proverb_pos_even_with_no_tags():
    assert matches_category({"pos": "phrase", "tags": []}, "idiom_slang") is True
    assert matches_category({"pos": "proverb", "tags": []}, "idiom_slang") is True


def test_matches_category_idiom_slang_excludes_the_broader_informal_tag():
    """A deliberate scope decision: 'informal' is real and common in the
    raw data, but including it would make Idioms/Slang match far too much
    of the dictionary to be a useful filter."""
    assert matches_category({"pos": "noun", "tags": ["informal"]}, "idiom_slang") is False


def test_matches_category_old_matches_via_register_tags():
    assert matches_category({"pos": "noun", "tags": ["archaic"]}, "old") is True
    assert matches_category({"pos": "noun", "tags": ["dated"]}, "old") is True
    assert matches_category({"pos": "noun", "tags": ["obsolete"]}, "old") is True
    assert matches_category({"pos": "noun", "tags": ["historical"]}, "old") is True
    assert matches_category({"pos": "noun", "tags": ["rare"]}, "old") is False


def test_matches_category_handles_a_record_with_no_tags_key_at_all():
    """Pre-reindex metadata rows (or any record built before this phase)
    have no 'tags' key at all -- must not KeyError, must simply not match
    the tag-based categories."""
    record = {"pos": "noun"}
    assert matches_category(record, "old") is False
    assert matches_category(record, "idiom_slang") is False
    assert matches_category(record, "noun") is True


def test_matches_category_raises_on_an_unknown_category():
    with pytest.raises(ValueError, match="Unknown category"):
        matches_category({"pos": "noun", "tags": []}, "verb_phrase")
