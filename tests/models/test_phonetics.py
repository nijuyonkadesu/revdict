from revdict.models import phonetics


def test_resolve_returns_the_expected_shape_for_cat():
    result = phonetics.resolve("cat", "noun")
    assert result == {
        "syllable_count": 1,
        "primary_vowel": "AE",
        "rhyme_key": "AE T",
        "meter": "/",
        "phonemes": ["K", "AE1", "T"],
    }


def test_resolve_distinguishes_the_record_noun_verb_heteronym_pair():
    noun = phonetics.resolve("record", "noun")
    verb = phonetics.resolve("record", "verb")
    assert noun["meter"] == "/x"
    assert verb["meter"] == "x/"
    assert noun["rhyme_key"] != verb["rhyme_key"]


def test_resolve_matches_the_pinned_meter_examples_from_the_plan():
    assert phonetics.resolve("happy", "adjective")["meter"] == "/x"
    assert phonetics.resolve("elephant", "noun")["meter"] == "/xx"
    assert phonetics.resolve("banana", "noun")["meter"] == "x/x"
    assert phonetics.resolve("photograph", "noun")["meter"] == "/x/"


def test_resolve_returns_none_for_a_multi_word_headword():
    assert phonetics.resolve("kick the bucket", "verb") is None


def test_resolve_returns_none_for_a_hyphenated_headword():
    assert phonetics.resolve("well-known", "adjective") is None


def test_resolve_never_raises_on_a_nonsense_word():
    """resolve() must be safe to call across the whole corpus during a
    reindex -- some malformed/unusual headword must never crash the whole
    build."""
    result = phonetics.resolve("", "noun")
    assert result is None or isinstance(result, dict)


def test_is_available_is_true_when_stressmark_is_importable_and_current():
    assert phonetics.is_available() is True
