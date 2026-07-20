# tests/data/test_build_index.py
from revdict.data.build_index import (
    build_metadata_record,
    estimate_full_duration,
    group_by_definition,
)


def test_estimate_full_duration_extrapolates_linearly_from_a_sample():
    assert estimate_full_duration(100, 10.0, 1000) == 100.0


def test_estimate_full_duration_handles_an_empty_sample():
    assert estimate_full_duration(0, 0.0, 1000) == 0.0


def test_group_by_definition_groups_identical_texts_and_preserves_first_seen_order():
    records = [{"definition": "a"}, {"definition": "b"}, {"definition": "a"}]

    unique_texts, index_groups = group_by_definition(records)

    assert unique_texts == ["a", "b"]
    assert index_groups == [[0, 2], [1]]


def test_build_metadata_record_includes_synonyms_when_present():
    """Fix 2: `load_wordnet_senses` already computes `synonyms`, but the
    metadata writer previously dropped it on the floor -- this locks in that
    the field actually makes it into the persisted meta dict."""
    record = {
        "headword": "happy",
        "pos": "adjective",
        "definition": "feeling great pleasure",
        "examples": ["a happy child"],
        "source": "wordnet",
        "sentiwordnet": {"pos": 0.8, "neg": 0.0, "obj": 0.2},
        "emolex": frozenset({"joy"}),
        "synonyms": ["glad", "content"],
    }

    meta = build_metadata_record(record)

    assert meta["synonyms"] == ["glad", "content"]
    assert meta["emolex"] == ["joy"]
    assert meta["sentiwordnet"] == {"pos": 0.8, "neg": 0.0, "obj": 0.2}


def test_build_metadata_record_handles_wiktionary_records_lacking_synonyms_field():
    """Wiktionary-sourced records have no `synonyms` key at all (only WordNet
    computes it) -- must not KeyError, and should persist as None."""
    record = {
        "headword": "widget",
        "pos": "noun",
        "definition": "a small device",
        "examples": [],
        "source": "wiktionary",
        "emolex": None,
    }

    meta = build_metadata_record(record)

    assert meta["synonyms"] is None
    assert meta["sentiwordnet"] is None
    assert meta["emolex"] is None


def test_build_metadata_record_includes_tags_when_present():
    record = {
        "headword": "thou",
        "pos": "pronoun",
        "definition": "the second-person singular pronoun",
        "examples": [],
        "source": "wiktionary",
        "tags": ["archaic", "singular"],
    }

    meta = build_metadata_record(record)

    assert meta["tags"] == ["archaic", "singular"]


def test_build_metadata_record_defaults_tags_to_an_empty_list_when_absent():
    """WordNet-sourced records never carry a `tags` key at all (only
    Wiktionary senses compute one) -- must not KeyError, and should persist
    as [] rather than None so downstream category matching never needs a
    None-check."""
    record = {
        "headword": "happy",
        "pos": "adjective",
        "definition": "feeling great pleasure",
        "examples": ["a happy child"],
        "source": "wordnet",
        "emolex": None,
    }

    meta = build_metadata_record(record)

    assert meta["tags"] == []
