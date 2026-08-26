# tests/data/test_build_index.py
from revdict.data.build_index import (
    build_metadata_record,
    build_statistics,
    definitions_and_mapping,
    estimate_full_duration,
    group_by_definition,
    validate_build_arrays,
    validate_records,
)
import numpy as np
import pytest


def test_estimate_full_duration_extrapolates_linearly_from_a_sample():
    assert estimate_full_duration(100, 10.0, 1000) == 100.0


def test_estimate_full_duration_handles_an_empty_sample():
    assert estimate_full_duration(0, 0.0, 1000) == 0.0


def test_declined_build_aborts_before_loading_or_downloading(monkeypatch, tmp_path, capsys):
    import revdict.data.build_index as build_index_module

    touched = []
    monkeypatch.setattr(build_index_module, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr("builtins.input", lambda _prompt: "no")
    monkeypatch.setattr(
        build_index_module,
        "load_wordnet_senses",
        lambda: touched.append("wordnet"),
    )

    build_index_module.build(skip_confirm=False)

    assert touched == []
    assert "Aborted" in capsys.readouterr().out


def test_group_by_definition_groups_identical_texts_and_preserves_first_seen_order():
    records = [{"definition": "a"}, {"definition": "b"}, {"definition": "a"}]

    unique_texts, index_groups = group_by_definition(records)

    assert unique_texts == ["a", "b"]
    assert index_groups == [[0, 2], [1]]


def test_definitions_and_mapping_stores_one_vector_reference_per_record():
    records = [{"definition": "a"}, {"definition": "b"}, {"definition": "a"}]

    unique, mapping = definitions_and_mapping(records)

    assert unique == ["a", "b"]
    assert mapping.dtype.name == "int32"
    assert mapping.tolist() == [0, 1, 0]


def test_validate_build_arrays_rejects_non_finite_vectors():
    records = [{"headword": "x", "pos": "noun", "definition": "x"}]
    with pytest.raises(ValueError, match="NaN or infinite"):
        validate_build_arrays(
            records,
            np.array([[np.nan]], dtype="float32"),
            np.array([0], dtype="int32"),
        )


def test_validate_records_rejects_invalid_fields_before_embedding():
    records = [{"headword": " ", "pos": "punct", "definition": "a mark"}]

    with pytest.raises(ValueError, match="Record 0 has an invalid headword"):
        validate_records(records)


def test_build_statistics_records_coverage_and_saved_vector_rows():
    records = [
        {
            "headword": "calm",
            "pos": "adjective",
            "definition": "quiet",
            "source": "wordnet",
            "examples": ["remain calm"],
            "emolex": {"trust"},
            "phonetics": {"syllable_count": 1},
        },
        {
            "headword": "placid",
            "pos": "adjective",
            "definition": "quiet",
            "source": "wiktionary",
            "examples": [],
            "emolex": None,
            "phonetics": None,
        },
    ]

    stats = build_statistics(records, unique_definition_count=1, literary_frequency={"calm": 4.0})

    assert stats["sources"] == {"wiktionary": 1, "wordnet": 1}
    assert stats["duplicate_embedding_rows_avoided"] == 1
    assert stats["records_with_emolex"] == 1
    assert stats["phonetics"] == {
        "eligible_records": 2,
        "resolved_records": 1,
        "failure_reasons": {},
    }


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


def test_build_metadata_record_persists_source_and_sense_provenance():
    record = {
        "headword": "calm",
        "pos": "adjective",
        "definition": "not excited",
        "examples": [],
        "source": "wordnet",
        "sources": ["wordnet", "wiktionary"],
        "synset": "calm.a.01",
        "wiktionary_sense_ids": ["en:calm-quiet"],
        "wikidata_ids": ["Q154729"],
        "topics": ["emotion"],
        "antonyms": ["agitated"],
    }

    meta = build_metadata_record(record)

    assert meta["sources"] == ["wordnet", "wiktionary"]
    assert meta["synset"] == "calm.a.01"
    assert meta["wiktionary_sense_ids"] == ["en:calm-quiet"]
    assert meta["wikidata_ids"] == ["Q154729"]
    assert meta["topics"] == ["emotion"]
    assert meta["antonyms"] == ["agitated"]


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


def test_build_metadata_record_includes_phonetics_when_present():
    record = {
        "headword": "cat",
        "pos": "noun",
        "definition": "a small domesticated carnivore",
        "examples": [],
        "source": "wordnet",
        "phonetics": {
            "syllable_count": 1,
            "primary_vowel": "AE",
            "rhyme_key": "AE T",
            "meter": "/",
            "phonemes": ["K", "AE1", "T"],
        },
    }

    meta = build_metadata_record(record)

    assert meta["phonetics"] == {
        "syllable_count": 1,
        "primary_vowel": "AE",
        "rhyme_key": "AE T",
        "meter": "/",
        "phonemes": ["K", "AE1", "T"],
    }


def test_build_metadata_record_defaults_phonetics_to_none_when_absent():
    record = {
        "headword": "kick the bucket",
        "pos": "verb",
        "definition": "to die",
        "examples": [],
        "source": "wordnet",
    }

    meta = build_metadata_record(record)

    assert meta["phonetics"] is None


def test_build_attaches_phonetics_to_every_record(monkeypatch, tmp_path):
    """The precomputation pass itself: build() must call
    phonetics.resolve_with_diagnostic(headword, pos) for every merged record and store the
    result on record["phonetics"] before build_metadata_record ever runs,
    not leave it to be computed lazily later."""
    import revdict.data.build_index as build_index_module

    fake_records = [
        {"headword": "cat", "pos": "noun", "definition": "d1", "examples": [], "source": "wordnet"},
        {"headword": "run", "pos": "verb", "definition": "d2", "examples": [], "source": "wordnet"},
    ]
    monkeypatch.setattr(build_index_module, "load_wordnet_senses", lambda: fake_records)
    monkeypatch.setattr(
        build_index_module, "download_raw_wiktextract", lambda path, refresh=False: None
    )
    monkeypatch.setattr(build_index_module, "stream_filtered_entries_from_gzip", lambda path: iter(()))

    calls = []

    def fake_resolve(word, pos):
        calls.append((word, pos))
        return {"syllable_count": 1, "primary_vowel": "AE", "rhyme_key": "AE T", "meter": "/", "phonemes": ["X"]}

    monkeypatch.setattr(
        build_index_module.phonetics,
        "resolve_with_diagnostic",
        lambda word, pos: (fake_resolve(word, pos), None),
    )

    # Stub every other slow/network-touching step so this test exercises
    # only the phonetics-attachment wiring, matching this file's existing
    # convention for build()-level tests (see the emolex/literary-frequency
    # stubs already used elsewhere in this test file for the same reason).
    monkeypatch.setattr(build_index_module, "load_emolex", lambda: {})
    monkeypatch.setattr(build_index_module, "lookup_emolex", lambda word, emolex, pos=None: None)
    monkeypatch.setattr(
        build_index_module, "download_raw_ngram_fiction", lambda path, refresh=False: None
    )
    monkeypatch.setattr(
        build_index_module,
        "download_raw_ngram_fiction_totalcounts",
        lambda path, refresh=False: None,
    )
    monkeypatch.setattr(build_index_module, "compute_literary_frequencies", lambda headwords, a, b: {})

    class FakeEmbedder:
        def encode_passages(self, texts):
            import numpy as np

            return np.zeros((len(texts), 4), dtype="float32")

    monkeypatch.setattr(build_index_module, "Embedder", lambda: FakeEmbedder())
    monkeypatch.setattr(build_index_module, "INDEX_DIR", tmp_path)

    build_index_module.build(skip_confirm=True)

    assert ("cat", "noun") in calls
    assert ("run", "verb") in calls


def test_build_reuses_cached_phonetics_for_repeated_headword_pos_pairs(monkeypatch, tmp_path):
    """Many records share the same headword under multiple senses of the
    same part of speech -- the whole point of the (headword.lower(), pos)
    cache is to resolve each such pair only once per build(). This pins
    that dedup actually happens rather than merely trusting the
    implementation by inspection."""
    import revdict.data.build_index as build_index_module

    fake_records = [
        {"headword": "cat", "pos": "noun", "definition": "d1", "examples": [], "source": "wordnet"},
        {"headword": "Cat", "pos": "noun", "definition": "d2", "examples": [], "source": "wordnet"},
        {"headword": "cat", "pos": "noun", "definition": "d3", "examples": [], "source": "wiktionary"},
    ]
    monkeypatch.setattr(build_index_module, "load_wordnet_senses", lambda: fake_records)
    monkeypatch.setattr(
        build_index_module, "download_raw_wiktextract", lambda path, refresh=False: None
    )
    monkeypatch.setattr(build_index_module, "stream_filtered_entries_from_gzip", lambda path: iter(()))

    calls = []

    def fake_resolve(word, pos):
        calls.append((word, pos))
        return {"syllable_count": 1, "primary_vowel": "AE", "rhyme_key": "AE T", "meter": "/", "phonemes": ["X"]}

    monkeypatch.setattr(
        build_index_module.phonetics,
        "resolve_with_diagnostic",
        lambda word, pos: (fake_resolve(word, pos), None),
    )

    monkeypatch.setattr(build_index_module, "load_emolex", lambda: {})
    monkeypatch.setattr(build_index_module, "lookup_emolex", lambda word, emolex, pos=None: None)
    monkeypatch.setattr(
        build_index_module, "download_raw_ngram_fiction", lambda path, refresh=False: None
    )
    monkeypatch.setattr(
        build_index_module,
        "download_raw_ngram_fiction_totalcounts",
        lambda path, refresh=False: None,
    )
    monkeypatch.setattr(build_index_module, "compute_literary_frequencies", lambda headwords, a, b: {})

    class FakeEmbedder:
        def encode_passages(self, texts):
            import numpy as np

            return np.zeros((len(texts), 4), dtype="float32")

    monkeypatch.setattr(build_index_module, "Embedder", lambda: FakeEmbedder())
    monkeypatch.setattr(build_index_module, "INDEX_DIR", tmp_path)

    build_index_module.build(skip_confirm=True)

    assert len(calls) == 1
    assert calls[0] == ("cat", "noun")
