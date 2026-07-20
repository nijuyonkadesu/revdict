import gzip
import itertools
import tempfile
from pathlib import Path

import pytest

from revdict.data.wiktionary_source import (
    _combine_glosses,
    parse_filtered_entries,
    stream_filtered_entries_from_gzip,
)
from revdict.paths import RAW_WIKTIONARY_PATH

ENGLISH_NOUN_LINE = (
    '{"word": "dictionary", "pos": "noun", "lang": "English", "lang_code": "en", '
    '"senses": [{"glosses": ["A reference work listing words and explaining their meanings."], '
    '"examples": [{"text": "a law dictionary"}]}]}'
)
ENGLISH_ADJ_LINE = (
    '{"word": "green with envy", "pos": "adj", "lang": "English", "lang_code": "en", '
    '"senses": [{"glosses": ["Very jealous."]}]}'
)
ENGLISH_FORM_OF_LINE = (
    '{"word": "dictionaries", "pos": "noun", "lang": "English", "lang_code": "en", '
    '"senses": [{"glosses": ["plural of dictionary"], "tags": ["form-of", "plural"], '
    '"form_of": [{"word": "dictionary"}]}]}'
)
# Real example found in the raw data: "bigge" tagged alt-of/obsolete, glosses
# self-referentially restating the target word ("Obsolete spelling of big.")
# -- these weren't caught by the form-of check (wiktextract treats
# alternative/obsolete spellings as a distinct tag from inflectional forms).
ENGLISH_ALT_OF_LINE = (
    '{"word": "bigge", "pos": "adj", "lang": "English", "lang_code": "en", '
    '"senses": [{"glosses": ["Obsolete spelling of big."], "tags": ["alt-of", "obsolete"]}]}'
)
NON_ENGLISH_LINE = (
    '{"word": "diccionario", "pos": "noun", "lang": "Spanish", "lang_code": "es", '
    '"senses": [{"glosses": ["Un diccionario"]}]}'
)
ENGLISH_ARCHAIC_LINE = (
    '{"word": "thou", "pos": "pron", "lang": "English", "lang_code": "en", '
    '"senses": [{"glosses": ["The second-person singular pronoun."], '
    '"tags": ["archaic", "singular"]}]}'
)
ENGLISH_NO_TAGS_LINE = (
    '{"word": "dog", "pos": "noun", "lang": "English", "lang_code": "en", '
    '"senses": [{"glosses": ["A domesticated canine."]}]}'
)


def test_parse_filtered_entries_keeps_english_drops_form_of_alt_of_and_non_english():
    lines = [
        ENGLISH_NOUN_LINE,
        ENGLISH_ADJ_LINE,
        ENGLISH_FORM_OF_LINE,
        ENGLISH_ALT_OF_LINE,
        NON_ENGLISH_LINE,
    ]
    records = parse_filtered_entries(lines)
    headwords = {r["headword"] for r in records}
    assert headwords == {"dictionary", "green with envy"}

    dictionary_record = next(r for r in records if r["headword"] == "dictionary")
    assert dictionary_record["pos"] == "noun"
    assert "reference work" in dictionary_record["definition"]
    assert dictionary_record["examples"] == ["a law dictionary"]
    assert dictionary_record["source"] == "wiktionary"

    idiom_record = next(r for r in records if r["headword"] == "green with envy")
    assert idiom_record["pos"] == "adjective"


def test_combine_glosses_returns_the_single_gloss_unchanged():
    assert _combine_glosses(["Very jealous."]) == "Very jealous."


def test_combine_glosses_joins_a_two_level_hierarchy():
    # Real example from the raw data: many senses of "free" all start with
    # this same broad category gloss -- keeping only glosses[0] would make
    # them all read identically and collapse together in corpus.py's dedup.
    result = _combine_glosses(["Unconstrained.", "Not imprisoned or enslaved."])
    assert result == "Unconstrained; Not imprisoned or enslaved."


def test_combine_glosses_joins_a_three_level_hierarchy():
    result = _combine_glosses(
        ["Terms relating to animals.", "A mammal of the family Felidae.", "A house pet."]
    )
    assert result == "Terms relating to animals; A mammal of the family Felidae; A house pet."


def test_stream_filtered_entries_from_gzip_reads_a_real_gz_file():
    with tempfile.TemporaryDirectory() as tmp:
        gz_path = Path(tmp) / "sample.jsonl.gz"
        with gzip.open(gz_path, "wt", encoding="utf-8") as f:
            f.write(ENGLISH_NOUN_LINE + "\n")
            f.write(NON_ENGLISH_LINE + "\n")
        records = list(stream_filtered_entries_from_gzip(str(gz_path)))
        assert len(records) == 1
        assert records[0]["headword"] == "dictionary"


def test_parse_filtered_entries_captures_the_tags_field_when_present():
    records = parse_filtered_entries([ENGLISH_ARCHAIC_LINE])
    assert records[0]["tags"] == ["archaic", "singular"]


def test_parse_filtered_entries_defaults_tags_to_an_empty_list_when_absent():
    records = parse_filtered_entries([ENGLISH_NO_TAGS_LINE])
    assert records[0]["tags"] == []


_REAL_RAW_WIKTIONARY_PATH = RAW_WIKTIONARY_PATH


@pytest.mark.skipif(
    not _REAL_RAW_WIKTIONARY_PATH.exists(),
    reason="requires the real cached Wiktionary dump (present after running `revdict build-index` once)",
)
def test_stream_filtered_entries_captures_real_tags_from_the_actual_dump():
    """Regression guard against the real data source's actual shape, not
    just a synthetic fixture. Scoped to the first 25K raw lines (not the
    full ~2.7GB file) to keep this fast on every test run -- verified by
    direct sampling that this slice alone contains hundreds of tagged
    senses (archaic: 448, slang: 777, as of the 2026-07 dump), so a
    zero-tag result here would be a genuine capture bug, not a sampling
    fluke. (A one-time broader sample, taken manually while writing this
    plan, already confirmed the full tag vocabulary -- this committed test
    only needs to catch a regression, not re-discover the vocabulary, so it
    stays small.)"""
    with gzip.open(_REAL_RAW_WIKTIONARY_PATH, "rt", encoding="utf-8") as f:
        lines = list(itertools.islice(f, 25_000))
    records = parse_filtered_entries(lines)
    seen_tags = {tag for record in records for tag in record["tags"]}
    assert "archaic" in seen_tags
    assert "slang" in seen_tags
