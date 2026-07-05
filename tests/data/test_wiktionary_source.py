import gzip
import tempfile
from pathlib import Path

from revdict.data.wiktionary_source import (
    _combine_glosses,
    parse_filtered_entries,
    stream_filtered_entries_from_gzip,
)

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
