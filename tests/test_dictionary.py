import json
from pathlib import Path

from revdict.dictionary import load_metadata, load_word_index, lookup_exact


def test_lookup_exact_returns_all_senses_case_insensitively():
    metadata = [
        {
            "headword": "bank",
            "pos": "noun",
            "definition": "a financial institution",
            "examples": [],
            "source": "wordnet",
        },
        {
            "headword": "bank",
            "pos": "noun",
            "definition": "the land alongside a river",
            "examples": [],
            "source": "wordnet",
        },
    ]
    word_index = {"bank": [0, 1]}

    result = lookup_exact("Bank", word_index, metadata)

    assert result["headword"] == "Bank"
    assert len(result["senses"]) == 2
    assert result["senses"][0]["definition"] == "a financial institution"


def test_lookup_exact_carries_raw_sentiwordnet_emolex_and_synonyms_fields_per_sense():
    """dictionary.py stays a raw lookup: it must surface the persisted
    sentiwordnet/emolex/synonyms fields per-sense so search.py can tag emotion
    and display synonyms without dictionary.py reaching into emotion.py itself."""
    metadata = [
        {
            "headword": "happy",
            "pos": "adjective",
            "definition": "feeling great pleasure",
            "examples": [],
            "source": "wordnet",
            "sentiwordnet": {"pos": 0.8, "neg": 0.0, "obj": 0.2},
            "emolex": ["joy", "positive"],
            "synonyms": ["glad", "content"],
        },
        {
            "headword": "happy",
            "pos": "adjective",
            "definition": "willing to do something",
            "examples": [],
            "source": "wiktionary",
        },
    ]
    word_index = {"happy": [0, 1]}

    result = lookup_exact("happy", word_index, metadata)

    first, second = result["senses"]
    assert first["sentiwordnet"] == {"pos": 0.8, "neg": 0.0, "obj": 0.2}
    assert first["emolex"] == ["joy", "positive"]
    assert first["synonyms"] == ["glad", "content"]
    # Wiktionary-sourced records lack these fields entirely -- must not KeyError.
    assert second["sentiwordnet"] is None
    assert second["emolex"] is None
    assert second["synonyms"] is None


def test_lookup_exact_returns_none_for_unknown_word():
    assert lookup_exact("zzznotarealword", {}, []) is None


def test_load_word_index_and_metadata_read_real_files(tmp_path):
    (tmp_path / "word_index.json").write_text(json.dumps({"bank": [0]}), encoding="utf-8")
    (tmp_path / "metadata.jsonl").write_text(
        json.dumps(
            {
                "headword": "bank",
                "pos": "noun",
                "definition": "a financial institution",
                "examples": [],
                "source": "wordnet",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    word_index = load_word_index(Path(tmp_path))
    metadata = load_metadata(Path(tmp_path))

    assert word_index == {"bank": [0]}
    assert metadata[0]["headword"] == "bank"
