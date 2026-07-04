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
