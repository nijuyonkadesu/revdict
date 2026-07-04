from revdict.data.corpus import merge_records


def test_merge_records_drops_exact_duplicate_and_keeps_distinct_senses():
    wordnet_records = [
        {
            "headword": "happy",
            "pos": "adjective",
            "definition": "Feeling or showing pleasure.",
            "examples": [],
            "source": "wordnet",
        }
    ]
    wiktionary_records = [
        {
            "headword": "happy",
            "pos": "adjective",
            "definition": "  feeling OR showing pleasure. ",
            "examples": [],
            "source": "wiktionary",
        },
        {
            "headword": "happy",
            "pos": "adjective",
            "definition": "Fortunate and convenient.",
            "examples": [],
            "source": "wiktionary",
        },
    ]

    merged = merge_records(wordnet_records, wiktionary_records)

    assert len(merged) == 2
    assert merged[0]["source"] == "wordnet"
    assert merged[1]["definition"] == "Fortunate and convenient."
