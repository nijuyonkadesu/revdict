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
    assert merged[0]["sources"] == ["wordnet", "wiktionary"]
    assert merged[1]["definition"] == "Fortunate and convenient."


def test_merge_identity_includes_part_of_speech():
    common = {
        "headword": "headfirst",
        "definition": "with the head foremost",
        "examples": [],
        "source": "wordnet",
    }

    merged = merge_records(
        [{**common, "pos": "adjective"}, {**common, "pos": "adverb"}], []
    )

    assert [record["pos"] for record in merged] == ["adjective", "adverb"]


def test_merge_deduplicates_within_wordnet_and_combines_useful_fields():
    first = {
        "headword": "incorrupt",
        "pos": "adjective",
        "definition": "Free of corruption or immorality.",
        "examples": ["first example"],
        "source": "wordnet",
        "synset": "incorrupt.a.01",
        "synonyms": ["honest"],
    }
    duplicate = {
        "headword": "INCORRUPT",
        "pos": "adjective",
        "definition": "free of corruption or immorality",
        "examples": ["second example"],
        "source": "wordnet",
        "synonyms": ["honorable"],
    }

    merged = merge_records([first, duplicate], [])

    assert len(merged) == 1
    assert merged[0]["examples"] == ["first example", "second example"]
    assert merged[0]["synonyms"] == ["honest", "honorable"]
    assert merged[0]["synset"] == "incorrupt.a.01"


def test_cross_source_duplicate_preserves_tags_relations_and_identifiers():
    wordnet = {
        "headword": "happy",
        "pos": "adjective",
        "definition": "Feeling pleasure.",
        "examples": [],
        "source": "wordnet",
        "synset": "happy.a.01",
        "synonyms": ["glad"],
    }
    wiktionary = {
        "headword": "happy",
        "pos": "adjective",
        "definition": "Feeling pleasure",
        "examples": ["a happy child"],
        "source": "wiktionary",
        "synonyms": ["content"],
        "antonyms": ["sad"],
        "tags": ["informal"],
        "topics": ["emotion"],
        "wiktionary_sense_ids": ["en:happy-feeling"],
    }

    record = merge_records([wordnet], [wiktionary])[0]

    assert record["source"] == "wordnet"
    assert record["sources"] == ["wordnet", "wiktionary"]
    assert record["examples"] == ["a happy child"]
    assert record["synonyms"] == ["glad", "content"]
    assert record["antonyms"] == ["sad"]
    assert record["tags"] == ["informal"]
    assert record["wiktionary_sense_ids"] == ["en:happy-feeling"]
