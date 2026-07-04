from revdict.data.wordnet_source import load_wordnet_senses


def test_load_wordnet_senses_includes_known_word_with_expected_fields():
    records = load_wordnet_senses()
    happy = [r for r in records if r["headword"] == "happy" and r["synset"] == "happy.a.01"]
    assert len(happy) == 1
    r = happy[0]
    assert r["pos"] == "adjective"
    assert "pleasure" in r["definition"] or "joy" in r["definition"]
    assert r["source"] == "wordnet"
    assert r["sentiwordnet"] is not None
    assert r["sentiwordnet"]["pos"] > r["sentiwordnet"]["neg"]
    assert "a happy smile" in r["examples"]


def test_load_wordnet_senses_expands_multi_lemma_synsets_to_one_record_per_word():
    records = load_wordnet_senses()
    car_synonyms = [r for r in records if r["synset"] == "car.n.01"]
    headwords = {r["headword"] for r in car_synonyms}
    assert "car" in headwords
    assert "automobile" in headwords
    for r in car_synonyms:
        assert r["definition"] == car_synonyms[0]["definition"]
