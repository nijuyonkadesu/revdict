# tests/data/test_nrc_emolex_source.py
from revdict.data.nrc_emolex_source import load_emolex, lookup_emolex


def test_lookup_emolex_is_case_insensitive_and_returns_none_for_unknown_word():
    fake_emolex = {"abandon": frozenset({"fear", "negative", "sadness"})}
    assert lookup_emolex("Abandon", fake_emolex) == frozenset({"fear", "negative", "sadness"})
    assert lookup_emolex("zzznotarealword", fake_emolex) is None


def test_load_emolex_returns_real_bundled_lexicon():
    emolex = load_emolex()
    assert len(emolex) > 1000
    assert "abandon" in emolex
    assert "fear" in emolex["abandon"] or "sadness" in emolex["abandon"]
