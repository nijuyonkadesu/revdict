# tests/models/test_emotion.py
from revdict.models.emotion import (
    label_from_emolex,
    polarity_from_sentiwordnet,
    tag_emotion,
)


class FakeClassifier:
    def __init__(self, result):
        self._result = result
        self.calls = 0

    def classify(self, text):
        self.calls += 1
        return self._result


class FakeClassifierFactory:
    """Tracks how many times it was actually invoked to construct a classifier,
    so tests can assert the (expensive, lazy) classifier is only ever built
    when tag_emotion has decided it's actually needed."""

    def __init__(self, classifier):
        self._classifier = classifier
        self.construct_calls = 0

    def __call__(self):
        self.construct_calls += 1
        return self._classifier


def test_polarity_from_sentiwordnet_prefers_the_higher_score():
    assert polarity_from_sentiwordnet({"pos": 0.8, "neg": 0.0, "obj": 0.2}) == "positive"
    assert polarity_from_sentiwordnet({"pos": 0.0, "neg": 0.6, "obj": 0.4}) == "negative"
    assert polarity_from_sentiwordnet({"pos": 0.0, "neg": 0.0, "obj": 1.0}) == "neutral"


def test_label_from_emolex_prefers_specific_emotion_over_bare_sentiment_flag():
    label, polarity = label_from_emolex(frozenset({"fear", "negative", "sadness"}))
    assert label == "fear"
    assert polarity == "negative"


def test_label_from_emolex_falls_back_to_bare_sentiment_flag():
    assert label_from_emolex(frozenset({"positive"})) == ("positive", "positive")


def test_tag_emotion_uses_sentiwordnet_polarity_with_emolex_category():
    record = {
        "source": "wordnet",
        "definition": "x",
        "sentiwordnet": {"pos": 0.9, "neg": 0.0, "obj": 0.1},
        "emolex": frozenset({"joy"}),
    }
    result = tag_emotion(record, classifier_factory=None)
    assert result == {"label": "joy", "polarity": "positive", "emotion_source": "emolex"}


def test_tag_emotion_builds_classifier_only_when_emolex_has_no_specific_category():
    record_needs_classifier = {
        "source": "wiktionary",
        "definition": "x",
        "sentiwordnet": None,
        "emolex": None,
    }
    factory = FakeClassifierFactory(FakeClassifier(("anger", "negative")))
    result = tag_emotion(record_needs_classifier, classifier_factory=factory)
    assert result == {"label": "anger", "polarity": "negative", "emotion_source": "classifier"}
    assert factory.construct_calls == 1
    assert factory._classifier.calls == 1

    record_emolex_specific = {
        "source": "wiktionary",
        "definition": "x",
        "sentiwordnet": None,
        "emolex": frozenset({"joy"}),
    }
    factory_should_be_skipped = FakeClassifierFactory(FakeClassifier(("anger", "negative")))
    result = tag_emotion(record_emolex_specific, classifier_factory=factory_should_be_skipped)
    assert result["emotion_source"] == "emolex"
    assert factory_should_be_skipped.construct_calls == 0


def test_tag_emotion_invokes_classifier_when_emolex_has_only_bare_sentiment_flag():
    """A bare EmoLex sentiment flag (e.g. {"positive"}) is not a *specific*
    emotion category, so the classifier fallback must still fire to supply the
    category label -- this is the exact scenario the classifier-fallback design
    correction targeted, and it must stay covered so a future simplification of
    `_emolex_has_specific_category` (e.g. to `bool(emolex_labels)`) can't silently
    reintroduce the bug."""
    record = {
        "source": "wiktionary",
        "definition": "x",
        "sentiwordnet": None,
        "emolex": frozenset({"positive"}),
    }
    factory = FakeClassifierFactory(FakeClassifier(("fear", "negative")))
    result = tag_emotion(record, classifier_factory=factory)
    assert factory.construct_calls == 1
    assert factory._classifier.calls == 1
    # Category comes from the classifier fallback; polarity still comes from
    # EmoLex's bare "positive" flag, since _resolve_polarity checks EmoLex
    # before ever looking at the classifier result.
    assert result == {"label": "fear", "polarity": "positive", "emotion_source": "classifier"}


def test_tag_emotion_returns_neutral_none_when_nothing_available():
    record = {"source": "wiktionary", "definition": "x", "sentiwordnet": None, "emolex": None}
    result = tag_emotion(record, classifier_factory=None)
    assert result == {"label": "neutral", "polarity": "neutral", "emotion_source": "none"}
