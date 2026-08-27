# tests/models/test_emotion.py
import pytest

from revdict.models.emotion import (
    EmotionClassifier,
    format_emotion_label,
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


def classifier_from_scores(scores):
    return FakeClassifier(EmotionClassifier._prediction(scores))


def test_polarity_from_sentiwordnet_prefers_the_higher_score():
    assert polarity_from_sentiwordnet({"pos": 0.8, "neg": 0.0, "obj": 0.2}) == "positive"
    assert polarity_from_sentiwordnet({"pos": 0.0, "neg": 0.6, "obj": 0.4}) == "negative"
    assert polarity_from_sentiwordnet({"pos": 0.0, "neg": 0.0, "obj": 1.0}) == "neutral"
    assert polarity_from_sentiwordnet({"pos": 0.125, "neg": 0.0, "obj": 0.875}) == "neutral"


def test_label_from_emolex_preserves_ambiguity_instead_of_alphabetically_selecting():
    label, polarity = label_from_emolex(frozenset({"fear", "negative", "sadness"}))
    assert label == "mixed"
    assert polarity == "negative"


def test_label_from_emolex_returns_the_only_specific_emotion():
    assert label_from_emolex(frozenset({"joy", "positive"})) == ("joy", "positive")


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
    assert result["label"] == "joy"
    assert result["labels"] == ["joy"]
    assert result["polarity"] == "positive"
    assert result["category_source"] == "emolex-prior"
    assert result["polarity_source"] == "emolex-prior"


def test_tag_emotion_uses_sense_classifier_and_retains_lexicon_evidence():
    record_needs_classifier = {
        "source": "wiktionary",
        "definition": "a feeling of anger and rage",
        "sentiwordnet": None,
        "emolex": None,
    }
    factory = FakeClassifierFactory(FakeClassifier(("anger", "negative")))
    result = tag_emotion(record_needs_classifier, classifier_factory=factory)
    assert result["label"] == "anger"
    assert result["labels"] == ["anger"]
    assert result["polarity"] == "negative"
    assert result["emotion_source"] == "classifier"
    assert factory.construct_calls == 1
    assert factory._classifier.calls == 1

    record_emolex_specific = {
        "source": "wiktionary",
        "definition": "a feeling of joy and delight",
        "sentiwordnet": None,
        "emolex": frozenset({"joy"}),
    }
    specific_factory = FakeClassifierFactory(FakeClassifier(("anger", "negative")))
    result = tag_emotion(record_emolex_specific, classifier_factory=specific_factory)
    assert result["emotion_source"] == "definition-cue"
    assert result["label"] == "joy"
    assert result["labels"] == ["joy"]
    assert result["evidence"]["classifier"]["accepted"] is False
    assert result["evidence"]["classifier"]["reason"] == "lexicon-conflict"
    assert specific_factory.construct_calls == 1


def test_tag_emotion_invokes_classifier_when_emolex_has_only_bare_sentiment_flag():
    """A bare EmoLex sentiment flag (e.g. {"positive"}) is not a *specific*
    emotion category, so the classifier fallback must still fire to supply the
    category label -- this is the exact scenario the classifier-fallback design
    correction targeted, and it must stay covered so a future simplification of
    `_emolex_has_specific_category` (e.g. to `bool(emolex_labels)`) can't silently
    reintroduce the bug."""
    record = {
        "source": "wiktionary",
        "definition": "a feeling of terror and fear",
        "sentiwordnet": None,
        "emolex": frozenset({"positive"}),
    }
    factory = FakeClassifierFactory(FakeClassifier(("fear", "negative")))
    result = tag_emotion(record, classifier_factory=factory)
    assert factory.construct_calls == 1
    assert factory._classifier.calls == 1
    # A primary badge must be coherent: the retained NRC positive flag remains
    # available as evidence, but cannot turn a `fear` badge positive.
    assert result["label"] == "fear"
    assert result["polarity"] == "negative"
    assert result["emolex_labels"] == ["positive"]
    assert result["category_source"] == "classifier"


def test_tag_emotion_returns_neutral_none_when_nothing_available():
    record = {"source": "wiktionary", "definition": "x", "sentiwordnet": None, "emolex": None}
    result = tag_emotion(record, classifier_factory=None)
    assert result["label"] == "neutral"
    assert result["labels"] == []
    assert result["polarity"] == "neutral"
    assert result["emotion_source"] == "none"
    assert result["category_source"] is None
    assert result["status"] == "abstained"


def test_multi_label_emolex_uses_classifier_for_sense_but_retains_all_labels():
    record = {
        "source": "wiktionary",
        "definition": "a state of grief",
        "sentiwordnet": None,
        "emolex": frozenset({"fear", "sadness", "negative"}),
    }
    factory = FakeClassifierFactory(FakeClassifier(("sadness", "negative")))

    result = tag_emotion(record, classifier_factory=factory)

    assert result["label"] == "sadness"
    assert result["labels"] == ["sadness"]
    assert result["emolex_labels"] == ["fear", "negative", "sadness"]
    assert result["polarity"] == "negative"
    assert factory._classifier.calls == 1


def test_low_confidence_classifier_falls_back_to_unambiguous_emolex_label():
    class LowConfidenceClassifier:
        def classify(self, _text):
            return {
                "label": "anger",
                "polarity": "negative",
                "confidence": 0.3,
                "scores": {"anger": 0.3},
            }

    record = {
        "source": "wiktionary",
        "definition": "a feeling of delight",
        "sentiwordnet": None,
        "emolex": frozenset({"joy", "positive"}),
    }

    result = tag_emotion(record, classifier_factory=lambda: LowConfidenceClassifier())

    assert result["label"] == "joy"
    assert result["polarity"] == "positive"
    assert result["category_source"] == "definition-cue"


def test_happiness_conflict_resolves_to_positive_mixed_prior_not_negative():
    record = {
        "headword": "happiness",
        "source": "wordnet",
        "definition": "emotions experienced when in a state of well-being",
        "sentiwordnet": {"pos": 0.125, "neg": 0.0, "obj": 0.875},
        "emolex": frozenset({"anticipation", "joy", "positive"}),
    }
    classifier = classifier_from_scores(
        {"sadness": 0.40, "neutral": 0.35, "joy": 0.15, "fear": 0.10}
    )

    result = tag_emotion(record, classifier_factory=lambda: classifier)

    assert result["label"] == "mixed"
    assert result["labels"] == ["anticipation", "joy"]
    assert result["polarity"] == "positive"
    assert format_emotion_label(result["label"], result["labels"]) == (
        "mixed (anticipation/joy)"
    )
    assert result["evidence"]["classifier"]["accepted"] is False


def test_objective_table_sense_rejects_overconfident_domain_mismatch():
    record = {
        "headword": "table",
        "source": "wordnet",
        "definition": "a piece of furniture having a smooth flat top",
        "sentiwordnet": {"pos": 0.0, "neg": 0.0, "obj": 1.0},
        "emolex": None,
    }
    classifier = classifier_from_scores(
        {"disgust": 0.919, "neutral": 0.035, "fear": 0.015, "surprise": 0.011}
    )

    result = tag_emotion(record, classifier_factory=lambda: classifier)

    assert result["label"] == "neutral"
    assert result["labels"] == []
    assert result["status"] == "abstained"
    assert result["evidence"]["classifier"]["reason"] == "objective-sense"


def test_full_taxonomy_cue_can_override_negative_word_prior():
    record = {
        "headword": "fear",
        "source": "wordnet",
        "definition": "a feeling of profound respect for someone or something",
        "sentiwordnet": {"pos": 0.5, "neg": 0.0, "obj": 0.5},
        "emolex": frozenset({"anger", "fear", "negative"}),
    }
    classifier = classifier_from_scores(
        {"joy": 0.989, "neutral": 0.006, "fear": 0.005}
    )

    result = tag_emotion(record, classifier_factory=lambda: classifier)

    assert result["label"] == "trust"
    assert result["labels"] == ["trust"]
    assert result["polarity"] == "positive"
    assert format_emotion_label(result["label"], result["labels"]) == "trust"
    assert result["emolex_labels"] == ["anger", "fear", "negative"]
    assert result["evidence"]["classifier"]["reason"] == (
        "full-taxonomy-cue-preferred"
    )


def test_canonical_emotion_headword_wins_over_same_polarity_sibling():
    record = {
        "headword": "disgust",
        "source": "wordnet",
        "definition": "cause aversion in; offend the moral sense of",
        "sentiwordnet": {"pos": 0.125, "neg": 0.625, "obj": 0.25},
        "emolex": frozenset({"anger", "disgust", "fear", "negative", "sadness"}),
    }
    classifier = classifier_from_scores(
        {"anger": 0.936, "disgust": 0.03, "fear": 0.02, "neutral": 0.014}
    )

    result = tag_emotion(record, classifier_factory=lambda: classifier)

    assert result["label"] == "disgust"
    assert result["labels"] == ["disgust"]
    assert result["polarity"] == "negative"
    assert result["evidence"]["classifier"]["reason"] == "canonical-prior-preferred"


def test_trust_prior_is_kept_for_emotional_sense_but_not_corporate_sense():
    trust_classifier = classifier_from_scores(
        {"joy": 0.506, "neutral": 0.30, "fear": 0.10, "anger": 0.094}
    )
    emotional = {
        "headword": "trust",
        "source": "wordnet",
        "definition": "the trait of believing in the honesty and reliability of others",
        "sentiwordnet": {"pos": 0.625, "neg": 0.0, "obj": 0.375},
        "emolex": frozenset({"trust"}),
    }
    corporate = {
        "headword": "trust",
        "source": "wordnet",
        "definition": "a consortium of independent organizations formed to limit competition",
        "sentiwordnet": {"pos": 0.0, "neg": 0.0, "obj": 1.0},
        "emolex": frozenset({"trust"}),
    }

    emotional_result = tag_emotion(emotional, classifier_factory=lambda: trust_classifier)
    corporate_result = tag_emotion(corporate, classifier_factory=lambda: trust_classifier)

    assert emotional_result["label"] == "trust"
    assert emotional_result["polarity"] == "positive"
    assert corporate_result["label"] == "neutral"
    assert corporate_result["status"] == "abstained"

    neutral_classifier = classifier_from_scores(
        {"neutral": 0.548, "joy": 0.216, "fear": 0.083, "anger": 0.083}
    )
    confidence_sense = {
        "headword": "trust",
        "source": "wordnet",
        "definition": "complete confidence in a person or plan",
        "sentiwordnet": {"pos": 0.0, "neg": 0.0, "obj": 1.0},
        "emolex": frozenset({"trust"}),
    }
    confidence_result = tag_emotion(
        confidence_sense, classifier_factory=lambda: neutral_classifier
    )

    assert confidence_result["label"] == "trust"
    assert confidence_result["evidence"]["classifier"]["reason"] == (
        "lexicon-cue-preferred"
    )


def test_classifier_requires_margin_and_entropy_not_confidence_alone():
    record = {
        "headword": "plain",
        "source": "wiktionary",
        "definition": "an emotional feeling",
        "sentiwordnet": None,
        "emolex": None,
    }
    classifier = classifier_from_scores(
        {"joy": 0.40, "sadness": 0.35, "fear": 0.25}
    )

    result = tag_emotion(record, classifier_factory=lambda: classifier)

    assert result["label"] == "neutral"
    assert result["evidence"]["classifier"]["accepted"] is False
    assert result["evidence"]["classifier"]["margin"] == pytest.approx(0.05)


def test_classifier_predictions_persist_by_model_revision_and_definition(tmp_path):
    inference_calls = []

    def pipeline_factory(*_args, **_kwargs):
        def classify(text):
            inference_calls.append(text)
            return [[
                {"label": "joy", "score": 0.8},
                {"label": "neutral", "score": 0.2},
            ]]

        return classify

    cache_path = tmp_path / "emotion.sqlite3"
    first = EmotionClassifier(cache_path=cache_path, pipeline_factory=pipeline_factory)
    second = EmotionClassifier(cache_path=cache_path, pipeline_factory=pipeline_factory)

    assert first.classify("feeling delighted")["label"] == "joy"
    assert second.classify("feeling delighted")["label"] == "joy"
    assert inference_calls == [["feeling delighted"]]


def test_classifier_batches_uncached_definitions_and_reuses_memory_cache(tmp_path):
    batches = []

    def pipeline_factory(*_args, **_kwargs):
        def classify(texts):
            batches.append(texts)
            return [
                [{"label": "joy", "score": 0.9}, {"label": "neutral", "score": 0.1}]
                for _text in texts
            ]

        return classify

    classifier = EmotionClassifier(
        cache_path=tmp_path / "emotion.sqlite3", pipeline_factory=pipeline_factory
    )

    predictions = classifier.classify_many(["delight", "grief", "delight"])
    classifier.classify_many(["grief", "delight"])

    assert [prediction["label"] for prediction in predictions] == ["joy", "joy", "joy"]
    assert batches == [["delight", "grief"]]


def test_corrupt_prediction_cache_is_quarantined_not_deleted(tmp_path):
    cache_path = tmp_path / "emotion.sqlite3"
    cache_path.write_text("not a sqlite database", encoding="utf-8")

    def pipeline_factory(*_args, **_kwargs):
        return lambda _texts: [[{"label": "neutral", "score": 1.0}]]

    classifier = EmotionClassifier(cache_path=cache_path, pipeline_factory=pipeline_factory)

    assert classifier.classify("plain definition")["label"] == "neutral"
    assert cache_path.is_file()
    quarantined = list(tmp_path.glob("emotion.sqlite3.corrupt-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_text(encoding="utf-8") == "not a sqlite database"
