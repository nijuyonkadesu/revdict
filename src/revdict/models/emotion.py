from typing import cast

EMOTION_POLARITY = {
    "joy": "positive",
    "trust": "positive",
    "anticipation": "positive",
    "anger": "negative",
    "disgust": "negative",
    "fear": "negative",
    "sadness": "negative",
    "surprise": "neutral",
}

_SENTIMENT_FLAGS = {"positive", "negative", "neutral"}


def polarity_from_sentiwordnet(scores: dict) -> str:
    pos, neg = scores["pos"], scores["neg"]
    if pos == neg:
        return "neutral"
    return "positive" if pos > neg else "negative"


def label_from_emolex(labels: frozenset[str]) -> tuple[str, str]:
    specific = sorted(label for label in labels if label not in _SENTIMENT_FLAGS)
    if specific:
        label = specific[0]
        return label, EMOTION_POLARITY.get(label, "neutral")
    if "positive" in labels:
        return "positive", "positive"
    if "negative" in labels:
        return "negative", "negative"
    return "neutral", "neutral"


class EmotionClassifier:
    def __init__(self):
        from transformers import pipeline

        self._pipe = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
        )

    def classify(self, text: str) -> tuple[str, str]:
        # transformers' pipeline stubs don't precisely type the top_k=None
        # shape; at runtime this is a list of {"label": str, "score": float}
        # dicts for the single input text (verified by the real model calls
        # exercised in Task 14's manual validation).
        results = cast(list[dict], self._pipe(text)[0])
        top = max(results, key=lambda item: item["score"])
        label = top["label"].lower()
        return label, EMOTION_POLARITY.get(label, "neutral")


def _emolex_has_specific_category(emolex_labels: frozenset[str] | None) -> bool:
    if not emolex_labels:
        return False
    label, _ = label_from_emolex(emolex_labels)
    return label not in _SENTIMENT_FLAGS


def _resolve_polarity(record: dict, emolex_labels, classifier_result) -> tuple[str, str]:
    sentiwordnet = record.get("sentiwordnet")
    if record.get("source") == "wordnet" and sentiwordnet is not None:
        polarity = polarity_from_sentiwordnet(sentiwordnet)
        if polarity != "neutral":
            return polarity, "sentiwordnet"
    if emolex_labels:
        _, polarity = label_from_emolex(emolex_labels)
        if polarity != "neutral":
            return polarity, "emolex"
    if classifier_result is not None:
        _, polarity = classifier_result
        return polarity, "classifier"
    return "neutral", "none"


def _resolve_category(emolex_labels, classifier_result) -> tuple[str | None, str | None]:
    if emolex_labels:
        label, _ = label_from_emolex(emolex_labels)
        if label not in _SENTIMENT_FLAGS:
            return label, "emolex"
    if classifier_result is not None:
        label, _ = classifier_result
        return label, "classifier"
    return None, None


def tag_emotion(record: dict, classifier_factory) -> dict:
    """classifier_factory is a zero-argument callable returning an
    EmotionClassifier (typically memoizing), or None to disable the
    classifier fallback entirely. It is only called when EmoLex doesn't
    already supply a specific emotion category for this record."""
    emolex_labels = record.get("emolex")

    classifier_result = None
    if not _emolex_has_specific_category(emolex_labels) and classifier_factory is not None:
        classifier = classifier_factory()
        classifier_result = classifier.classify(record["definition"])

    polarity, polarity_source = _resolve_polarity(record, emolex_labels, classifier_result)
    category, category_source = _resolve_category(emolex_labels, classifier_result)

    label = category or polarity
    source = category_source or polarity_source
    return {"label": label, "polarity": polarity, "emotion_source": source}
