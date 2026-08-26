from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import cast

from revdict.paths import EMOTION_CACHE_PATH

EMOTION_POLARITY = {
    "joy": "positive",
    "trust": "positive",
    "anticipation": "positive",
    "anger": "negative",
    "disgust": "negative",
    "fear": "negative",
    "sadness": "negative",
    "surprise": "neutral",
    "neutral": "neutral",
}

_SENTIMENT_FLAGS = {"positive", "negative", "neutral"}
CLASSIFIER_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
CLASSIFIER_MODEL_REVISION = "0e1cd914e3d46199ed785853e12b57304e04178b"
SENTIWORDNET_MIN_MARGIN = 0.25
CLASSIFIER_MIN_CONFIDENCE = 0.45


def polarity_from_sentiwordnet(scores: dict) -> str:
    pos, neg = float(scores["pos"]), float(scores["neg"])
    if abs(pos - neg) < SENTIWORDNET_MIN_MARGIN:
        return "neutral"
    return "positive" if pos > neg else "negative"


def specific_emolex_labels(labels: frozenset[str] | None) -> list[str]:
    return sorted(label for label in labels or () if label not in _SENTIMENT_FLAGS)


def format_emotion_label(primary: str, labels: list[str]) -> str:
    if len(labels) <= 1:
        return primary
    return f"{primary} ({'/'.join(labels)})"


def label_from_emolex(labels: frozenset[str]) -> tuple[str, str]:
    """Return a deterministic summary without discarding multi-label ambiguity."""
    specific = specific_emolex_labels(labels)
    if len(specific) == 1:
        label = specific[0]
        return label, EMOTION_POLARITY.get(label, "neutral")
    if len(specific) > 1:
        return "mixed", _polarity_from_emolex_flags(labels)
    polarity = _polarity_from_emolex_flags(labels)
    return polarity, polarity


def _polarity_from_emolex_flags(labels: frozenset[str] | None) -> str:
    labels = labels or frozenset()
    positive = "positive" in labels
    negative = "negative" in labels
    if positive == negative:
        return "neutral"
    return "positive" if positive else "negative"


class EmotionClassifier:
    """Sense-text classifier backed by a revision-specific persistent cache."""

    def __init__(self, cache_path: Path | None = None, pipeline_factory=None):
        if pipeline_factory is None:
            from transformers import pipeline

            pipeline_factory = pipeline
        self._pipeline_factory = pipeline_factory
        self._pipe = None
        self._cache_path = Path(EMOTION_CACHE_PATH if cache_path is None else cache_path)
        self._lock = threading.Lock()
        self._memory_cache: dict[str, dict[str, float]] = {}

    def _pipeline(self):
        if self._pipe is None:
            self._pipe = self._pipeline_factory(
                "text-classification",
                model=CLASSIFIER_MODEL_NAME,
                revision=CLASSIFIER_MODEL_REVISION,
                top_k=None,
            )
        return self._pipe

    @staticmethod
    def _cache_key(text: str) -> str:
        material = f"{CLASSIFIER_MODEL_REVISION}\0{text}".encode("utf-8")
        return hashlib.sha256(material).hexdigest()

    def _read_cache(self, key: str) -> dict[str, float] | None:
        if not self._cache_path.is_file():
            return None
        with sqlite3.connect(self._cache_path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS predictions "
                "(cache_key TEXT PRIMARY KEY, scores_json TEXT NOT NULL)"
            )
            row = connection.execute(
                "SELECT scores_json FROM predictions WHERE cache_key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        try:
            value = json.loads(row[0])
        except json.JSONDecodeError:
            return None
        if not isinstance(value, dict):
            return None
        return {str(label): float(score) for label, score in value.items()}

    def _write_cache(self, key: str, scores: dict[str, float]) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._cache_path) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS predictions "
                "(cache_key TEXT PRIMARY KEY, scores_json TEXT NOT NULL)"
            )
            connection.execute(
                "INSERT OR REPLACE INTO predictions(cache_key, scores_json) VALUES (?, ?)",
                (key, json.dumps(scores, sort_keys=True)),
            )

    def _quarantine_corrupt_cache(self) -> None:
        marker = uuid.uuid4().hex
        for path in (
            self._cache_path,
            Path(str(self._cache_path) + "-wal"),
            Path(str(self._cache_path) + "-shm"),
        ):
            if path.exists() or path.is_symlink():
                os.replace(path, path.with_name(f"{path.name}.corrupt-{marker}"))

    def classify(self, text: str) -> dict:
        return self.classify_many([text])[0]

    def classify_many(self, texts: list[str]) -> list[dict]:
        """Classify uncached definitions in one transformer batch."""
        unique_texts = list(dict.fromkeys(texts))
        predictions: dict[str, dict] = {}
        missing = []
        with self._lock:
            for text in unique_texts:
                key = self._cache_key(text)
                scores = self._memory_cache.get(key)
                if scores is None:
                    try:
                        scores = self._read_cache(key)
                    except sqlite3.DatabaseError:
                        self._quarantine_corrupt_cache()
                        scores = None
                if scores is None:
                    missing.append(text)
                    continue
                self._memory_cache[key] = scores
                predictions[text] = self._prediction(scores)

            if missing:
                raw_groups = self._pipeline()(missing)
                if len(missing) == 1 and raw_groups and isinstance(raw_groups[0], dict):
                    raw_groups = [raw_groups]
                for text, results in zip(missing, raw_groups, strict=True):
                    scores = {
                        str(item["label"]).lower(): float(item["score"])
                        for item in cast(list[dict], results)
                    }
                    if not scores:
                        raise RuntimeError("Emotion classifier returned no labels")
                    key = self._cache_key(text)
                    self._memory_cache[key] = scores
                    try:
                        self._write_cache(key, scores)
                    except sqlite3.DatabaseError:
                        self._quarantine_corrupt_cache()
                        self._write_cache(key, scores)
                    predictions[text] = self._prediction(scores)
        return [predictions[text] for text in texts]

    @staticmethod
    def _prediction(scores: dict[str, float]) -> dict:
        label, confidence = max(scores.items(), key=lambda item: item[1])
        return {
            "label": label,
            "polarity": EMOTION_POLARITY.get(label, "neutral"),
            "confidence": confidence,
            "scores": scores,
        }

def _normalize_classifier_result(result) -> dict | None:
    if result is None:
        return None
    if isinstance(result, tuple):
        label, polarity = result
        return {"label": label, "polarity": polarity, "confidence": None, "scores": {}}
    return result


def _fallback_polarity(record: dict, emolex_labels: frozenset[str] | None) -> tuple[str, str]:
    sentiwordnet = record.get("sentiwordnet")
    if record.get("source") == "wordnet" and sentiwordnet is not None:
        polarity = polarity_from_sentiwordnet(sentiwordnet)
        if polarity != "neutral":
            return polarity, "sentiwordnet"
    polarity = _polarity_from_emolex_flags(emolex_labels)
    if polarity != "neutral":
        return polarity, "emolex"
    return "neutral", "none"


def tag_emotion(record: dict, classifier_factory) -> dict:
    """Resolve a coherent primary badge while preserving all lexicon labels.

    A single NRC category remains authoritative and cheap. Multiple NRC
    categories are retained and disambiguated by the sense definition; absent
    categories also use that sense-level classifier. Polarity follows the
    chosen category, preventing badges such as ``fear · positive``.
    """
    emolex_labels = record.get("emolex")
    specific = specific_emolex_labels(emolex_labels)
    classifier_result = None
    if classifier_factory is not None:
        classifier_result = _normalize_classifier_result(
            classifier_factory().classify(record["definition"])
        )

    classifier_is_confident = classifier_result is not None and (
        classifier_result.get("confidence") is None
        or classifier_result["confidence"] >= CLASSIFIER_MIN_CONFIDENCE
    )
    if classifier_is_confident:
        label = classifier_result["label"]
        category_source = "classifier"
        confidence = classifier_result.get("confidence")
    elif len(specific) == 1:
        label = specific[0]
        category_source = "emolex"
        confidence = None
    elif specific:
        label = "mixed"
        category_source = "emolex"
        confidence = None
    else:
        label = None
        category_source = None
        confidence = None

    if label in EMOTION_POLARITY:
        polarity = EMOTION_POLARITY[label]
        polarity_source = category_source
    elif classifier_result is not None:
        polarity = classifier_result["polarity"]
        polarity_source = "classifier"
    else:
        polarity, polarity_source = _fallback_polarity(record, emolex_labels)

    if label is None:
        label = polarity
    preserved_labels = specific or (
        [label] if label not in _SENTIMENT_FLAGS and label != "mixed" else []
    )
    return {
        "label": label,
        "labels": preserved_labels,
        "polarity": polarity,
        "category_source": category_source,
        "polarity_source": polarity_source,
        "emotion_source": category_source or polarity_source,
        "confidence": confidence,
        "emolex_labels": sorted(emolex_labels or ()),
    }
