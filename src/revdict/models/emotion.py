from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import threading
import uuid
from collections import OrderedDict
from contextlib import closing
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
    "unknown": "neutral",
}

_SENTIMENT_FLAGS = {"positive", "negative", "neutral"}
_SPECIFIC_EMOTIONS = frozenset(EMOTION_POLARITY) - {"neutral", "unknown"}
CLASSIFIER_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
CLASSIFIER_MODEL_REVISION = "0e1cd914e3d46199ed785853e12b57304e04178b"
CLASSIFIER_BATCH_SIZE = 32
CLASSIFIER_MEMORY_CACHE_SIZE = 4096
SENTIWORDNET_MIN_MARGIN = 0.25
CLASSIFIER_MIN_CONFIDENCE = 0.70
CLASSIFIER_MIN_MARGIN = 0.20
CLASSIFIER_MAX_ENTROPY = 0.78
CLASSIFIER_ALIGNED_MIN_CONFIDENCE = 0.55
CLASSIFIER_ALIGNED_MIN_MARGIN = 0.15
CLASSIFIER_NEUTRAL_MIN_CONFIDENCE = 0.50
CLASSIFIER_NEUTRAL_MIN_MARGIN = 0.15

_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
_GENERAL_EMOTION_CUES = frozenset(
    {
        "affect",
        "affective",
        "emotion",
        "emotional",
        "emotions",
        "feeling",
        "feelings",
        "mood",
        "sentiment",
    }
)
_EMOTION_CUES = {
    "anger": frozenset(
        {"anger", "angry", "enraged", "fury", "hostile", "hostility", "irate", "rage", "wrath"}
    ),
    "anticipation": frozenset(
        {"anticipate", "anticipation", "eager", "expect", "expectation", "foresee", "suspense"}
    ),
    "disgust": frozenset(
        {
            "aversion",
            "disgust",
            "disgusted",
            "dislike",
            "distaste",
            "loathing",
            "nauseated",
            "repelled",
            "revulsion",
        }
    ),
    "fear": frozenset(
        {"afraid", "anxious", "anxiety", "dread", "fear", "fearful", "fright", "scared", "terror"}
    ),
    "joy": frozenset(
        {"delight", "elation", "euphoric", "euphoria", "glad", "happiness", "happy", "joy", "joyful"}
    ),
    "sadness": frozenset(
        {"dejected", "despair", "grief", "melancholy", "mourn", "sad", "sadness", "sorrow", "unhappy"}
    ),
    "surprise": frozenset(
        {"amazed", "astonished", "astonishment", "startled", "surprise", "surprised", "unexpected"}
    ),
    "trust": frozenset(
        {
            "awe",
            "believe",
            "believing",
            "confidence",
            "faith",
            "honest",
            "honesty",
            "reliability",
            "reliable",
            "respect",
            "reverence",
            "trust",
        }
    ),
}


def polarity_from_sentiwordnet(scores: dict) -> str:
    pos, neg = float(scores["pos"]), float(scores["neg"])
    if abs(pos - neg) < SENTIWORDNET_MIN_MARGIN:
        return "neutral"
    return "positive" if pos > neg else "negative"


def specific_emolex_labels(labels: frozenset[str] | None) -> list[str]:
    return sorted(label for label in labels or () if label not in _SENTIMENT_FLAGS)


def format_emotion_label(primary: str, labels: list[str]) -> str:
    if primary == "mixed" and len(labels) > 1:
        return f"mixed ({'/'.join(labels)})"
    return primary


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
        self._memory_cache: OrderedDict[str, dict[str, float]] = OrderedDict()

    def _remember(self, key: str, scores: dict[str, float]) -> None:
        self._memory_cache[key] = scores
        self._memory_cache.move_to_end(key)
        while len(self._memory_cache) > CLASSIFIER_MEMORY_CACHE_SIZE:
            self._memory_cache.popitem(last=False)

    def _pipeline(self):
        if self._pipe is None:
            self._pipe = self._pipeline_factory(
                "text-classification",
                model=CLASSIFIER_MODEL_NAME,
                revision=CLASSIFIER_MODEL_REVISION,
                top_k=None,
                # Transformers otherwise defaults list inference to batches
                # of one.  The result definitions are independent inputs, so
                # true CPU batching preserves the pinned model's scores while
                # avoiding one forward-pass setup per displayed record.
                batch_size=CLASSIFIER_BATCH_SIZE,
            )
        return self._pipe

    @staticmethod
    def _cache_key(text: str) -> str:
        material = f"{CLASSIFIER_MODEL_REVISION}\0{text}".encode("utf-8")
        return hashlib.sha256(material).hexdigest()

    def _read_cache(self, key: str) -> dict[str, float] | None:
        return self._read_cache_many([key]).get(key)

    def _read_cache_many(self, keys: list[str]) -> dict[str, dict[str, float]]:
        if not keys:
            return {}
        if not self._cache_path.is_file():
            return {}
        rows = []
        with closing(sqlite3.connect(self._cache_path)) as connection:
            with connection:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS predictions "
                    "(cache_key TEXT PRIMARY KEY, scores_json TEXT NOT NULL)"
                )
                # Stay below SQLite builds' conservative host-parameter limit.
                for start in range(0, len(keys), 500):
                    chunk = keys[start : start + 500]
                    placeholders = ",".join("?" for _key in chunk)
                    rows.extend(
                        connection.execute(
                            "SELECT cache_key, scores_json FROM predictions "
                            f"WHERE cache_key IN ({placeholders})",
                            chunk,
                        )
                    )
        found = {}
        for cache_key, raw_scores in rows:
            try:
                value = json.loads(raw_scores)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                found[cache_key] = {
                    str(label): float(score) for label, score in value.items()
                }
        return found

    def _write_cache(self, key: str, scores: dict[str, float]) -> None:
        self._write_cache_many({key: scores})

    def _write_cache_many(self, entries: dict[str, dict[str, float]]) -> None:
        if not entries:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self._cache_path)) as connection:
            with connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS predictions "
                    "(cache_key TEXT PRIMARY KEY, scores_json TEXT NOT NULL)"
                )
                connection.executemany(
                    "INSERT OR REPLACE INTO predictions(cache_key, scores_json) VALUES (?, ?)",
                    (
                        (key, json.dumps(scores, sort_keys=True))
                        for key, scores in entries.items()
                    ),
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
        with self._lock:
            keys = {text: self._cache_key(text) for text in unique_texts}
            disk_keys = [key for key in keys.values() if key not in self._memory_cache]
            try:
                disk_scores = self._read_cache_many(disk_keys)
            except sqlite3.DatabaseError:
                self._quarantine_corrupt_cache()
                disk_scores = {}

            missing = []
            for text in unique_texts:
                key = keys[text]
                scores = self._memory_cache.get(key)
                if scores is not None:
                    self._memory_cache.move_to_end(key)
                if scores is None:
                    scores = disk_scores.get(key)
                if scores is None:
                    missing.append(text)
                    continue
                self._remember(key, scores)
                predictions[text] = self._prediction(scores)

            if missing:
                raw_groups = self._pipeline()(missing)
                if len(missing) == 1 and raw_groups and isinstance(raw_groups[0], dict):
                    raw_groups = [raw_groups]
                pending_writes = {}
                for text, results in zip(missing, raw_groups, strict=True):
                    scores = {
                        str(item["label"]).lower(): float(item["score"])
                        for item in cast(list[dict], results)
                    }
                    if not scores:
                        raise RuntimeError("Emotion classifier returned no labels")
                    key = keys[text]
                    self._remember(key, scores)
                    pending_writes[key] = scores
                    predictions[text] = self._prediction(scores)
                try:
                    self._write_cache_many(pending_writes)
                except sqlite3.DatabaseError:
                    self._quarantine_corrupt_cache()
                    self._write_cache_many(pending_writes)
        return [predictions[text] for text in texts]

    @staticmethod
    def _prediction(scores: dict[str, float]) -> dict:
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        label, confidence = ranked[0]
        margin = confidence - ranked[1][1] if len(ranked) > 1 else confidence
        total = sum(max(score, 0.0) for score in scores.values())
        probabilities = [max(score, 0.0) / total for score in scores.values()] if total else []
        entropy = -sum(p * math.log(p) for p in probabilities if p > 0.0)
        normalized_entropy = entropy / math.log(len(probabilities)) if len(probabilities) > 1 else 0.0
        return {
            "label": label,
            "polarity": EMOTION_POLARITY.get(label, "neutral"),
            "confidence": confidence,
            "margin": margin,
            "entropy": normalized_entropy,
            "scores": scores,
        }

def _normalize_classifier_result(result) -> dict | None:
    if result is None:
        return None
    if isinstance(result, tuple):
        label, polarity = result
        return {
            "label": label,
            "polarity": polarity,
            "confidence": None,
            "margin": None,
            "entropy": None,
            "scores": {},
        }
    normalized = dict(result)
    scores = normalized.get("scores") or {}
    if "margin" not in normalized:
        ranked = sorted((float(score) for score in scores.values()), reverse=True)
        normalized["margin"] = ranked[0] - ranked[1] if len(ranked) > 1 else None
    if "entropy" not in normalized:
        total = sum(max(float(score), 0.0) for score in scores.values())
        probabilities = [max(float(score), 0.0) / total for score in scores.values()] if total else []
        entropy = -sum(p * math.log(p) for p in probabilities if p > 0.0)
        normalized["entropy"] = (
            entropy / math.log(len(probabilities)) if len(probabilities) > 1 else None
        )
    return normalized


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


def _definition_has_emotion_cue(definition: str, labels: list[str] | None = None) -> bool:
    words = frozenset(_WORD_RE.findall(definition.casefold()))
    if words & _GENERAL_EMOTION_CUES:
        return True
    relevant = labels or list(_SPECIFIC_EMOTIONS)
    return any(words & _EMOTION_CUES.get(label, frozenset()) for label in relevant)


def _definition_emotion_labels(definition: str) -> list[str]:
    words = frozenset(_WORD_RE.findall(definition.casefold()))
    return sorted(label for label, cues in _EMOTION_CUES.items() if words & cues)


def _classifier_passes(
    result: dict,
    *,
    min_confidence: float,
    min_margin: float,
    max_entropy: float = CLASSIFIER_MAX_ENTROPY,
) -> bool:
    confidence = result.get("confidence")
    if confidence is None:
        # Compatibility for deterministic/test classifiers that return a
        # category-polarity tuple without a probability distribution.
        return True
    if float(confidence) < min_confidence:
        return False
    margin = result.get("margin")
    if margin is not None and float(margin) < min_margin:
        return False
    entropy = result.get("entropy")
    return entropy is None or float(entropy) <= max_entropy


def _sentiwordnet_evidence(record: dict) -> tuple[str, bool]:
    scores = record.get("sentiwordnet")
    if record.get("source") != "wordnet" or scores is None:
        return "neutral", False
    return polarity_from_sentiwordnet(scores), True


def _lexicon_polarity(labels: frozenset[str] | None, specific: list[str]) -> str:
    flagged = _polarity_from_emolex_flags(labels)
    if flagged != "neutral":
        return flagged
    polarities = {EMOTION_POLARITY[label] for label in specific if label in EMOTION_POLARITY}
    return polarities.pop() if len(polarities) == 1 else "neutral"


def _classifier_decision(
    record: dict,
    result: dict | None,
    specific: list[str],
    senti_polarity: str,
    has_sentiwordnet: bool,
) -> tuple[bool, str]:
    if result is None:
        return False, "unavailable"
    label = str(result.get("label", "")).casefold()
    if label not in EMOTION_POLARITY:
        return False, "unsupported-label"
    definition_labels = _definition_emotion_labels(record["definition"])

    if label == "neutral":
        accepted = _classifier_passes(
            result,
            min_confidence=CLASSIFIER_NEUTRAL_MIN_CONFIDENCE,
            min_margin=CLASSIFIER_NEUTRAL_MIN_MARGIN,
            max_entropy=0.85,
        )
        if accepted and (
            definition_labels
            or (specific and _definition_has_emotion_cue(record["definition"], specific))
        ):
            return False, "lexicon-cue-preferred"
        return accepted, "neutral-abstention" if accepted else "weak-neutral"

    aligned = label in specific
    accepted_quality = _classifier_passes(
        result,
        min_confidence=(
            CLASSIFIER_ALIGNED_MIN_CONFIDENCE if aligned else CLASSIFIER_MIN_CONFIDENCE
        ),
        min_margin=CLASSIFIER_ALIGNED_MIN_MARGIN if aligned else CLASSIFIER_MIN_MARGIN,
        max_entropy=0.85 if aligned else CLASSIFIER_MAX_ENTROPY,
    )
    if not accepted_quality:
        return False, "weak-prediction"

    classifier_polarity = EMOTION_POLARITY[label]
    canonical_headword = str(record.get("headword", "")).strip().casefold()
    canonical_prior = canonical_headword if canonical_headword in specific else None

    unsupported_full_taxonomy_cues = [
        cue
        for cue in definition_labels
        if cue in {"trust", "anticipation"} and cue != label
    ]
    if unsupported_full_taxonomy_cues and any(
        not has_sentiwordnet
        or senti_polarity == "neutral"
        or EMOTION_POLARITY[cue] == senti_polarity
        for cue in unsupported_full_taxonomy_cues
    ):
        return False, "full-taxonomy-cue-preferred"

    if aligned:
        # For words that are themselves canonical emotion names, prefer that
        # exact category over a sibling category with the same polarity. This
        # prevents senses of "disgust" from becoming "anger" merely because
        # the seven-label classifier conflates nearby negative emotions.
        if (
            canonical_prior is not None
            and label != canonical_prior
            and EMOTION_POLARITY[canonical_prior] == classifier_polarity
        ):
            return False, "canonical-prior-preferred"
        return True, "classifier-lexicon-agreement"

    if specific:
        # A conflicting sense prediction may override the word-level prior
        # only when sense-specific SentiWordNet independently supports its
        # polarity. This retains legitimate polysemy such as fear=reverence
        # while rejecting unsupported category swaps.
        if (
            has_sentiwordnet
            and senti_polarity != "neutral"
            and senti_polarity == classifier_polarity
        ):
            return True, "classifier-sense-override"
        return False, "lexicon-conflict"

    if has_sentiwordnet:
        if senti_polarity != "neutral" and senti_polarity == classifier_polarity:
            return True, "classifier-sentiwordnet-agreement"
        return False, "objective-sense"

    # Wiktionary senses have no SentiWordNet evidence. Require both a very
    # strong distribution and explicit affective language in the definition;
    # confidence alone is unsafe under the dictionary-definition domain shift.
    strong = _classifier_passes(
        result, min_confidence=0.85, min_margin=0.40, max_entropy=0.60
    )
    if strong and _definition_has_emotion_cue(record["definition"], [label]):
        return True, "classifier-definition-cue"
    return False, "unsupported-domain-prediction"


def _resolved_labels(label: str, specific: list[str]) -> list[str]:
    if label == "mixed":
        return specific
    if label in _SPECIFIC_EMOTIONS:
        return [label]
    return []


def tag_emotion(record: dict, classifier_factory) -> dict:
    """Resolve one coherent sense badge from independent evidence.

    NRC EmoLex is a word-level prior, never a second set of display labels.
    The classifier must pass confidence, separation, entropy, and independent
    sense-evidence gates. Unsupported conflicts abstain instead of being
    forced into a plausible-looking but contradictory category.
    """
    raw_emolex = record.get("emolex")
    emolex_labels = frozenset(raw_emolex) if raw_emolex else None
    specific = specific_emolex_labels(emolex_labels)
    classifier_result = None
    if classifier_factory is not None:
        classifier_result = _normalize_classifier_result(
            classifier_factory().classify(record["definition"])
        )

    senti_polarity, has_sentiwordnet = _sentiwordnet_evidence(record)
    classifier_accepted, classifier_reason = _classifier_decision(
        record, classifier_result, specific, senti_polarity, has_sentiwordnet
    )

    if classifier_accepted:
        label = str(classifier_result["label"]).casefold()
        category_source = "classifier"
        confidence = classifier_result.get("confidence")
    else:
        confidence = None
        category_source = None
        canonical_headword = str(record.get("headword", "")).strip().casefold()
        cue_labels = [canonical_headword] if canonical_headword in specific else specific
        has_definition_cue = _definition_has_emotion_cue(record["definition"], cue_labels)
        definition_labels = _definition_emotion_labels(record["definition"])
        lexicon_polarity = _lexicon_polarity(emolex_labels, specific)

        supported_definition_labels = [
            definition_label
            for definition_label in definition_labels
            if (
                definition_label in specific
                or not specific
                or (
                    has_sentiwordnet
                    and senti_polarity != "neutral"
                    and EMOTION_POLARITY[definition_label] == senti_polarity
                )
            )
            and not (
                has_sentiwordnet
                and senti_polarity != "neutral"
                and EMOTION_POLARITY[definition_label] != senti_polarity
            )
        ]

        if len(supported_definition_labels) == 1:
            label = supported_definition_labels[0]
            category_source = "definition-cue"
        elif has_sentiwordnet and senti_polarity == "neutral" and not has_definition_cue:
            label = "neutral"
            category_source = "abstained"
            classifier_reason = "objective-sense"
        elif len(specific) == 1 and (
            has_definition_cue
            or (
                has_sentiwordnet
                and senti_polarity != "neutral"
                and senti_polarity == lexicon_polarity
            )
        ):
            label = specific[0]
            category_source = "emolex-prior"
        elif len(specific) > 1 and has_definition_cue:
            label = canonical_headword if canonical_headword in specific else "mixed"
            category_source = "emolex-prior"
        elif specific:
            label = "neutral" if senti_polarity == "neutral" else "unknown"
            category_source = "abstained"
        elif has_sentiwordnet and senti_polarity != "neutral":
            label = "unknown"
            category_source = "abstained"
        else:
            label = "neutral"
            category_source = "abstained" if classifier_result is not None else None

    if label in _SPECIFIC_EMOTIONS or label in {"neutral", "unknown"}:
        polarity = EMOTION_POLARITY[label]
        polarity_source = category_source
    elif label == "mixed":
        polarity = _lexicon_polarity(emolex_labels, specific)
        polarity_source = "emolex-prior"
    else:
        polarity, polarity_source = _fallback_polarity(record, emolex_labels)

    resolved_labels = _resolved_labels(label, specific)
    status = (
        "resolved"
        if label in _SPECIFIC_EMOTIONS
        else "ambiguous"
        if label == "mixed"
        else "abstained"
    )
    classifier_evidence = None
    if classifier_result is not None:
        classifier_evidence = {
            "label": classifier_result.get("label"),
            "confidence": classifier_result.get("confidence"),
            "margin": classifier_result.get("margin"),
            "entropy": classifier_result.get("entropy"),
            "accepted": classifier_accepted,
            "reason": classifier_reason,
        }
    return {
        "label": label,
        "labels": resolved_labels,
        "polarity": polarity,
        "category_source": category_source,
        "polarity_source": polarity_source,
        "emotion_source": category_source or "none",
        "confidence": confidence,
        "emolex_labels": sorted(emolex_labels or ()),
        "status": status,
        "evidence": {
            "classifier": classifier_evidence,
            "emolex": {
                "specific_labels": specific,
                "sentiment": _polarity_from_emolex_flags(emolex_labels),
            }
            if emolex_labels
            else None,
            "sentiwordnet": {
                "polarity": senti_polarity,
                "available": has_sentiwordnet,
            },
        },
    }
