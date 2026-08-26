import json
import fcntl
import random
import time
from collections import Counter
from contextlib import contextmanager

import numpy as np

from revdict.data.corpus import merge_records
from revdict.data.literary_frequency_source import (
    NGRAM_RELEASE,
    YEAR_RANGE_END,
    YEAR_RANGE_START,
    compute_literary_frequencies,
    download_raw_ngram_fiction,
    download_raw_ngram_fiction_totalcounts,
)
from revdict.data.nrc_emolex_source import emolex_provenance, load_emolex, lookup_emolex
from revdict.data.wiktionary_source import (
    download_raw_wiktextract,
    stream_filtered_entries_from_gzip,
)
from revdict.data.wordnet_source import load_wordnet_senses, wordnet_provenance
from revdict.index_bundle import (
    build_manifest,
    create_staging_index,
    discard_staging_index,
    publish_staged_index,
    validate_index_directory,
    write_manifest,
)
from revdict.models import phonetics
from revdict.models.embedder import MODEL_NAME as EMBEDDING_MODEL_NAME
from revdict.models.embedder import MODEL_REVISION as EMBEDDING_MODEL_REVISION
from revdict.models.embedder import Embedder
from revdict.models.emotion import CLASSIFIER_MODEL_NAME, CLASSIFIER_MODEL_REVISION
from revdict.models.reranker import MODEL_NAME as RERANKER_MODEL_NAME
from revdict.models.reranker import MODEL_REVISION as RERANKER_MODEL_REVISION
from revdict.paths import (
    INDEX_DIR,
    RAW_NGRAM_FICTION_PATH,
    RAW_NGRAM_FICTION_TOTALCOUNTS_PATH,
    RAW_WIKTIONARY_PATH,
)


def estimate_full_duration(sample_count: int, sample_seconds: float, total_count: int) -> float:
    if sample_count == 0:
        return 0.0
    rate = sample_count / sample_seconds
    return total_count / rate


def group_by_definition(records: list[dict]) -> tuple[list[str], list[list[int]]]:
    text_to_group: dict[str, int] = {}
    unique_texts: list[str] = []
    index_groups: list[list[int]] = []
    for position, record in enumerate(records):
        text = record["definition"]
        if text not in text_to_group:
            text_to_group[text] = len(unique_texts)
            unique_texts.append(text)
            index_groups.append([])
        index_groups[text_to_group[text]].append(position)
    return unique_texts, index_groups


def definitions_and_mapping(records: list[dict]) -> tuple[list[str], np.ndarray]:
    """Deduplicate definitions without allocating one Python list per group."""
    text_to_index: dict[str, int] = {}
    unique_texts: list[str] = []
    record_embedding_indices = np.empty(len(records), dtype="int32")
    for position, record in enumerate(records):
        text = record["definition"]
        embedding_index = text_to_index.get(text)
        if embedding_index is None:
            embedding_index = len(unique_texts)
            text_to_index[text] = embedding_index
            unique_texts.append(text)
        record_embedding_indices[position] = embedding_index
    return unique_texts, record_embedding_indices


def build_metadata_record(record: dict) -> dict:
    return {
        "headword": record["headword"],
        "pos": record["pos"],
        "definition": record["definition"],
        "examples": record["examples"],
        "source": record["source"],
        "sources": record.get("sources") or [record["source"]],
        "sentiwordnet": record.get("sentiwordnet"),
        "emolex": sorted(record["emolex"]) if record.get("emolex") else None,
        "synonyms": record.get("synonyms"),
        "tags": record.get("tags") or [],
        "phonetics": record.get("phonetics"),
        "phonetics_error": record.get("phonetics_error"),
        "synset": record.get("synset"),
        "antonyms": record.get("antonyms") or [],
        "topics": record.get("topics") or [],
        "wiktionary_sense_ids": record.get("wiktionary_sense_ids") or [],
        "wikidata_ids": record.get("wikidata_ids") or [],
        "etymology_number": record.get("etymology_number"),
    }


def build_statistics(
    records: list[dict],
    unique_definition_count: int,
    literary_frequency: dict[str, float],
) -> dict:
    source_counts = Counter(record["source"] for record in records)
    eligible_phonetics = sum(
        " " not in record["headword"] and "-" not in record["headword"]
        for record in records
    )
    phonetics_count = sum(record.get("phonetics") is not None for record in records)
    phonetics_failures = Counter(
        record["phonetics_error"]
        for record in records
        if record.get("phonetics_error") is not None
    )
    return {
        "sources": dict(sorted(source_counts.items())),
        "unique_headwords": len({record["headword"].casefold() for record in records}),
        "unique_definitions": unique_definition_count,
        "duplicate_embedding_rows_avoided": len(records) - unique_definition_count,
        "records_with_examples": sum(bool(record.get("examples")) for record in records),
        "records_with_emolex": sum(bool(record.get("emolex")) for record in records),
        "phonetics": {
            "eligible_records": eligible_phonetics,
            "resolved_records": phonetics_count,
            "failure_reasons": dict(sorted(phonetics_failures.items())),
        },
        "literary_frequency_headwords": len(literary_frequency),
    }


def validate_records(records: list[dict]) -> None:
    if not records:
        raise ValueError("Refusing to publish an empty index")
    for position, record in enumerate(records):
        for field in ("headword", "pos", "definition"):
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise ValueError(f"Record {position} has an invalid {field}")


def validate_build_arrays(
    records: list[dict], embeddings: np.ndarray, record_embedding_indices: np.ndarray
) -> None:
    validate_records(records)
    if embeddings.ndim != 2 or embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        raise ValueError(f"Embedding model returned an invalid shape: {embeddings.shape}")
    if record_embedding_indices.shape != (len(records),):
        raise ValueError("Record-to-embedding mapping has the wrong length")
    if int(record_embedding_indices.min()) < 0 or int(record_embedding_indices.max()) >= len(embeddings):
        raise ValueError("Record-to-embedding mapping contains an invalid reference")
    for start in range(0, len(embeddings), 16_384):
        if not np.isfinite(embeddings[start : start + 16_384]).all():
            raise ValueError("Embedding model returned NaN or infinite values")


@contextmanager
def _build_lock(index_root):
    index_root.mkdir(parents=True, exist_ok=True)
    lock_path = index_root / "build.lock"
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def build(skip_confirm: bool = False, refresh_data: bool = False) -> None:
    if not skip_confirm:
        answer = input(
            "Build a new index? This validates/downloads several GB of datasets "
            "and may take tens of minutes. [y/N] "
        )
        if answer.strip().lower() not in ("y", "yes"):
            print("Aborted. Re-run `revdict build-index` when ready.")
            return
    with _build_lock(INDEX_DIR):
        _build(refresh_data=refresh_data)


def _build(refresh_data: bool = False) -> None:
    print("Loading WordNet + SentiWordNet...")
    try:
        wordnet_records = load_wordnet_senses()
    except Exception as error:
        raise RuntimeError(
            "Failed to load WordNet/SentiWordNet via NLTK (this downloads ~35MB on "
            f"first run — check your internet connection and retry): {error}"
        ) from error

    print("Downloading/streaming Wiktionary data (this may take a while on first run)...")
    try:
        wiktionary_provenance = download_raw_wiktextract(
            str(RAW_WIKTIONARY_PATH), refresh=refresh_data
        )
        wiktionary_records = stream_filtered_entries_from_gzip(str(RAW_WIKTIONARY_PATH))
    except Exception as error:
        raise RuntimeError(
            "Failed to download or parse the Wiktionary dump from kaikki.org "
            f"(the verified cache was left intact; check the error and retry): {error}"
        ) from error

    print("Streaming and merging corpus...")
    try:
        records = merge_records(wordnet_records, wiktionary_records)
    except Exception as error:
        raise RuntimeError(
            f"Failed while parsing the verified Wiktionary dump: {error}"
        ) from error
    validate_records(records)
    print(f"Merged corpus: {len(records)} sense records.")

    print("Precomputing phonetics (syllables, rhyme key, meter)...")
    phonetics_cache: dict[tuple[str, str], tuple[dict | None, str | None]] = {}
    for record in records:
        cache_key = (record["headword"].lower(), record["pos"])
        if cache_key not in phonetics_cache:
            phonetics_cache[cache_key] = phonetics.resolve_with_diagnostic(
                record["headword"], record["pos"]
            )
        record["phonetics"], record["phonetics_error"] = phonetics_cache[cache_key]

    print("Attaching NRC EmoLex tags...")
    emolex = load_emolex()
    for record in records:
        record["emolex"] = lookup_emolex(record["headword"], emolex, record["pos"])

    print(
        "Downloading/streaming Google Books Ngram English Fiction data "
        "(this may take a while on first run)..."
    )
    try:
        ngram_provenance = download_raw_ngram_fiction(
            str(RAW_NGRAM_FICTION_PATH), refresh=refresh_data
        )
        totalcounts_provenance = download_raw_ngram_fiction_totalcounts(
            str(RAW_NGRAM_FICTION_TOTALCOUNTS_PATH), refresh=refresh_data
        )
        headwords = {record["headword"] for record in records}
        literary_frequency = compute_literary_frequencies(
            headwords, str(RAW_NGRAM_FICTION_PATH), str(RAW_NGRAM_FICTION_TOTALCOUNTS_PATH)
        )
    except Exception as error:
        raise RuntimeError(
            "Failed to download or process the Google Books Ngram English Fiction "
            f"data (the verified cache was left intact; check the error and retry): {error}"
        ) from error
    print(f"Literary frequency computed for {len(literary_frequency)} headwords.")

    try:
        embedder = Embedder()
    except Exception as error:
        raise RuntimeError(
            "Failed to load the BAAI/bge-small-en-v1.5 embedding model (first run "
            f"downloads it from Hugging Face — check your internet connection): {error}"
        ) from error

    unique_texts, record_embedding_indices = definitions_and_mapping(records)
    sample = random.Random(42).sample(records, min(1000, len(records)))
    sample_texts, _ = group_by_definition(sample)
    start = time.time()
    embedder.encode_passages(sample_texts)
    elapsed = time.time() - start
    eta_seconds = estimate_full_duration(len(sample_texts), elapsed, len(unique_texts))
    print(
        f"Benchmark: encoded {len(sample_texts)} unique definitions from a random "
        f"sample in {elapsed:.1f}s -> estimated {eta_seconds / 60:.1f} min for the "
        f"full {len(unique_texts)}-definition corpus ({len(records)} records)."
    )

    print("Embedding full corpus...")
    embeddings = embedder.encode_passages(unique_texts)
    validate_build_arrays(records, embeddings, record_embedding_indices)
    statistics = build_statistics(records, len(unique_texts), literary_frequency)
    phonetic_stats = statistics["phonetics"]
    if phonetic_stats["eligible_records"] and not phonetic_stats["resolved_records"]:
        print("Warning: no eligible headwords received phonetic data; phonetic filters will be empty.")

    print("Writing and validating a staged index...")
    staging_dir = create_staging_index(INDEX_DIR)
    try:
        np.save(staging_dir / "embeddings.npy", embeddings)
        np.save(staging_dir / "record_embeddings.npy", record_embedding_indices)

        word_index: dict[str, list[int]] = {}
        with (staging_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
            for position, record in enumerate(records):
                meta = build_metadata_record(record)
                f.write(json.dumps(meta) + "\n")
                word_index.setdefault(record["headword"].lower(), []).append(position)

        with (staging_dir / "word_index.json").open("w", encoding="utf-8") as f:
            json.dump(word_index, f)

        with (staging_dir / "literary_frequency.json").open("w", encoding="utf-8") as f:
            json.dump(literary_frequency, f)

        manifest = build_manifest(
            staging_dir,
            record_count=len(records),
            definition_count=len(unique_texts),
            embeddings=embeddings,
            record_embedding_indices=record_embedding_indices,
            datasets={
                "wiktionary": wiktionary_provenance or {},
                "google_books_english_fiction_1gram": {
                    **(ngram_provenance or {}),
                    "release": NGRAM_RELEASE,
                    "year_range": [YEAR_RANGE_START, YEAR_RANGE_END],
                },
                "google_books_english_fiction_totalcounts": {
                    **(totalcounts_provenance or {}),
                    "release": NGRAM_RELEASE,
                    "year_range": [YEAR_RANGE_START, YEAR_RANGE_END],
                },
                "wordnet": wordnet_provenance(),
                "nrc_emolex": emolex_provenance(),
            },
            models={
                "embedding": {
                    "name": EMBEDDING_MODEL_NAME,
                    "revision": EMBEDDING_MODEL_REVISION,
                },
                "reranker": {
                    "name": RERANKER_MODEL_NAME,
                    "revision": RERANKER_MODEL_REVISION,
                },
                "emotion_classifier": {
                    "name": CLASSIFIER_MODEL_NAME,
                    "revision": CLASSIFIER_MODEL_REVISION,
                },
            },
            statistics=statistics,
        )
        write_manifest(staging_dir, manifest)
        validate_index_directory(staging_dir, verify_hashes=True)
        published_dir = publish_staged_index(INDEX_DIR, staging_dir, manifest)
    except Exception:
        discard_staging_index(INDEX_DIR, staging_dir)
        raise

    print(f"Done. Index build {manifest['build_id']} published from {published_dir}")
