import json
import random
import time

import numpy as np

from revdict.data.corpus import merge_records
from revdict.data.nrc_emolex_source import load_emolex, lookup_emolex
from revdict.data.wiktionary_source import (
    download_raw_wiktextract,
    stream_filtered_entries_from_gzip,
)
from revdict.data.wordnet_source import load_wordnet_senses
from revdict.models.embedder import Embedder
from revdict.paths import INDEX_DIR, RAW_WIKTIONARY_PATH


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


def build_metadata_record(record: dict) -> dict:
    return {
        "headword": record["headword"],
        "pos": record["pos"],
        "definition": record["definition"],
        "examples": record["examples"],
        "source": record["source"],
        "sentiwordnet": record.get("sentiwordnet"),
        "emolex": list(record["emolex"]) if record.get("emolex") else None,
        "synonyms": record.get("synonyms"),
    }


def build(skip_confirm: bool = False) -> None:
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
        download_raw_wiktextract(str(RAW_WIKTIONARY_PATH))
        wiktionary_records = list(stream_filtered_entries_from_gzip(str(RAW_WIKTIONARY_PATH)))
    except Exception as error:
        raise RuntimeError(
            "Failed to download or parse the Wiktionary dump from kaikki.org "
            "(check your internet connection; if a partial download is stuck, "
            f"delete {RAW_WIKTIONARY_PATH} and retry): {error}"
        ) from error

    print("Merging corpus...")
    records = merge_records(wordnet_records, wiktionary_records)
    print(f"Merged corpus: {len(records)} sense records.")

    print("Attaching NRC EmoLex tags...")
    emolex = load_emolex()
    for record in records:
        record["emolex"] = lookup_emolex(record["headword"], emolex)

    try:
        embedder = Embedder()
    except Exception as error:
        raise RuntimeError(
            "Failed to load the BAAI/bge-small-en-v1.5 embedding model (first run "
            f"downloads it from Hugging Face — check your internet connection): {error}"
        ) from error

    sample = random.Random(42).sample(records, min(1000, len(records)))
    sample_texts, _ = group_by_definition(sample)
    start = time.time()
    embedder.encode_passages(sample_texts)
    elapsed = time.time() - start
    eta_seconds = estimate_full_duration(len(sample_texts), elapsed, len(records))
    print(
        f"Benchmark: encoded {len(sample_texts)} unique definitions from a random "
        f"sample in {elapsed:.1f}s -> estimated {eta_seconds / 60:.1f} min for the "
        f"full {len(records)}-record corpus."
    )

    if not skip_confirm:
        answer = input("Proceed with the full build? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            print("Aborted. Re-run `revdict build-index` when ready.")
            return

    print("Embedding full corpus...")
    unique_texts, index_groups = group_by_definition(records)
    vectors = embedder.encode_passages(unique_texts)
    embeddings = np.zeros((len(records), vectors.shape[1]), dtype="float32")
    for group_index, positions in enumerate(index_groups):
        for position in positions:
            embeddings[position] = vectors[group_index]

    print("Writing index to disk...")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_DIR / "embeddings.npy", embeddings)

    word_index: dict[str, list[int]] = {}
    with (INDEX_DIR / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for position, record in enumerate(records):
            meta = build_metadata_record(record)
            f.write(json.dumps(meta) + "\n")
            word_index.setdefault(record["headword"].lower(), []).append(position)

    with (INDEX_DIR / "word_index.json").open("w", encoding="utf-8") as f:
        json.dump(word_index, f)

    print(f"Done. Index written to {INDEX_DIR}")
