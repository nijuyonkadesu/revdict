# Reverse Dictionary CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `revdict`, a local offline CLI that (1) shows standard dictionary entries for real words, (2) always additionally runs a semantic reverse-dictionary search that suggests candidate words matching a description of meaning, and (3) tags every word shown with an emotion/connotation badge — all served from a prebuilt local index, no network calls at query time.

**Architecture:** A one-time `revdict build-index` command merges WordNet (via NLTK) and a filtered Wiktionary extract (via kaikki.org's raw wiktextract dump) into one corpus, embeds every definition with a small bi-encoder, and writes the result to `~/.cache/rev_dictionary/`. Everyday queries embed the input, retrieve top-75 candidates by cosine similarity, rerank with a cross-encoder, dedupe by headword, tag emotion (SentiWordNet + NRC EmoLex first, a small classifier only where those lexicons don't cover a word), and present results through a real `fzf` picker with a live preview pane.

**Tech Stack:** Python 3.13, `nltk` (WordNet + SentiWordNet), a filtered kaikki.org wiktextract dump, `sentence-transformers` (bi-encoder `BAAI/bge-small-en-v1.5` + cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2`), `nrclex` (bundled NRC EmoLex data), `transformers` (`j-hartmann/emotion-english-distilroberta-base`, fallback only), `numpy`, `rich`, real `fzf` binary (already installed), `pytest`.

## Global Constraints

- CPU-only inference — this machine has no NVIDIA GPU (AMD integrated graphics only). Always install `torch` via `pip install torch --index-url https://download.pytorch.org/whl/cpu` **before** installing the rest of the project's dependencies, so pip never pulls the default CUDA-bundled torch wheel and its multi-GB `nvidia-*` dependency packages.
- No network calls at query time — network access only happens inside `revdict build-index` (downloading NLTK corpora, the kaikki.org dump, and HuggingFace model weights on first use).
- Emotion tagging: **SentiWordNet supplies polarity** (its real strength — it's a real-valued score per WordNet sense), **NRC EmoLex supplies specific emotion categories** where it has the word (~6,500 words), and the **transformer classifier is the fallback for a specific category only when EmoLex doesn't cover the word** — these three sources are *combined*, not tried as a strict fallback chain, per the corrected design in Task 8.
- Every part-of-speech string shown to the user uses the vocabulary `noun`/`verb`/`adjective`/`adverb` (WordNet's native vocabulary) — Wiktionary's abbreviated `adj`/`adv` codes are normalized to match; other Wiktionary POS values (e.g. `article`, `intj`, `prefix`) pass through unchanged since WordNet has no equivalent to normalize against.
- Index and raw-data caches live under `~/.cache/rev_dictionary/` (outside the git repo) — never commit generated index artifacts.
- Deviation from the approved spec, noted here for the record: the spec suggested WordNet via the `wn` package. This plan uses NLTK's `wordnet` + `sentiwordnet` corpora instead, verified empirically to share synset identifiers directly (e.g. `happy.a.01`) — the `wn` package's Open English WordNet has diverged from Princeton WordNet 3.0 since forking, which SentiWordNet is built against, and mixing them would need a lossy ID-mapping step. This still delivers the spec's actual requirement ("SentiWordNet aligns directly with the WordNet senses already loaded") more faithfully than the literal package name it mentioned.
- Second deviation from the spec's component list, noted for the record: the spec listed `sentiwordnet_source.py` as its own file. This plan folds SentiWordNet loading directly into `wordnet_source.py` (Task 2) instead, since NLTK loads both corpora from the same synset iteration — writing them to two files would mean iterating all ~117k synsets twice for no benefit.
- Any failure to download data or model weights during `revdict build-index` (WordNet/SentiWordNet via NLTK, the Wiktionary dump, or the bi-encoder from Hugging Face) must surface as a clear error naming which specific resource failed, per the spec's error-handling section — see the `try`/`except` wrapping in Task 10.

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/revdict/__init__.py`
- Create: `src/revdict/paths.py`
- Create: `src/revdict/data/__init__.py`
- Create: `src/revdict/models/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/data/__init__.py`
- Create: `tests/models/__init__.py`

**Interfaces:**
- Produces: `revdict.paths.CACHE_DIR: Path`, `revdict.paths.INDEX_DIR: Path`, `revdict.paths.RAW_WIKTIONARY_PATH: Path` — every later task that touches disk storage imports these three constants from here instead of redefining them.

- [ ] **Step 1: Create the directory layout**

```bash
mkdir -p src/revdict/data src/revdict/models tests/data tests/models
touch src/revdict/__init__.py src/revdict/data/__init__.py src/revdict/models/__init__.py
touch tests/__init__.py tests/data/__init__.py tests/models/__init__.py
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "revdict"
version = "0.1.0"
description = "Local offline reverse-dictionary CLI"
requires-python = ">=3.11"
dependencies = [
    "nltk>=3.9",
    "sentence-transformers>=5.0",
    "transformers>=5.0",
    "nrclex>=4.1",
    "numpy>=1.26",
    "rich>=13.9",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[project.scripts]
revdict = "revdict.cli:main"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 3: Write `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
```

- [ ] **Step 4: Write `src/revdict/paths.py`**

```python
from pathlib import Path

CACHE_DIR = Path.home() / ".cache" / "rev_dictionary"
INDEX_DIR = CACHE_DIR / "index"
RAW_WIKTIONARY_PATH = CACHE_DIR / "raw-wiktextract-data.jsonl.gz"
```

- [ ] **Step 5: Create the virtual environment**

```bash
python3 -m venv .venv
```

- [ ] **Step 6: Install torch (CPU-only build) before anything else**

```bash
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Expected: installs `torch` (a `+cpu`-less but CPU-only manylinux wheel from this dedicated index, no `nvidia-*` packages pulled in). Confirmed available for this machine's `cp313`/`x86_64` combination.

- [ ] **Step 7: Install the project in editable mode with dev dependencies**

```bash
.venv/bin/pip install -e ".[dev]"
```

Expected: pulls in `nltk`, `sentence-transformers`, `transformers`, `nrclex`, `numpy`, `rich`, `pytest` — and since `torch` is already installed satisfying `sentence-transformers`'/`transformers`' version requirements, pip will not replace it with the default CUDA-bundled wheel.

- [ ] **Step 8: Verify the package imports**

```bash
.venv/bin/python -c "import revdict; from revdict.paths import INDEX_DIR; print(INDEX_DIR)"
```

Expected: prints something like `/home/shichika/.cache/rev_dictionary/index` with no errors.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml .gitignore src tests
git commit -m "Scaffold revdict package, venv, and CPU-only dependency setup"
```

---

### Task 2: WordNet + SentiWordNet loader

**Files:**
- Create: `src/revdict/data/wordnet_source.py`
- Test: `tests/data/test_wordnet_source.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (uses `nltk` directly).
- Produces: `load_wordnet_senses() -> list[dict]`, where each dict has keys `headword: str`, `pos: str` (one of `noun`/`verb`/`adjective`/`adverb`), `definition: str`, `examples: list[str]`, `source: "wordnet"`, `synset: str`, `synonyms: list[str]`, `sentiwordnet: {"pos": float, "neg": float, "obj": float} | None`. Later tasks (`corpus.py`, `build_index.py`) consume this list directly.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_wordnet_source.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/data/test_wordnet_source.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'revdict.data.wordnet_source'`

- [ ] **Step 3: Write the implementation**

```python
# src/revdict/data/wordnet_source.py
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn

_POS_NAMES = {"n": "noun", "v": "verb", "a": "adjective", "s": "adjective", "r": "adverb"}


def _ensure_nltk_data() -> None:
    for package in ("wordnet", "sentiwordnet"):
        try:
            nltk.data.find(f"corpora/{package}")
        except LookupError:
            nltk.download(package, quiet=True)


def load_wordnet_senses() -> list[dict]:
    _ensure_nltk_data()
    records = []
    for synset in wn.all_synsets():
        pos = _POS_NAMES.get(synset.pos(), synset.pos())
        definition = synset.definition()
        examples = list(synset.examples())
        lemma_names = [name.replace("_", " ") for name in synset.lemma_names()]

        try:
            senti = swn.senti_synset(synset.name())
            sentiwordnet = {
                "pos": senti.pos_score(),
                "neg": senti.neg_score(),
                "obj": senti.obj_score(),
            }
        except (LookupError, ValueError):
            sentiwordnet = None

        for lemma in lemma_names:
            records.append(
                {
                    "headword": lemma,
                    "pos": pos,
                    "definition": definition,
                    "examples": examples,
                    "source": "wordnet",
                    "synset": synset.name(),
                    "synonyms": [name for name in lemma_names if name != lemma],
                    "sentiwordnet": sentiwordnet,
                }
            )
    return records
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/data/test_wordnet_source.py -v`
Expected: PASS (this run downloads `wordnet` and `sentiwordnet` NLTK corpora on first execution, ~35MB; the function itself takes a few seconds to iterate all ~117k synsets — this is expected and only needs to run once per test session)

- [ ] **Step 5: Commit**

```bash
git add src/revdict/data/wordnet_source.py tests/data/test_wordnet_source.py
git commit -m "Add WordNet + SentiWordNet loader"
```

---

### Task 3: NRC EmoLex loader

**Files:**
- Create: `src/revdict/data/nrc_emolex_source.py`
- Test: `tests/data/test_nrc_emolex_source.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (uses the `nrclex` package's bundled data file directly, not its `NRCLex` text-analysis class).
- Produces: `load_emolex() -> dict[str, frozenset[str]]` (word, lowercase, to a set of labels drawn from `{anger, anticipation, disgust, fear, joy, sadness, surprise, trust, positive, negative}`), `lookup_emolex(word: str, emolex: dict[str, frozenset[str]]) -> frozenset[str] | None`. Task 8 (`emotion.py`) and Task 10 (`build_index.py`) consume both functions.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/data/test_nrc_emolex_source.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'revdict.data.nrc_emolex_source'`

- [ ] **Step 3: Write the implementation**

```python
# src/revdict/data/nrc_emolex_source.py
import importlib.resources
import json


def load_emolex() -> dict[str, frozenset[str]]:
    data_path = importlib.resources.files("nrclex.data").joinpath("nrc_en.json")
    raw = json.loads(data_path.read_text(encoding="utf-8"))
    return {word: frozenset(labels) for word, labels in raw.items()}


def lookup_emolex(word: str, emolex: dict[str, frozenset[str]]) -> frozenset[str] | None:
    return emolex.get(word.lower())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/data/test_nrc_emolex_source.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/revdict/data/nrc_emolex_source.py tests/data/test_nrc_emolex_source.py
git commit -m "Add NRC EmoLex loader"
```

---

### Task 4: Wiktionary source loader and filter

**Files:**
- Create: `src/revdict/data/wiktionary_source.py`
- Test: `tests/data/test_wiktionary_source.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `iter_filtered_entries(lines: Iterable[str]) -> Iterator[dict]`, `parse_filtered_entries(lines: Iterable[str]) -> list[dict]`, `stream_filtered_entries_from_gzip(path: str) -> Iterator[dict]`, `download_raw_wiktextract(dest_path: str) -> None`. Each yielded/returned dict has keys `headword: str`, `pos: str` (normalized to `noun`/`verb`/`adjective`/`adverb` where applicable, otherwise passed through as-is), `definition: str`, `examples: list[str]`, `source: "wiktionary"`. Task 5 (`corpus.py`) and Task 10 (`build_index.py`) consume these.

- [ ] **Step 1: Write the failing test**

Real schema confirmed against the live kaikki.org dump (`https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz`): top-level keys include `word`, `pos`, `lang`, `lang_code`, `senses` (each with `glosses`, `tags`, `examples`), `forms`. Inflected/derived forms are marked with `"form-of"` in a sense's `tags` list and carry a `form_of` key. English POS values seen in the wild: `adj`, `adv`, `article`, `character`, `det`, `intj`, `name`, `noun`, `num`, `particle`, `phrase`, `prefix`, `prep`, `pron`, `symbol`, `verb`.

```python
# tests/data/test_wiktionary_source.py
import gzip
import tempfile
from pathlib import Path

from revdict.data.wiktionary_source import (
    parse_filtered_entries,
    stream_filtered_entries_from_gzip,
)

ENGLISH_NOUN_LINE = (
    '{"word": "dictionary", "pos": "noun", "lang": "English", "lang_code": "en", '
    '"senses": [{"glosses": ["A reference work listing words and explaining their meanings."], '
    '"examples": [{"text": "a law dictionary"}]}]}'
)
ENGLISH_ADJ_LINE = (
    '{"word": "green with envy", "pos": "adj", "lang": "English", "lang_code": "en", '
    '"senses": [{"glosses": ["Very jealous."]}]}'
)
ENGLISH_FORM_OF_LINE = (
    '{"word": "dictionaries", "pos": "noun", "lang": "English", "lang_code": "en", '
    '"senses": [{"glosses": ["plural of dictionary"], "tags": ["form-of", "plural"], '
    '"form_of": [{"word": "dictionary"}]}]}'
)
NON_ENGLISH_LINE = (
    '{"word": "diccionario", "pos": "noun", "lang": "Spanish", "lang_code": "es", '
    '"senses": [{"glosses": ["Un diccionario"]}]}'
)


def test_parse_filtered_entries_keeps_english_drops_form_of_and_non_english():
    lines = [ENGLISH_NOUN_LINE, ENGLISH_ADJ_LINE, ENGLISH_FORM_OF_LINE, NON_ENGLISH_LINE]
    records = parse_filtered_entries(lines)
    headwords = {r["headword"] for r in records}
    assert headwords == {"dictionary", "green with envy"}

    dictionary_record = next(r for r in records if r["headword"] == "dictionary")
    assert dictionary_record["pos"] == "noun"
    assert "reference work" in dictionary_record["definition"]
    assert dictionary_record["examples"] == ["a law dictionary"]
    assert dictionary_record["source"] == "wiktionary"

    idiom_record = next(r for r in records if r["headword"] == "green with envy")
    assert idiom_record["pos"] == "adjective"


def test_stream_filtered_entries_from_gzip_reads_a_real_gz_file():
    with tempfile.TemporaryDirectory() as tmp:
        gz_path = Path(tmp) / "sample.jsonl.gz"
        with gzip.open(gz_path, "wt", encoding="utf-8") as f:
            f.write(ENGLISH_NOUN_LINE + "\n")
            f.write(NON_ENGLISH_LINE + "\n")
        records = list(stream_filtered_entries_from_gzip(str(gz_path)))
        assert len(records) == 1
        assert records[0]["headword"] == "dictionary"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/data/test_wiktionary_source.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'revdict.data.wiktionary_source'`

- [ ] **Step 3: Write the implementation**

```python
# src/revdict/data/wiktionary_source.py
import gzip
import json
import urllib.request
from pathlib import Path
from typing import Iterable, Iterator

RAW_WIKTEXTRACT_URL = "https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz"

_POS_NORMALIZATION = {"adj": "adjective", "adv": "adverb"}


def _normalize_pos(pos: str) -> str:
    return _POS_NORMALIZATION.get(pos, pos)


def iter_filtered_entries(lines: Iterable[str]) -> Iterator[dict]:
    for line in lines:
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        if entry.get("lang_code") != "en":
            continue
        word = entry.get("word")
        pos = entry.get("pos")
        if not word or not pos:
            continue
        for sense in entry.get("senses", []):
            tags = sense.get("tags") or []
            if "form-of" in tags or "form_of" in sense:
                continue
            glosses = sense.get("glosses") or []
            if not glosses:
                continue
            examples = [
                example.get("text", "")
                for example in sense.get("examples", [])
                if example.get("text")
            ]
            yield {
                "headword": word,
                "pos": _normalize_pos(pos),
                "definition": glosses[0],
                "examples": examples,
                "source": "wiktionary",
            }


def parse_filtered_entries(lines: Iterable[str]) -> list[dict]:
    return list(iter_filtered_entries(lines))


def stream_filtered_entries_from_gzip(path: str) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        yield from iter_filtered_entries(f)


def download_raw_wiktextract(dest_path: str) -> None:
    dest = Path(dest_path)
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(RAW_WIKTEXTRACT_URL, tmp)
    tmp.rename(dest)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/data/test_wiktionary_source.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/revdict/data/wiktionary_source.py tests/data/test_wiktionary_source.py
git commit -m "Add Wiktionary (kaikki.org) loader with English/form-of filtering"
```

---

### Task 5: Corpus merge

**Files:**
- Create: `src/revdict/data/corpus.py`
- Test: `tests/data/test_corpus.py`

**Interfaces:**
- Consumes: lists of dicts shaped like Task 2's and Task 4's output (only the `headword` and `definition` keys are inspected; everything else passes through untouched).
- Produces: `merge_records(wordnet_records: list[dict], wiktionary_records: list[dict]) -> list[dict]`. Task 10 (`build_index.py`) consumes this.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_corpus.py
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
    assert merged[1]["definition"] == "Fortunate and convenient."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/data/test_corpus.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'revdict.data.corpus'`

- [ ] **Step 3: Write the implementation**

```python
# src/revdict/data/corpus.py
def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def merge_records(wordnet_records: list[dict], wiktionary_records: list[dict]) -> list[dict]:
    seen = {
        (_normalize(record["headword"]), _normalize(record["definition"]))
        for record in wordnet_records
    }
    merged = list(wordnet_records)
    for record in wiktionary_records:
        key = (_normalize(record["headword"]), _normalize(record["definition"]))
        if key in seen:
            continue
        seen.add(key)
        merged.append(record)
    return merged
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/data/test_corpus.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/revdict/data/corpus.py tests/data/test_corpus.py
git commit -m "Add corpus merge/dedupe logic"
```

---

### Task 6: Bi-encoder embedder wrapper

**Files:**
- Create: `src/revdict/models/embedder.py`
- Test: `tests/models/test_embedder.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `format_query_text(query: str) -> str`, class `Embedder` with `encode_passages(texts: list[str]) -> np.ndarray` (shape `(N, 384)`, `float32`, L2-normalized, no instruction prefix) and `encode_query(query: str) -> np.ndarray` (shape `(384,)`, `float32`, L2-normalized, instruction-prefixed). Task 10 (`build_index.py`) and Task 11 (`search.py`) consume `Embedder`.

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_embedder.py
from revdict.models.embedder import format_query_text


def test_format_query_text_prepends_the_bge_retrieval_instruction():
    result = format_query_text("feeling of intense annoyance")
    assert result == (
        "Represent this sentence for searching relevant passages: "
        "feeling of intense annoyance"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/models/test_embedder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'revdict.models.embedder'`

- [ ] **Step 3: Write the implementation**

```python
# src/revdict/models/embedder.py
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def format_query_text(query: str) -> str:
    return QUERY_INSTRUCTION + query


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME):
        self._model = SentenceTransformer(model_name)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return vectors.astype("float32")

    def encode_query(self, query: str) -> np.ndarray:
        vector = self._model.encode(
            [format_query_text(query)],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        return vector.astype("float32")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/models/test_embedder.py -v`
Expected: PASS (this test only exercises `format_query_text`, so it does not download the model)

- [ ] **Step 5: Commit**

```bash
git add src/revdict/models/embedder.py tests/models/test_embedder.py
git commit -m "Add bge-small-en-v1.5 bi-encoder wrapper"
```

---

### Task 7: Cross-encoder reranker wrapper

**Files:**
- Create: `src/revdict/models/reranker.py`
- Test: `tests/models/test_reranker.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `build_pairs(query: str, definitions: list[str]) -> list[tuple[str, str]]`, class `Reranker` with `score(query: str, definitions: list[str]) -> list[float]`. Task 11 (`search.py`) consumes `Reranker`.

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_reranker.py
from revdict.models.reranker import build_pairs


def test_build_pairs_pairs_the_query_with_each_definition_in_order():
    pairs = build_pairs("joy", ["feeling happy", "a legal document"])
    assert pairs == [("joy", "feeling happy"), ("joy", "a legal document")]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/models/test_reranker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'revdict.models.reranker'`

- [ ] **Step 3: Write the implementation**

```python
# src/revdict/models/reranker.py
from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def build_pairs(query: str, definitions: list[str]) -> list[tuple[str, str]]:
    return [(query, definition) for definition in definitions]


class Reranker:
    def __init__(self, model_name: str = MODEL_NAME):
        self._model = CrossEncoder(model_name)

    def score(self, query: str, definitions: list[str]) -> list[float]:
        pairs = build_pairs(query, definitions)
        scores = self._model.predict(pairs)
        return [float(score) for score in scores]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/models/test_reranker.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/revdict/models/reranker.py tests/models/test_reranker.py
git commit -m "Add ms-marco-MiniLM cross-encoder reranker wrapper"
```

---

### Task 8: Emotion tagging (SentiWordNet + EmoLex combined, classifier fallback)

This is the corrected design: the three emotion sources are **combined**, not tried as a strict priority chain. SentiWordNet supplies polarity (its real strength: a real-valued positive/negative/objective score per WordNet sense — but it has no concept of specific emotion categories, and most synsets score as objective/neutral). NRC EmoLex supplies a specific emotion category (joy, anger, fear, etc.) for the ~6,500 words it covers. The transformer classifier is invoked **only when EmoLex doesn't have a specific category for the word**, to still deliver "richer emotion categories" corpus-wide as the user asked for — it is explicitly a best-effort signal on the definition text, not the word's connotation, and is skipped whenever EmoLex already answers the category question (which also keeps it off the hot path for the ~6,500 EmoLex-covered words).

**Files:**
- Create: `src/revdict/models/emotion.py`
- Test: `tests/models/test_emotion.py`

**Interfaces:**
- Consumes: a `record: dict` with keys `source: str`, `definition: str`, `sentiwordnet: {"pos": float, "neg": float, "obj": float} | None`, `emolex: frozenset[str] | None` (the same shape Task 10 writes into each corpus record).
- Produces: `polarity_from_sentiwordnet(scores: dict) -> str`, `label_from_emolex(labels: frozenset[str]) -> tuple[str, str]`, class `EmotionClassifier` with `classify(text: str) -> tuple[str, str]`, `tag_emotion(record: dict, classifier_factory: "Callable[[], EmotionClassifier] | None") -> dict` returning `{"label": str, "polarity": str, "emotion_source": str}`. **Note the parameter is a zero-argument factory callable, not a classifier instance** — `tag_emotion` only calls `classifier_factory()` when it has already determined (via EmoLex) that it actually needs a classification, so the caller can pass a lazily-memoizing factory without needing to duplicate that "is it actually needed" decision itself. Task 11 (`search.py`) consumes `tag_emotion` and `EmotionClassifier`, and must supply a memoizing factory rather than deciding up front whether to construct one.

- [ ] **Step 1: Write the failing test**

```python
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


def test_tag_emotion_returns_neutral_none_when_nothing_available():
    record = {"source": "wiktionary", "definition": "x", "sentiwordnet": None, "emolex": None}
    result = tag_emotion(record, classifier_factory=None)
    assert result == {"label": "neutral", "polarity": "neutral", "emotion_source": "none"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/models/test_emotion.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'revdict.models.emotion'`

- [ ] **Step 3: Write the implementation**

```python
# src/revdict/models/emotion.py
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
        results = self._pipe(text)[0]
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/models/test_emotion.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/revdict/models/emotion.py tests/models/test_emotion.py
git commit -m "Add combined SentiWordNet+EmoLex emotion tagging with classifier fallback"
```

---

### Task 9: Exact-match dictionary lookup

**Files:**
- Create: `src/revdict/dictionary.py`
- Test: `tests/test_dictionary.py`

**Interfaces:**
- Consumes: `revdict.paths.INDEX_DIR` (Task 1); on-disk files `word_index.json` and `metadata.jsonl` written by Task 10's `build()`.
- Produces: `load_word_index(index_dir: Path = INDEX_DIR) -> dict[str, list[int]]`, `load_metadata(index_dir: Path = INDEX_DIR) -> list[dict]`, `lookup_exact(word: str, word_index: dict[str, list[int]], metadata: list[dict]) -> dict | None` returning `{"headword": str, "senses": [{"pos": str, "definition": str, "examples": list[str], "source": str}, ...]}`. Task 11 (`search.py`) and Task 13 (`cli.py`) consume all three functions.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dictionary.py
import json
from pathlib import Path

from revdict.dictionary import load_metadata, load_word_index, lookup_exact


def test_lookup_exact_returns_all_senses_case_insensitively():
    metadata = [
        {
            "headword": "bank",
            "pos": "noun",
            "definition": "a financial institution",
            "examples": [],
            "source": "wordnet",
        },
        {
            "headword": "bank",
            "pos": "noun",
            "definition": "the land alongside a river",
            "examples": [],
            "source": "wordnet",
        },
    ]
    word_index = {"bank": [0, 1]}

    result = lookup_exact("Bank", word_index, metadata)

    assert result["headword"] == "Bank"
    assert len(result["senses"]) == 2
    assert result["senses"][0]["definition"] == "a financial institution"


def test_lookup_exact_returns_none_for_unknown_word():
    assert lookup_exact("zzznotarealword", {}, []) is None


def test_load_word_index_and_metadata_read_real_files(tmp_path):
    (tmp_path / "word_index.json").write_text(json.dumps({"bank": [0]}), encoding="utf-8")
    (tmp_path / "metadata.jsonl").write_text(
        json.dumps(
            {
                "headword": "bank",
                "pos": "noun",
                "definition": "a financial institution",
                "examples": [],
                "source": "wordnet",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    word_index = load_word_index(Path(tmp_path))
    metadata = load_metadata(Path(tmp_path))

    assert word_index == {"bank": [0]}
    assert metadata[0]["headword"] == "bank"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_dictionary.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'revdict.dictionary'`

- [ ] **Step 3: Write the implementation**

```python
# src/revdict/dictionary.py
import json
from pathlib import Path

from revdict.paths import INDEX_DIR


def load_word_index(index_dir: Path = INDEX_DIR) -> dict[str, list[int]]:
    path = Path(index_dir) / "word_index.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_metadata(index_dir: Path = INDEX_DIR) -> list[dict]:
    path = Path(index_dir) / "metadata.jsonl"
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def lookup_exact(word: str, word_index: dict[str, list[int]], metadata: list[dict]) -> dict | None:
    indices = word_index.get(word.lower())
    if not indices:
        return None
    senses = [
        {
            "pos": metadata[i]["pos"],
            "definition": metadata[i]["definition"],
            "examples": metadata[i]["examples"],
            "source": metadata[i]["source"],
        }
        for i in indices
    ]
    return {"headword": word, "senses": senses}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_dictionary.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/revdict/dictionary.py tests/test_dictionary.py
git commit -m "Add exact-match dictionary lookup"
```

---

### Task 10: Index builder (`revdict build-index`)

**Files:**
- Create: `src/revdict/data/build_index.py`
- Test: `tests/data/test_build_index.py`

**Interfaces:**
- Consumes: `load_wordnet_senses` (Task 2), `load_emolex`/`lookup_emolex` (Task 3), `download_raw_wiktextract`/`stream_filtered_entries_from_gzip` (Task 4), `merge_records` (Task 5), `Embedder` (Task 6), `revdict.paths.INDEX_DIR`/`RAW_WIKTIONARY_PATH` (Task 1).
- Produces: `estimate_full_duration(sample_count: int, sample_seconds: float, total_count: int) -> float`, `group_by_definition(records: list[dict]) -> tuple[list[str], list[list[int]]]`, `build(skip_confirm: bool = False) -> None`. `build()` writes `embeddings.npy`, `metadata.jsonl`, `word_index.json` under `INDEX_DIR` — Task 9's `load_word_index`/`load_metadata` and Task 11's `search()` read these files. Each `metadata.jsonl` line has keys `headword`, `pos`, `definition`, `examples`, `source`, `sentiwordnet` (dict or `null`), `emolex` (list of strings or `null`) — this is the exact shape Task 11 must reconstruct into `emotion.tag_emotion`'s expected `record` shape (turning the `emolex` list back into a `frozenset`).

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_build_index.py
from revdict.data.build_index import estimate_full_duration, group_by_definition


def test_estimate_full_duration_extrapolates_linearly_from_a_sample():
    assert estimate_full_duration(100, 10.0, 1000) == 100.0


def test_estimate_full_duration_handles_an_empty_sample():
    assert estimate_full_duration(0, 0.0, 1000) == 0.0


def test_group_by_definition_groups_identical_texts_and_preserves_first_seen_order():
    records = [{"definition": "a"}, {"definition": "b"}, {"definition": "a"}]

    unique_texts, index_groups = group_by_definition(records)

    assert unique_texts == ["a", "b"]
    assert index_groups == [[0, 2], [1]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/data/test_build_index.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'revdict.data.build_index'`

- [ ] **Step 3: Write the implementation**

```python
# src/revdict/data/build_index.py
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
            meta = {
                "headword": record["headword"],
                "pos": record["pos"],
                "definition": record["definition"],
                "examples": record["examples"],
                "source": record["source"],
                "sentiwordnet": record.get("sentiwordnet"),
                "emolex": list(record["emolex"]) if record.get("emolex") else None,
            }
            f.write(json.dumps(meta) + "\n")
            word_index.setdefault(record["headword"].lower(), []).append(position)

    with (INDEX_DIR / "word_index.json").open("w", encoding="utf-8") as f:
        json.dump(word_index, f)

    print(f"Done. Index written to {INDEX_DIR}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/data/test_build_index.py -v`
Expected: PASS (these tests only exercise the two pure helpers, not the real `build()` function — running the real build end-to-end is Task 14's manual validation step)

- [ ] **Step 5: Commit**

```bash
git add src/revdict/data/build_index.py tests/data/test_build_index.py
git commit -m "Add build-index orchestration with random-sample throughput benchmark"
```

---

### Task 11: Search pipeline

**Files:**
- Create: `src/revdict/search.py`
- Test: `tests/test_search.py`

**Interfaces:**
- Consumes: `Embedder` (Task 6), `Reranker` (Task 7), `EmotionClassifier`/`tag_emotion` (Task 8), `load_metadata`/`load_word_index`/`lookup_exact` (Task 9), `revdict.paths.INDEX_DIR` (Task 1).
- Produces: `cosine_top_k(query_vec: np.ndarray, matrix: np.ndarray, k: int) -> list[tuple[int, float]]`, `dedupe_by_headword(scored_rows: list[tuple[int, float]], metadata: list[dict]) -> list[tuple[int, float]]`, `relative_relevance(scores: list[float]) -> list[int]`, `search(query: str, top_n: int = 10) -> dict` returning `{"exact_match": dict | None, "candidates": list[dict]}` where each candidate has keys `headword`, `pos`, `definition`, `examples`, `label`, `polarity`, `relevance` (0-100 int). Task 12 (`picker.py`) and Task 13 (`cli.py`) consume `search()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_search.py
import numpy as np

from revdict.search import cosine_top_k, dedupe_by_headword, relative_relevance


def test_cosine_top_k_ranks_the_most_similar_vector_first():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]], dtype="float32")
    query = np.array([1.0, 0.0], dtype="float32")

    results = cosine_top_k(query, matrix, k=2)

    assert results[0][0] == 0
    assert results[1][0] == 2


def test_dedupe_by_headword_keeps_the_best_scoring_sense_per_word_case_insensitively():
    metadata = [{"headword": "Happy"}, {"headword": "happy"}, {"headword": "joyful"}]
    scored = [(0, 0.5), (1, 0.9), (2, 0.7)]

    result = dedupe_by_headword(scored, metadata)

    assert result == [(1, 0.9), (2, 0.7)]


def test_relative_relevance_min_max_scales_and_handles_equal_scores():
    assert relative_relevance([0.2, 0.6, 1.0]) == [0, 50, 100]
    assert relative_relevance([0.5, 0.5]) == [50, 50]
    assert relative_relevance([]) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_search.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'revdict.search'`

- [ ] **Step 3: Write the implementation**

```python
# src/revdict/search.py
import numpy as np

from revdict import dictionary
from revdict.models.embedder import Embedder
from revdict.models.emotion import EmotionClassifier, tag_emotion
from revdict.models.reranker import Reranker
from revdict.paths import INDEX_DIR

_state: dict = {}


def cosine_top_k(query_vec: np.ndarray, matrix: np.ndarray, k: int) -> list[tuple[int, float]]:
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    matrix_norms = np.linalg.norm(matrix, axis=1) + 1e-12
    scores = (matrix @ query_norm) / matrix_norms
    k = min(k, len(scores))
    top_indices = np.argpartition(-scores, k - 1)[:k]
    top_indices = top_indices[np.argsort(-scores[top_indices])]
    return [(int(i), float(scores[i])) for i in top_indices]


def dedupe_by_headword(
    scored_rows: list[tuple[int, float]], metadata: list[dict]
) -> list[tuple[int, float]]:
    best: dict[str, tuple[int, float]] = {}
    for index, score in scored_rows:
        key = metadata[index]["headword"].lower()
        if key not in best or score > best[key][1]:
            best[key] = (index, score)
    return sorted(best.values(), key=lambda pair: -pair[1])


def relative_relevance(scores: list[float]) -> list[int]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [50] * len(scores)
    return [round(100 * (score - lo) / (hi - lo)) for score in scores]


def _load_state() -> dict:
    if not _state:
        _state["embeddings"] = np.load(INDEX_DIR / "embeddings.npy")
        _state["metadata"] = dictionary.load_metadata(INDEX_DIR)
        _state["word_index"] = dictionary.load_word_index(INDEX_DIR)
        _state["embedder"] = Embedder()
        _state["reranker"] = Reranker()
        _state["classifier"] = None
    return _state


def _get_classifier(state: dict) -> EmotionClassifier:
    if state["classifier"] is None:
        state["classifier"] = EmotionClassifier()
    return state["classifier"]


def search(query: str, top_n: int = 10) -> dict:
    state = _load_state()
    metadata = state["metadata"]

    query_vec = state["embedder"].encode_query(query)
    retrieved = cosine_top_k(query_vec, state["embeddings"], k=75)
    definitions = [metadata[index]["definition"] for index, _ in retrieved]
    rerank_scores = state["reranker"].score(query, definitions)
    scored = [(retrieved[i][0], rerank_scores[i]) for i in range(len(retrieved))]

    deduped = dedupe_by_headword(scored, metadata)[:top_n]
    relevances = relative_relevance([score for _, score in deduped])

    candidates = []
    for (row_index, _), relevance in zip(deduped, relevances):
        record = dict(metadata[row_index])
        if record.get("emolex"):
            record["emolex"] = frozenset(record["emolex"])
        emotion = tag_emotion(record, classifier_factory=lambda: _get_classifier(state))
        candidates.append(
            {
                "headword": record["headword"],
                "pos": record["pos"],
                "definition": record["definition"],
                "examples": record["examples"],
                "label": emotion["label"],
                "polarity": emotion["polarity"],
                "relevance": relevance,
            }
        )

    exact_match = dictionary.lookup_exact(query.strip(), state["word_index"], metadata)
    return {"exact_match": exact_match, "candidates": candidates}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_search.py -v`
Expected: PASS (these tests only exercise the three pure functions; `search()` itself needs a real built index and is exercised manually in Task 14)

- [ ] **Step 5: Commit**

```bash
git add src/revdict/search.py tests/test_search.py
git commit -m "Add search pipeline: cosine retrieval, rerank, dedupe, emotion tagging"
```

---

### Task 12: fzf picker

**Files:**
- Create: `src/revdict/picker.py`
- Test: `tests/test_picker.py`

**Interfaces:**
- Consumes: candidate dicts shaped like Task 11's `search()["candidates"]` entries, and `exact_match` dicts shaped like Task 9's `lookup_exact()` return value.
- Produces: `format_candidate_line(headword: str, pos: str, definition: str, emotion_label: str, polarity: str, relevance: int, index: int, is_exact: bool = False) -> str`, `parse_selection(fzf_stdout: str) -> int | None`, `run_picker(candidates: list[dict], exact_match: dict | None) -> str | None` (returns the selected headword, or `None` if the user cancelled or `fzf` isn't installed). Task 13 (`cli.py`) consumes `run_picker`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_picker.py
from revdict.picker import format_candidate_line, parse_selection


def test_format_candidate_line_has_five_tab_fields_and_marks_exact_match():
    line = format_candidate_line(
        "happy", "adjective", "feeling pleasure", "Joy", "positive", 92, index=3, is_exact=True
    )
    fields = line.split("\t")
    assert len(fields) == 5
    assert fields[-1] == "3"
    assert fields[0].startswith("★")


def test_format_candidate_line_truncates_long_definitions():
    long_definition = "x" * 200
    line = format_candidate_line(
        "word", "noun", long_definition, "neutral", "neutral", 50, index=0
    )
    gloss_field = line.split("\t")[1]
    assert len(gloss_field) < 100


def test_parse_selection_extracts_trailing_index_or_none_for_empty_input():
    line = format_candidate_line("joyful", "adjective", "x", "Joy", "positive", 80, index=5)
    assert parse_selection(line + "\n") == 5
    assert parse_selection("") is None
    assert parse_selection("   ") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_picker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'revdict.picker'`

- [ ] **Step 3: Write the implementation**

```python
# src/revdict/picker.py
import shutil
import subprocess
import tempfile
from pathlib import Path


def format_candidate_line(
    headword: str,
    pos: str,
    definition: str,
    emotion_label: str,
    polarity: str,
    relevance: int,
    index: int,
    is_exact: bool = False,
) -> str:
    marker = "★" if is_exact else " "
    gloss = definition if len(definition) <= 70 else definition[:67] + "..."
    fields = [
        f"{marker} {headword}",
        f"({pos}) {gloss}",
        f"[{emotion_label} · {polarity}]",
        f"{relevance}%",
        str(index),
    ]
    return "\t".join(fields)


def parse_selection(fzf_stdout: str) -> int | None:
    line = fzf_stdout.strip()
    if not line:
        return None
    return int(line.rsplit("\t", 1)[1])


def _render_exact_preview(exact_match: dict) -> str:
    lines = [f"Exact match — {exact_match['headword']}", ""]
    for sense in exact_match["senses"]:
        lines.append(f"({sense['pos']}) {sense['definition']}")
        for example in sense["examples"]:
            lines.append(f'    "{example}"')
        lines.append("")
    return "\n".join(lines)


def _render_candidate_preview(candidate: dict) -> str:
    lines = [
        f"{candidate['headword']} ({candidate['pos']})",
        "",
        candidate["definition"],
        "",
        f"Emotion: {candidate['label']} · {candidate['polarity']}",
        f"Relative match: {candidate['relevance']}%",
    ]
    if candidate["examples"]:
        lines.append("")
        for example in candidate["examples"]:
            lines.append(f'"{example}"')
    return "\n".join(lines)


def run_picker(candidates: list[dict], exact_match: dict | None) -> str | None:
    if shutil.which("fzf") is None:
        return None

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        lines = []
        index = 0

        if exact_match is not None:
            first_sense = exact_match["senses"][0]
            (tmp_path / f"{index}.txt").write_text(
                _render_exact_preview(exact_match), encoding="utf-8"
            )
            lines.append(
                format_candidate_line(
                    exact_match["headword"],
                    first_sense["pos"],
                    first_sense["definition"],
                    "exact match",
                    "n/a",
                    100,
                    index=index,
                    is_exact=True,
                )
            )
            index += 1

        for candidate in candidates:
            (tmp_path / f"{index}.txt").write_text(
                _render_candidate_preview(candidate), encoding="utf-8"
            )
            lines.append(
                format_candidate_line(
                    candidate["headword"],
                    candidate["pos"],
                    candidate["definition"],
                    candidate["label"],
                    candidate["polarity"],
                    candidate["relevance"],
                    index=index,
                )
            )
            index += 1

        input_text = "\n".join(lines) + "\n"
        result = subprocess.run(
            [
                "fzf",
                "--delimiter",
                "\t",
                "--with-nth=1,2,3,4",
                "--preview",
                f"cat {tmp_path}/{{5}}.txt",
                "--preview-window",
                "right:60%:wrap",
                "--bind",
                "?:toggle-preview",
            ],
            input=input_text,
            capture_output=True,
            text=True,
        )

        selection_index = parse_selection(result.stdout)
        if selection_index is None:
            return None
        if exact_match is not None:
            if selection_index == 0:
                return exact_match["headword"]
            return candidates[selection_index - 1]["headword"]
        return candidates[selection_index]["headword"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_picker.py -v`
Expected: PASS (these tests only exercise the pure formatting/parsing functions; `run_picker`'s real `fzf` subprocess invocation is exercised manually in Task 14)

- [ ] **Step 5: Commit**

```bash
git add src/revdict/picker.py tests/test_picker.py
git commit -m "Add fzf-based interactive picker with live preview pane"
```

---

### Task 13: CLI entry point

**Files:**
- Create: `src/revdict/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `build` (Task 10), `search` (Task 11), `run_picker` (Task 12), `revdict.paths.INDEX_DIR` (Task 1).
- Produces: `main(argv: list[str] | None = None) -> int`, wired as the `revdict` console-script entry point declared in Task 1's `pyproject.toml`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
from revdict import cli


def test_main_prints_error_and_returns_1_when_index_missing(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_index_exists", lambda: False)

    code = cli.main(["happy"])

    captured = capsys.readouterr()
    assert code == 1
    assert "build-index" in captured.out


def test_main_routes_the_build_index_subcommand(monkeypatch):
    called = {}

    def fake_build(skip_confirm):
        called["skip_confirm"] = skip_confirm

    monkeypatch.setattr(cli, "build", fake_build)

    code = cli.main(["build-index", "--yes"])

    assert code == 0
    assert called["skip_confirm"] is True


def test_run_query_warns_and_returns_0_on_blank_query(capsys):
    code = cli._run_query("   ", top_n=10, interactive=False)

    captured = capsys.readouterr()
    assert code == 0
    assert "word or phrase" in captured.out


def test_run_query_prints_static_results_when_not_interactive(monkeypatch, capsys):
    fake_result = {
        "exact_match": None,
        "candidates": [
            {
                "headword": "joyful",
                "pos": "adjective",
                "definition": "feeling great happiness",
                "examples": [],
                "label": "joy",
                "polarity": "positive",
                "relevance": 90,
            }
        ],
    }
    monkeypatch.setattr(cli.search_mod, "search", lambda query, top_n: fake_result)

    code = cli._run_query("happy", top_n=10, interactive=False)

    captured = capsys.readouterr()
    assert code == 0
    assert "joyful" in captured.out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'revdict.cli'`

- [ ] **Step 3: Write the implementation**

```python
# src/revdict/cli.py
import shutil
import sys

from rich.console import Console
from rich.table import Table

from revdict import search as search_mod
from revdict.data.build_index import build
from revdict.paths import INDEX_DIR
from revdict.picker import run_picker

console = Console()


def _index_exists() -> bool:
    return (INDEX_DIR / "embeddings.npy").exists()


def _fzf_missing() -> bool:
    return shutil.which("fzf") is None


def _print_no_index_error() -> None:
    console.print("[bold red]No index found.[/bold red] Run: [bold]revdict build-index[/bold]")


def _print_static_results(result: dict) -> None:
    if result["exact_match"] is not None:
        table = Table(title=f"Exact match — {result['exact_match']['headword']}")
        table.add_column("POS")
        table.add_column("Definition")
        for sense in result["exact_match"]["senses"]:
            table.add_row(sense["pos"], sense["definition"])
        console.print(table)

    table = Table(title="Related words you might mean")
    table.add_column("#")
    table.add_column("Word")
    table.add_column("Definition")
    table.add_column("Emotion")
    table.add_column("Relevance")
    for position, candidate in enumerate(result["candidates"], start=1):
        table.add_row(
            str(position),
            candidate["headword"],
            candidate["definition"],
            f"{candidate['label']} · {candidate['polarity']}",
            f"{candidate['relevance']}%",
        )
    console.print(table)


def _run_query(query: str, top_n: int, interactive: bool) -> int:
    if not query.strip():
        console.print("[yellow]Please enter a word or phrase.[/yellow]")
        return 0

    result = search_mod.search(query, top_n=top_n)

    if interactive:
        selected = run_picker(result["candidates"], result["exact_match"])
        if selected is None and _fzf_missing():
            _print_static_results(result)
            return 0
        if selected:
            print(selected)
        return 0

    _print_static_results(result)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    if argv and argv[0] == "build-index":
        build(skip_confirm="--yes" in argv)
        return 0

    if not argv:
        if not _index_exists():
            _print_no_index_error()
            return 1
        query = console.input("[bold]> [/bold]")
        return _run_query(query, top_n=10, interactive=True)

    no_interactive = "--no-interactive" in argv
    args = [arg for arg in argv if arg != "--no-interactive"]

    top_n = 10
    if "-n" in args:
        position = args.index("-n")
        top_n = int(args[position + 1])
        args = args[:position] + args[position + 2 :]

    query = " ".join(args)

    if not _index_exists():
        _print_no_index_error()
        return 1

    interactive = not no_interactive and sys.stdout.isatty()
    return _run_query(query, top_n, interactive)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/revdict/cli.py tests/test_cli.py
git commit -m "Add revdict CLI entry point (build-index / one-shot / interactive)"
```

---

### Task 14: End-to-end build and manual validation

This task has no new unit tests — per the approved spec, ML ranking/embedding quality is not meaningfully unit-testable and is validated manually here. This is also where the full pipeline runs for real for the first time (all earlier tasks tested pure logic in isolation, never the real 2.6GB Wiktionary download, the real ~117k-synset WordNet load together with it, or real model inference).

**Files:**
- None created — this task runs the already-built CLI against real data.

- [ ] **Step 1: Run the full test suite once to confirm every task's tests still pass together**

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: all tests PASS (the WordNet-related tests will re-download/re-read the ~35MB NLTK corpora if not already cached from Task 2).

- [ ] **Step 2: Run the real index build**

```bash
.venv/bin/revdict build-index
```

Expected: prints progress through WordNet loading, Wiktionary download (~2.6GB, first run only) and filtering, corpus merge size, EmoLex tagging, then the benchmark line with an estimated full-build ETA. At the `Proceed with the full build? [y/N]` prompt, answer `y` to continue, or Ctrl-C and re-run with `--yes` later once satisfied with the ETA. If the estimated time is unreasonably long (multiple hours), stop here and reduce scope by filtering the Wiktionary corpus further in `wiktionary_source.py` (e.g., drop rare/archaic-tagged senses) before re-running — do not silently let a multi-hour build run unattended without the user's awareness.

- [ ] **Step 3: Confirm the index files were written**

```bash
ls -la ~/.cache/rev_dictionary/index/
```

Expected: `embeddings.npy`, `metadata.jsonl`, `word_index.json` all present with non-trivial sizes.

- [ ] **Step 4: Exercise a plain word query**

```bash
.venv/bin/revdict "happy" --no-interactive
```

Expected: an "Exact match" table for `happy` (adjective, "enjoying or showing or marked by joy or pleasure") followed by a "Related words you might mean" table with near-synonyms (e.g. `joyful`, `content`, `cheerful`) each showing an emotion badge and a relevance percentage.

- [ ] **Step 5: Exercise a descriptive-phrase query with no single matching word**

```bash
.venv/bin/revdict "feeling of intense annoyance" --no-interactive
```

Expected: no "Exact match" section (the literal input isn't a headword), and a candidate list headed by words like `irritation`, `exasperation`, or `annoyance`.

- [ ] **Step 6: Exercise the interactive fzf picker**

```bash
.venv/bin/revdict "happy"
```

Expected: `fzf` opens with the exact-match entry pinned first (marked with `★`), a fuzzy-filterable candidate list below it, and a live preview pane on the right showing full detail for whichever entry is highlighted. Confirm arrowing up/down updates the preview live, typing filters the list, and pressing `?` toggles the preview pane on/off. Press Enter on a candidate and confirm the selected word prints to stdout after fzf closes.

- [ ] **Step 7: Exercise a low-confidence/gibberish query**

```bash
.venv/bin/revdict "asdkjfhqwoeiruty" --no-interactive
```

Expected: still prints a top-10 candidate list (never a hard refusal), but with visibly low relevance percentages across the board.

- [ ] **Step 8: Record any quality issues as follow-up, do not block on them**

If specific candidates look wrong (e.g., a POS mismatch, an unhelpful Wiktionary sense slipping through the form-of filter), note them for a future corpus-filtering iteration — this task's job is to confirm the pipeline runs correctly end-to-end, not to hand-tune ranking quality.

- [ ] **Step 9: Commit the final validation note**

```bash
git commit --allow-empty -m "Validate end-to-end build and query pipeline manually"
```
