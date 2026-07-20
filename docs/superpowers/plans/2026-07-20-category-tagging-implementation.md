# Category Tagging Implementation Plan (Phase 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add full category/register filtering (All/Noun/Adjective/Verb/Adverb/Idioms-Slang/Old) to revdict, per TODO.md's feature group 3 and the roadmap's Phase 3 row (`docs/superpowers/plans/2026-07-19-onelook-feature-parity-roadmap.md`).

**Architecture:** A new `src/revdict/category.py` module (mirrors `sort.py`'s shape exactly: a `CATEGORIES` tuple + a pure `matches_category(record, category) -> bool` predicate). `noun`/`adjective`/`verb`/`adverb`/`all` work immediately off the existing `pos` metadata field. `idiom_slang`/`old` need a NEW metadata field (`tags`, Wiktionary's per-sense register tags — currently downloaded but discarded) and therefore require a reindex; until reindexed they gracefully match nothing rather than erroring. Filtering is threaded as a new `category` parameter alongside the existing `sort_mode` parameter through `search()`, the daemon wire protocol, and a new `--category` CLI flag — applied to the scored candidate pool BEFORE `top_n` truncation at every code path, so a category filter never silently shrinks a result set below what was asked for.

**Tech Stack:** Python 3.11+, existing revdict stack (no new dependencies).

## Global Constraints

- Seven categories, this exact tuple and order: `("all", "noun", "adjective", "verb", "adverb", "idiom_slang", "old")`. `all` is the default (no filtering).
- **Single `--category` CLI flag**, not two separate `--pos`/`--category` flags. TODO.md presents these seven as one mutually-exclusive selector list (not two independent facets a user combines) — the roadmap's "`--pos`/`--category`" phrasing was shorthand for "some CLI surface for this," not a firm two-flag commitment. One flag with seven choices matches TODO.md's actual presentation and mirrors `--sort`'s established precedent exactly.
- Real Wiktionary tag vocabulary, confirmed by directly sampling the cached raw dump (`~/.cache/rev_dictionary/raw-wiktextract-data.jsonl.gz`, ~371K senses scanned): `archaic` 6003, `dated` 4139, `obsolete` 13038, `historical` 4918 (→ `old`); `idiomatic` 5789, `slang` 12203, `vulgar` 1580, `colloquial` 3450, plus real `pos` values `phrase` 679 and `proverb` 280 (→ `idiom_slang`). These are NOT guesses — use exactly these tag/pos sets.
- `informal` (7536 occurrences, also real and common) is deliberately EXCLUDED from `idiom_slang` — it's a broad, common register marker that would dilute the category far past what "Idioms/Slang" is meant to mean. This is a disclosed interpretation call, not a spec requirement — document it with a comment where the tag set is defined.
- `noun`/`adjective`/`verb`/`adverb`/`all` categories need **no reindex** — `pos` already exists in every metadata row today. `idiom_slang`/`old` need **a reindex** — `tags` is a brand-new metadata field. Until a user runs `revdict build-index` again, `idiom_slang`/`old` must gracefully return zero candidates (via `.get("tags") or []`, never `record["tags"]`), not crash. Document this precisely in the README.
- Category filtering applies to the **candidate list only**, never to the exact-match direct lookup of the typed word (`dictionary.lookup_exact`/`tag_exact_match_senses` stay completely untouched by `category`). A query like `run --category noun` still shows the verb sense of "run" in the exact-match panel — the user typed that exact word on purpose.
- Category filtering must be applied to the scored/ranked candidate pool **before** any `top_n` slicing, at every code path (meaning/combined mode in `search.py`, structural/expand/phrase_contains mode in `structural_search.py`). Filtering after truncation would silently return fewer than `top_n` results whenever non-matching rows occupied truncated-away slots, even when enough real matches existed further down the ranked list.
- Do **not** add a `"tags"` field to `build_candidate()`'s output. Filtering happens before candidates are built, so the wire-locked 9-field candidate shape (`headword`/`pos`/`definition`/`examples`/`label`/`polarity`/`relevance`/`stress`/`synonyms`) stays exactly as-is this phase — confirmed unchanged by an existing test in `tests/test_structural_search.py` (`test_run_structural_builds_full_candidate_records_for_every_match`, asserting the exact key set). Adding an unused `tags` field would be YAGNI and would break that test for no functional benefit.
- Do **not** trigger a real `revdict build-index` reindex as part of any task in this plan — it's a multi-hour, resource-heavy operation the user runs manually when ready. All tests use small in-memory fixtures, matching every prior phase's convention (no test in this codebase touches the real ~787K-row index).
- Mirror `sort.py`/`--sort`'s already-shipped wiring pattern exactly: `SORT_MODES` tuple ↔ `CATEGORIES` tuple, `apply_sort` ↔ `matches_category`, daemon `"sort"` JSON field ↔ `"category"` JSON field, `--sort` CLI flag ↔ `--category` CLI flag. Every new parameter is optional and defaults to `None`/no-op, so an old client talking to a new daemon (or vice versa) keeps working exactly as `sort_mode` already does.
- The live-session/revdict.nvim-facing entrypoints (`_run_query_only`, `_run_jsonl_query` in `cli.py`) are driven by typed query text during a live picker session, not CLI flags — leave them completely unchanged, matching Phase 2's identical precedent for `--sort`. Category selection during live use is a later (TUI/nvim-follow-up) concern, not this phase's.

---

### Task 1: Capture Wiktionary's `tags` field into metadata

**Files:**
- Modify: `src/revdict/data/wiktionary_source.py:55-61`
- Modify: `src/revdict/data/build_index.py:49-59`
- Test: `tests/data/test_wiktionary_source.py`
- Test: `tests/data/test_build_index.py`

**Interfaces:**
- Consumes: nothing new (first task).
- Produces: every record yielded by `iter_filtered_entries`/`parse_filtered_entries`/`stream_filtered_entries_from_gzip` now carries a `"tags": list[str]` key (the sense's raw Wiktionary tags, e.g. `["archaic", "singular"]`, or `[]` if the sense has none). Every dict returned by `build_metadata_record` now carries a `"tags": list[str]` key — `[]` for WordNet-sourced records (which never compute tags) and for any record missing the key entirely.

- [ ] **Step 1: Write the failing tests**

Add to `tests/data/test_wiktionary_source.py` (after the existing line constants, before the first test function):

```python
ENGLISH_ARCHAIC_LINE = (
    '{"word": "thou", "pos": "pron", "lang": "English", "lang_code": "en", '
    '"senses": [{"glosses": ["The second-person singular pronoun."], '
    '"tags": ["archaic", "singular"]}]}'
)
ENGLISH_NO_TAGS_LINE = (
    '{"word": "dog", "pos": "noun", "lang": "English", "lang_code": "en", '
    '"senses": [{"glosses": ["A domesticated canine."]}]}'
)
```

Add these test functions:

```python
def test_parse_filtered_entries_captures_the_tags_field_when_present():
    records = parse_filtered_entries([ENGLISH_ARCHAIC_LINE])
    assert records[0]["tags"] == ["archaic", "singular"]


def test_parse_filtered_entries_defaults_tags_to_an_empty_list_when_absent():
    records = parse_filtered_entries([ENGLISH_NO_TAGS_LINE])
    assert records[0]["tags"] == []
```

Also add this real-data regression test, guarded to skip when the cached
dump isn't present (portable across machines/CI that never ran
`revdict build-index`). Add these imports at the top of the file alongside
the existing `gzip`/`tempfile`/`Path` imports: `import itertools` and
`import pytest`.

```python
_REAL_RAW_WIKTIONARY_PATH = Path.home() / ".cache" / "rev_dictionary" / "raw-wiktextract-data.jsonl.gz"


@pytest.mark.skipif(
    not _REAL_RAW_WIKTIONARY_PATH.exists(),
    reason="requires the real cached Wiktionary dump (present after running `revdict build-index` once)",
)
def test_stream_filtered_entries_captures_real_tags_from_the_actual_dump():
    """Regression guard against the real data source's actual shape, not
    just a synthetic fixture. Scoped to the first 25K raw lines (not the
    full ~2.7GB file) to keep this fast on every test run -- verified by
    direct sampling that this slice alone contains hundreds of tagged
    senses (archaic: 448, slang: 777, as of the 2026-07 dump), so a
    zero-tag result here would be a genuine capture bug, not a sampling
    fluke. (A one-time broader sample, taken manually while writing this
    plan, already confirmed the full tag vocabulary -- this committed test
    only needs to catch a regression, not re-discover the vocabulary, so it
    stays small.)"""
    with gzip.open(_REAL_RAW_WIKTIONARY_PATH, "rt", encoding="utf-8") as f:
        lines = list(itertools.islice(f, 25_000))
    records = parse_filtered_entries(lines)
    seen_tags = {tag for record in records for tag in record["tags"]}
    assert "archaic" in seen_tags
    assert "slang" in seen_tags
```

Add to `tests/data/test_build_index.py`:

```python
def test_build_metadata_record_includes_tags_when_present():
    record = {
        "headword": "thou",
        "pos": "pronoun",
        "definition": "the second-person singular pronoun",
        "examples": [],
        "source": "wiktionary",
        "tags": ["archaic", "singular"],
    }

    meta = build_metadata_record(record)

    assert meta["tags"] == ["archaic", "singular"]


def test_build_metadata_record_defaults_tags_to_an_empty_list_when_absent():
    """WordNet-sourced records never carry a `tags` key at all (only
    Wiktionary senses compute one) -- must not KeyError, and should persist
    as [] rather than None so downstream category matching never needs a
    None-check."""
    record = {
        "headword": "happy",
        "pos": "adjective",
        "definition": "feeling great pleasure",
        "examples": ["a happy child"],
        "source": "wordnet",
        "emolex": None,
    }

    meta = build_metadata_record(record)

    assert meta["tags"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/data/test_wiktionary_source.py tests/data/test_build_index.py -v`
Expected: the four new non-skipped tests FAIL with `KeyError: 'tags'`.

- [ ] **Step 3: Implement**

In `src/revdict/data/wiktionary_source.py`, inside `iter_filtered_entries`,
change the yielded dict (the `tags` local variable already exists two lines
above this, from the existing `form-of`/`alt-of` check):

```python
            yield {
                "headword": word,
                "pos": _normalize_pos(pos),
                "definition": _combine_glosses(glosses),
                "examples": examples,
                "source": "wiktionary",
                "tags": tags,
            }
```

In `src/revdict/data/build_index.py`, change `build_metadata_record`:

```python
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
        "tags": record.get("tags") or [],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/test_wiktionary_source.py tests/data/test_build_index.py -v`
Expected: all PASS (the real-dump test either passes or skips, depending on whether the cache file exists on this machine).

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `pytest`
Expected: same pass count as before plus the new tests, same 2 pre-existing `FORCE_COLOR` failures (`test_main_error_message_is_not_mangled_by_rich_markup`, `test_main_routes_daemon_status`) and nothing else.

- [ ] **Step 6: Commit**

```bash
git add src/revdict/data/wiktionary_source.py src/revdict/data/build_index.py tests/data/test_wiktionary_source.py tests/data/test_build_index.py
git commit -m "data: capture Wiktionary's per-sense tags field into metadata"
```

---

### Task 2: `category.py` — categories and the matching predicate

**Files:**
- Create: `src/revdict/category.py`
- Test: `tests/test_category.py`

**Interfaces:**
- Consumes: nothing (pure module, no dependency on Task 1's data changes — operates on whatever `pos`/`tags` a record dict happens to have).
- Produces: `CATEGORIES: tuple[str, ...]` (7 values, `"all"` first) and `matches_category(record: dict, category: str | None) -> bool`, raising `ValueError` for any string not in `CATEGORIES`. Both are imported by Tasks 3, 4, and 6.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_category.py`:

```python
import pytest

from revdict.category import CATEGORIES, matches_category


def test_categories_lists_all_seven_values_in_a_stable_order():
    assert CATEGORIES == ("all", "noun", "adjective", "verb", "adverb", "idiom_slang", "old")


def test_matches_category_all_accepts_everything_including_a_bare_record():
    assert matches_category({}, "all") is True
    assert matches_category({"pos": "noun", "tags": ["archaic"]}, "all") is True


def test_matches_category_none_is_treated_the_same_as_all():
    assert matches_category({"pos": "noun"}, None) is True


@pytest.mark.parametrize("pos", ["noun", "adjective", "verb", "adverb"])
def test_matches_category_pos_buckets_require_an_exact_pos_match(pos):
    assert matches_category({"pos": pos, "tags": []}, pos) is True
    other_pos = "noun" if pos != "noun" else "verb"
    assert matches_category({"pos": other_pos, "tags": []}, pos) is False


def test_matches_category_idiom_slang_matches_via_tags():
    assert matches_category({"pos": "noun", "tags": ["slang"]}, "idiom_slang") is True
    assert matches_category({"pos": "noun", "tags": ["colloquial", "rare"]}, "idiom_slang") is True
    assert matches_category({"pos": "noun", "tags": ["formal"]}, "idiom_slang") is False


def test_matches_category_idiom_slang_matches_via_phrase_or_proverb_pos_even_with_no_tags():
    assert matches_category({"pos": "phrase", "tags": []}, "idiom_slang") is True
    assert matches_category({"pos": "proverb", "tags": []}, "idiom_slang") is True


def test_matches_category_idiom_slang_excludes_the_broader_informal_tag():
    """A deliberate scope decision: 'informal' is real and common in the
    raw data, but including it would make Idioms/Slang match far too much
    of the dictionary to be a useful filter."""
    assert matches_category({"pos": "noun", "tags": ["informal"]}, "idiom_slang") is False


def test_matches_category_old_matches_via_register_tags():
    assert matches_category({"pos": "noun", "tags": ["archaic"]}, "old") is True
    assert matches_category({"pos": "noun", "tags": ["dated"]}, "old") is True
    assert matches_category({"pos": "noun", "tags": ["obsolete"]}, "old") is True
    assert matches_category({"pos": "noun", "tags": ["historical"]}, "old") is True
    assert matches_category({"pos": "noun", "tags": ["rare"]}, "old") is False


def test_matches_category_handles_a_record_with_no_tags_key_at_all():
    """Pre-reindex metadata rows (or any record built before this phase)
    have no 'tags' key at all -- must not KeyError, must simply not match
    the tag-based categories."""
    record = {"pos": "noun"}
    assert matches_category(record, "old") is False
    assert matches_category(record, "idiom_slang") is False
    assert matches_category(record, "noun") is True


def test_matches_category_raises_on_an_unknown_category():
    with pytest.raises(ValueError, match="Unknown category"):
        matches_category({"pos": "noun", "tags": []}, "verb_phrase")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_category.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'revdict.category'`.

- [ ] **Step 3: Implement**

Create `src/revdict/category.py`:

```python
CATEGORIES = ("all", "noun", "adjective", "verb", "adverb", "idiom_slang", "old")

_POS_CATEGORIES = {"noun", "adjective", "verb", "adverb"}

# Confirmed against the real Wiktionary dump (2026-07 sample, ~371K
# senses): archaic 6003, dated 4139, obsolete 13038, historical 4918.
_OLD_TAGS = {"archaic", "dated", "obsolete", "historical"}

# Confirmed against the same sample: idiomatic 5789, slang 12203, vulgar
# 1580, colloquial 3450; pos values phrase 679, proverb 280. "informal"
# (7536, also real) is deliberately excluded -- it's a broad, common
# register marker that would dilute this category far past what
# "Idioms/Slang" is meant to mean.
_IDIOM_SLANG_TAGS = {"idiomatic", "slang", "vulgar", "colloquial"}
_IDIOM_SLANG_POS = {"phrase", "proverb"}


def matches_category(record: dict, category: str | None) -> bool:
    if not category or category == "all":
        return True
    if category in _POS_CATEGORIES:
        return record.get("pos") == category
    if category == "idiom_slang":
        return record.get("pos") in _IDIOM_SLANG_POS or bool(
            set(record.get("tags") or []) & _IDIOM_SLANG_TAGS
        )
    if category == "old":
        return bool(set(record.get("tags") or []) & _OLD_TAGS)
    raise ValueError(f"Unknown category: {category!r}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_category.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `pytest`
Expected: previous pass count + these new tests, same 2 pre-existing `FORCE_COLOR` failures.

- [ ] **Step 6: Commit**

```bash
git add src/revdict/category.py tests/test_category.py
git commit -m "feat: add category.py with the 7 category buckets and matching predicate"
```

---

### Task 3: Wire category filtering into `search()`'s meaning/combined path

**Files:**
- Modify: `src/revdict/search.py`
- Test: `tests/test_search.py`

**Interfaces:**
- Consumes: `revdict.category.matches_category(record: dict, category: str | None) -> bool` (Task 2).
- Produces: `search(query: str, top_n: int = 10, sort_mode: str | None = None, category: str | None = None) -> dict` — new 4th parameter. Also produces `filter_by_category(scored_rows: list[tuple[int, float]], metadata: list[dict], category: str | None) -> list[tuple[int, float]]` in `search.py`, next to the existing `dedupe_by_headword`/`exclude_headword` helpers.
- **Scope note:** this task covers ONLY the meaning/combined-mode path (the code after the `if parsed.mode in ("structural", "expand", "phrase_contains")` branch). The structural/expand/phrase_contains branch's call to `structural_search.run_structural(parsed, state, top_n)` is left completely unchanged in this task — it does not yet accept or apply `category`. That is Task 4's responsibility. This is intentional, not a gap: it mirrors this project's established practice of splitting a cross-cutting feature by file/code-path (see Phase 1's Task 6 vs Task 7 split for the same reason). A call like `search("blue*", category="noun")` after this task alone will simply not filter by category yet; it will after Task 4 lands.

- [ ] **Step 1: Write the failing tests**

Add `import pytest` to the top of `tests/test_search.py` (it isn't imported there yet). Then add these test functions (place them near the end of the file, after the existing sort-mode tests):

```python
def test_search_category_none_returns_candidates_of_every_part_of_speech(monkeypatch):
    """category=None (the default) must not filter anything -- proven with
    a fixture containing multiple different POS values, all of which must
    still appear."""
    metadata = [
        {
            "headword": "bluebird", "pos": "noun", "definition": "a songbird",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
        },
        {
            "headword": "blue", "pos": "adjective", "definition": "the color of the sky",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
        },
    ]
    state = {
        "metadata": metadata,
        "word_index": {"bluebird": [0], "blue": [1]},
        "literary_frequency": {},
        "classifier": None,
    }
    import numpy as np

    class FakeEmbedder:
        def encode_query(self, query):
            return np.array([1.0, 0.0], dtype="float32")

    class FakeReranker:
        def score(self, query, definitions):
            return [1.0 for _ in definitions]

    state["embedder"] = FakeEmbedder()
    state["reranker"] = FakeReranker()
    state["embeddings"] = np.array([[1.0, 0.0], [1.0, 0.0]], dtype="float32")
    state["embedding_norms"] = np.array([1.0, 1.0])
    monkeypatch.setattr(search_mod, "_load_state", lambda: state)

    result = search_mod.search("sky color", top_n=10)

    headwords = {c["headword"] for c in result["candidates"]}
    assert headwords == {"bluebird", "blue"}


def test_search_category_filters_meaning_mode_candidates_to_the_matching_pos(monkeypatch):
    metadata = [
        {
            "headword": "bluebird", "pos": "noun", "definition": "a songbird",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
        },
        {
            "headword": "blue", "pos": "adjective", "definition": "the color of the sky",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
        },
    ]
    state = {
        "metadata": metadata,
        "word_index": {"bluebird": [0], "blue": [1]},
        "literary_frequency": {},
        "classifier": None,
    }
    import numpy as np

    class FakeEmbedder:
        def encode_query(self, query):
            return np.array([1.0, 0.0], dtype="float32")

    class FakeReranker:
        def score(self, query, definitions):
            return [1.0 for _ in definitions]

    state["embedder"] = FakeEmbedder()
    state["reranker"] = FakeReranker()
    state["embeddings"] = np.array([[1.0, 0.0], [1.0, 0.0]], dtype="float32")
    state["embedding_norms"] = np.array([1.0, 1.0])
    monkeypatch.setattr(search_mod, "_load_state", lambda: state)

    result = search_mod.search("sky color", top_n=10, category="noun")

    assert [c["headword"] for c in result["candidates"]] == ["bluebird"]


def test_search_category_filters_before_top_n_truncation_so_real_matches_are_not_dropped(monkeypatch):
    """The category filter must apply to the FULL scored candidate pool
    before slicing to top_n, not after -- otherwise a high-scoring
    non-matching row occupying a top_n slot would silently squeeze out a
    real match ranked just below it. Fixture: the single highest-scoring
    candidate is an adjective (excluded by category='noun'); two lower-
    scoring nouns follow. Asking for top_n=2 nouns must return BOTH of
    them, not just one."""
    metadata = [
        {
            "headword": "bluely", "pos": "adjective", "definition": "a common adjective sense",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
        },
        {
            "headword": "blueness", "pos": "noun", "definition": "a rare noun sense one",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
        },
        {
            "headword": "bluebell", "pos": "noun", "definition": "a rare noun sense two",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
        },
    ]
    state = {
        "metadata": metadata,
        "word_index": {"bluely": [0], "blueness": [1], "bluebell": [2]},
        "literary_frequency": {},
        "classifier": None,
    }
    import numpy as np

    class FakeEmbedder:
        def encode_query(self, query):
            return np.array([1.0, 0.0], dtype="float32")

    class FakeReranker:
        def score(self, query, definitions):
            score_by_definition = {
                "a common adjective sense": 5.0,
                "a rare noun sense one": 3.0,
                "a rare noun sense two": 2.0,
            }
            return [score_by_definition[d] for d in definitions]

    state["embedder"] = FakeEmbedder()
    state["reranker"] = FakeReranker()
    state["embeddings"] = np.array([[1.0, 0.0]] * 3, dtype="float32")
    state["embedding_norms"] = np.array([1.0, 1.0, 1.0])
    monkeypatch.setattr(search_mod, "_load_state", lambda: state)

    result = search_mod.search("blue things", top_n=2, category="noun")

    assert {c["headword"] for c in result["candidates"]} == {"blueness", "bluebell"}


def test_search_category_does_not_filter_the_exact_match_panel(monkeypatch):
    """category narrows the candidate list only -- the exact-match block
    always shows the typed word's own senses regardless of category, since
    the user explicitly typed that exact word."""
    metadata = [
        {
            "headword": "run", "pos": "verb", "definition": "to move fast on foot",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
        },
    ]
    state = {
        "metadata": metadata,
        "word_index": {"run": [0]},
        "literary_frequency": {},
        "classifier": None,
    }
    import numpy as np

    class FakeEmbedder:
        def encode_query(self, query):
            return np.array([1.0], dtype="float32")

    class FakeReranker:
        def score(self, query, definitions):
            return [1.0 for _ in definitions]

    state["embedder"] = FakeEmbedder()
    state["reranker"] = FakeReranker()
    state["embeddings"] = np.array([[1.0]], dtype="float32")
    state["embedding_norms"] = np.array([1.0])
    monkeypatch.setattr(search_mod, "_load_state", lambda: state)

    result = search_mod.search("run", top_n=10, category="noun")

    assert result["exact_match"]["headword"] == "run"


def test_search_category_applies_to_combined_mode_too(monkeypatch):
    """Combined mode ('blue*:snow') restricts the retrieval pool by
    structural pattern first; category must further narrow the SAME final
    candidate list, not be bypassed by the combined-mode code path."""
    metadata = [
        {
            "headword": "bluebird", "pos": "noun", "definition": "a songbird",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
        },
        {
            "headword": "bluely", "pos": "adjective", "definition": "in a blue manner",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
        },
    ]
    state = {
        "metadata": metadata,
        "word_index": {"bluebird": [0], "bluely": [1]},
        "literary_frequency": {},
        "classifier": None,
    }
    import numpy as np

    class FakeEmbedder:
        def encode_query(self, query):
            return np.array([1.0, 0.0], dtype="float32")

    class FakeReranker:
        def score(self, query, definitions):
            return [1.0 for _ in definitions]

    state["embedder"] = FakeEmbedder()
    state["reranker"] = FakeReranker()
    state["embeddings"] = np.array([[1.0, 0.0], [1.0, 0.0]], dtype="float32")
    state["embedding_norms"] = np.array([1.0, 1.0])
    monkeypatch.setattr(search_mod, "_load_state", lambda: state)

    result = search_mod.search("blue*:snow", top_n=10, category="noun")

    assert [c["headword"] for c in result["candidates"]] == ["bluebird"]


def test_search_unknown_category_raises_value_error(monkeypatch):
    metadata = [
        {
            "headword": "bluebird", "pos": "noun", "definition": "a songbird",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
        },
    ]
    state = {
        "metadata": metadata,
        "word_index": {"bluebird": [0]},
        "literary_frequency": {},
        "classifier": None,
    }
    import numpy as np

    class FakeEmbedder:
        def encode_query(self, query):
            return np.array([1.0], dtype="float32")

    class FakeReranker:
        def score(self, query, definitions):
            return [1.0 for _ in definitions]

    state["embedder"] = FakeEmbedder()
    state["reranker"] = FakeReranker()
    state["embeddings"] = np.array([[1.0]], dtype="float32")
    state["embedding_norms"] = np.array([1.0])
    monkeypatch.setattr(search_mod, "_load_state", lambda: state)

    with pytest.raises(ValueError, match="Unknown category"):
        search_mod.search("bluebird", top_n=10, category="verb_phrase")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_search.py -k category -v`
Expected: FAIL with `TypeError: search() got an unexpected keyword argument 'category'`.

- [ ] **Step 3: Implement**

In `src/revdict/search.py`, add the import alongside the existing `from revdict import ...` lines:

```python
from revdict import category as category_module
```

(Aliased to `category_module` because `search()`'s new parameter is itself named `category` — importing the module as plain `category` would shadow the parameter inside the function body.)

Add `filter_by_category` next to `exclude_headword`:

```python
def filter_by_category(
    scored_rows: list[tuple[int, float]], metadata: list[dict], category: str | None
) -> list[tuple[int, float]]:
    """Filters BEFORE any top_n truncation happens -- applying this after
    truncation would silently return fewer than top_n results whenever
    non-matching rows occupied slots that got cut, even though more real
    matches existed further down the ranked list."""
    if not category or category == "all":
        return scored_rows
    return [
        (index, score)
        for index, score in scored_rows
        if category_module.matches_category(metadata[index], category)
    ]
```

Change `search()`'s signature:

```python
def search(
    query: str, top_n: int = 10, sort_mode: str | None = None, category: str | None = None
) -> dict:
```

In the meaning/combined-mode branch, change:

```python
    deduped = dedupe_by_headword(scored, metadata)
    deduped = exclude_headword(deduped, metadata, exact_headword)[:top_n]
```

to:

```python
    deduped = dedupe_by_headword(scored, metadata)
    deduped = exclude_headword(deduped, metadata, exact_headword)
    # category never filters the exact-match panel above -- it narrows the
    # candidate list only, so a query like "run" --category noun still
    # shows the verb sense of "run" in the exact-match block.
    deduped = filter_by_category(deduped, metadata, category)[:top_n]
```

Leave every other line of `search()` (including the structural-mode dispatch branch at the top) untouched.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_search.py -v`
Expected: all PASS, including every pre-existing test in the file (the new `category` parameter must not change any test that doesn't pass it).

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `pytest`
Expected: previous pass count + these new tests, same 2 pre-existing `FORCE_COLOR` failures.

- [ ] **Step 6: Commit**

```bash
git add src/revdict/search.py tests/test_search.py
git commit -m "feat: apply category filtering to search()'s meaning/combined-mode path"
```

---

### Task 4: Wire category filtering into structural mode (`structural_search.py`)

**Files:**
- Modify: `src/revdict/structural_search.py`
- Modify: `src/revdict/search.py` (one line: the structural-mode dispatch call)
- Test: `tests/test_structural_search.py`
- Test: `tests/test_search.py`

**Interfaces:**
- Consumes: `revdict.category.matches_category` (Task 2); `search()`'s `category` parameter (Task 3, already threaded through the function signature, just not yet passed into the structural dispatch call).
- Produces: `run_structural(parsed: ParsedQuery, state: dict, top_n: int, category: str | None = None) -> dict` — new 4th parameter. Completes the wiring Task 3 deferred: after this task, `search()` passes `category=category` into every structural/expand/phrase_contains call, so category filtering works uniformly across all query modes.

- [ ] **Step 1: Write the failing tests**

Add `import pytest` to the top of `tests/test_structural_search.py` if it
isn't already imported there. Then add these tests (near the other
`run_structural` tests):

```python
def test_run_structural_filters_by_category_before_top_n_truncation():
    """Mirrors search.py's equivalent guarantee: category must narrow the
    matched-headword pool before truncating to top_n, not after -- a
    fixture where the single most-frequent match is the wrong category
    proves this."""
    metadata = [
        {
            "headword": "blueadverbially", "pos": "adverb", "definition": "in a blue manner",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
        },
        {
            "headword": "bluebird", "pos": "noun", "definition": "an American songbird",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
        },
        {
            "headword": "blueprint", "pos": "noun", "definition": "a technical drawing",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
        },
    ]
    word_index = {"blueadverbially": [0], "bluebird": [1], "blueprint": [2]}
    literary_frequency = {"blueadverbially": 9.0, "bluebird": 1.5, "blueprint": 1.0}
    state = {
        "metadata": metadata,
        "word_index": word_index,
        "literary_frequency": literary_frequency,
        "classifier": None,
    }
    parsed = ParsedQuery(mode="structural", pattern_clauses=["blue*"])

    result = run_structural(parsed, state, top_n=2, category="noun")

    assert {c["headword"] for c in result["candidates"]} == {"bluebird", "blueprint"}


def test_run_structural_category_none_matches_every_part_of_speech():
    parsed = ParsedQuery(mode="structural", pattern_clauses=["blue*"])
    state = _build_state()

    result = run_structural(parsed, state, top_n=10, category=None)

    assert {c["headword"] for c in result["candidates"]} == {"bluebird", "blueprint"}


def test_run_structural_category_all_matches_every_part_of_speech():
    parsed = ParsedQuery(mode="structural", pattern_clauses=["blue*"])
    state = _build_state()

    result = run_structural(parsed, state, top_n=10, category="all")

    assert {c["headword"] for c in result["candidates"]} == {"bluebird", "blueprint"}


def test_run_structural_unknown_category_raises_value_error():
    parsed = ParsedQuery(mode="structural", pattern_clauses=["blue*"])
    state = _build_state()

    with pytest.raises(ValueError, match="Unknown category"):
        run_structural(parsed, state, top_n=10, category="verb_phrase")
```

Add this test to `tests/test_search.py` (proves the end-to-end wiring through `search()`, not just `run_structural` in isolation):

```python
def test_search_category_filters_structural_mode_candidates_too(monkeypatch):
    metadata = [
        {
            "headword": "bluebird", "pos": "noun", "definition": "an American songbird",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
        },
        {
            "headword": "bluely", "pos": "adverb", "definition": "in a blue manner",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
        },
    ]
    state = {
        "metadata": metadata,
        "word_index": {"bluebird": [0], "bluely": [1]},
        "literary_frequency": {},
        "classifier": None,
    }
    monkeypatch.setattr(search_mod, "_load_state", lambda: state)

    result = search_mod.search("blue*", top_n=10, category="noun")

    assert [c["headword"] for c in result["candidates"]] == ["bluebird"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_structural_search.py tests/test_search.py -k category -v`
Expected: the four new `test_structural_search.py` tests FAIL with `TypeError: run_structural() got an unexpected keyword argument 'category'`. `test_search_category_filters_structural_mode_candidates_too` FAILS with an assertion error (both headwords come back, since category isn't threaded into the dispatch yet).

- [ ] **Step 3: Implement**

In `src/revdict/structural_search.py`, add the import at the top:

```python
from revdict import category as category_module
```

(Same aliasing reason as Task 3: the new parameter is named `category`.)

Change `run_structural`'s signature and body:

```python
def run_structural(parsed: ParsedQuery, state: dict, top_n: int, category: str | None = None) -> dict:
    # Deferred import: search.py imports structural_search for dispatch
    # (Task 6), so importing search.py at module load time here would be
    # circular. Matches the lazy-import pattern already used elsewhere in
    # this codebase (cli.py's _local_search_fallback, daemon.py's
    # run_server) to defer a heavy/cyclic import until it's actually needed.
    from revdict.search import build_candidate, relative_relevance

    word_index = state["word_index"]
    metadata = state["metadata"]
    literary_frequency = state["literary_frequency"]

    headwords = matching_headwords(parsed, word_index)
    if category and category != "all":
        headwords = [
            word
            for word in headwords
            if category_module.matches_category(metadata[word_index[word][0]], category)
        ]
    ranked = _score_and_sort(headwords, literary_frequency)[:top_n]
    relevances = relative_relevance([score for _, score in ranked])

    candidates = [
        build_candidate(metadata[word_index[headword][0]], relevance, state)
        for (headword, _), relevance in zip(ranked, relevances)
    ]

    return {"exact_match": None, "candidates": candidates}
```

In `src/revdict/search.py`, change the structural-mode dispatch call:

```python
    if parsed.mode in ("structural", "expand", "phrase_contains"):
        result = structural_search.run_structural(parsed, state, top_n, category=category)
```

(was `structural_search.run_structural(parsed, state, top_n)`, without `category`)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_structural_search.py tests/test_search.py -v`
Expected: all PASS, including every pre-existing test.

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `pytest`
Expected: previous pass count + these new tests, same 2 pre-existing `FORCE_COLOR` failures.

- [ ] **Step 6: Commit**

```bash
git add src/revdict/structural_search.py src/revdict/search.py tests/test_structural_search.py tests/test_search.py
git commit -m "feat: apply category filtering to structural/expand/phrase_contains modes"
```

---

### Task 5: Daemon wire protocol — `"category"` field

**Files:**
- Modify: `src/revdict/daemon.py`
- Test: `tests/test_daemon.py`

**Interfaces:**
- Consumes: `search_mod.search(query, top_n, sort_mode=None, category=None)` (Tasks 3+4, already fully wired).
- Produces: `send_query(query: str, top_n: int, sort_mode: str | None = None, category: str | None = None, timeout: float = 30.0) -> dict | None`; `_handle_request` now reads `request.get("category")` and passes it to `search_fn`.

- [ ] **Step 1: Write the failing tests**

`tests/test_daemon.py` has **exactly four existing** `search_fn`/`fake_search` definitions that must each gain a `category` parameter, or they will raise `TypeError` once `_handle_request` starts calling them with `category=...`. These are the exact current lines (verify with `grep -n "sort_mode=None):\|sort_mode):" tests/test_daemon.py` if this plan and the file have drifted):

1. Line 104 area, inside `test_handle_request_calls_search_fn_with_parsed_args_and_returns_json_result`:
   ```python
   def fake_search(query, top_n, sort_mode):
   ```
2. Line 119 area, inside `test_handle_request_returns_error_payload_when_search_fn_raises`:
   ```python
   def failing_search(query, top_n, sort_mode):
   ```
3. Line 372 area, inside `test_handle_request_passes_sort_mode_through_to_search_fn`:
   ```python
   def fake_search(query, top_n, sort_mode):
   ```
4. Line 389 area, inside `test_handle_request_defaults_sort_mode_to_none_for_requests_without_it`:
   ```python
   def fake_search(query, top_n, sort_mode):
   ```

Change all four to add `category` as a fourth parameter: `def fake_search(query, top_n, sort_mode, category):` (and `failing_search` likewise). For #1, also capture it and extend the assertion:

```python
def test_handle_request_calls_search_fn_with_parsed_args_and_returns_json_result():
    calls = {}

    def fake_search(query, top_n, sort_mode, category):
        calls["query"] = query
        calls["top_n"] = top_n
        calls["sort_mode"] = sort_mode
        calls["category"] = category
        return {"exact_match": None, "candidates": []}

    request_text = json.dumps({"query": "happy", "top_n": 10})

    response_text = daemon._handle_request(request_text, fake_search)

    assert calls == {"query": "happy", "top_n": 10, "sort_mode": None, "category": None}
    assert json.loads(response_text) == {"exact_match": None, "candidates": []}
```

Also update the two existing `send_query` payload assertions, since the
request JSON will now always include a `"category"` key (dict equality is
exact, so a missing key would fail these):

```python
def test_send_query_includes_sort_mode_in_the_request_payload(tmp_path, monkeypatch):
    ...
    daemon.send_query("happy", 10, sort_mode="alpha", timeout=2.0)

    server_thread.join(timeout=2)
    assert received["request"] == {"query": "happy", "top_n": 10, "sort": "alpha", "category": None}


def test_send_query_defaults_sort_mode_to_none_when_omitted(tmp_path, monkeypatch):
    ...
    daemon.send_query("happy", 10, timeout=2.0)

    server_thread.join(timeout=2)
    assert received["request"] == {"query": "happy", "top_n": 10, "sort": None, "category": None}
```

(Only the `assert received["request"] == ...` lines change in these two; everything above them stays as-is.)

Now add four new tests, mirroring the sort-mode wire-protocol tests exactly. Place them near the existing `send_query`/`_handle_request` sort tests:

```python
def test_send_query_includes_category_in_the_request_payload(tmp_path, monkeypatch):
    """The wire-protocol extension: a non-default category must actually
    reach the server in the request JSON, not get silently dropped."""
    socket_path = tmp_path / "daemon.sock"
    monkeypatch.setattr(daemon, "DAEMON_SOCKET_PATH", socket_path)
    received = {}

    ready_event = threading.Event()
    server_thread = threading.Thread(
        target=_run_capturing_server, args=(socket_path, received, ready_event)
    )
    server_thread.start()
    ready_event.wait(timeout=2)

    daemon.send_query("happy", 10, category="noun", timeout=2.0)

    server_thread.join(timeout=2)
    assert received["request"] == {"query": "happy", "top_n": 10, "sort": None, "category": "noun"}


def test_send_query_defaults_category_to_none_when_omitted(tmp_path, monkeypatch):
    """Backward compatibility for the CLIENT side: an existing call site
    that doesn't pass category at all must still send a well-formed
    request (with "category": null), matching what an updated server
    expects."""
    socket_path = tmp_path / "daemon.sock"
    monkeypatch.setattr(daemon, "DAEMON_SOCKET_PATH", socket_path)
    received = {}

    ready_event = threading.Event()
    server_thread = threading.Thread(
        target=_run_capturing_server, args=(socket_path, received, ready_event)
    )
    server_thread.start()
    ready_event.wait(timeout=2)

    daemon.send_query("happy", 10, timeout=2.0)

    server_thread.join(timeout=2)
    assert received["request"] == {"query": "happy", "top_n": 10, "sort": None, "category": None}


def test_handle_request_passes_category_through_to_search_fn():
    calls = {}

    def fake_search(query, top_n, sort_mode, category):
        calls["category"] = category
        return {"exact_match": None, "candidates": []}

    request_text = json.dumps({"query": "happy", "top_n": 10, "category": "noun"})

    daemon._handle_request(request_text, fake_search)

    assert calls == {"category": "noun"}


def test_handle_request_defaults_category_to_none_for_requests_without_it():
    """Backward compatibility for the SERVER side: an OLD client's request
    (no "category" key at all, not even null) must still work, with
    category defaulting to None."""
    calls = {}

    def fake_search(query, top_n, sort_mode, category):
        calls["category"] = category
        return {"exact_match": None, "candidates": []}

    request_text = json.dumps({"query": "happy", "top_n": 10})

    daemon._handle_request(request_text, fake_search)

    assert calls == {"category": None}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_daemon.py -v`
Expected: the four modified tests and the new tests FAIL (`TypeError` for the `fake_search` signature mismatches once Step 3 lands `_handle_request`'s new call; assertion mismatches for the payload-shape tests before Step 3's `send_query` change).

- [ ] **Step 3: Implement**

In `src/revdict/daemon.py`, change `send_query`:

```python
def send_query(
    query: str,
    top_n: int,
    sort_mode: str | None = None,
    category: str | None = None,
    timeout: float = 30.0,
) -> dict | None:
    if not DAEMON_SOCKET_PATH.exists():
        return None
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(str(DAEMON_SOCKET_PATH))
            request = json.dumps(
                {"query": query, "top_n": top_n, "sort": sort_mode, "category": category}
            )
            sock.sendall(request.encode("utf-8"))
```

(Only the signature and the `request = json.dumps(...)` line change; everything else in the function body is unchanged.)

Change `_handle_request`:

```python
def _handle_request(request_text: str, search_fn) -> str:
    try:
        request = json.loads(request_text)
        result = search_fn(
            request["query"],
            top_n=request["top_n"],
            sort_mode=request.get("sort"),
            category=request.get("category"),
        )
    except Exception as error:
        return json.dumps({"error": str(error)})
    return json.dumps(result)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_daemon.py -v`
Expected: all PASS, including every pre-existing test.

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `pytest`
Expected: previous pass count + these new tests, same 2 pre-existing `FORCE_COLOR` failures.

- [ ] **Step 6: Commit**

```bash
git add src/revdict/daemon.py tests/test_daemon.py
git commit -m "feat: thread category through the daemon wire protocol"
```

---

### Task 6: `--category` CLI flag + README

**Files:**
- Modify: `src/revdict/cli.py`
- Modify: `README.md`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `revdict.category.CATEGORIES` (Task 2); `daemon.send_query(..., category=None)` / `_handle_request` (Task 5); `search_mod.search(..., category=None)` (Tasks 3+4).
- Produces: `--category` CLI flag with `choices=list(category.CATEGORIES)`; `_local_search_fallback`, `_get_search_result`, `_run_query` all gain `category: str | None = None`; `main()`'s post-parse dispatch passes `category=args.category`.

- [ ] **Step 1: Write the failing tests**

`tests/test_cli.py` has **exactly 18** existing mock sites that pass
`sort_mode=None` and must each also gain `category=None`, since
`_get_search_result` (and, one level deeper, `daemon.send_query` /
`_local_search_fallback`) will require it once Step 3 lands. Run this to
get the authoritative, current list before editing (it must match — if it
doesn't, the file has drifted since this plan was written and the new list
governs):

```bash
grep -n "sort_mode=None" tests/test_cli.py
```

At plan-writing time this returned exactly these 18 lines (line numbers may drift slightly after earlier edits in this same file, but the count and shapes should match):

```
153:        lambda query, top_n, sort_mode=None: {"exact_match": None, "candidates": []},
169:    def fake_send_query(query, top_n, sort_mode=None):
185:    monkeypatch.setattr(cli.daemon, "send_query", lambda query, top_n, sort_mode=None: None)
190:        cli, "_local_search_fallback", lambda query, top_n, sort_mode=None: fake_result
237:    monkeypatch.setattr(cli, "_get_search_result", lambda query, top_n, sort_mode=None: fake_result)
268:    monkeypatch.setattr(cli, "_get_search_result", lambda query, top_n, sort_mode=None: fake_result)
294:    monkeypatch.setattr(cli, "_get_search_result", lambda query, top_n, sort_mode=None: fake_result)
318:    monkeypatch.setattr(cli, "_get_search_result", lambda query, top_n, sort_mode=None: fake_result)
344:    monkeypatch.setattr(cli, "_get_search_result", lambda query, top_n, sort_mode=None: _FAKE_INTERACTIVE_RESULT)
359:    monkeypatch.setattr(cli, "_get_search_result", lambda query, top_n, sort_mode=None: _FAKE_INTERACTIVE_RESULT)
376:    monkeypatch.setattr(cli, "_get_search_result", lambda query, top_n, sort_mode=None: _FAKE_INTERACTIVE_RESULT)
435:    monkeypatch.setattr(cli, "_get_search_result", lambda query, top_n, sort_mode=None: fake_result)
471:    monkeypatch.setattr(cli, "_get_search_result", lambda query, top_n, sort_mode=None: fake_result)
503:    monkeypatch.setattr(cli, "_get_search_result", lambda query, top_n, sort_mode=None: fake_result)
542:    monkeypatch.setattr(cli, "_get_search_result", lambda query, top_n, sort_mode=None: fake_result)
775:        cli, "_get_search_result", lambda query, top_n, sort_mode=None: {"exact_match": None, "candidates": []}
827:    def fake_get_search_result(query, top_n, sort_mode=None):
845:    def fake_get_search_result(query, top_n, sort_mode=None):
```

For every one of these 18 sites, add `category=None` as a new trailing
keyword parameter: `lambda query, top_n, sort_mode=None: X` becomes
`lambda query, top_n, sort_mode=None, category=None: X`; `def fake_x(query, top_n, sort_mode=None):` becomes `def fake_x(query, top_n, sort_mode=None, category=None):`. None of these need to actually use the new parameter (it just needs to be accepted) — **except** lines 827 and 845, which are inside the sort-flag-passthrough tests; leave their bodies as-is (they only read `sort_mode`), the added `category=None` parameter is enough to keep them callable.

Do NOT touch line 402's `def fake_run_query(query, top_n, interactive):` — that mocks `_run_query` itself (called only from the no-argv stdin-read path in `main()`, which never passes `sort_mode`/`category`) and is unrelated to this list.

Now add new tests, placed near the existing `--sort` tests
(`test_query_parser_accepts_all_seven_sort_modes`,
`test_query_parser_rejects_an_invalid_sort_mode`,
`test_query_parser_sort_defaults_to_none`,
`test_main_passes_sort_flag_through_to_get_search_result`,
`test_main_without_sort_flag_passes_none`):

```python
def test_query_parser_accepts_all_seven_categories():
    from revdict import cli

    parser = cli._query_parser()

    for value in ("all", "noun", "adjective", "verb", "adverb", "idiom_slang", "old"):
        args = parser.parse_args(["happy", "--category", value])
        assert args.category == value


def test_query_parser_rejects_an_invalid_category():
    import pytest

    from revdict import cli

    parser = cli._query_parser()

    with pytest.raises(cli._ArgumentError):
        parser.parse_args(["happy", "--category", "nonsense"])


def test_query_parser_category_defaults_to_none():
    from revdict import cli

    parser = cli._query_parser()

    args = parser.parse_args(["happy"])
    assert args.category is None


def test_main_passes_category_flag_through_to_get_search_result(monkeypatch):
    from revdict import cli

    monkeypatch.setattr(cli, "_index_exists", lambda: True)
    calls = {}

    def fake_get_search_result(query, top_n, sort_mode=None, category=None):
        calls["category"] = category
        return {"exact_match": None, "candidates": []}

    monkeypatch.setattr(cli, "_get_search_result", fake_get_search_result)

    code = cli.main(["happy", "--category", "noun", "--no-interactive"])

    assert code == 0
    assert calls["category"] == "noun"


def test_main_without_category_flag_passes_none(monkeypatch):
    from revdict import cli

    monkeypatch.setattr(cli, "_index_exists", lambda: True)
    calls = {}

    def fake_get_search_result(query, top_n, sort_mode=None, category=None):
        calls["category"] = category
        return {"exact_match": None, "candidates": []}

    monkeypatch.setattr(cli, "_get_search_result", fake_get_search_result)

    code = cli.main(["happy", "--no-interactive"])

    assert code == 0
    assert calls["category"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py -v`
Expected: the 5 new category tests FAIL (`AttributeError`/`TypeError`/argparse errors — no `--category` flag exists yet). The 18 modified mock sites do not yet cause failures on their own (they're not exercised until Step 3 changes `_get_search_result`'s real body to require `category`), so most of the file still passes at this point — that's expected; Step 4 is the real regression check.

- [ ] **Step 3: Implement**

In `src/revdict/cli.py`, add the import alongside `from revdict import sort`:

```python
from revdict import category
```

(No aliasing needed here — none of the functions that reference the `category` module also have a `category` parameter in the same scope as that reference; `_query_parser()` is a zero-arg function.)

Add the flag in `_query_parser()`, alongside `--sort`:

```python
    parser.add_argument(
        "--category",
        choices=list(category.CATEGORIES),
        default=None,
        help="Filter results by category (default: all).",
    )
```

Change `_local_search_fallback`:

```python
def _local_search_fallback(
    query: str, top_n: int, sort_mode: str | None = None, category: str | None = None
) -> dict:
    from revdict.query_env import configure_offline_quiet_env

    configure_offline_quiet_env()
    from revdict import search as search_mod

    return search_mod.search(query, top_n=top_n, sort_mode=sort_mode, category=category)
```

Change `_get_search_result`:

```python
def _get_search_result(
    query: str, top_n: int, sort_mode: str | None = None, category: str | None = None
) -> dict:
    result = daemon.send_query(query, top_n, sort_mode=sort_mode, category=category)
    if result is not None:
        return result
    if daemon.ensure_daemon_running():
        result = daemon.send_query(query, top_n, sort_mode=sort_mode, category=category)
        if result is not None:
            return result
    return _local_search_fallback(query, top_n, sort_mode=sort_mode, category=category)
```

Change `_run_query`:

```python
def _run_query(
    query: str,
    top_n: int,
    interactive: bool,
    sort_mode: str | None = None,
    category: str | None = None,
) -> int:
    if not query.strip():
        console.print("[yellow]Please enter a word or phrase.[/yellow]")
        return 0

    result = _get_search_result(query, top_n, sort_mode=sort_mode, category=category)
```

(The rest of `_run_query`'s body is unchanged.)

In `main()`, change the final dispatch line:

```python
    interactive = not args.no_interactive and sys.stdout.isatty()
    return _run_query(query, args.n, interactive, sort_mode=args.sort, category=args.category)
```

Leave `_run_query_only`, `_run_jsonl_query`, and the no-argv stdin-read
path in `main()` completely untouched — none of them parse `--sort` or
`--category`, matching Phase 2's established precedent.

Add a new section to `README.md`, after the existing "## Sort order" section:

```markdown
## Category filter

Results default to matching any part of speech or register. Narrow them with `--category`:

| `--category` value | Matches |
|---|---|
| `all` (default) | Everything |
| `noun` | Nouns only |
| `adjective` | Adjectives only |
| `verb` | Verbs only |
| `adverb` | Adverbs only |
| `idiom_slang` | Idiomatic phrases, slang, vulgar, and colloquial senses |
| `old` | Archaic, dated, obsolete, and historical senses |

```bash
revdict "feeling of intense annoyance" --category adjective --no-interactive
```

`noun`/`adjective`/`verb`/`adverb`/`all` work with any existing index. `idiom_slang` and `old` rely on Wiktionary's register tags, which are only captured starting with this version — run `revdict build-index` to rebuild your index before those two categories will return results (they'll simply come back empty on an older index, not error).
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v`
Expected: all PASS, including every pre-existing test in the file. This is the point where a missed mock from the Step 1 list would surface as a `TypeError` — if anything fails here, re-run `grep -n "sort_mode=None" tests/test_cli.py` to find what was missed.

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `pytest`
Expected: previous pass count + these new tests, same 2 pre-existing `FORCE_COLOR` failures and nothing else.

- [ ] **Step 6: Commit**

```bash
git add src/revdict/cli.py README.md tests/test_cli.py
git commit -m "feat: add --category CLI flag and document it in the README"
```

---

## Self-review notes (fixed before dispatch)

1. **Reindex requirement scoped correctly.** Initially considered a single "requires reindex" caveat for the whole phase; corrected after checking `build_metadata_record`'s current (pre-Task-1) output directly — `pos` is already persisted today, so `noun`/`adjective`/`verb`/`adverb`/`all` need no reindex at all. Only `idiom_slang`/`old` (which need the new `tags` field) do. Global Constraints and the README section both state this precisely rather than over-broadly.
2. **Tag vocabulary grounded in real data, not guessed.** The `_OLD_TAGS`/`_IDIOM_SLANG_TAGS` sets in Task 2 were verified by directly sampling the actual cached Wiktionary dump (two separate samples: ~2M lines for overall vocabulary, first 200K lines for the Task 1 regression test's specific slice) rather than assumed from memory — a wrong guess here would only have surfaced after a multi-hour reindex, so this was checked before writing any task code.
3. **Task 3/Task 4 split avoids an inter-task regression.** An earlier draft had Task 3 change `search()`'s structural-mode dispatch line to pass `category=category` in the same task that added the `category` parameter — but that would call `structural_search.run_structural(parsed, state, top_n, category=category)` before Task 4 (in the very next task) adds `category` to `run_structural`'s signature, breaking every existing structural-mode test for the duration between the two tasks. Fixed by explicitly scoping Task 3 to the meaning/combined path only, leaving the structural dispatch call unchanged until Task 4 lands both the `run_structural` signature change and the dispatch-line update together.
4. **`build_candidate()` deliberately NOT extended with a `tags` field.** Considered adding it for potential future TUI/nvim display, but filtering happens before candidates are built (at the metadata-row/headword level), so nothing in this phase needs it there — confirmed via `tests/test_structural_search.py`'s existing exact-key-set assertion, which would need an unrelated update for zero functional gain. Left out (YAGNI); a future phase can add it when a real consumer needs it.
5. **`exact_match` × `category` interaction made explicit.** Neither TODO.md nor the roadmap states whether category should filter the exact-match panel. Decided category filters candidates only — `dictionary.lookup_exact`/`tag_exact_match_senses` are never touched by any task in this plan — since the user typed that exact word on purpose. Documented in Global Constraints, a code comment at the call site (Task 3), and a dedicated test (`test_search_category_does_not_filter_the_exact_match_panel`).
6. **Single `--category` flag, not `--pos`+`--category`.** The roadmap row literally says "`--pos`/`--category` CLI flags," which could be read as two flags. Re-checked TODO.md directly: it presents all seven values (All/Nouns/Adjectives/Verbs/Adverbs/Idioms-Slang/Old) as one mutually-exclusive list, not two independent facets — so one flag matching `--sort`'s precedent is the correct reading. Documented as a disclosed interpretation in Global Constraints rather than silently picked.
7. **Spec coverage check:** TODO.md feature group 3's full list (All/Nouns/Adjectives/Verbs/Adverbs/Idioms-Slang/Old) — Task 2 (`CATEGORIES` + `matches_category`). "figure out how to properly implement category filtering (likely needs its own data/tagging layer, not just fzf filtering)" — Task 1 (new `tags` metadata field) + Tasks 3/4 (server-side filtering in `search()`, not fzf-side). Roadmap Phase 3 row's "`search()` param" — Tasks 3/4. "CLI flags" — Task 6. No gaps found.
8. **Placeholder scan:** no TBD/"add error handling"/"similar to Task N" text anywhere in the six tasks above; every step has complete, concrete code.
9. **Type/name consistency check:** `category: str | None = None` used identically across `matches_category`, `filter_by_category`, `search()`, `run_structural()`, `send_query()`, `_handle_request`'s `search_fn` contract, `_local_search_fallback`, `_get_search_result`, `_run_query`, and the CLI's `args.category` — no renaming drift between tasks.
