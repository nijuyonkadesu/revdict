# Phase 5: Remaining Sort Modes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 4 new `--sort` values -- `most_formal`, `oldest`, `most_modern`, `most_lyrical` -- completing Phase 5 of the OneLook-feature-parity roadmap (`docs/superpowers/plans/2026-07-19-onelook-feature-parity-roadmap.md`).

**Architecture:** `most_formal`/`oldest`/`most_modern` rank candidates by the Wiktionary register tags (`formal`/`archaic`/`dated`/`obsolete`/`historical`/`slang`/`vulgar`/`colloquial`/`idiomatic`/`informal`) already captured by Phase 3 into each metadata row's `tags` field -- no new data capture, no new reindex requirement beyond what Phase 3 already needs. `most_lyrical` ranks by average consonant-cluster length computed live from Phase 4's precomputed `phonetics.phonemes` field -- no new precompute, no new reindex beyond Phase 4's. All 4 modes need `apply_sort()` to read `tags`/`phonetics` off each candidate dict, which requires growing `build_candidate()`'s fixed 9-key output shape to 11 keys -- this is the one real architectural change in this phase, and it fixes a correctness property (see Task 1) rather than just adding a feature.

**Tech Stack:** Pure Python (`src/revdict/sort.py`, `src/revdict/search.py`), pytest.

## Global Constraints

- Exactly 4 new `SORT_MODES` values: `most_formal`, `oldest`, `most_modern`, `most_lyrical` -- no others (no "most-funny-sounding", no emotional-bucket sort; both remain deferred per the roadmap).
- `most_formal` is **register-only** -- ranks by the formal/informal Wiktionary tag spectrum, NOT genuine legal-domain detection. The "(legal)" qualifier in `TODO.md`'s naming is deliberately dropped from this implementation; capturing Wiktionary's `topics`/`categories` fields for real legal-domain detection is an explicit backlog item, decided via direct user confirmation, not silently scoped out.
- `most_formal` promotes the `formal` tag, demotes any of `{slang, vulgar, colloquial, idiomatic, informal}` (call this set `_INFORMAL_REGISTER_TAGS`), and treats everything else (no tag, or only `archaic`/`dated`/`obsolete`/`historical`/`literary`/`poetic`) as neutral, tied at relevance. This deliberately includes `"informal"` even though `category.py`'s `idiom_slang` CATEGORY grouping excludes it (`category.py:9-13`) -- a sort axis and a category filter are allowed to define "informal" differently; this must be stated explicitly in code comments and the README, not left as a silent inconsistency.
- `oldest`/`most_modern` use the exact same tag vocabulary as `category.py`'s existing `"old"` category: `{archaic, dated, obsolete, historical}` (call this set `_OLD_REGISTER_TAGS`). `oldest` ranks tagged senses first; `most_modern` is the exact mirror (untagged/non-archaic senses first). Neither blends with `literary_frequency` -- doing so would make them a near-duplicate of the existing `most_common`/`least_common` sorts, which are frequency-driven on a different axis entirely.
- `oldest`/`most_modern` must be documented (README + a code comment) as the softest-grounded of the 4 new sorts: a real spot check against actual search() output (12 diverse queries, 360 total candidates, checked against the real cached Wiktionary dump) found 60% of candidates carry SOME register tag somewhere in their senses, but the sort only reads the ONE matched sense's tags (via `build_candidate`), which is frequently untagged even when a word has an archaic sense elsewhere (e.g. "glad" surfaces via its plain untagged adjective sense even though the same headword also has separate obsolete/archaic/informal senses). This is why untagged candidates must tie at relevance rather than being pushed to an arbitrary position -- ties preserve the original relevance order because Python's `sorted()` is stable and no secondary sort key is used.
- `most_lyrical` ranks candidates by **ascending average consonant-cluster length**, computed from `candidate["phonetics"]["phonemes"]` (Phase 4's raw ARPAbet phoneme list). Grounded via real measurement (see roadmap decision log / this plan's Task 3): lyrical words (melody, luminous, flow, moon, gleam) measured avg cluster length 1.0-2.0; clunky words (strengths, angst, twelfths, sixths) measured 2.5-4.0 -- controlled for syllable count (monosyllabic smooth words vs. monosyllabic clunky words still separate cleanly). A candidate's stress-position variance was tested and rejected as a second scoring dimension: it is mathematically always 0 for words with <=2 stressed syllables (the overwhelming majority of English headwords), so it carries no real signal and would only add dimensional noise.
- `most_lyrical` candidates with `phonetics=None` (multi-word/hyphenated headwords, or a candidate from an index built before the Phase 4 reindex) tie at the worst (last) position, mirroring the existing precedent in `most_common`/`least_common` where a missing `literary_frequency` entry defaults to the worst-case value (0.0) rather than being dropped or given an arbitrary rank.
- `build_candidate()` in `src/revdict/search.py` gains exactly 2 new keys: `"tags"` (the matched sense's raw tags list, `record.get("tags") or []`) and `"phonetics"` (the matched sense's precomputed phonetics dict or `None`, `record.get("phonetics")`) -- taken directly from the metadata row already in hand at candidate-build time, never via a separate headword-keyed relookup. This is a backward-compatible shape growth: the daemon wire protocol and CLI/nvim JSON consumers ignore unknown keys, so nothing downstream breaks.
- No changes needed to `cli.py` (the `--sort` flag's `choices=list(sort.SORT_MODES)` already reads the tuple dynamically -- confirmed by reading `cli.py:97-102`) or `daemon.py` (the `"sort"` wire field is already a passthrough string -- confirmed by reading `daemon.py:55,74,166`). Do not add any new CLI flags or wire-protocol fields in this phase.
- Full test suite baseline before this phase: 316 passed + 2 known pre-existing `FORCE_COLOR`-environment-artifact failures in `test_cli.py` (`test_main_error_message_is_not_mangled_by_rich_markup`, `test_main_routes_daemon_status`) -- confirmed unrelated to any code in this project's history. Never treat these 2 as a regression caused by this phase's changes.

---

### Task 1: Extend `build_candidate()` to carry the matched sense's tags and phonetics

**Files:**
- Modify: `src/revdict/search.py:243-258` (the `build_candidate` function)
- Modify: `tests/test_structural_search.py:117-129` (the one test that pins the exact candidate key set)
- Test: `tests/test_search.py` (new direct unit tests for `build_candidate`)

**Interfaces:**
- Consumes: nothing new -- `record` (a metadata row dict that already has `"tags"` and `"phonetics"` keys, populated by Phase 3/Phase 4's `build_index.py`, though possibly absent on older test fixtures -- use `.get()` with defaults), `relevance: int`, `state: dict` (unchanged).
- Produces: `build_candidate(record, relevance, state) -> dict` now returns an 11-key dict (the existing 9 keys plus `"tags": list[str]` and `"phonetics": dict | None`). Task 2 and Task 3 read `candidate["tags"]` and `candidate["phonetics"]` directly in `sort.py`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_search.py` (near the top-level function tests, after the existing `combine_score` tests around line 253):

```python
def test_build_candidate_carries_the_matched_senses_tags_and_phonetics():
    """Phase 5's tag/phonetics-driven sorts (most_formal/oldest/most_modern/
    most_lyrical) read these straight off the candidate dict -- they must
    come from the exact matched sense's row (the one passed into
    build_candidate), never a separate headword-keyed relookup, since tags
    and phonetics are per-sense and a relookup could silently grab a
    different sense of the same headword (e.g. Wiktionary's "tort" has a
    law-topic sense with no register tag AND a separate obsolete-tagged
    adjective sense -- a relookup keyed only on the headword "tort" could
    return either one, independent of which sense actually matched)."""
    record = {
        "headword": "writ", "pos": "noun", "definition": "a legal document",
        "examples": [], "source": "wiktionary", "sentiwordnet": None,
        "emolex": ["joy"], "synonyms": None,
        "tags": ["archaic"],
        "phonetics": {
            "syllable_count": 1, "primary_vowel": "IH", "rhyme_key": "IH T",
            "meter": "/", "phonemes": ["R", "IH1", "T"],
        },
    }
    state = {"classifier": None}

    candidate = search_mod.build_candidate(record, relevance=80, state=state)

    assert candidate["tags"] == ["archaic"]
    assert candidate["phonetics"] == {
        "syllable_count": 1, "primary_vowel": "IH", "rhyme_key": "IH T",
        "meter": "/", "phonemes": ["R", "IH1", "T"],
    }


def test_build_candidate_defaults_tags_to_empty_list_and_phonetics_to_none():
    """A record with no tags/phonetics keys at all (e.g. an older test
    fixture, or a WordNet-sourced row that never had tags to begin with)
    must not KeyError -- tags defaults to [], phonetics defaults to None,
    matching category.py's and phonetics.py's own `.get(...) or []`/
    `.get(...)` conventions for the same fields."""
    record = {
        "headword": "plain", "pos": "adjective", "definition": "simple",
        "examples": [], "source": "wordnet", "sentiwordnet": None,
        "emolex": ["joy"], "synonyms": None,
    }
    state = {"classifier": None}

    candidate = search_mod.build_candidate(record, relevance=50, state=state)

    assert candidate["tags"] == []
    assert candidate["phonetics"] is None
```

`emolex` is set to `["joy"]` (a specific EmoLex category) rather than `None` in both fixtures -- deliberately, matching the existing convention elsewhere in this file (e.g. `test_search_category_none_returns_candidates_of_every_part_of_speech`). `tag_emotion()` only invokes the classifier factory when EmoLex does NOT already supply a specific category (`src/revdict/models/emotion.py:99`); with `emolex=None` these tests would trigger a real `EmotionClassifier()` model load, which is slow and not what this test is checking.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/shichika/redacted/rev-dictionary && source .venv/bin/activate && pytest tests/test_search.py::test_build_candidate_carries_the_matched_senses_tags_and_phonetics tests/test_search.py::test_build_candidate_defaults_tags_to_empty_list_and_phonetics_to_none -v`
Expected: both FAIL with `KeyError: 'tags'` (assertion on a dict key that doesn't exist yet).

- [ ] **Step 3: Extend `build_candidate()`**

In `src/revdict/search.py`, replace the `build_candidate` function (currently lines 243-258):

```python
def build_candidate(record: dict, relevance: int, state: dict) -> dict:
    record = dict(record)
    if record.get("emolex"):
        record["emolex"] = frozenset(record["emolex"])
    emotion = tag_emotion(record, classifier_factory=lambda: get_classifier(state))
    return {
        "headword": record["headword"],
        "pos": record["pos"],
        "definition": record["definition"],
        "examples": record["examples"],
        "label": emotion["label"],
        "polarity": emotion["polarity"],
        "relevance": relevance,
        "stress": stress.mark(record["headword"], record["pos"]),
        "synonyms": record.get("synonyms"),
        "tags": record.get("tags") or [],
        "phonetics": record.get("phonetics"),
    }
```

(The only change is the two new trailing keys.)

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `pytest tests/test_search.py::test_build_candidate_carries_the_matched_senses_tags_and_phonetics tests/test_search.py::test_build_candidate_defaults_tags_to_empty_list_and_phonetics_to_none -v`
Expected: both PASS.

- [ ] **Step 5: Update the pinned candidate key-set test**

`tests/test_structural_search.py:117-129` currently asserts candidates have exactly 9 keys. Update it:

```python
def test_run_structural_builds_full_candidate_records_for_every_match():
    parsed = ParsedQuery(mode="structural", pattern_clauses=["blue*"])
    state = _build_state()

    result = run_structural(parsed, state, top_n=10)

    headwords = {candidate["headword"] for candidate in result["candidates"]}
    assert headwords == {"bluebird", "blueprint"}
    for candidate in result["candidates"]:
        assert set(candidate.keys()) == {
            "headword", "pos", "definition", "examples",
            "label", "polarity", "relevance", "stress", "synonyms",
            "tags", "phonetics",
        }
```

- [ ] **Step 6: Run the full search + structural_search test files**

Run: `pytest tests/test_search.py tests/test_structural_search.py -v`
Expected: all PASS (no regressions -- this confirms the shape change doesn't break any existing candidate-shape assumption elsewhere in these two files).

- [ ] **Step 7: Commit**

```bash
git add src/revdict/search.py tests/test_search.py tests/test_structural_search.py
git commit -m "Carry the matched sense's tags/phonetics onto every candidate"
```

---

### Task 2: Register-tag-driven sort modes -- `most_formal`, `oldest`, `most_modern`

**Files:**
- Modify: `src/revdict/sort.py`
- Test: `tests/test_sort.py`
- Test: `tests/test_search.py` (one end-to-end `search()` integration test)

**Interfaces:**
- Consumes: `candidate["tags"]` (a `list[str]`, always present per Task 1 -- but this task's code must still use `.get("tags") or []` for defensiveness, matching `category.py`'s convention).
- Produces: `SORT_MODES` gains `"most_formal"`, `"oldest"`, `"most_modern"` (3 of the 4 new values -- `most_lyrical` is Task 3). `apply_sort()` handles all 3.

- [ ] **Step 1: Write the failing tests**

Replace the top of `tests/test_sort.py` -- update the existing `SORT_MODES` pin (currently `test_sort_modes_contains_exactly_the_seven_documented_modes`, lines 10-19) and add a candidate-with-tags helper and new tests. The full new top section of the file:

```python
import pytest

from revdict.sort import SORT_MODES, apply_sort


def _candidates(*headwords):
    return [{"headword": hw} for hw in headwords]


def _candidate(headword, tags=None, phonetics=None):
    return {"headword": headword, "tags": tags or [], "phonetics": phonetics}


def test_sort_modes_contains_exactly_the_ten_documented_modes():
    assert SORT_MODES == (
        "relevance",
        "alpha",
        "alpha_desc",
        "shortest",
        "longest",
        "most_common",
        "least_common",
        "most_formal",
        "oldest",
        "most_modern",
    )
```

(This replaces the old 7-item assertion with a 10-item one -- `most_lyrical` is Task 3's addition, not this task's, so it is deliberately not in this tuple yet. Leave the rest of the existing file -- the `none_sort_mode`/`alpha`/`shortest`/`longest`/`most_common`/`least_common`/`unknown_sort_mode` tests -- unchanged; they still pass unmodified. `apply_sort`'s existing 3-arg signature is unchanged in this task, since these new sort modes read tags off the candidate dicts themselves rather than needing a new parameter.)

Append these new tests to the end of `tests/test_sort.py`:

```python
def test_most_formal_ranks_formal_tagged_first():
    candidates = [
        _candidate("khazi", tags=["slang"]),
        _candidate("lavatory", tags=["formal"]),
        _candidate("toilet"),
    ]

    result = apply_sort(candidates, "most_formal", {})

    assert [c["headword"] for c in result] == ["lavatory", "toilet", "khazi"]


@pytest.mark.parametrize("tag", ["slang", "vulgar", "colloquial", "idiomatic", "informal"])
def test_most_formal_treats_every_informal_register_tag_as_informal(tag):
    """"informal" is included deliberately even though category.py's
    idiom_slang CATEGORY grouping excludes it (category.py:9-13) -- a sort
    axis and a category filter are allowed to define "informal"
    differently; this test locks in that most_formal's definition covers
    all 5 tags, not just the 4 category.py happens to use."""
    candidates = [_candidate("plain"), _candidate("marked", tags=[tag])]

    result = apply_sort(candidates, "most_formal", {})

    assert [c["headword"] for c in result] == ["plain", "marked"]


def test_most_formal_treats_archaic_and_dated_tags_as_neutral_not_informal():
    """archaic/dated/obsolete/historical belong to the oldest/most_modern
    axis, not the formal/informal axis -- a purely archaic-tagged sense
    must tie with an untagged sense here, not get demoted like slang."""
    candidates = [
        _candidate("zebra", tags=["archaic"]),
        _candidate("apple"),
    ]

    result = apply_sort(candidates, "most_formal", {})

    assert [c["headword"] for c in result] == ["zebra", "apple"]


def test_most_formal_preserves_relevance_order_within_a_tie():
    candidates = [_candidate("zebra"), _candidate("apple"), _candidate("mango")]

    result = apply_sort(candidates, "most_formal", {})

    assert [c["headword"] for c in result] == ["zebra", "apple", "mango"]


@pytest.mark.parametrize("tag", ["archaic", "dated", "obsolete", "historical"])
def test_oldest_ranks_any_old_register_tag_first(tag):
    candidates = [_candidate("plain"), _candidate("marked", tags=[tag])]

    result = apply_sort(candidates, "oldest", {})

    assert [c["headword"] for c in result] == ["marked", "plain"]


def test_oldest_preserves_relevance_order_among_untagged_candidates():
    candidates = [_candidate("zebra"), _candidate("apple"), _candidate("mango")]

    result = apply_sort(candidates, "oldest", {})

    assert [c["headword"] for c in result] == ["zebra", "apple", "mango"]


def test_most_modern_is_the_exact_reverse_of_oldest():
    candidates = [
        _candidate("plain"),
        _candidate("marked", tags=["archaic"]),
        _candidate("other"),
    ]

    oldest_order = [c["headword"] for c in apply_sort(candidates, "oldest", {})]
    modern_order = [c["headword"] for c in apply_sort(candidates, "most_modern", {})]

    assert oldest_order == ["marked", "plain", "other"]
    assert modern_order == ["plain", "other", "marked"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_sort.py -v`
Expected: the `SORT_MODES` pin test fails (tuple only has 7 items so far), and every new `most_formal`/`oldest`/`most_modern` test fails with `ValueError: Unknown sort mode`.

- [ ] **Step 3: Implement the 3 new sort modes**

Replace `src/revdict/sort.py` in full:

```python
SORT_MODES = (
    "relevance",
    "alpha",
    "alpha_desc",
    "shortest",
    "longest",
    "most_common",
    "least_common",
    "most_formal",
    "oldest",
    "most_modern",
)

# Wiktionary sense tags treated as the informal end of the formal/informal
# register spectrum for --sort most_formal. Deliberately includes
# "informal" even though category.py's idiom_slang CATEGORY grouping
# excludes it (category.py:9-13, where "informal" is left out as "too
# broad" for that narrower category) -- a sort axis and a category filter
# are allowed to define "informal" differently; --category idiom_slang and
# --sort most_formal are not the same axis and must not be conflated.
_INFORMAL_REGISTER_TAGS = {"slang", "vulgar", "colloquial", "idiomatic", "informal"}

# Same vocabulary as category.py's "old" category (archaic/dated/obsolete/
# historical) -- reused here for the oldest/most_modern sort axis.
_OLD_REGISTER_TAGS = {"archaic", "dated", "obsolete", "historical"}


def _formality_rank(candidate: dict) -> int:
    """0 (formal, ranks first) / 1 (neutral -- no tag, or only an
    old-register/literary tag) / 2 (informal, ranks last). Real spot check
    against actual search() candidate pools (12 diverse queries, 360
    candidates checked against the real Wiktionary dump) found the
    explicit "formal" tag on only ~2% of candidates but an informal-family
    tag on a large share of register-rich queries (e.g. "toilet" surfaces
    khazi/biffy/pisser via their own slang senses) -- so this rank is
    built around demoting the common informal signal, not promoting the
    rare formal one."""
    tags = set(candidate.get("tags") or [])
    if "formal" in tags:
        return 0
    if tags & _INFORMAL_REGISTER_TAGS:
        return 2
    return 1


def _oldness_rank(candidate: dict) -> int:
    """0 (old-tagged, ranks first) / 1 (not old-tagged). Untagged
    candidates tie with each other in their original relevance order
    (Python's sorted() is stable and this uses no secondary key) -- this
    is deliberate: a word's matched sense is frequently untagged even when
    the same headword has a separate archaic sense elsewhere, so there is
    no reliable secondary signal to break the tie on. See this plan's
    Global Constraints for the measured tag-density numbers behind this
    call."""
    tags = set(candidate.get("tags") or [])
    return 0 if tags & _OLD_REGISTER_TAGS else 1


def _modernness_rank(candidate: dict) -> int:
    """Exact mirror of _oldness_rank -- not-old-tagged ranks first."""
    return 1 - _oldness_rank(candidate)


def apply_sort(
    candidates: list[dict], sort_mode: str | None, literary_frequency: dict[str, float]
) -> list[dict]:
    if not sort_mode or sort_mode == "relevance":
        return candidates
    if sort_mode == "alpha":
        return sorted(candidates, key=lambda c: c["headword"].lower())
    if sort_mode == "alpha_desc":
        return sorted(candidates, key=lambda c: c["headword"].lower(), reverse=True)
    if sort_mode == "shortest":
        return sorted(candidates, key=lambda c: (len(c["headword"]), c["headword"].lower()))
    if sort_mode == "longest":
        return sorted(candidates, key=lambda c: (-len(c["headword"]), c["headword"].lower()))
    if sort_mode == "most_common":
        return sorted(
            candidates,
            key=lambda c: (
                -literary_frequency.get(c["headword"].lower(), 0.0),
                c["headword"].lower(),
            ),
        )
    if sort_mode == "least_common":
        return sorted(
            candidates,
            key=lambda c: (
                literary_frequency.get(c["headword"].lower(), 0.0),
                c["headword"].lower(),
            ),
        )
    if sort_mode == "most_formal":
        return sorted(candidates, key=_formality_rank)
    if sort_mode == "oldest":
        return sorted(candidates, key=_oldness_rank)
    if sort_mode == "most_modern":
        return sorted(candidates, key=_modernness_rank)
    raise ValueError(f"Unknown sort mode: {sort_mode!r}")
```

`most_lyrical` is deliberately NOT added here -- it's a phonetics-driven axis with its own independent grounding story, added in Task 3 together with its own tests so that task's implementation and tests land in the same reviewable unit, rather than splitting an implementation from the tests that exercise it across two review gates.

- [ ] **Step 4: Run the sort tests to verify they pass**

Run: `pytest tests/test_sort.py -v`
Expected: all PASS, including the parametrized ones (5 informal-tag cases, 4 old-tag cases).

- [ ] **Step 5: Write one end-to-end `search()` integration test**

Add to `tests/test_search.py` (near the existing `test_search_alpha_sort_mode_reorders_meaning_mode_candidates` test, reusing its exact `FakeEmbedder`/`FakeReranker`/state-construction pattern):

```python
def test_search_most_formal_sort_mode_reorders_meaning_mode_candidates(monkeypatch):
    metadata = [
        {
            "headword": "khazi", "pos": "noun", "definition": "a toilet",
            "examples": [], "source": "wiktionary", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": ["slang"], "phonetics": None,
        },
        {
            "headword": "lavatory", "pos": "noun", "definition": "a toilet",
            "examples": [], "source": "wiktionary", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": ["formal"], "phonetics": None,
        },
    ]
    state = {
        "metadata": metadata,
        "word_index": {"khazi": [0], "lavatory": [1]},
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

    result = search_mod.search("toilet", top_n=10, sort_mode="most_formal")

    assert [c["headword"] for c in result["candidates"]] == ["lavatory", "khazi"]
```

- [ ] **Step 6: Run it to verify it passes**

Run: `pytest tests/test_search.py::test_search_most_formal_sort_mode_reorders_meaning_mode_candidates -v`
Expected: PASS.

- [ ] **Step 7: Run the full test_sort.py and test_search.py files**

Run: `pytest tests/test_sort.py tests/test_search.py -v`
Expected: all PASS, no regressions.

- [ ] **Step 8: Commit**

```bash
git add src/revdict/sort.py tests/test_sort.py tests/test_search.py
git commit -m "Add most_formal/oldest/most_modern register-tag-driven sort modes"
```

---

### Task 3: Phonetics-driven sort mode -- `most_lyrical`

**Files:**
- Modify: `src/revdict/sort.py`
- Test: `tests/test_sort.py`
- Test: `tests/test_search.py` (one end-to-end `search()` integration test)

**Interfaces:**
- Consumes: `candidate["phonetics"]["phonemes"]` (a `list[str]` of raw ARPAbet phonemes with stress digits, e.g. `["S", "T", "R", "EH1", "NG", "K", "TH", "S"]`), or `candidate["phonetics"] is None`. Both already produced by Task 1's `build_candidate` change and Phase 4's `phonetics.resolve()`.
- Produces: `SORT_MODES` gains `"most_lyrical"`, the last of the 4 new values. `apply_sort()` handles it.

- [ ] **Step 1: Write the failing tests**

Update the `SORT_MODES` pin test in `tests/test_sort.py` from Task 2's 10-item version to the full 11-item version:

```python
def test_sort_modes_contains_exactly_the_eleven_documented_modes():
    assert SORT_MODES == (
        "relevance",
        "alpha",
        "alpha_desc",
        "shortest",
        "longest",
        "most_common",
        "least_common",
        "most_formal",
        "oldest",
        "most_modern",
        "most_lyrical",
    )
```

(This replaces Task 2's `test_sort_modes_contains_exactly_the_ten_documented_modes` -- rename the test and add the final entry.)

Append to `tests/test_sort.py`:

```python
def test_most_lyrical_ranks_lower_average_consonant_cluster_length_first():
    """Real measurement (see this plan's Global Constraints): "moon"
    (phonemes M-UW1-N) has consonant clusters [1, 1] -> average 1.0;
    "strengths" (phonemes S-T-R-EH1-NG-K-TH-S) has clusters [3, 4] ->
    average 3.5. moon is the more lyrical (lower-cluster) word and must
    rank first."""
    candidates = [
        _candidate("strengths", phonetics={
            "phonemes": ["S", "T", "R", "EH1", "NG", "K", "TH", "S"],
        }),
        _candidate("moon", phonetics={"phonemes": ["M", "UW1", "N"]}),
    ]

    result = apply_sort(candidates, "most_lyrical", {})

    assert [c["headword"] for c in result] == ["moon", "strengths"]


def test_most_lyrical_treats_missing_phonetics_as_the_least_lyrical():
    """Mirrors most_common/least_common's convention of defaulting a
    missing signal to the worst-case value rather than dropping the
    candidate or giving it an arbitrary rank."""
    candidates = [
        _candidate("strengths", phonetics={
            "phonemes": ["S", "T", "R", "EH1", "NG", "K", "TH", "S"],
        }),
        _candidate("unresolved", phonetics=None),
    ]

    result = apply_sort(candidates, "most_lyrical", {})

    assert [c["headword"] for c in result] == ["strengths", "unresolved"]


def test_most_lyrical_treats_a_word_with_no_consonants_as_maximally_lyrical():
    candidates = [
        _candidate("strengths", phonetics={
            "phonemes": ["S", "T", "R", "EH1", "NG", "K", "TH", "S"],
        }),
        _candidate("aye", phonetics={"phonemes": ["AY1"]}),
    ]

    result = apply_sort(candidates, "most_lyrical", {})

    assert [c["headword"] for c in result] == ["aye", "strengths"]


def test_most_lyrical_preserves_relevance_order_within_a_tie():
    """Real phonemes (confirmed via stressmark): "flow" is F-L-OW1 and
    "glow" is G-L-OW1 -- both have exactly one cluster of length 2 (the
    two consonants before the vowel) and nothing after, so both average
    2.0. A genuine tie, so relevance order (input order) must be
    preserved."""
    candidates = [
        _candidate("flow", phonetics={"phonemes": ["F", "L", "OW1"]}),
        _candidate("glow", phonetics={"phonemes": ["G", "L", "OW1"]}),
    ]

    result = apply_sort(candidates, "most_lyrical", {})

    assert [c["headword"] for c in result] == ["flow", "glow"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_sort.py -v`
Expected: the `SORT_MODES` pin test fails (tuple only has Task 2's 10 items so far), and every new `most_lyrical` test fails with `ValueError: Unknown sort mode`.

- [ ] **Step 3: Implement `most_lyrical`**

In `src/revdict/sort.py`: add `"most_lyrical"` as the 11th entry of `SORT_MODES` (append after `"most_modern"`), add the two helper functions below (after `_modernness_rank`, before `apply_sort`), and add the dispatch branch to `apply_sort` (after the `most_modern` branch, before `raise ValueError`):

```python
def _consonant_cluster_lengths(phonemes: list[str]) -> list[int]:
    lengths = []
    run = 0
    for phoneme in phonemes:
        if phoneme[-1].isdigit():
            if run:
                lengths.append(run)
            run = 0
        else:
            run += 1
    if run:
        lengths.append(run)
    return lengths


def _lyrical_rank(candidate: dict) -> float:
    """Ascending average consonant-cluster length -- lower is more
    lyrical/euphonious. Grounded via real measurement (see this plan's
    Global Constraints): lyrical words (melody, luminous, flow, moon)
    measured 1.0-2.0; clunky words (strengths, angst, twelfths) measured
    2.5-4.0, controlled for syllable count. A candidate with no phonetics
    data (multi-word/hyphenated headword, or an index without a Phase 4
    reindex) ranks last (float("inf")) -- the same "missing data sorts to
    the worst end" convention most_common/least_common already use for a
    missing literary_frequency entry."""
    phonetics_data = candidate.get("phonetics")
    if not phonetics_data:
        return float("inf")
    phonemes = phonetics_data.get("phonemes")
    if not phonemes:
        return float("inf")
    lengths = _consonant_cluster_lengths(phonemes)
    if not lengths:
        return 0.0
    return sum(lengths) / len(lengths)
```

And in `apply_sort`, add this branch immediately after the `most_modern` branch:

```python
    if sort_mode == "most_lyrical":
        return sorted(candidates, key=_lyrical_rank)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_sort.py -v`
Expected: all PASS.

- [ ] **Step 5: Write one end-to-end `search()` integration test**

Add to `tests/test_search.py`:

```python
def test_search_most_lyrical_sort_mode_reorders_meaning_mode_candidates(monkeypatch):
    metadata = [
        {
            "headword": "strengths", "pos": "noun", "definition": "plural of strength",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
            "phonetics": {
                "syllable_count": 1, "primary_vowel": "EH", "rhyme_key": "EH NG K TH S",
                "meter": "/", "phonemes": ["S", "T", "R", "EH1", "NG", "K", "TH", "S"],
            },
        },
        {
            "headword": "moon", "pos": "noun", "definition": "the moon",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
            "phonetics": {
                "syllable_count": 1, "primary_vowel": "UW", "rhyme_key": "UW N",
                "meter": "/", "phonemes": ["M", "UW1", "N"],
            },
        },
    ]
    state = {
        "metadata": metadata,
        "word_index": {"strengths": [0], "moon": [1]},
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

    result = search_mod.search("night sky", top_n=10, sort_mode="most_lyrical")

    assert [c["headword"] for c in result["candidates"]] == ["moon", "strengths"]
```

- [ ] **Step 6: Run it to verify it passes**

Run: `pytest tests/test_search.py::test_search_most_lyrical_sort_mode_reorders_meaning_mode_candidates -v`
Expected: PASS.

- [ ] **Step 7: Run the full test_sort.py and test_search.py files**

Run: `pytest tests/test_sort.py tests/test_search.py -v`
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add src/revdict/sort.py tests/test_sort.py tests/test_search.py
git commit -m "Add the most_lyrical phonetics-driven sort mode"
```

---

### Task 4: README documentation

**Files:**
- Modify: `README.md` (the existing "Sort order" table and section, currently lines 127-150)

**Interfaces:**
- Consumes: nothing code-level -- this task is documentation-only, describing the behavior Tasks 1-3 already implemented and tested.
- Produces: nothing consumed by later tasks -- this is the last task in the phase.

- [ ] **Step 1: Update the "Sort order" table and add the caveat paragraphs**

In `README.md`, replace the existing table (lines 132-141) and the paragraph immediately after it (lines 143-150) with:

```markdown
| `--sort` value | Order |
|---|---|
| `relevance` (default) | Most similar first (semantic match quality) |
| `alpha` | A → Z |
| `alpha_desc` | Z → A |
| `shortest` | Shortest word first |
| `longest` | Longest word first |
| `most_common` | Most common in modern published fiction first |
| `least_common` | Least common in modern published fiction first |
| `most_formal` | Most formal-register first (e.g. "lavatory" before "toilet" before "khazi") |
| `oldest` | Most archaic/dated/obsolete/historical-tagged first |
| `most_modern` | Least archaic/dated/obsolete/historical-tagged first |
| `most_lyrical` | Smoothest-sounding (fewest/shortest consonant clusters) first -- experimental |

```bash
revdict "happy" --sort alpha --no-interactive
revdict "blue*" --sort longest --no-interactive
revdict "toilet" --sort most_formal --no-interactive
```

`most_common`/`least_common` reuse the same literary-frequency data that
already nudges the default relevance ranking — a word with no frequency
data at all (very rare hyphenated/multi-word entries) sorts as if it had
zero frequency.

`most_formal`/`oldest`/`most_modern` reuse the same Wiktionary register
tags `--category old`/`--category idiom_slang` are built on (see below) —
they need the same reindex those categories need on an older index.
`most_formal` ranks by the formal ↔ informal spectrum only, not by
subject-matter domain — a legal term like "writ" is not specifically
detected as legal, only as (in this case) untagged/neutral register; true
topic/domain detection (e.g. distinguishing legal terms specifically) is
not yet implemented. These three sorts rank by whichever single sense of
a word actually matched your query, not by whether the word has *any*
tagged sense anywhere — a word like "glad" can have separate obsolete and
informal senses in Wiktionary, but if your query matches its plain,
untagged sense, it sorts as neutral/formal-tied rather than as old or
informal. Untagged/tied candidates keep their original relevance order
rather than moving to an arbitrary position.

`most_lyrical` is an experimental approximation of "smooth/euphonious
sounding" based on average consonant-cluster length in the word's
pronunciation — it needs the same `revdict build-index` reindex
`--syllables`/`--meter`/etc. need (see "Phonetic filters" below), and
words without precomputed phonetics data (multi-word/hyphenated
headwords, or an un-reindexed older index) sort last rather than being
excluded.
```

- [ ] **Step 2: Proofread against the actual implementation**

Re-read `src/revdict/sort.py` (as it stands after Tasks 2-3) side by side with this new README section, and re-read the existing "Category filter" and "Phonetic filters" sections just above/below this one for consistency of tone and cross-references. Confirm every claim in the new paragraphs (register-tag reuse, reindex requirement, matched-sense-only caveat, phonetics-missing-sorts-last) is literally true of the code as written -- fix any drift before committing.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Document the most_formal/oldest/most_modern/most_lyrical sort modes"
```

---

## Post-plan note for the final whole-branch reviewer

Per this project's established Phase 3/4 precedent, the final whole-branch review should include a real-data spot check: run a handful of real queries (e.g. "toilet", "drunk", "die", "happy") with `--sort most_formal`, `--sort oldest`, `--sort most_lyrical` against a rebuilt index and confirm the ordering is plausible and non-trivial (not a no-op), not just that the unit tests pass. This requires a `revdict build-index` reindex to populate `tags`/`phonetics` on the local test environment's index if it hasn't been rebuilt since Phase 3/4 shipped -- check `has_tags_nonnull`/`has_phonetics_nonnull` counts on the local metadata.jsonl first (as this plan's own scoping work did) to confirm whether a reindex is actually needed before spending the ~30 minutes on one.
