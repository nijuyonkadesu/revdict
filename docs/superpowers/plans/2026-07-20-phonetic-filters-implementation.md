# Phonetic Filters Implementation Plan (Phase 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 5 independently-combinable phonetic advanced filters to revdict — syllable count, primary vowel, rhymes-with, sounds-like, and meter (stress pattern) — per TODO.md's feature group 2 and the roadmap's Phase 4 row (`docs/superpowers/plans/2026-07-19-onelook-feature-parity-roadmap.md`).

**Architecture:** Phonetic data (syllable count, primary vowel, rhyme key, stress-pattern meter string, raw ARPAbet phonemes) is **precomputed at index-build time** and stored as a new `"phonetics"` metadata field — the same architecture Phase 3 used for `tags`, and for the same reason: filtering the candidate pool before `top_n` truncation needs a field already sitting on every metadata row, not a live per-candidate model call across thousands of rows per query. This requires a small, minimal extension to the sibling `stressmark` library (`../emphasis`) to expose raw phonemes on its `WordResult`, committed and pushed to that repo's own origin before any revdict code depends on it. The one place phonetics stays query-time-live is resolving the *target* word of `--rhymes-with`/`--sounds-like` (arbitrary, typed per query, unprecomputable) — this keeps a live stressmark dependency at query time for exactly those two flags, with an eager, clearly-worded failure if stressmark isn't available.

**Tech Stack:** Python 3.11+, existing revdict + stressmark stack. No new dependencies (edit-distance and all derivation logic are pure stdlib, confirmed below).

## Global Constraints

- Five independently-combinable filters (a candidate must satisfy every filter that was actually passed; filters the user didn't pass are no-ops): `--syllables N` (exact integer match), `--primary-vowel VOWEL` (an ARPAbet vowel symbol, case-insensitive, e.g. `AE`), `--rhymes-with WORD`, `--sounds-like WORD`, `--meter PATTERN` (a string of `/` and `x`, one char per syllable, e.g. `/x`, `x/`, `/xx`, `x/x`).
- **Scope decision, confirmed with the user before this plan was written:** TODO.md's "Starts with / Ends with / Letters count" advanced filters are NOT part of this phase (or any phase) — Phase 1's query syntax DSL already covers them exactly (`blue*`, `*bird`, `bl????rd`, even combined with meaning search via `bl*:snow`), and the user confirmed the DSL is sufficient; no dedicated CLI flags are being added for them.
- **Scope decision:** "Also related to" (TODO.md's remaining feature-group-2 item) is explicitly **deferred out of this phase** — it's a retrieval-path change (combining two query embedding vectors, or unioning two searches), unrelated code and risk profile to the phonetics-filter work here, sharing none of this phase's precomputed substrate. Not scheduled; pick it up as its own small plan whenever it's wanted.
- **`||` (the poetry caesura/phrase-boundary symbol in TODO.md's meter examples) is out of scope.** revdict filters single dictionary headwords, not full lines of poetry; `||` only makes sense across multiple words. `--meter` matches a single headword's own per-syllable stress pattern.
- **Phonetics precomputation architecture:** `"phonetics"` is a new metadata field, either a dict `{"syllable_count": int, "primary_vowel": str, "rhyme_key": str, "meter": str, "phonemes": list[str]}` or `None`. Computed once per metadata record at `revdict build-index` time (this phase requires a reindex, exactly like Phase 3's `tags` field — `noun`/`adjective`/etc. category filtering still needs no reindex; phonetic filters do).
- **Phonetics is `None` for multi-word headwords (contains a space) and hyphenated headwords (contains a hyphen).** This is not a simplification for convenience — it's a **measured, confirmed bug boundary**: `stressmark.engine.resolve_word_by_pos()` (the function this whole phase builds on) produces genuinely malformed output for both. Directly verified: `resolve_word_by_pos("kick the bucket", "noun")` returns syllables `['kick ', 'the ', 'buc', 'ket']` (nonsense fragments, meaningless `primary` index); `resolve_word_by_pos("well-known", "adjective")` returns syllables `['well', '', 'known']` — a literal empty-string fragment. Root cause: stressmark's hyphen/multi-word compound handling lives entirely in `stressmark.engine.analyze()`'s sentence-level post-processing pass (lines ~454-486 of `engine.py`), which `resolve_word_by_pos()` never invokes (it calls `resolve_word()` directly). This is a pre-existing stressmark limitation, not something this plan fixes — fixing it is future stressmark work, out of scope here. Excluding these headwords sidesteps it entirely; it does not create new risk (well-formed data or no data, never garbage data).
- **Measured, real numbers this plan's cost estimate is grounded in** (measured directly against the real on-disk index and real stressmark calls before this plan was written, not assumed): the corpus has 801,725 unique headwords; 565,911 of them (70.6%) are "clean" (no space, no hyphen) and get real phonetics; the other 235,814 (29.4%) get `phonetics: None`. Resolving all 565,911 clean headwords via `resolve_word_by_pos` measured at **0.68ms/word average** (3000-word and 2000-word real samples, zero malformed results after excluding multi-word/hyphenated) — **~6.4 minutes wall-clock for the full clean set**, negligible next to the existing embedding pass. Of a random sample of single-token headwords, only ~16% hit CMUdict directly; ~84% fall through to G2P prediction — expected for an exhaustive Wiktionary-backed corpus (chemistry terms, neologisms, rare/dialectal words, proper nouns), and already accounted for in the measured timing above (the sample was not filtered to only "easy" words).
- **Rhyme-key definition, pinned with real examples** (from the primary-stressed vowel, ARPAbet stress digits stripped, to the end of the word — the standard rhyming-dictionary convention): `"cat"` → phonemes `K AE1 T` → rhyme key `"AE T"`; `"hat"` → `HH AE1 T` → `"AE T"` — these rhyme (keys match). `"record"` (**noun**, heteronym-resolved) → `R EH1 K ER0 D` → rhyme key `"EH K ER D"`; `"chord"` → `K AO1 R D` → `"AO R D"` — these do **not** rhyme (keys differ), correctly. `"record"` (**verb**, heteronym-resolved) → `R IH0 K AO1 R D` → rhyme key `"AO R D"`; `"afford"` → `AH0 F AO1 R D` → `"AO R D"` — these **do** rhyme (keys match). The noun/verb divergence on the same spelling is a deliberate, correct consequence of using `resolve_word_by_pos`'s existing heteronym resolution rather than a naive CMUdict-first-entry lookup.
- **Meter-string definition, pinned with real examples** (one syllable = one character; `result.primary` and every index in `result.secondary` both render as `/`, everything else as `x` — both primary and secondary stress count as a strong beat, matching standard poetic-foot scanning): `"happy"` (adjective) → `/x` (trochee); `"record"` (verb) → `x/` (iamb); `"elephant"` → `/xx` (dactyl); `"banana"` → `x/x` (amphibrach); `"photograph"` → `/x/` (primary on syllable 0, secondary on syllable 2). All five confirmed directly against real `resolve_word_by_pos` output before writing this plan, and all five match TODO.md's own example patterns (`/x`, `x/`, `/xx`, `x/x`) exactly.
- **Sounds-like definition, pinned with real examples:** normalized Levenshtein edit distance over stress-stripped ARPAbet phoneme sequences (`edit_distance / max(len_a, len_b)`), threshold **≤ 0.34**. True homophones score 0.00 (confirmed: night/knight, night/nite, there/their, two/too, bear/bare, sight/site — all exactly 0.00). A one-phoneme substitution in a short word (`cat`/`bat`) scores 0.33 — intentionally still a match. A one-letter misspelling (`elephant`/`elifant`) scores 0.14. Unrelated words score far above the threshold: `cat`/`elephant` 0.86, `phone`/`photograph` 0.75. 0.34 sits just above the cat/bat case and well below every unrelated pair tested — pure stdlib Levenshtein, no new dependency.
- **`--rhymes-with`/`--sounds-like` still need stressmark at query time** — their target word is arbitrary, typed per query, and cannot be precomputed. If stressmark is unavailable, or the target word itself can't be resolved (e.g. it's multi-word/hyphenated, or stressmark is too old to expose `.phonemes`), `search()` must raise a clear, immediate `ValueError` — **not** silently return zero candidates. A silent-empty result reads as "nothing rhymes with X," which is a materially different (and wrong) message from "this feature needs stressmark installed." The target word's part of speech is not collected from the user (the CLI only takes a bare word) — it defaults to `"noun"` for resolution purposes; document this.
- **The `stressmark` extension (Task 1) must be committed AND pushed to its own GitHub origin** (`github.com/nijuyonkadesu/emphasis`), not left as a local-only change relying on this session's editable install. `import stressmark` in this venv currently resolves directly to `/home/shichika/redacted/emphasis/src/stressmark/__init__.py` via an editable install — convenient for this session, but every other real install (including a future fresh `revdict` checkout) needs the pushed commit to get the same code. Bump `emphasis/pyproject.toml`'s version (`0.1.0` → `0.2.0`) as part of this — gives future revdict work something concrete to reference even though `stressmark` still isn't a declared dependency (that gap is tracked separately, see the `revdict-backlog` memory / roadmap section 7).
- **Every new `search()`/daemon/CLI parameter defaults to `None`/no-op**, mirroring `sort_mode`/`category`'s existing contract exactly — all pre-existing behavior must be unchanged when none of the 5 phonetic flags are passed.
- **The final whole-branch review must include a real-data spot check**, not just the task-level tests below (which necessarily use synthetic fixtures or, for one integration test in Task 4, a handful of real single words — never a real reindexed corpus, since a full reindex is a multi-minute-to-real operation no task in this plan triggers). Every synthetic fixture is a claim about what real corpus data looks like; the 0.34 sounds-like threshold and the rhyme/meter definitions were tuned against 10-15 hand-picked word pairs, not the actual distribution of ~566K real headwords. Whoever runs the final review should: build (or have available) a real reindexed corpus, run each of the 5 filters at least once against it, and sanity-check the result counts are plausible (a few to a few dozen matches for a specific `--rhymes-with`/`--sounds-like` query — not zero, and not thousands, which would mean the threshold or a key-matching contract is wrong in a way no synthetic test could catch). This mirrors exactly how Phase 3's final review caught a real README inaccuracy specifically by checking the real on-disk index rather than trusting the task-level tests alone.
- **Do not add a `"phonetics"` field to `build_candidate()`'s output.** Filtering happens on raw metadata rows before candidates are ever built (same reasoning as Phase 3's `"tags"` decision) — YAGNI until a real consumer (e.g. a future TUI showing "rhymes with: ...") needs it displayed.

---

### Task 1 (cross-repo — `../emphasis`): Expose raw phonemes on `WordResult`

**Files (all in `/home/shichika/redacted/emphasis`, a sibling repo, NOT `rev-dictionary`):**
- Modify: `src/stressmark/engine.py`
- Modify: `pyproject.toml`
- Test: `tests/test_integration.py`

**Interfaces:**
- Consumes: nothing new (first task).
- Produces: every `WordResult` returned by `resolve_word_by_pos(word, pos)` (and by `resolve_word`/`analyze`, though this plan only consumes the former) now carries a `.phonemes: list[str]` attribute — the raw ARPAbet phoneme list of whichever pronunciation variant was ultimately used to resolve stress (heteronym-resolved variant, first CMUdict entry, or G2P prediction), or `[]` for reducible function words (which never get a phoneme lookup at all, unchanged from today).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_integration.py` (in the `emphasis` repo):

```python
from stressmark.engine import resolve_word_by_pos


def test_resolve_word_by_pos_exposes_phonemes_for_a_dictionary_word():
    result = resolve_word_by_pos("cat", "noun")
    assert result.phonemes == ["K", "AE1", "T"]


def test_resolve_word_by_pos_exposes_the_heteronym_resolved_phonemes():
    """record(noun) and record(verb) must expose DIFFERENT phoneme lists --
    proves .phonemes reflects the POS-resolved variant, not a naive
    first-CMUdict-entry lookup that would return the same phonemes
    regardless of POS."""
    noun = resolve_word_by_pos("record", "noun")
    verb = resolve_word_by_pos("record", "verb")
    assert noun.phonemes == ["R", "EH1", "K", "ER0", "D"]
    assert verb.phonemes == ["R", "IH0", "K", "AO1", "R", "D"]
    assert noun.phonemes != verb.phonemes


def test_resolve_word_by_pos_exposes_phonemes_for_a_g2p_predicted_word():
    """A word with essentially no chance of being in CMUdict -- proves the
    G2P-prediction branch also populates .phonemes, not just the two
    dictionary-lookup branches."""
    result = resolve_word_by_pos("zxqvorplitude", "noun")
    assert isinstance(result.phonemes, list)
    assert len(result.phonemes) > 0


def test_resolve_word_by_pos_gives_an_empty_phonemes_list_for_a_reducible_word():
    """Function words never get a phoneme lookup at all -- .phonemes must
    default to [] rather than being missing or None, so callers can always
    do `if result.phonemes:` without a hasattr/None check."""
    result = resolve_word_by_pos("the", "noun")
    assert result.cls == "reducible"
    assert result.phonemes == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run (from `/home/shichika/redacted/emphasis`): `pytest tests/test_integration.py -v -k phonemes`
Expected: FAIL with `AttributeError: 'WordResult' object has no attribute 'phonemes'`.

- [ ] **Step 3: Implement**

In `src/stressmark/engine.py`, add a default to `WordResult.__init__`:

```python
class WordResult:
    def __init__(self, raw):
        self.raw = raw
        self.is_word = False
        self.syllables = []
        self.primary = -1
        self.secondary = set()
        self.confidence = None
        self.tier = None
        self.rule = None
        self.cls = None
        self.tag = None
        self.compound_parts = None
        self.phonemes = []
```

In `resolve_word()`, the heteronym branch — add one line before its `return r`:

```python
    if lower in HETERONYMS:
        want_verb = tag in VERB_TAGS
        variant = HETERONYMS[lower]["verb"] if want_verb else HETERONYMS[lower]["noun"]
        phonetic_n = sum(1 for p in variant if p[-1].isdigit())
        ortho = syllabify(raw, min_syllables=phonetic_n)
        r.syllables = ortho
        primary, secondary = stress_positions_for_pron(variant, len(ortho), lower)
        r.primary, r.secondary, r.confidence = primary, secondary, "dict-pos-resolved"
        r.phonemes = variant
        return r
```

The plain dictionary-lookup branch — add one line before its `return r`:

```python
    if lower in _cmu:
        prons = _cmu[lower]
        pron = prons[0]
        phonetic_n = sum(1 for p in pron if p[-1].isdigit())
        ortho = syllabify(raw, min_syllables=phonetic_n)
        r.syllables = ortho
        primary, secondary = stress_positions_for_pron(pron, len(ortho), lower)
        r.primary, r.secondary = primary, secondary
        primaries = set(_primary_positions_with_one(prons))
        r.confidence = "dict-flagged" if (len(ortho) > 1 and len(primaries) > 1) else "dict"
        r.phonemes = pron
        return r
```

The G2P/predicted branch — add one line right after the `try`/`except` that computes `phones` (covers both the early-return single-syllable path and the final return, since `phones` is set either way, including to `[]` on exception):

```python
    try:
        phones = [p for p in _g2p(raw) if p != " "]
        phonetic_n = sum(1 for p in phones if p[-1].isdigit())
    except Exception:
        phones, phonetic_n = [], 1
    r.phonemes = phones

    ortho = syllabify(raw, min_syllables=max(phonetic_n, 1))
    r.syllables = ortho

    if phonetic_n <= 1 or len(ortho) == 1:
        r.primary, r.confidence = 0, ("dict" if phonetic_n <= 1 else "predicted")
        return r

    primary, secondary = stress_positions_for_pron(phones, len(ortho), lower)
    r.primary, r.secondary, r.confidence = primary, secondary, "predicted"
    return r
```

The reducible branch needs no change — `r.phonemes` already defaults to `[]` via the `__init__` change above.

In `pyproject.toml`, bump the version:

```toml
[project]
name = "stressmark"
version = "0.2.0"
```

(Only the `version` line changes; everything else in the file is unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_integration.py -v -k phonemes`
Expected: all PASS.

- [ ] **Step 5: Run emphasis's full test suite to confirm no regressions**

Run: `pytest` (from `/home/shichika/redacted/emphasis`)
Expected: all pre-existing tests still pass — this change is purely additive (one new field, four one-line assignments), nothing removed or restructured.

- [ ] **Step 6: Commit and push**

```bash
cd /home/shichika/redacted/emphasis
git add src/stressmark/engine.py pyproject.toml tests/test_integration.py
git commit -m "feat: expose raw ARPAbet phonemes on WordResult for downstream phonetic filtering"
git push origin main
```

Confirm the push succeeded (`git log origin/main -1` should show this commit) before starting Task 2 — every later task in this plan depends on this being real, pushed, external state, not just what happens to be on disk in this session's editable install.

---

### Task 2: `src/revdict/models/phonetics.py` — stressmark-wrapping resolution

**Files:**
- Create: `src/revdict/models/phonetics.py`
- Test: `tests/models/test_phonetics.py`

**Interfaces:**
- Consumes: `stressmark.engine.resolve_word_by_pos` with the `.phonemes` field from Task 1 (already committed and pushed there; this repo's editable install already sees it immediately).
- Produces: `is_available() -> bool`, `resolve(word: str, pos: str) -> dict | None`. `resolve()`'s return shape (when not `None`): `{"syllable_count": int, "primary_vowel": str, "rhyme_key": str, "meter": str, "phonemes": list[str]}`. This is the single function both Task 3 (index-build-time precomputation, called for every clean headword) and Task 5 (query-time target-word resolution for `--rhymes-with`/`--sounds-like`) call.

- [ ] **Step 1: Write the failing tests**

Create `tests/models/test_phonetics.py` (create the `tests/models/` directory if it doesn't already exist — check first: `ls tests/models/` — this repo's existing `src/revdict/models/` has no matching `tests/models/` directory yet, so also add `tests/models/__init__.py` if the directory needs one to be importable; check whether `tests/data/` — the existing sibling test directory — has an `__init__.py` to match this repo's convention before deciding):

```python
from revdict.models import phonetics


def test_resolve_returns_the_expected_shape_for_cat():
    result = phonetics.resolve("cat", "noun")
    assert result == {
        "syllable_count": 1,
        "primary_vowel": "AE",
        "rhyme_key": "AE T",
        "meter": "/",
        "phonemes": ["K", "AE1", "T"],
    }


def test_resolve_distinguishes_the_record_noun_verb_heteronym_pair():
    noun = phonetics.resolve("record", "noun")
    verb = phonetics.resolve("record", "verb")
    assert noun["meter"] == "/x"
    assert verb["meter"] == "x/"
    assert noun["rhyme_key"] != verb["rhyme_key"]


def test_resolve_matches_the_pinned_meter_examples_from_the_plan():
    assert phonetics.resolve("happy", "adjective")["meter"] == "/x"
    assert phonetics.resolve("elephant", "noun")["meter"] == "/xx"
    assert phonetics.resolve("banana", "noun")["meter"] == "x/x"
    assert phonetics.resolve("photograph", "noun")["meter"] == "/x/"


def test_resolve_returns_none_for_a_multi_word_headword():
    assert phonetics.resolve("kick the bucket", "verb") is None


def test_resolve_returns_none_for_a_hyphenated_headword():
    assert phonetics.resolve("well-known", "adjective") is None


def test_resolve_never_raises_on_a_nonsense_word():
    """resolve() must be safe to call across the whole corpus during a
    reindex -- some malformed/unusual headword must never crash the whole
    build."""
    result = phonetics.resolve("", "noun")
    assert result is None or isinstance(result, dict)


def test_is_available_is_true_when_stressmark_is_importable_and_current():
    assert phonetics.is_available() is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/models/test_phonetics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'revdict.models.phonetics'`.

- [ ] **Step 3: Implement**

Create `src/revdict/models/phonetics.py`:

```python
try:
    import stressmark.engine as _engine
except ImportError:
    _engine = None


def is_available() -> bool:
    """True only when stressmark is installed AND new enough to expose
    .phonemes on its WordResult (Task 1 of this plan) -- an older
    stressmark install would otherwise silently produce phonetics-free
    results instead of a clear signal that an upgrade is needed."""
    if _engine is None:
        return False
    try:
        probe = _engine.WordResult("")
    except Exception:
        return False
    return hasattr(probe, "phonemes")


def _strip_stress(phoneme: str) -> str:
    return phoneme.rstrip("012")


def _phonetic_primary_index(phonemes: list[str]) -> int:
    """Mirrors stressmark.engine.stress_positions_for_pron's own
    fallback-to-0 convention (the first phoneme marked stress '1', or
    syllable 0 if none is marked) -- kept consistent with how stressmark
    itself picks a word's primary-stressed syllable."""
    for i, p in enumerate(phonemes):
        if p[-1] == "1":
            return i
    return 0


def resolve(word: str, pos: str) -> dict | None:
    """Full phonetic resolution for a single word -- used both at index-
    build time (every clean headword in the corpus) and at query time (an
    arbitrary --rhymes-with/--sounds-like target). Returns None, and never
    raises, when: stressmark is unavailable or too old (is_available() is
    False), `word` contains a space or hyphen (resolve_word_by_pos
    produces malformed syllable/stress data for both -- confirmed directly
    against real headwords like "kick the bucket" and "well-known" before
    this module was written, see the plan's Global Constraints), or any
    other unexpected failure occurs resolving this specific word. This
    must never crash a reindex, or a live query, over one weird headword.
    """
    if not is_available():
        return None
    if " " in word or "-" in word:
        return None
    try:
        result = _engine.resolve_word_by_pos(word, pos)
        phonemes = result.phonemes
        if not phonemes:
            return None
        syllable_count = len(result.syllables)
        idx = _phonetic_primary_index(phonemes)
        primary_vowel = _strip_stress(phonemes[idx])
        rhyme_key = " ".join(_strip_stress(p) for p in phonemes[idx:])
        stressed = {result.primary} | set(result.secondary)
        meter = "".join("/" if i in stressed else "x" for i in range(syllable_count))
        return {
            "syllable_count": syllable_count,
            "primary_vowel": primary_vowel,
            "rhyme_key": rhyme_key,
            "meter": meter,
            "phonemes": list(phonemes),
        }
    except Exception:
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/models/test_phonetics.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `pytest`
Expected: previous pass count + these new tests, same 2 pre-existing `FORCE_COLOR` failures (`test_main_error_message_is_not_mangled_by_rich_markup`, `test_main_routes_daemon_status`) and nothing else.

- [ ] **Step 6: Commit**

```bash
git add src/revdict/models/phonetics.py tests/models/test_phonetics.py
git commit -m "feat: add models/phonetics.py, a stressmark-wrapping phonetic resolver"
```

(If `tests/models/__init__.py` was needed per Step 1's directory check, include it in this commit too.)

---

### Task 3: Precompute phonetics into metadata at index-build time

**Files:**
- Modify: `src/revdict/data/build_index.py`
- Test: `tests/data/test_build_index.py`

**Interfaces:**
- Consumes: `revdict.models.phonetics.resolve(word: str, pos: str) -> dict | None` (Task 2).
- Produces: `build_metadata_record()`'s output gains a `"phonetics"` key (the dict from Task 2's `resolve()`, or `None`). Every row written to `metadata.jsonl` after a `revdict build-index` run now carries this field.

- [ ] **Step 1: Write the failing tests**

Add to `tests/data/test_build_index.py`:

```python
def test_build_metadata_record_includes_phonetics_when_present():
    record = {
        "headword": "cat",
        "pos": "noun",
        "definition": "a small domesticated carnivore",
        "examples": [],
        "source": "wordnet",
        "phonetics": {
            "syllable_count": 1,
            "primary_vowel": "AE",
            "rhyme_key": "AE T",
            "meter": "/",
            "phonemes": ["K", "AE1", "T"],
        },
    }

    meta = build_metadata_record(record)

    assert meta["phonetics"] == {
        "syllable_count": 1,
        "primary_vowel": "AE",
        "rhyme_key": "AE T",
        "meter": "/",
        "phonemes": ["K", "AE1", "T"],
    }


def test_build_metadata_record_defaults_phonetics_to_none_when_absent():
    record = {
        "headword": "kick the bucket",
        "pos": "verb",
        "definition": "to die",
        "examples": [],
        "source": "wordnet",
    }

    meta = build_metadata_record(record)

    assert meta["phonetics"] is None
```

Add one integration-level test proving `build()` actually calls the resolver and attaches results (mock `phonetics.resolve` rather than calling the real, slow-ish resolver in a unit test):

```python
def test_build_attaches_phonetics_to_every_record(monkeypatch, tmp_path):
    """The precomputation pass itself: build() must call
    phonetics.resolve(headword, pos) for every merged record and store the
    result on record["phonetics"] before build_metadata_record ever runs,
    not leave it to be computed lazily later."""
    import revdict.data.build_index as build_index_module

    fake_records = [
        {"headword": "cat", "pos": "noun", "definition": "d1", "examples": [], "source": "wordnet"},
        {"headword": "run", "pos": "verb", "definition": "d2", "examples": [], "source": "wordnet"},
    ]
    monkeypatch.setattr(build_index_module, "load_wordnet_senses", lambda: fake_records)
    monkeypatch.setattr(build_index_module, "download_raw_wiktextract", lambda path: None)
    monkeypatch.setattr(build_index_module, "stream_filtered_entries_from_gzip", lambda path: iter(()))

    calls = []

    def fake_resolve(word, pos):
        calls.append((word, pos))
        return {"syllable_count": 1, "primary_vowel": "AE", "rhyme_key": "AE T", "meter": "/", "phonemes": ["X"]}

    monkeypatch.setattr(build_index_module.phonetics, "resolve", fake_resolve)

    # Stub every other slow/network-touching step so this test exercises
    # only the phonetics-attachment wiring, matching this file's existing
    # convention for build()-level tests (see the emolex/literary-frequency
    # stubs already used elsewhere in this test file for the same reason).
    monkeypatch.setattr(build_index_module, "load_emolex", lambda: {})
    monkeypatch.setattr(build_index_module, "lookup_emolex", lambda word, emolex: None)
    monkeypatch.setattr(build_index_module, "download_raw_ngram_fiction", lambda path: None)
    monkeypatch.setattr(build_index_module, "download_raw_ngram_fiction_totalcounts", lambda path: None)
    monkeypatch.setattr(build_index_module, "compute_literary_frequencies", lambda headwords, a, b: {})

    class FakeEmbedder:
        def encode_passages(self, texts):
            import numpy as np

            return np.zeros((len(texts), 4), dtype="float32")

    monkeypatch.setattr(build_index_module, "Embedder", lambda: FakeEmbedder())
    monkeypatch.setattr(build_index_module, "INDEX_DIR", tmp_path)

    build_index_module.build(skip_confirm=True)

    assert ("cat", "noun") in calls
    assert ("run", "verb") in calls
```

If any of the monkeypatched names above don't match `build_index.py`'s actual current imports (check with `grep -n "^from\|^import" src/revdict/data/build_index.py` first), adjust the patch targets to match reality — the intent (stub every slow/network step, verify `phonetics.resolve` is called once per record with the right args) is what matters, not the exact patch list if the module's imports have shifted.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/data/test_build_index.py -v`
Expected: the two `build_metadata_record` tests FAIL with a `KeyError`/assertion mismatch (no `"phonetics"` key yet); the `build()` integration test FAILS with an `AttributeError` (`build_index_module.phonetics` doesn't exist yet).

- [ ] **Step 3: Implement**

In `src/revdict/data/build_index.py`, add the import alongside the existing `from revdict...` imports:

```python
from revdict.models import phonetics
```

Change `build_metadata_record`:

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
        "phonetics": record.get("phonetics"),
    }
```

In `build()`, add the precomputation pass. Place it after the `merge_records(...)` call and before the EmoLex-attachment loop (read the current file first to find the exact surrounding lines — `grep -n "Merging corpus\|Attaching NRC EmoLex" src/revdict/data/build_index.py` — and insert between them):

```python
    print("Precomputing phonetics (syllables, rhyme key, meter)...")
    phonetics_cache: dict[tuple[str, str], dict | None] = {}
    for record in records:
        cache_key = (record["headword"].lower(), record["pos"])
        if cache_key not in phonetics_cache:
            phonetics_cache[cache_key] = phonetics.resolve(record["headword"], record["pos"])
        record["phonetics"] = phonetics_cache[cache_key]
```

(The `(headword.lower(), pos)` cache avoids redundant resolution work when the same headword appears under the same part of speech across multiple WordNet/Wiktionary senses — many words do. This is a plain memory-dict cache scoped to one `build()` call, nothing persisted.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/test_build_index.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `pytest`
Expected: previous pass count + these new tests, same 2 pre-existing `FORCE_COLOR` failures.

- [ ] **Step 6: Commit**

```bash
git add src/revdict/data/build_index.py tests/data/test_build_index.py
git commit -m "feat: precompute phonetics (syllables, rhyme key, meter) into metadata at index-build time"
```

---

### Task 4: `src/revdict/phonetics.py` — matching predicates

**Files:**
- Create: `src/revdict/phonetics.py`
- Test: `tests/test_phonetics.py`

**Interfaces:**
- Consumes: nothing new for the unit tests (pure functions operating on whatever `"phonetics"` dict a metadata record happens to carry). The one exception is this task's final integration test, which deliberately DOES depend on Task 2's real `models.phonetics.resolve()` — see Step 1's last test and the note below it.
- Produces: `matches_syllable_count(record, syllables) -> bool`, `matches_primary_vowel(record, vowel) -> bool`, `matches_rhyme(record, target_rhyme_key) -> bool`, `matches_meter(record, target_meter) -> bool`, `matches_sounds_like(record, target_phonemes) -> bool`, `SOUNDS_LIKE_THRESHOLD: float`. Consumed by Tasks 5 and 6.
- **Why this task also tests against Task 2's real output, not just synthetic fixtures:** every other test in this file hand-writes its own `"phonetics"` dict, and every test in Task 2 checks `resolve()`'s output shape in isolation — nothing anywhere asserts that the dict `resolve()` *actually produces* is the dict these predicates *actually read*. If the two tasks silently disagreed on a key name (e.g. one used `"syllables"`, the other reads `"syllable_count"`), every test in both tasks would still pass (each checks its own shape), and the real, integrated behavor would be "every phonetic filter silently matches nothing" — the exact silent-empty failure mode this plan works hard to avoid for the stressmark-unavailable case, hiding instead at this module seam. One integration test closes it for good.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_phonetics.py`:

```python
import pytest

from revdict.phonetics import (
    SOUNDS_LIKE_THRESHOLD,
    matches_meter,
    matches_primary_vowel,
    matches_rhyme,
    matches_sounds_like,
    matches_syllable_count,
)

_CAT = {"phonetics": {"syllable_count": 1, "primary_vowel": "AE", "rhyme_key": "AE T", "meter": "/", "phonemes": ["K", "AE1", "T"]}}
_BAT = {"phonetics": {"syllable_count": 1, "primary_vowel": "AE", "rhyme_key": "AE T", "meter": "/", "phonemes": ["B", "AE1", "T"]}}
_DOG = {"phonetics": {"syllable_count": 1, "primary_vowel": "AO", "rhyme_key": "AO G", "meter": "/", "phonemes": ["D", "AO1", "G"]}}
_ELEPHANT = {"phonetics": {"syllable_count": 3, "primary_vowel": "EH", "rhyme_key": "EH L AH F AH N T", "meter": "/xx", "phonemes": ["EH1", "L", "AH0", "F", "AH0", "N", "T"]}}
_NO_PHONETICS = {"phonetics": None}


def test_matches_syllable_count_none_is_a_noop():
    assert matches_syllable_count(_CAT, None) is True
    assert matches_syllable_count(_NO_PHONETICS, None) is True


def test_matches_syllable_count_exact_match_only():
    assert matches_syllable_count(_CAT, 1) is True
    assert matches_syllable_count(_ELEPHANT, 1) is False
    assert matches_syllable_count(_ELEPHANT, 3) is True


def test_matches_syllable_count_false_when_phonetics_is_none():
    assert matches_syllable_count(_NO_PHONETICS, 1) is False


def test_matches_primary_vowel_none_or_empty_is_a_noop():
    assert matches_primary_vowel(_CAT, None) is True
    assert matches_primary_vowel(_CAT, "") is True


def test_matches_primary_vowel_case_insensitive_exact_match():
    assert matches_primary_vowel(_CAT, "AE") is True
    assert matches_primary_vowel(_CAT, "ae") is True
    assert matches_primary_vowel(_DOG, "AE") is False


def test_matches_primary_vowel_false_when_phonetics_is_none():
    assert matches_primary_vowel(_NO_PHONETICS, "AE") is False


def test_matches_rhyme_none_or_empty_is_a_noop():
    assert matches_rhyme(_CAT, None) is True
    assert matches_rhyme(_CAT, "") is True


def test_matches_rhyme_exact_key_match():
    assert matches_rhyme(_CAT, "AE T") is True
    assert matches_rhyme(_BAT, "AE T") is True
    assert matches_rhyme(_DOG, "AE T") is False


def test_matches_meter_none_or_empty_is_a_noop():
    assert matches_meter(_ELEPHANT, None) is True
    assert matches_meter(_ELEPHANT, "") is True


def test_matches_meter_exact_pattern_match():
    assert matches_meter(_ELEPHANT, "/xx") is True
    assert matches_meter(_ELEPHANT, "/x") is False
    assert matches_meter(_CAT, "/") is True


def test_matches_sounds_like_none_or_empty_target_is_a_noop():
    assert matches_sounds_like(_CAT, None) is True
    assert matches_sounds_like(_CAT, []) is True


def test_matches_sounds_like_exact_homophone_matches():
    # "cat" against its own phonemes -- distance 0, must match regardless
    # of threshold value.
    assert matches_sounds_like(_CAT, ["K", "AE1", "T"]) is True


def test_matches_sounds_like_one_phoneme_substitution_matches():
    # cat vs bat: real measured normalized distance 0.33, which is <=
    # SOUNDS_LIKE_THRESHOLD (0.34) -- pinned in the plan's Global
    # Constraints as an intentional match.
    assert matches_sounds_like(_CAT, ["B", "AE1", "T"]) is True


def test_matches_sounds_like_unrelated_word_does_not_match():
    # cat vs elephant: real measured normalized distance 0.86, far above
    # threshold.
    assert matches_sounds_like(_CAT, ["EH1", "L", "AH0", "F", "AH0", "N", "T"]) is False


def test_matches_sounds_like_false_when_phonetics_is_none():
    assert matches_sounds_like(_NO_PHONETICS, ["K", "AE1", "T"]) is False


def test_sounds_like_threshold_is_the_pinned_value():
    assert SOUNDS_LIKE_THRESHOLD == 0.34


def test_matching_predicates_actually_consume_real_resolve_output():
    """Closes the one seam no other test in this plan covers: every other
    test here hand-writes its own "phonetics" dict, and Task 2's own tests
    check resolve()'s output shape in isolation -- nothing asserts the
    dict resolve() PRODUCES is the dict these predicates ACTUALLY READ. If
    the two tasks silently disagreed on a key name, every test in both
    tasks would still pass individually, and the real integrated behavior
    would be "every phonetic filter matches nothing" -- a silent failure
    that would only surface after a real multi-hour reindex. This test
    uses Task 2's real resolve() (skipped if stressmark isn't installed --
    same guard as this plan's other real-stressmark tests), builds a
    metadata list from its real output, and runs the real predicates
    end-to-end."""
    import pytest as _pytest

    from revdict.models import phonetics as phonetics_models

    if not phonetics_models.is_available():
        _pytest.skip("requires stressmark to be installed")

    cat = {"phonetics": phonetics_models.resolve("cat", "noun")}
    dog = {"phonetics": phonetics_models.resolve("dog", "noun")}
    hat = {"phonetics": phonetics_models.resolve("hat", "noun")}

    assert matches_rhyme(cat, hat["phonetics"]["rhyme_key"]) is True
    assert matches_rhyme(dog, hat["phonetics"]["rhyme_key"]) is False
    assert matches_syllable_count(cat, 1) is True
    assert matches_sounds_like(cat, hat["phonetics"]["phonemes"]) is True
    assert matches_sounds_like(cat, dog["phonetics"]["phonemes"]) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_phonetics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'revdict.phonetics'`.

- [ ] **Step 3: Implement**

Create `src/revdict/phonetics.py`:

```python
def _levenshtein(a: list[str], b: list[str]) -> int:
    """Standard edit distance over two phoneme-symbol sequences. Pure
    stdlib -- no new dependency, as the roadmap anticipated for this
    phase."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


SOUNDS_LIKE_THRESHOLD = 0.34
"""Normalized edit distance (edit_distance / max(len_a, len_b)) at or below
which two words are considered to "sound like" each other. Calibrated
against real ARPAbet phoneme sequences (stress digits stripped): true
homophones score 0.00 (night/knight, there/their, two/too, bear/bare,
sight/site all measured exactly 0.00); a one-phoneme substitution in a
short word like cat/bat measures 0.33 and is intentionally still a match;
a one-letter misspelling like elephant/elifant measures 0.14. Unrelated
words measure far higher: cat/elephant 0.86, phone/photograph 0.75. 0.34
sits just above the cat/bat case and well below every unrelated pair
tested."""


def matches_syllable_count(record: dict, syllables: int | None) -> bool:
    if syllables is None:
        return True
    phonetics = record.get("phonetics")
    return phonetics is not None and phonetics["syllable_count"] == syllables


def matches_primary_vowel(record: dict, vowel: str | None) -> bool:
    if not vowel:
        return True
    phonetics = record.get("phonetics")
    return phonetics is not None and phonetics["primary_vowel"] == vowel.upper()


def matches_rhyme(record: dict, target_rhyme_key: str | None) -> bool:
    if not target_rhyme_key:
        return True
    phonetics = record.get("phonetics")
    return phonetics is not None and phonetics["rhyme_key"] == target_rhyme_key


def matches_meter(record: dict, target_meter: str | None) -> bool:
    if not target_meter:
        return True
    phonetics = record.get("phonetics")
    return phonetics is not None and phonetics["meter"] == target_meter


def matches_sounds_like(record: dict, target_phonemes: list[str] | None) -> bool:
    if not target_phonemes:
        return True
    phonetics = record.get("phonetics")
    if phonetics is None:
        return False
    candidate = [p.rstrip("012") for p in phonetics["phonemes"]]
    target = [p.rstrip("012") for p in target_phonemes]
    distance = _levenshtein(candidate, target)
    return distance / max(len(candidate), len(target), 1) <= SOUNDS_LIKE_THRESHOLD
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_phonetics.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `pytest`
Expected: previous pass count + these new tests, same 2 pre-existing `FORCE_COLOR` failures.

- [ ] **Step 6: Commit**

```bash
git add src/revdict/phonetics.py tests/test_phonetics.py
git commit -m "feat: add phonetics.py matching predicates for the 5 phonetic filters"
```

---

### Task 5: Wire phonetic filters into `search()`'s meaning/combined path

**Files:**
- Modify: `src/revdict/search.py`
- Test: `tests/test_search.py`

**Interfaces:**
- Consumes: `revdict.phonetics.matches_syllable_count/matches_primary_vowel/matches_rhyme/matches_meter/matches_sounds_like` (Task 4); `revdict.models.phonetics.resolve/is_available` (Task 2).
- Produces: `search(query, top_n=10, sort_mode=None, category=None, syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None) -> dict` — 5 new parameters. Also produces `filter_by_phonetics(scored_rows, metadata, syllables, primary_vowel, rhyme_key, sounds_like_phonemes, meter)` next to `filter_by_category`, and `resolve_phonetic_target(word, flag_name)` (a small helper that resolves `--rhymes-with`/`--sounds-like`'s raw word argument, raising a clear `ValueError` on failure).
- **Scope note, matching Phase 3's Task 3/Task 4 split precedent exactly:** this task covers ONLY the meaning/combined-mode path. The structural/expand/phrase_contains dispatch branch is left completely unchanged here — Task 6 wires phonetics into `structural_search.run_structural` separately, to avoid a signature-mismatch regression if both were touched in the same task (see the roadmap's Phase 3 precedent for why this split matters: a task that adds a new keyword to a call site before the callee accepts it breaks every test exercising that call site for the duration between the two tasks).

- [ ] **Step 1: Write the failing tests**

Add these test functions to `tests/test_search.py` (place them after the existing category tests):

```python
def _phonetics_dict(syllable_count, primary_vowel, rhyme_key, meter, phonemes):
    return {
        "syllable_count": syllable_count,
        "primary_vowel": primary_vowel,
        "rhyme_key": rhyme_key,
        "meter": meter,
        "phonemes": phonemes,
    }


def test_search_phonetic_filters_none_is_a_noop(monkeypatch):
    metadata = [
        {
            "headword": "bluebird", "pos": "noun", "definition": "a songbird",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
            "phonetics": _phonetics_dict(2, "UW", "UW B ER D", "/x", ["B", "L", "UW1", "B", "ER0", "D"]),
        },
        {
            "headword": "blue", "pos": "adjective", "definition": "the color of the sky",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
            "phonetics": _phonetics_dict(1, "UW", "UW", "/", ["B", "L", "UW1"]),
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

    assert {c["headword"] for c in result["candidates"]} == {"bluebird", "blue"}


def test_search_syllables_filter_restricts_candidates(monkeypatch):
    metadata = [
        {
            "headword": "bluebird", "pos": "noun", "definition": "a songbird",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
            "phonetics": _phonetics_dict(2, "UW", "UW B ER D", "/x", ["B", "L", "UW1", "B", "ER0", "D"]),
        },
        {
            "headword": "blue", "pos": "adjective", "definition": "the color of the sky",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
            "phonetics": _phonetics_dict(1, "UW", "UW", "/", ["B", "L", "UW1"]),
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

    result = search_mod.search("sky color", top_n=10, syllables=1)

    assert [c["headword"] for c in result["candidates"]] == ["blue"]


def test_search_phonetic_filters_apply_before_top_n_truncation(monkeypatch):
    """Same ordering guarantee as category: the highest-scoring candidate
    fails the syllables filter, two lower-scoring ones pass -- top_n=2
    must return both, not just one, proving the filter runs before
    truncation, not after."""
    metadata = [
        {
            "headword": "bluely", "pos": "adverb", "definition": "a common sense",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
            "phonetics": _phonetics_dict(2, "UW", "UW L IY", "/x", ["B", "L", "UW1", "L", "IY0"]),
        },
        {
            "headword": "blueness", "pos": "noun", "definition": "a rare noun sense one",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
            "phonetics": _phonetics_dict(1, "UW", "UW N AH S", "/", ["B", "L", "UW1", "N", "AH0", "S"]),
        },
        {
            "headword": "bluebell", "pos": "noun", "definition": "a rare noun sense two",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
            "phonetics": _phonetics_dict(1, "UW", "UW B EH L", "/", ["B", "L", "UW1", "B", "EH1", "L"]),
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
            score_by_definition = {"a common sense": 5.0, "a rare noun sense one": 3.0, "a rare noun sense two": 2.0}
            return [score_by_definition[d] for d in definitions]

    state["embedder"] = FakeEmbedder()
    state["reranker"] = FakeReranker()
    state["embeddings"] = np.array([[1.0, 0.0]] * 3, dtype="float32")
    state["embedding_norms"] = np.array([1.0, 1.0, 1.0])
    monkeypatch.setattr(search_mod, "_load_state", lambda: state)

    result = search_mod.search("blue things", top_n=2, syllables=1)

    assert {c["headword"] for c in result["candidates"]} == {"blueness", "bluebell"}


def test_search_meter_filter_restricts_candidates(monkeypatch):
    metadata = [
        {
            "headword": "happy", "pos": "adjective", "definition": "feeling joy",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
            "phonetics": _phonetics_dict(2, "AE", "AE P IY", "/x", ["HH", "AE1", "P", "IY0"]),
        },
        {
            "headword": "glad", "pos": "adjective", "definition": "feeling joy too",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
            "phonetics": _phonetics_dict(1, "AE", "AE D", "/", ["G", "L", "AE1", "D"]),
        },
    ]
    state = {
        "metadata": metadata,
        "word_index": {"happy": [0], "glad": [1]},
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

    result = search_mod.search("feeling joy", top_n=10, meter="/x")

    assert [c["headword"] for c in result["candidates"]] == ["happy"]


def test_search_rhymes_with_resolves_the_target_and_filters(monkeypatch):
    metadata = [
        {
            "headword": "cat", "pos": "noun", "definition": "a small carnivore",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
            "phonetics": _phonetics_dict(1, "AE", "AE T", "/", ["K", "AE1", "T"]),
        },
        {
            "headword": "dog", "pos": "noun", "definition": "a small carnivore too",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
            "phonetics": _phonetics_dict(1, "AO", "AO G", "/", ["D", "AO1", "G"]),
        },
    ]
    state = {
        "metadata": metadata,
        "word_index": {"cat": [0], "dog": [1]},
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

    result = search_mod.search("small carnivore", top_n=10, rhymes_with="hat")

    assert [c["headword"] for c in result["candidates"]] == ["cat"]


def test_search_rhymes_with_raises_when_stressmark_is_unavailable(monkeypatch):
    state = {
        "metadata": [],
        "word_index": {},
        "literary_frequency": {},
        "classifier": None,
    }
    monkeypatch.setattr(search_mod, "_load_state", lambda: state)
    monkeypatch.setattr(search_mod.phonetics_models, "is_available", lambda: False)

    with pytest.raises(ValueError, match="stressmark"):
        search_mod.search("anything", top_n=10, rhymes_with="hat")


def test_search_category_and_phonetics_filters_combine(monkeypatch):
    """Filters from different phases must AND together, not override each
    other -- category (Phase 3) and syllables (Phase 4) both apply."""
    metadata = [
        {
            "headword": "bluebird", "pos": "noun", "definition": "a songbird",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
            "phonetics": _phonetics_dict(2, "UW", "UW B ER D", "/x", ["B", "L", "UW1", "B", "ER0", "D"]),
        },
        {
            "headword": "blue", "pos": "adjective", "definition": "the color of the sky",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
            "phonetics": _phonetics_dict(1, "UW", "UW", "/", ["B", "L", "UW1"]),
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

    result = search_mod.search("sky color", top_n=10, category="noun", syllables=2)

    assert [c["headword"] for c in result["candidates"]] == ["bluebird"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_search.py -k "phonetic or syllables or meter or rhymes" -v`
Expected: FAIL with `TypeError: search() got an unexpected keyword argument 'syllables'` (and similarly for the other new keywords).

- [ ] **Step 3: Implement**

In `src/revdict/search.py`, add imports alongside the existing `from revdict import ...` lines:

```python
from revdict import phonetics
from revdict.models import phonetics as phonetics_models
```

(Two separate imports because Task 2's module lives at `revdict.models.phonetics` — the stressmark-wrapping resolver — and Task 4's lives at `revdict.phonetics` — the pure matching predicates. Aliasing the models one to `phonetics_models` avoids colliding with the top-level `phonetics` import.)

Add `filter_by_phonetics` and `resolve_phonetic_target` next to `filter_by_category`:

```python
def filter_by_phonetics(
    scored_rows: list[tuple[int, float]],
    metadata: list[dict],
    syllables: int | None,
    primary_vowel: str | None,
    rhyme_key: str | None,
    sounds_like_phonemes: list[str] | None,
    meter: str | None,
) -> list[tuple[int, float]]:
    """Same before-top_n-truncation contract as filter_by_category -- see
    that function's docstring. All 5 filters AND together; each is
    individually a no-op when its argument is falsy/None."""
    if not any([syllables, primary_vowel, rhyme_key, sounds_like_phonemes, meter]):
        return scored_rows
    return [
        (index, score)
        for index, score in scored_rows
        if phonetics.matches_syllable_count(metadata[index], syllables)
        and phonetics.matches_primary_vowel(metadata[index], primary_vowel)
        and phonetics.matches_rhyme(metadata[index], rhyme_key)
        and phonetics.matches_sounds_like(metadata[index], sounds_like_phonemes)
        and phonetics.matches_meter(metadata[index], meter)
    ]


def resolve_phonetic_target(word: str, flag_name: str) -> dict:
    """Resolves an arbitrary user-typed word (the target of --rhymes-with
    or --sounds-like) into its phonetic data at QUERY time -- this is the
    one place phonetic filtering still depends on stressmark being
    available live, since the target is unprecomputable. Raises
    ValueError (never returns None) so a missing/outdated stressmark, or
    an unresolvable target word, surfaces as a clear error message instead
    of silently behaving like "nothing matches"."""
    if not phonetics_models.is_available():
        raise ValueError(
            f"--{flag_name} requires the stressmark library (>= 0.2.0) to be installed and importable."
        )
    resolved = phonetics_models.resolve(word, "noun")
    if resolved is None:
        raise ValueError(f"Could not resolve a pronunciation for --{flag_name} target {word!r}.")
    return resolved
```

Change `search()`'s signature:

```python
def search(
    query: str,
    top_n: int = 10,
    sort_mode: str | None = None,
    category: str | None = None,
    syllables: int | None = None,
    primary_vowel: str | None = None,
    rhymes_with: str | None = None,
    sounds_like: str | None = None,
    meter: str | None = None,
) -> dict:
```

Right after the existing category eager-guard (before `parsed = query_syntax.parse_query(query)`), add the target-word resolution for `--rhymes-with`/`--sounds-like` — eager, so it applies uniformly regardless of `parsed.mode`, exactly like the category guard:

```python
    rhyme_key = None
    if rhymes_with:
        rhyme_key = resolve_phonetic_target(rhymes_with, "rhymes-with")["rhyme_key"]

    sounds_like_phonemes = None
    if sounds_like:
        sounds_like_phonemes = resolve_phonetic_target(sounds_like, "sounds-like")["phonemes"]
```

In the structural-mode dispatch branch, leave it completely unchanged (Task 6's job):

```python
    parsed = query_syntax.parse_query(query)
    if parsed.mode in ("structural", "expand", "phrase_contains"):
        result = structural_search.run_structural(parsed, state, top_n, category=category)
        result["candidates"] = sort.apply_sort(
            result["candidates"], sort_mode, state["literary_frequency"]
        )
        return result
```

In the meaning/combined-mode branch, change:

```python
    deduped = dedupe_by_headword(scored, metadata)
    deduped = exclude_headword(deduped, metadata, exact_headword)
    # category never filters the exact-match panel above -- it narrows the
    # candidate list only, so a query like "run" --category noun still
    # shows the verb sense of "run" in the exact-match block.
    deduped = filter_by_category(deduped, metadata, category)[:top_n]
```

to:

```python
    deduped = dedupe_by_headword(scored, metadata)
    deduped = exclude_headword(deduped, metadata, exact_headword)
    # category/phonetics never filter the exact-match panel above -- they
    # narrow the candidate list only, so a query like "run" --category
    # noun still shows the verb sense of "run" in the exact-match block.
    deduped = filter_by_category(deduped, metadata, category)
    deduped = filter_by_phonetics(
        deduped, metadata, syllables, primary_vowel, rhyme_key, sounds_like_phonemes, meter
    )[:top_n]
```

Leave every other line of `search()` untouched.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_search.py -v`
Expected: all PASS, including every pre-existing test.

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `pytest`
Expected: previous pass count + these new tests, same 2 pre-existing `FORCE_COLOR` failures.

- [ ] **Step 6: Commit**

```bash
git add src/revdict/search.py tests/test_search.py
git commit -m "feat: apply phonetic filters to search()'s meaning/combined-mode path"
```

---

### Task 6: Wire phonetic filters into structural mode (`structural_search.py`)

**Files:**
- Modify: `src/revdict/structural_search.py`
- Modify: `src/revdict/search.py` (one line: the structural-mode dispatch call)
- Test: `tests/test_structural_search.py`
- Test: `tests/test_search.py`

**Interfaces:**
- Consumes: `revdict.phonetics.matches_*` (Task 4); `search()`'s 5 new parameters and `resolve_phonetic_target` (Task 5, already resolves `rhymes_with`/`sounds_like` into `rhyme_key`/`sounds_like_phonemes` before dispatch).
- Produces: `run_structural(parsed, state, top_n, category=None, syllables=None, primary_vowel=None, rhyme_key=None, sounds_like_phonemes=None, meter=None) -> dict` — 5 new parameters, matching Task 4's category precedent of filtering the headword pool before `_score_and_sort(...)[:top_n]`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_structural_search.py` (near the existing category tests):

```python
def test_run_structural_filters_by_syllables_before_top_n_truncation():
    metadata = [
        {
            "headword": "blueadverbially", "pos": "adverb", "definition": "in a blue manner",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
            "phonetics": {"syllable_count": 5, "primary_vowel": "UW", "rhyme_key": "X", "meter": "xxxx/", "phonemes": []},
        },
        {
            "headword": "bluebird", "pos": "noun", "definition": "an American songbird",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
            "phonetics": {"syllable_count": 2, "primary_vowel": "UW", "rhyme_key": "Y", "meter": "/x", "phonemes": []},
        },
        {
            "headword": "blueprint", "pos": "noun", "definition": "a technical drawing",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
            "phonetics": {"syllable_count": 2, "primary_vowel": "UW", "rhyme_key": "Z", "meter": "/x", "phonemes": []},
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

    result = run_structural(parsed, state, top_n=2, syllables=2)

    assert {c["headword"] for c in result["candidates"]} == {"bluebird", "blueprint"}


def test_run_structural_no_phonetic_filters_matches_everything():
    parsed = ParsedQuery(mode="structural", pattern_clauses=["blue*"])
    state = _build_state()

    result = run_structural(parsed, state, top_n=10)

    assert {c["headword"] for c in result["candidates"]} == {"bluebird", "blueprint"}
```

Add this end-to-end wiring test to `tests/test_search.py`:

```python
def test_search_syllables_filters_structural_mode_candidates_too(monkeypatch):
    metadata = [
        {
            "headword": "bluebird", "pos": "noun", "definition": "an American songbird",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
            "phonetics": {"syllable_count": 2, "primary_vowel": "UW", "rhyme_key": "Y", "meter": "/x", "phonemes": []},
        },
        {
            "headword": "bluely", "pos": "adverb", "definition": "in a blue manner",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None, "tags": [],
            "phonetics": {"syllable_count": 3, "primary_vowel": "UW", "rhyme_key": "Z", "meter": "/xx", "phonemes": []},
        },
    ]
    state = {
        "metadata": metadata,
        "word_index": {"bluebird": [0], "bluely": [1]},
        "literary_frequency": {},
        "classifier": None,
    }
    monkeypatch.setattr(search_mod, "_load_state", lambda: state)

    result = search_mod.search("blue*", top_n=10, syllables=2)

    assert [c["headword"] for c in result["candidates"]] == ["bluebird"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_structural_search.py tests/test_search.py -k "syllables" -v`
Expected: FAIL with `TypeError: run_structural() got an unexpected keyword argument 'syllables'`.

- [ ] **Step 3: Implement**

In `src/revdict/structural_search.py`, add the import:

```python
from revdict import phonetics
```

Change `run_structural`'s signature and body:

```python
def run_structural(
    parsed: ParsedQuery,
    state: dict,
    top_n: int,
    category: str | None = None,
    syllables: int | None = None,
    primary_vowel: str | None = None,
    rhyme_key: str | None = None,
    sounds_like_phonemes: list[str] | None = None,
    meter: str | None = None,
) -> dict:
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
    if any([syllables, primary_vowel, rhyme_key, sounds_like_phonemes, meter]):
        headwords = [
            word
            for word in headwords
            if phonetics.matches_syllable_count(metadata[word_index[word][0]], syllables)
            and phonetics.matches_primary_vowel(metadata[word_index[word][0]], primary_vowel)
            and phonetics.matches_rhyme(metadata[word_index[word][0]], rhyme_key)
            and phonetics.matches_sounds_like(metadata[word_index[word][0]], sounds_like_phonemes)
            and phonetics.matches_meter(metadata[word_index[word][0]], meter)
        ]
    ranked = _score_and_sort(headwords, literary_frequency)[:top_n]
    relevances = relative_relevance([score for _, score in ranked])

    candidates = [
        build_candidate(metadata[word_index[headword][0]], relevance, state)
        for (headword, _), relevance in zip(ranked, relevances)
    ]

    return {"exact_match": None, "candidates": candidates}
```

(`category_module` here is the existing import already present in `structural_search.py` from Phase 3 — confirm it's still there via `grep -n "^from revdict import category" src/revdict/structural_search.py` before assuming; do not re-add it if already present.)

In `src/revdict/search.py`, change the structural-mode dispatch call:

```python
    if parsed.mode in ("structural", "expand", "phrase_contains"):
        result = structural_search.run_structural(
            parsed,
            state,
            top_n,
            category=category,
            syllables=syllables,
            primary_vowel=primary_vowel,
            rhyme_key=rhyme_key,
            sounds_like_phonemes=sounds_like_phonemes,
            meter=meter,
        )
```

(was: `structural_search.run_structural(parsed, state, top_n, category=category)`. `rhyme_key`/`sounds_like_phonemes` here are the already-resolved local variables Task 5 introduced above this dispatch point — not the raw `rhymes_with`/`sounds_like` string arguments.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_structural_search.py tests/test_search.py -v`
Expected: all PASS, including every pre-existing test.

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `pytest`
Expected: previous pass count + these new tests, same 2 pre-existing `FORCE_COLOR` failures.

- [ ] **Step 6: Commit**

```bash
git add src/revdict/structural_search.py src/revdict/search.py tests/test_structural_search.py tests/test_search.py
git commit -m "feat: apply phonetic filters to structural/expand/phrase_contains modes"
```

---

### Task 7: Daemon wire protocol — 5 new fields

**Files:**
- Modify: `src/revdict/daemon.py`
- Test: `tests/test_daemon.py`

**Interfaces:**
- Consumes: `search_mod.search(query, top_n, sort_mode=None, category=None, syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None)` (Tasks 5+6, already fully wired).
- Produces: `send_query(query, top_n, sort_mode=None, category=None, syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None, timeout=30.0) -> dict | None`; `_handle_request` reads all 5 new keys via `.get()`.

- [ ] **Step 1: Write the failing tests**

`tests/test_daemon.py` has mock `fake_search`/`failing_search` definitions (from Phase 3's Task 5) that need `category` extended to also accept the 5 new keywords. Run `grep -n "def fake_search\|def failing_search" tests/test_daemon.py` to find the current exact set before editing (Phase 3 left exactly 4; confirm this count still holds). For each, add the 5 new parameters as trailing keyword args with `=None` defaults, e.g. `def fake_search(query, top_n, sort_mode, category, syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None):`.

Update the two existing `send_query` payload-assertion tests (`test_send_query_includes_sort_mode_in_the_request_payload`, `test_send_query_defaults_sort_mode_to_none_when_omitted`) so their expected dict literals include all 5 new keys as `None`, e.g.:

```python
    assert received["request"] == {
        "query": "happy", "top_n": 10, "sort": "alpha", "category": None,
        "syllables": None, "primary_vowel": None, "rhymes_with": None,
        "sounds_like": None, "meter": None,
    }
```

Add new tests mirroring Phase 3's category wire-protocol tests, for one representative field (`syllables`) to keep this task's test volume proportionate — the mechanism is identical across all 5 new fields, so one thoroughly-tested representative plus a single combined smoke test covers the wiring risk without 5x redundant near-duplicate tests:

```python
def test_send_query_includes_all_five_phonetic_fields_in_the_request_payload(tmp_path, monkeypatch):
    socket_path = tmp_path / "daemon.sock"
    monkeypatch.setattr(daemon, "DAEMON_SOCKET_PATH", socket_path)
    received = {}

    ready_event = threading.Event()
    server_thread = threading.Thread(
        target=_run_capturing_server, args=(socket_path, received, ready_event)
    )
    server_thread.start()
    ready_event.wait(timeout=2)

    daemon.send_query(
        "happy", 10, syllables=2, primary_vowel="AE", rhymes_with="cat",
        sounds_like="bat", meter="/x", timeout=2.0,
    )

    server_thread.join(timeout=2)
    assert received["request"] == {
        "query": "happy", "top_n": 10, "sort": None, "category": None,
        "syllables": 2, "primary_vowel": "AE", "rhymes_with": "cat",
        "sounds_like": "bat", "meter": "/x",
    }


def test_send_query_defaults_all_five_phonetic_fields_to_none_when_omitted(tmp_path, monkeypatch):
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
    assert received["request"]["syllables"] is None
    assert received["request"]["primary_vowel"] is None
    assert received["request"]["rhymes_with"] is None
    assert received["request"]["sounds_like"] is None
    assert received["request"]["meter"] is None


def test_handle_request_passes_all_five_phonetic_fields_through_to_search_fn():
    calls = {}

    def fake_search(query, top_n, sort_mode, category, syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None):
        calls.update(
            syllables=syllables, primary_vowel=primary_vowel, rhymes_with=rhymes_with,
            sounds_like=sounds_like, meter=meter,
        )
        return {"exact_match": None, "candidates": []}

    request_text = json.dumps(
        {"query": "happy", "top_n": 10, "syllables": 2, "primary_vowel": "AE", "rhymes_with": "cat", "sounds_like": "bat", "meter": "/x"}
    )

    daemon._handle_request(request_text, fake_search)

    assert calls == {"syllables": 2, "primary_vowel": "AE", "rhymes_with": "cat", "sounds_like": "bat", "meter": "/x"}


def test_handle_request_defaults_all_five_phonetic_fields_to_none_for_requests_without_them():
    calls = {}

    def fake_search(query, top_n, sort_mode, category, syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None):
        calls.update(
            syllables=syllables, primary_vowel=primary_vowel, rhymes_with=rhymes_with,
            sounds_like=sounds_like, meter=meter,
        )
        return {"exact_match": None, "candidates": []}

    request_text = json.dumps({"query": "happy", "top_n": 10})

    daemon._handle_request(request_text, fake_search)

    assert calls == {"syllables": None, "primary_vowel": None, "rhymes_with": None, "sounds_like": None, "meter": None}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_daemon.py -v`
Expected: the new/modified tests FAIL (`TypeError` on the `fake_search` signature mismatches once `_handle_request`'s new call lands in Step 3; assertion mismatches on the payload-shape tests before `send_query`'s change).

- [ ] **Step 3: Implement**

In `src/revdict/daemon.py`, change `send_query`:

```python
def send_query(
    query: str,
    top_n: int,
    sort_mode: str | None = None,
    category: str | None = None,
    syllables: int | None = None,
    primary_vowel: str | None = None,
    rhymes_with: str | None = None,
    sounds_like: str | None = None,
    meter: str | None = None,
    timeout: float = 30.0,
) -> dict | None:
    if not DAEMON_SOCKET_PATH.exists():
        return None
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(str(DAEMON_SOCKET_PATH))
            request = json.dumps(
                {
                    "query": query,
                    "top_n": top_n,
                    "sort": sort_mode,
                    "category": category,
                    "syllables": syllables,
                    "primary_vowel": primary_vowel,
                    "rhymes_with": rhymes_with,
                    "sounds_like": sounds_like,
                    "meter": meter,
                }
            )
            sock.sendall(request.encode("utf-8"))
```

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
            syllables=request.get("syllables"),
            primary_vowel=request.get("primary_vowel"),
            rhymes_with=request.get("rhymes_with"),
            sounds_like=request.get("sounds_like"),
            meter=request.get("meter"),
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
git commit -m "feat: thread 5 phonetic filter fields through the daemon wire protocol"
```

---

### Task 8: CLI flags + README

**Files:**
- Modify: `src/revdict/cli.py`
- Modify: `README.md`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `search_mod.search(..., syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None)` (Tasks 5+6); `daemon.send_query`/`_handle_request` (Task 7).
- Produces: `--syllables N` (int), `--primary-vowel VOWEL`, `--rhymes-with WORD`, `--sounds-like WORD`, `--meter PATTERN` CLI flags; `_local_search_fallback`, `_get_search_result`, `_run_query` all gain the 5 new parameters; `main()`'s dispatch passes them all through. Also produces `_ARPABET_VOWELS` (a closed set) and `_validate_meter_pattern()` (an argparse `type=` callback) — `--primary-vowel` and `--meter` are the only two of the 5 new flags with a validatable closed/constrained format, so they get upfront argparse-level rejection instead of accepting a typo and silently matching zero candidates later (the same UX principle Task 5's `resolve_phonetic_target` already applies to a missing/outdated stressmark — a bad input should fail loudly, not read as "nothing matched").

- [ ] **Step 1: Write the failing tests**

Run `grep -n "category=None" tests/test_cli.py` first — every mock site that currently accepts `category=None` (Phase 3's exhaustive list) now also needs the 5 new keywords added as trailing `=None` params, e.g. `lambda query, top_n, sort_mode=None, category=None: X` becomes `lambda query, top_n, sort_mode=None, category=None, syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None: X`. Apply this to every match; do not skip any (this is the same mechanical-update discipline as Phase 3's Task 6 — re-run the grep against the CURRENT file, don't assume Phase 3's captured count still applies verbatim, since this task's own edits will also touch some of these same lines).

Add these test functions (mirroring Phase 3's `--category` test shapes, one representative field for the parser-level tests plus a combined dispatch test):

```python
def test_query_parser_accepts_all_five_phonetic_flags():
    from revdict import cli

    parser = cli._query_parser()

    args = parser.parse_args([
        "happy", "--syllables", "2", "--primary-vowel", "AE",
        "--rhymes-with", "cat", "--sounds-like", "bat", "--meter", "/x",
    ])
    assert args.syllables == 2
    assert args.primary_vowel == "AE"
    assert args.rhymes_with == "cat"
    assert args.sounds_like == "bat"
    assert args.meter == "/x"


def test_query_parser_phonetic_flags_default_to_none():
    from revdict import cli

    parser = cli._query_parser()

    args = parser.parse_args(["happy"])
    assert args.syllables is None
    assert args.primary_vowel is None
    assert args.rhymes_with is None
    assert args.sounds_like is None
    assert args.meter is None


def test_query_parser_rejects_a_non_integer_syllables_value():
    import pytest

    from revdict import cli

    parser = cli._query_parser()

    with pytest.raises(cli._ArgumentError):
        parser.parse_args(["happy", "--syllables", "two"])


def test_query_parser_rejects_an_invalid_primary_vowel():
    """--primary-vowel is a closed ARPAbet vowel set -- a typo or a stray
    stress digit (e.g. "AE1" instead of "AE") must fail loudly via
    argparse's choices=, not silently pass through and match nothing."""
    import pytest

    from revdict import cli

    parser = cli._query_parser()

    with pytest.raises(cli._ArgumentError):
        parser.parse_args(["happy", "--primary-vowel", "AE1"])


def test_query_parser_primary_vowel_is_case_insensitive():
    from revdict import cli

    parser = cli._query_parser()

    args = parser.parse_args(["happy", "--primary-vowel", "ae"])
    assert args.primary_vowel == "AE"


def test_query_parser_rejects_an_invalid_meter_pattern():
    """A --meter value with anything other than '/' and 'x' must fail
    loudly, not silently match nothing."""
    import pytest

    from revdict import cli

    parser = cli._query_parser()

    with pytest.raises(cli._ArgumentError):
        parser.parse_args(["happy", "--meter", "/-x"])


def test_main_passes_all_five_phonetic_flags_through_to_get_search_result(monkeypatch):
    from revdict import cli

    monkeypatch.setattr(cli, "_index_exists", lambda: True)
    calls = {}

    def fake_get_search_result(query, top_n, sort_mode=None, category=None, syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None):
        calls.update(
            syllables=syllables, primary_vowel=primary_vowel, rhymes_with=rhymes_with,
            sounds_like=sounds_like, meter=meter,
        )
        return {"exact_match": None, "candidates": []}

    monkeypatch.setattr(cli, "_get_search_result", fake_get_search_result)

    code = cli.main([
        "happy", "--syllables", "2", "--primary-vowel", "AE",
        "--rhymes-with", "cat", "--sounds-like", "bat", "--meter", "/x",
        "--no-interactive",
    ])

    assert code == 0
    assert calls == {"syllables": 2, "primary_vowel": "AE", "rhymes_with": "cat", "sounds_like": "bat", "meter": "/x"}


def test_main_without_phonetic_flags_passes_all_none(monkeypatch):
    from revdict import cli

    monkeypatch.setattr(cli, "_index_exists", lambda: True)
    calls = {}

    def fake_get_search_result(query, top_n, sort_mode=None, category=None, syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None):
        calls.update(
            syllables=syllables, primary_vowel=primary_vowel, rhymes_with=rhymes_with,
            sounds_like=sounds_like, meter=meter,
        )
        return {"exact_match": None, "candidates": []}

    monkeypatch.setattr(cli, "_get_search_result", fake_get_search_result)

    code = cli.main(["happy", "--no-interactive"])

    assert code == 0
    assert calls == {"syllables": None, "primary_vowel": None, "rhymes_with": None, "sounds_like": None, "meter": None}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py -v`
Expected: the new phonetic-flag tests FAIL (no such flags exist yet); other tests remain green until Step 3 changes `_get_search_result`'s real body, which is when a missed mock site would surface as a `TypeError`.

- [ ] **Step 3: Implement**

In `_query_parser()` in `src/revdict/cli.py`, add the 5 flags alongside `--category`:

```python
    parser.add_argument(
        "--syllables", type=int, default=None, metavar="N",
        help="Filter results to headwords with exactly N syllables.",
    )
    parser.add_argument(
        "--primary-vowel", choices=list(_ARPABET_VOWELS), default=None, metavar="VOWEL",
        type=str.upper,
        help="Filter results to headwords whose primary-stressed vowel is VOWEL (an ARPAbet vowel symbol, e.g. AE).",
    )
    parser.add_argument(
        "--rhymes-with", default=None, metavar="WORD",
        help="Filter results to headwords that rhyme with WORD.",
    )
    parser.add_argument(
        "--sounds-like", default=None, metavar="WORD",
        help="Filter results to headwords that sound phonetically similar to WORD.",
    )
    parser.add_argument(
        "--meter", default=None, metavar="PATTERN", type=_validate_meter_pattern,
        help='Filter results to headwords matching a stress pattern of "/" (stressed) and "x" (unstressed) per syllable, e.g. "/x".',
    )
```

Add these two module-level definitions above `_query_parser()` (find its current location with `grep -n "^def _query_parser" src/revdict/cli.py` and insert just before it):

```python
_ARPABET_VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
    "IH", "IY", "OW", "OY", "UH", "UW",
}


def _validate_meter_pattern(value: str) -> str:
    """argparse type= callback: rejects a --meter value containing anything
    other than '/' and 'x' up front, via an ArgumentTypeError (argparse's
    own convention for type= validation failures, converted by this file's
    _QuietArgumentParser into the same clean error path as an invalid
    --sort/--category choice), rather than silently accepting garbage that
    would then just never match any real headword's meter string."""
    if not value or any(ch not in "/x" for ch in value):
        raise argparse.ArgumentTypeError(
            f"invalid meter pattern {value!r}: must contain only '/' and 'x'"
        )
    return value
```

(`argparse` is already imported at the top of `cli.py`.) Note `--primary-vowel`'s `type=str.upper` runs BEFORE `choices` validation, so `--primary-vowel ae` and `--primary-vowel AE` both work identically — matches this flag's case-insensitive matching behavior in `phonetics.matches_primary_vowel`.


Change `_local_search_fallback`, `_get_search_result`, and `_run_query` to accept and thread through all 5 new parameters, following the exact same shape as `category` was added in Phase 3's Task 6 (each function gains the 5 params with `= None` defaults, in the same order as `search()`'s signature, and passes them by keyword to whatever it calls):

```python
def _local_search_fallback(
    query: str,
    top_n: int,
    sort_mode: str | None = None,
    category: str | None = None,
    syllables: int | None = None,
    primary_vowel: str | None = None,
    rhymes_with: str | None = None,
    sounds_like: str | None = None,
    meter: str | None = None,
) -> dict:
    from revdict.query_env import configure_offline_quiet_env

    configure_offline_quiet_env()
    from revdict import search as search_mod

    return search_mod.search(
        query,
        top_n=top_n,
        sort_mode=sort_mode,
        category=category,
        syllables=syllables,
        primary_vowel=primary_vowel,
        rhymes_with=rhymes_with,
        sounds_like=sounds_like,
        meter=meter,
    )


def _get_search_result(
    query: str,
    top_n: int,
    sort_mode: str | None = None,
    category: str | None = None,
    syllables: int | None = None,
    primary_vowel: str | None = None,
    rhymes_with: str | None = None,
    sounds_like: str | None = None,
    meter: str | None = None,
) -> dict:
    result = daemon.send_query(
        query, top_n, sort_mode=sort_mode, category=category, syllables=syllables,
        primary_vowel=primary_vowel, rhymes_with=rhymes_with, sounds_like=sounds_like, meter=meter,
    )
    if result is not None:
        return result
    if daemon.ensure_daemon_running():
        result = daemon.send_query(
            query, top_n, sort_mode=sort_mode, category=category, syllables=syllables,
            primary_vowel=primary_vowel, rhymes_with=rhymes_with, sounds_like=sounds_like, meter=meter,
        )
        if result is not None:
            return result
    return _local_search_fallback(
        query, top_n, sort_mode=sort_mode, category=category, syllables=syllables,
        primary_vowel=primary_vowel, rhymes_with=rhymes_with, sounds_like=sounds_like, meter=meter,
    )


def _run_query(
    query: str,
    top_n: int,
    interactive: bool,
    sort_mode: str | None = None,
    category: str | None = None,
    syllables: int | None = None,
    primary_vowel: str | None = None,
    rhymes_with: str | None = None,
    sounds_like: str | None = None,
    meter: str | None = None,
) -> int:
    if not query.strip():
        console.print("[yellow]Please enter a word or phrase.[/yellow]")
        return 0

    result = _get_search_result(
        query, top_n, sort_mode=sort_mode, category=category, syllables=syllables,
        primary_vowel=primary_vowel, rhymes_with=rhymes_with, sounds_like=sounds_like, meter=meter,
    )
```

(The rest of `_run_query`'s body, after this `result = ...` line, is unchanged.)

In `main()`, change the final dispatch line:

```python
    interactive = not args.no_interactive and sys.stdout.isatty()
    return _run_query(
        query, args.n, interactive, sort_mode=args.sort, category=args.category,
        syllables=args.syllables, primary_vowel=args.primary_vowel, rhymes_with=args.rhymes_with,
        sounds_like=args.sounds_like, meter=args.meter,
    )
```

Leave `_run_query_only`, `_run_jsonl_query`, and the no-argv stdin-read path completely untouched, matching Phase 2/3's identical precedent.

Add a new section to `README.md`, after the existing "## Category filter" section:

```markdown
## Phonetic filters

Five filters based on pronunciation, computed from a `revdict build-index` reindex (see below) — combine any of them, and combine them with `--category`/`--sort` too:

| Flag | Matches |
|---|---|
| `--syllables N` | Headwords with exactly N syllables |
| `--primary-vowel VOWEL` | Headwords whose stressed syllable's vowel is VOWEL (an ARPAbet vowel symbol — AA, AE, AH, AO, AW, AY, EH, ER, EY, IH, IY, OW, OY, UH, UW) |
| `--rhymes-with WORD` | Headwords that rhyme with WORD |
| `--sounds-like WORD` | Headwords that are phonetically close to WORD (not just spelled similarly) |
| `--meter PATTERN` | Headwords whose stressed/unstressed syllable pattern matches PATTERN — a string of `/` (stressed) and `x` (unstressed), one character per syllable, e.g. `/x` (trochee, like "happy"), `x/` (iamb, like "record" the verb), `/xx` (dactyl, like "elephant") |

```bash
revdict "feeling of intense annoyance" --syllables 2 --no-interactive
revdict "small carnivore" --rhymes-with hat --no-interactive
```

**Requires a reindex.** Unlike category filtering, none of these five work at all on an index built before this feature shipped — run `revdict build-index` to rebuild. Phonetic data is only computed for single-word headwords with no internal hyphen (multi-word phrases and hyphenated compounds are skipped — the underlying `stressmark` library doesn't reliably syllabify either yet); those headwords simply never match any phonetic filter, on any index.

`--rhymes-with`/`--sounds-like` additionally need the `stressmark` library installed and importable at query time (not just at index-build time) — they resolve your target word's pronunciation live, since it's not something a reindex could have precomputed. If `stressmark` isn't installed, these two flags fail with a clear error rather than silently returning no results.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v`
Expected: all PASS, including every pre-existing test. As with Phase 3's Task 6, a missed mock site from Step 1's grep would surface here as a `TypeError` — if anything fails, re-run `grep -n "category=None" tests/test_cli.py` to find what's missing a phonetic-keyword update.

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `pytest`
Expected: previous pass count + these new tests, same 2 pre-existing `FORCE_COLOR` failures and nothing else.

- [ ] **Step 6: Commit**

```bash
git add src/revdict/cli.py README.md tests/test_cli.py
git commit -m "feat: add 5 phonetic filter CLI flags and document them in the README"
```

---

## Self-review notes (fixed before dispatch)

1. **Cross-repo dependency correctly sequenced and made durable.** Task 1 lives entirely in `../emphasis`, is committed AND pushed to that repo's own GitHub origin before Task 2 begins — not left as a local-only change relying on this session's editable install (which would silently work for this session and silently break for anyone else, exactly the failure mode the existing `revdict-backlog` memory already flags for this dependency).
2. **Precompute-at-build-time architecture chosen over live-per-query resolution, and the cost was measured, not assumed.** An earlier version of this plan's thinking considered resolving phonetics live per candidate at query time; rejected once it became clear this would mean calling stressmark on thousands of rows per query (the retrieval pool, or the entire `word_index` for a broad structural pattern) — a real latency regression the daemon architecture specifically exists to avoid. Precomputing at index-build time (mirroring Phase 3's `tags` field exactly) moves this cost to a one-time reindex. That reindex cost was directly measured against the real corpus and real stressmark calls before writing this plan (not assumed by analogy to the embedding pass): 565,911 of 801,725 unique headwords are eligible (70.6%), and resolving all of them measures at ~6.4 minutes wall-clock.
3. **Multi-word and hyphenated headword exclusion is grounded in a measured, reproduced bug, not a guess.** Directly tested `resolve_word_by_pos` against real multi-word ("kick the bucket") and hyphenated ("well-known") headwords before writing this plan and found genuinely malformed output (empty-string syllable fragments, meaningless stress indices) in both cases — root-caused to stressmark's compound-word handling living only in its sentence-level `analyze()` pass, which `resolve_word_by_pos` never calls. Excluding both is a deliberate, disclosed scope boundary, not laziness — fixing the underlying stressmark limitation is future cross-repo work, out of scope here.
4. **Query-time stressmark dependency correctly identified as NOT fully eliminated by precomputation.** An earlier pass of this plan's design nearly missed that `--rhymes-with`/`--sounds-like`'s *target* word (arbitrary, typed per query) cannot be precomputed the way corpus headwords can — caught and fixed before writing Task 5's code: `resolve_phonetic_target()` resolves the target live and raises a clear, distinguishing `ValueError` (not a silent empty result) if stressmark is unavailable or the target itself can't be resolved.
5. **`--sounds-like` and `--rhymes-with`'s algorithms are pinned against real measured word pairs, not an arbitrary threshold.** Mirrors Phase 3's tag-vocabulary-grounding discipline exactly: the rhyme-key definition and the 0.34 sounds-like threshold were both validated against real ARPAbet output for known rhyming/homophone/near-miss/unrelated word pairs (see Global Constraints) before being written into any task's code or tests — including catching and correcting an initial mistake where an ungrounded test script used a naive CMUdict lookup instead of `resolve_word_by_pos`'s heteronym-aware resolution, which would have produced an incorrect "record rhymes with chord" example.
6. **Meter symbol semantics pinned with an example for every TODO.md-listed pattern.** Both primary AND secondary stress render as `/` (a deliberate simplification, disclosed in Global Constraints) — verified this produces exactly TODO.md's own listed examples (`/x`, `x/`, `/xx`, `x/x`) against real words (happy, record-verb, elephant, banana) before writing any task code.
7. **"Also related to" and TODO.md's "Starts with/Ends with/Letters count" are explicitly out of this plan's scope**, both confirmed rather than silently decided: the latter via a direct question to the user (confirmed: Phase 1's query DSL already covers it, no dedicated flags needed), the former based on it being a materially different code path (retrieval-vector combination, not a precomputed-field filter) sharing none of this phase's substrate.
8. **Task 5/6 split mirrors Phase 3's Task 3/Task 4 split for the identical reason** — Task 5 does not touch the structural-mode dispatch call site, avoiding the exact signature-mismatch regression class Phase 3 identified and worked around (a call site passing a new keyword before the callee accepts it breaks every test on that path for the task's duration).
9. **Spec coverage check:** TODO.md feature-group-2's phonetic items (Sounds like, Primary vowel, Rhymes with, Meters, Syllable count) — Tasks 2/4/5/6/7/8. "Also related to" — explicitly deferred (item 7 above). "Starts with/Ends with/Letters count" — explicitly out of scope, confirmed with the user (item 7 above). Roadmap's "stressmark API extension (phonemes, meter-pattern string, syllable count as structured data)" — Task 1 (phonemes on stressmark's side) + Task 2 (meter/syllable-count derivation on revdict's side, deliberately NOT pushed into stressmark itself, per the same minimal-cross-repo-surface reasoning Phase 3 and the roadmap's own Phase 6 TUI decision both already established for this codebase). No gaps found.
10. **The Task 2↔Task 4 module seam is explicitly tested, not just individually unit-tested.** Caught during advisor review: every task-level test hand-writes its own `"phonetics"` dict shape or checks `resolve()`'s output shape in isolation — nothing originally asserted that the dict `models.phonetics.resolve()` actually produces is the dict `phonetics.py`'s `matches_*` predicates actually read. A silent key-name mismatch between the two tasks would have made every test in both tasks pass individually while the real, integrated behavior became "every phonetic filter matches nothing" — surfacing only after a real user's multi-hour reindex. Fixed by adding `test_matching_predicates_actually_consume_real_resolve_output` to Task 4, which calls the real `models.phonetics.resolve()` (skipped if stressmark isn't installed) and runs the real predicates against its real output end-to-end.
11. **`--primary-vowel`/`--meter` typos now fail loudly instead of silently matching nothing.** Caught during advisor review: the plan already made a missing/outdated stressmark fail loudly for `--rhymes-with`/`--sounds-like` (Task 5's `resolve_phonetic_target`), but hadn't applied the same principle to `--primary-vowel`/`--meter`, which take literal targets with no query-time stressmark dependency at all — a typo like `--primary-vowel AE1` or `--meter /-x` would previously have parsed successfully and just returned zero results forever, indistinguishable from "no words happen to match." Fixed in Task 8 with `choices=` (a closed 15-symbol ARPAbet vowel set) for `--primary-vowel` and a `type=` validation callback for `--meter` (rejects anything outside `{/, x}`), both routing through argparse's existing error path the same way an invalid `--sort`/`--category` choice already does.
12. **Placeholder scan:** no TBD/"add error handling"/"similar to Task N" text anywhere in the eight tasks above; every step has complete, concrete code.
11. **Type/name consistency check:** `syllables: int | None`, `primary_vowel: str | None`, `rhymes_with`/`sounds_like: str | None` (raw target words), `rhyme_key`/`sounds_like_phonemes` (already-resolved forms), and `meter: str | None` are used identically in name and type across `phonetics.py`'s predicates, `models/phonetics.py`'s `resolve()` output keys, `search()`, `run_structural()`, `send_query()`/`_handle_request`'s wire fields, and the CLI's `args.*` — no renaming drift between tasks. Confirmed the wire-protocol JSON keys use the raw-target names (`"rhymes_with"`, `"sounds_like"`) matching the CLI flag names, while `search()`'s internal locals after resolution are named `rhyme_key`/`sounds_like_phonemes` to keep "the word the user typed" and "the phonetic data derived from it" visually distinct in the code.
