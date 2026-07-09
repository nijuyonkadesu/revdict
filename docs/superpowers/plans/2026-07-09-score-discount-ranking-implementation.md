# Score-Discount Ranking Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop obscure/archaic words from outranking common, natural-sounding synonyms in revdict's candidate list, without adding a general "how common is this word" boost that would incorrectly suppress genuinely fiction-common words like "murmured"/"scowled" that are simply rare in casual speech.

**Architecture:** Two independent, strictly-subtractive score adjustments applied to every candidate's raw reranker score, right after reranking and before dedup/truncation: a flat penalty for candidates with essentially no real-world word attestation (via the `wordfreq` package), and a discount scaled by how many times the query word literally appears in the candidate's own definition (the actual score-inflation mechanism found during investigation). Both adjustments can only lower a score, never raise one.

**Tech Stack:** `wordfreq` (new required dependency — offline, no network calls, ~58MB installed, ~0.4µs per lookup after a one-time ~0.12s load), Python's `re` module (stdlib).

## Global Constraints

- Both discounts are **strictly subtractive** — the combined adjustment applied to any raw score must never be positive. This is what guarantees the existing `absolute_relevance` "gibberish reads near-zero" property (validated during the daemon work) can't be broken by this feature.
- `ZERO_FREQUENCY_PENALTY = -3.0`, applied when `wordfreq.zipf_frequency(headword, "en") < 0.5` (a hard "essentially unattested" cutoff, not a general familiarity ranking — general frequency does not track literary register, confirmed during design: "murmured" (zipf 2.54) scores lower than "wherefore" (zipf 2.79) in `wordfreq`'s blended corpus).
- `OVERLAP_DISCOUNT_PER_OCCURRENCE = -3.5`, applied once per literal, case-insensitive, whole-word occurrence of the query string in the candidate's own definition text.
- Exact-match senses are never discounted — they're always pinned regardless of score; only candidate-list scoring is affected.
- `wordfreq` is a **required** dependency (added directly to `pyproject.toml`), not an optional plugin like the stressmark integration — this fixes a core ranking-quality issue.
- This plan pins concrete, real before/after numbers gathered during investigation as regression tests — not just the formula in the abstract.

---

### Task 1: Add `revdict.models.frequency` (the zero-attestation check)

**Files:**
- Modify: `pyproject.toml` (add `wordfreq` to `dependencies`)
- Create: `src/revdict/models/frequency.py`
- Test: `tests/models/test_frequency.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `revdict.models.frequency.is_essentially_unattested(word: str) -> bool`. Task 2 consumes this.

- [ ] **Step 1: Add `wordfreq` to `pyproject.toml`**

Find the `dependencies` list in `pyproject.toml` and add `"wordfreq>=3.1"` to it (alongside the existing entries like `nltk`, `sentence-transformers`, etc. — exact position in the list doesn't matter, keep the list readable).

- [ ] **Step 2: Install it and regenerate the lock file**

```bash
uv add wordfreq
```

Expected: updates `pyproject.toml` (if not already edited by hand — running `uv add` will add the dependency line itself, so Step 1 and this step are really one combined action; if you already hand-edited `pyproject.toml` in Step 1, run `uv lock && uv sync --all-extras` instead to pick up the manual edit) and `uv.lock`, installs `wordfreq` into `.venv`.

- [ ] **Step 3: Write the failing test**

```python
# tests/models/test_frequency.py
from revdict.models.frequency import is_essentially_unattested


def test_is_essentially_unattested_true_for_a_word_with_zero_real_world_attestation():
    # Real example from investigation: a genuinely obscure Wiktionary dialectal
    # entry with zero attestation across wordfreq's combined source corpora.
    assert is_essentially_unattested("wealful") is True


def test_is_essentially_unattested_false_for_a_common_word():
    assert is_essentially_unattested("happy") is False


def test_is_essentially_unattested_false_for_a_fiction_common_but_casually_rare_word():
    # The whole reason this is a hard "< 0.5" cutoff and not a general
    # familiarity ranking: "murmured" is rare in casual/social-media text
    # (low zipf) but is NOT "essentially unattested" -- it's a fiction
    # staple. A general frequency boost would have incorrectly suppressed
    # words like this; the cutoff must stay narrow enough not to.
    assert is_essentially_unattested("murmured") is False
```

- [ ] **Step 4: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/models/test_frequency.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'revdict.models.frequency'`

- [ ] **Step 5: Write the implementation**

```python
# src/revdict/models/frequency.py
from wordfreq import zipf_frequency

ZERO_FREQUENCY_THRESHOLD = 0.5


def is_essentially_unattested(word: str) -> bool:
    """True when `word` has no meaningful attestation across wordfreq's
    combined English source corpora (Wikipedia, subtitles, news, Google
    Books, web text, Twitter, Reddit) -- a reliable "this word is
    essentially unused" signal.

    Deliberately NOT used as a general familiarity ranking: general word
    frequency does not track literary/fiction register (verified during
    design -- "murmured" and "scowled" score lower than genuine
    Shakespeare-era archaisms like "wherefore"/"yonder" in wordfreq's
    blended corpus, since they're rare in casual speech/social media
    despite being fiction staples). This hard, narrow cutoff only fires
    for words that are close to actually unattested, not merely uncommon.
    """
    return zipf_frequency(word, "en") < ZERO_FREQUENCY_THRESHOLD
```

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/models/test_frequency.py -v`
Expected: PASS (all 3 tests)

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/revdict/models/frequency.py tests/models/test_frequency.py
git commit -m "$(cat <<'EOF'
Add revdict.models.frequency: a narrow zero-attestation check

Uses wordfreq's zipf_frequency with a hard <0.5 cutoff to catch words
with essentially no real-world attestation -- deliberately NOT used as
a general familiarity ranking, since general word frequency doesn't
track literary register (verified: "murmured" scores lower than
"wherefore" in wordfreq's blended corpus).
EOF
)"
```

---

### Task 2: Wire the two discounts into `search.py`

**Files:**
- Modify: `src/revdict/search.py`
- Test: `tests/test_search.py`

**Interfaces:**
- Consumes: `revdict.models.frequency.is_essentially_unattested(word: str) -> bool` (Task 1).
- Produces: `discount_score(raw_score: float, query: str, headword: str, definition: str) -> float` in `search.py`. `search()`'s candidate ranking now uses discount-adjusted scores throughout (dedup selection, truncation, and the displayed `relevance` percentage all reflect the adjustment).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_search.py -- add these (keep all existing tests in the file unchanged)
def test_discount_score_applies_zero_frequency_penalty_for_unattested_words():
    from revdict.search import discount_score

    # "wealful" has zero real-world attestation (see test_frequency.py) and
    # its gloss doesn't contain the query word, so only the zero-frequency
    # penalty applies, not the overlap discount.
    result = discount_score(6.157, "smiling", "wealful", "Prosperous.")

    assert result == 6.157 - 3.0


def test_discount_score_applies_overlap_discount_once_per_literal_occurrence():
    from revdict.search import discount_score

    # "glad" has real-world attestation (no zero-frequency penalty); its
    # gloss contains the query word "happy" once.
    result = discount_score(5.424, "happy", "glad", "Pleased; happy; gratified.")

    assert result == 5.424 - 3.5


def test_discount_score_applies_overlap_discount_multiple_times_for_multiple_occurrences():
    from revdict.search import discount_score

    # Real example from investigation: "happy" appears twice in this gloss.
    result = discount_score(7.280, "happy", "twinkly-eyed", "happy, of a happy character.")

    assert result == 7.280 - 2 * 3.5


def test_discount_score_applies_both_penalties_when_both_conditions_hold():
    from revdict.search import discount_score

    # Real example from investigation: "wealful" is both unattested AND its
    # gloss contains the query word once.
    result = discount_score(6.157, "happy", "wealful", "Happy; joyful; felicitous.")

    assert result == 6.157 - 3.0 - 3.5


def test_discount_score_is_unchanged_when_neither_condition_applies():
    from revdict.search import discount_score

    result = discount_score(-4.897, "happy", "cheerful", "being full of or promoting cheer")

    assert result == -4.897


def test_discount_score_overlap_match_is_case_insensitive_and_whole_word():
    from revdict.search import discount_score

    # "Happy" (capitalized) should still match; "happiness" must NOT match
    # (it's a different word, not the query "happy" as a substring).
    result = discount_score(5.0, "happy", "glad", "Happy and happiness go together.")

    assert result == 5.0 - 3.5  # only the one whole-word "Happy" match counts


def test_discount_score_never_raises_the_raw_score():
    """The load-bearing invariant this whole feature depends on: neither
    discount can ever be positive, for any input, so absolute_relevance's
    "gibberish reads near-zero" guarantee can't be broken by this function."""
    from revdict.search import discount_score

    cases = [
        (10.0, "happy", "joyful", "Feeling or causing joy."),
        (-10.0, "xqzflorp", "narcoxyl", "some definition"),
        (0.0, "test", "test", "test test test"),
    ]
    for raw, query, headword, definition in cases:
        assert discount_score(raw, query, headword, definition) <= raw


def test_discount_score_real_happy_candidate_set_reorders_as_calibrated():
    """Regression guard pinning the exact investigation finding: after
    discounting, "twinkly-eyed" (raw 7.28, the worst offender -- "happy"
    appears twice in its own gloss) must rank BELOW "good-humored" and
    "glad" (both common, real synonyms), even though its raw score was
    higher than both."""
    from revdict.search import discount_score

    candidates = [
        (7.280, "twinkly-eyed", "happy, of a happy character."),
        (7.003, "vogie", "Happy; pleased or well-disposed."),
        (6.637, "happies", "The act or state of being happy; happiness."),
        (6.468, "good-humored", "Happy, cheerful, amiable."),
        (6.157, "wealful", "Happy; joyful; felicitous."),
        (5.424, "glad", "Pleased; happy; gratified."),
    ]
    query = "happy"

    adjusted = [
        (discount_score(raw, query, headword, definition), headword)
        for raw, headword, definition in candidates
    ]
    adjusted.sort(reverse=True)
    order = [headword for _, headword in adjusted]

    twinkly_eyed_rank = order.index("twinkly-eyed")
    good_humored_rank = order.index("good-humored")
    glad_rank = order.index("glad")

    assert good_humored_rank < twinkly_eyed_rank
    assert glad_rank < twinkly_eyed_rank
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_search.py -v -k discount_score`
Expected: FAIL with `ImportError: cannot import name 'discount_score' from 'revdict.search'`

- [ ] **Step 3: Add `discount_score` and wire it into `search()`**

Add this import near the top of `src/revdict/search.py` (alongside the existing `from revdict.models import stress` line):

```python
import re
```

and

```python
from revdict.models import frequency
```

Add these two module-level constants near the top of the file (after the imports, before `_state: dict = {}`):

```python
ZERO_FREQUENCY_PENALTY = -3.0
OVERLAP_DISCOUNT_PER_OCCURRENCE = -3.5
```

Add this function (a good place is right after `absolute_relevance`, before `_load_state`):

```python
def discount_score(raw_score: float, query: str, headword: str, definition: str) -> float:
    """Adjusts a raw reranker score with two strictly-subtractive discounts
    -- this can only ever lower a score, never raise it, which is what
    guarantees absolute_relevance's "gibberish reads near-zero" property
    can't be broken by this function.

    1. A flat penalty when the candidate headword has essentially no
       real-world attestation (frequency.is_essentially_unattested).
    2. A discount scaled by how many times the query literally appears (as
       a whole word, case-insensitive) in the candidate's own definition --
       directly targets the real score-inflation mechanism found during
       investigation: a candidate's own gloss restating the query word
       artificially inflates its cross-encoder score (e.g. "twinkly-eyed"
       scoring 7.28 for the query "happy" because its gloss is literally
       "happy, of a happy character").
    """
    score = raw_score
    if frequency.is_essentially_unattested(headword):
        score += ZERO_FREQUENCY_PENALTY
    occurrences = len(re.findall(rf"\b{re.escape(query)}\b", definition, re.IGNORECASE))
    score += OVERLAP_DISCOUNT_PER_OCCURRENCE * occurrences
    return score
```

In `search()`, find this line:

```python
    scored = [(retrieved[i][0], rerank_scores[i]) for i in range(len(retrieved))]
```

Replace it with:

```python
    scored = []
    for i in range(len(retrieved)):
        row_index = retrieved[i][0]
        record = metadata[row_index]
        adjusted = discount_score(rerank_scores[i], query, record["headword"], record["definition"])
        scored.append((row_index, adjusted))
```

(Everything else in `search()` — `dedupe_by_headword`, `exclude_headword`, the `absolute_relevance` call, the candidate-building loop — is unchanged. They already operate on whatever scores are in `scored`/`deduped`, so they automatically pick up the discount-adjusted values without further edits.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_search.py -v`
Expected: PASS (all tests in the file, including the new ones)

- [ ] **Step 5: Run the full test suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all tests pass (no existing test asserts exact score values from `search()`'s candidate loop, since `search()` itself isn't directly unit-tested — matches the established pattern of not unit-testing the real-model-dependent parts of this codebase)

- [ ] **Step 6: Commit**

```bash
git add src/revdict/search.py tests/test_search.py
git commit -m "$(cat <<'EOF'
Discount candidates for zero-attestation and query-term self-reference

Wires two strictly-subtractive score adjustments into search()'s
candidate ranking, applied right after reranking so dedup, truncation,
and the displayed relevance percentage all consistently reflect the
same adjusted score. Pins the real investigation finding as a
regression test: "twinkly-eyed" (raw score 7.28, "happy" appears twice
in its own gloss) now ranks below "good-humored" and "glad".
EOF
)"
```

---

### Task 3: Manual end-to-end validation

No new unit tests — validates the real, calibrated behavior against the live rebuilt index and re-confirms the "gibberish reads near-zero" property still holds with real (not synthetic) data.

**Files:** None created.

- [ ] **Step 1: Confirm the full suite passes and the daemon picks up the new code**

```bash
cd /home/shichika/redacted/rev-dictionary
.venv/bin/python -m pytest tests/ -q
revdict daemon stop 2>&1 || true
```

Expected: all tests pass; daemon stop message (or "not running", either is fine — this just ensures the next query starts a fresh daemon process with the new code loaded, not a stale one).

- [ ] **Step 2: Re-run the investigation's original word list and confirm the concrete improvement**

```bash
for q in happy sad big beautiful angry; do
  echo "=== $q ==="
  revdict "$q" --no-interactive -n 8 2>&1 | grep -A20 "Related words" | grep -E "^│ [0-9]" | awk -F'│' '{print $3}'
done
```

Expected: compare against the investigation's original findings (recorded in the design spec) — common, natural-sounding synonyms should now appear noticeably higher in the list than before this fix (e.g. for "happy", "good-humored" and "glad" should rank ahead of "twinkly-eyed"). Not every single obscure word needs to disappear — the fix is calibrated to be conservative, not to eliminate every rare word — but the overall shift toward common vocabulary should be visible.

- [ ] **Step 3: Re-confirm the gibberish-query near-zero relevance property still holds**

```bash
revdict "asdkjfhqwoeiruty" --no-interactive -n 5 2>&1 | tail -10
```

Expected: all candidates still show 0% relevance (or very close to it) — this property is structurally guaranteed by the "discounts can only subtract" design, but confirming it live with real data is still worth doing before calling this done.

- [ ] **Step 4: Spot-check a query where the fix should NOT meaningfully change anything**

```bash
revdict "container for holding liquid" --no-interactive -n 5 2>&1 | tail -12
```

Expected: results still look sensible and high-quality (e.g. "bottle", "glass", "cask" from earlier validation work) — this fix should have no visible effect here, since none of these strong, legitimate matches are artificially inflated by query-term self-reference or zero attestation.

- [ ] **Step 5: Clean up and commit the validation note**

```bash
revdict daemon stop 2>&1 || true
git commit --allow-empty -m "Validate score-discount ranking fix against the real live index"
git push
```

If any step in this task didn't match its expected outcome, do not treat this feature as complete — investigate and fix before moving on.
