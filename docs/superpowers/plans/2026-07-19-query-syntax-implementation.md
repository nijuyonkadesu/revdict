# Query Syntax (OneLook Feature Group 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement TODO.md's "Searching capabilities" feature group (exact/prefix/suffix/wildcard/letter-position/letters-count/contains-letters/exclude-letters/restrict-letters/anagram/acronym-expand/phrase-contains-word/meaning-combined queries) as a query-syntax layer in front of revdict's existing search pipeline.

**Architecture:** A new `query_syntax.py` parses a raw query string into a `ParsedQuery`, routing to one of five modes: `meaning` (today's unchanged semantic path), `structural` (pure pattern matching over the headword vocabulary, bypasses embedding entirely), `expand` (acronym-initials matching), `phrase_contains` (whole-word-within-a-phrase matching), or `combined` (a pattern narrows the candidate pool, then the existing embed/rerank pipeline runs only over that narrowed pool). A new `pattern_matcher.py` compiles individual comma-separated pattern clauses into predicates via regex translation (`*`/`?`/`#`/`@`) and set operations (letter restrict/exclude/anagram). A new `structural_search.py` runs the pure-pattern modes end-to-end, reusing the existing emotion-tagging/stress-marking/relevance-scaling machinery so `cli.py`/`picker.py`/`daemon.py` need zero changes — the new modes produce the exact same `{"exact_match": ..., "candidates": [...]}` shape `search()` already returns.

**Tech Stack:** Pure Python stdlib (`re`, `dataclasses`) — no new dependencies.

## Global Constraints

- No change to `revdict.cli`'s argument parsing, `daemon.py`'s wire protocol, or `picker.py`/`revdict.nvim`'s consumption of `search()`'s output shape — every new mode returns the same `{"exact_match": dict | None, "candidates": list[dict]}` shape with the same candidate fields (`headword, pos, definition, examples, label, polarity, relevance, stress, synonyms`).
- Plain queries with no special characters (e.g. `"bluebird"`, `"a feeling of intense joy"`) must produce **functionally identical** results to before this plan (same retrieval, reranking, and exact-match behavior) — this is the hard backward-compatibility bar; Task 6's regression tests exist specifically to prove it. The one accepted difference: the meaning text used internally is `.strip()`-ed before being embedded/reranked, where the old code passed the raw unstripped query through to the embedder/reranker (only the exact-match lookup stripped before). This cannot change the exact-match behavior (identical either way) and does not change the retrieved candidate set in practice (the embedding/reranker models tokenize past leading/trailing whitespace), so it's treated as a non-issue rather than something to special-case away.
- Master headword vocabulary for structural matching is `state["word_index"]` (already loaded warm by both the daemon and the local fallback in `search._load_state()`) — no new index file, no new on-disk artifact. Benchmarked: a full linear scan with a compiled regex over the current 787,590-headword index averages ~57ms, well within interactive latency.
- `y` is treated as a consonant (not a vowel) for the `#`/`@` wildcards — an explicit interpretation call (TODO.md's legend doesn't specify), documented inline where the vowel/consonant sets are defined.
- `//abcd` and `//abcd//` (closing slashes optional) are both accepted as the anagram/unscramble syntax — confirmed equivalent via the TODO.md example itself (`//fuljyo` sorts to the same letters as `joyful`).

---

### Task 1: `query_syntax.py` — `ParsedQuery` and backward-compatible meaning-mode detection

**Files:**
- Create: `src/revdict/query_syntax.py`
- Test: `tests/test_query_syntax.py`

**Interfaces:**
- Produces: `ParsedQuery` dataclass with fields `mode: str`, `pattern_clauses: list[str]`, `meaning_text: str | None`, `expand_target: str | None`, `phrase_word: str | None`. Produces: `parse_query(raw: str) -> ParsedQuery`.
- Consumes: nothing (this task has no dependencies on other new modules).

- [ ] **Step 1: Write the failing tests for the dataclass and the two "meaning" cases**

```python
# tests/test_query_syntax.py
from revdict.query_syntax import ParsedQuery, parse_query


def test_parsed_query_defaults_have_empty_pattern_clauses_and_none_fields():
    parsed = ParsedQuery(mode="meaning")

    assert parsed.pattern_clauses == []
    assert parsed.meaning_text is None
    assert parsed.expand_target is None
    assert parsed.phrase_word is None


def test_plain_word_with_no_special_characters_parses_as_meaning_mode():
    """Backward compatibility: revdict's existing default behavior for a
    plain query like "bluebird" or a full descriptive phrase must be
    completely unaffected by the new query DSL."""
    parsed = parse_query("bluebird")

    assert parsed.mode == "meaning"
    assert parsed.meaning_text == "bluebird"


def test_plain_meaning_query_is_stripped_of_surrounding_whitespace():
    parsed = parse_query("  a feeling of intense joy  ")

    assert parsed.mode == "meaning"
    assert parsed.meaning_text == "a feeling of intense joy"


def test_colon_prefix_with_empty_pattern_part_is_meaning_mode():
    """':snow' -> list words related to snow (TODO.md line 12) -- same
    result as typing 'snow' directly; the colon here is just the
    degenerate case of the pattern:meaning separator with an empty
    pattern part."""
    parsed = parse_query(":snow")

    assert parsed.mode == "meaning"
    assert parsed.meaning_text == "snow"


def test_colon_prefix_meaning_text_can_contain_spaces():
    """':winter sport' -> related to the concept winter sport (TODO.md line 13)."""
    parsed = parse_query(":winter sport")

    assert parsed.mode == "meaning"
    assert parsed.meaning_text == "winter sport"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_query_syntax.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'revdict.query_syntax'`

- [ ] **Step 3: Write the minimal implementation**

```python
# src/revdict/query_syntax.py
from dataclasses import dataclass, field


@dataclass
class ParsedQuery:
    mode: str  # "meaning" | "structural" | "combined" | "expand" | "phrase_contains"
    pattern_clauses: list[str] = field(default_factory=list)
    meaning_text: str | None = None
    expand_target: str | None = None
    phrase_word: str | None = None


def parse_query(raw: str) -> ParsedQuery:
    text = raw.strip()

    if ":" in text:
        pattern_part, meaning_part = text.split(":", 1)
        pattern_part = pattern_part.strip()
        meaning_part = meaning_part.strip()
        if not pattern_part:
            return ParsedQuery(mode="meaning", meaning_text=meaning_part)

    return ParsedQuery(mode="meaning", meaning_text=text)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_query_syntax.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/revdict/query_syntax.py tests/test_query_syntax.py
git commit -m "Add ParsedQuery and backward-compatible meaning-mode query parsing"
```

---

### Task 2: `query_syntax.py` — expand, phrase-contains, structural, and combined modes

**Files:**
- Modify: `src/revdict/query_syntax.py`
- Test: `tests/test_query_syntax.py`

**Interfaces:**
- Consumes: `ParsedQuery` from Task 1.
- Produces: `parse_query` now returns all five modes; later tasks (`pattern_matcher.py`, `structural_search.py`) consume `parsed.pattern_clauses` (list of raw clause strings, comma-split), `parsed.expand_target` (lowercased string), `parsed.phrase_word` (lowercased string).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_query_syntax.py (append)


def test_expand_prefix_parses_the_target_letters_lowercased():
    """'expand:nasa' -> phrases that spell out n.a.s.a. (TODO.md line 15)."""
    parsed = parse_query("expand:NASA")

    assert parsed.mode == "expand"
    assert parsed.expand_target == "nasa"


def test_double_star_wrapped_word_parses_as_phrase_contains():
    """'**winter**' -> phrases that contain the word winter (TODO.md line 14)."""
    parsed = parse_query("**winter**")

    assert parsed.mode == "phrase_contains"
    assert parsed.phrase_word == "winter"


def test_prefix_wildcard_parses_as_a_single_structural_clause():
    """'blue*' -> list words that start with blue (TODO.md line 6)."""
    parsed = parse_query("blue*")

    assert parsed.mode == "structural"
    assert parsed.pattern_clauses == ["blue*"]


def test_suffix_wildcard_parses_as_a_single_structural_clause():
    """'*bird' -> ...that end with bird (TODO.md line 7)."""
    parsed = parse_query("*bird")

    assert parsed.mode == "structural"
    assert parsed.pattern_clauses == ["*bird"]


def test_letter_position_wildcard_parses_as_a_single_structural_clause():
    """'bl????rd' -> start with bl, end with rd, 4 letters between (TODO.md line 8)."""
    parsed = parse_query("bl????rd")

    assert parsed.mode == "structural"
    assert parsed.pattern_clauses == ["bl????rd"]


def test_double_slash_contains_letters_parses_as_a_single_structural_clause():
    """'//fuljyo' -> have the letters "fuljyo" (TODO.md line 9)."""
    parsed = parse_query("//fuljyo")

    assert parsed.mode == "structural"
    assert parsed.pattern_clauses == ["//fuljyo"]


def test_comma_separated_clauses_split_into_multiple_pattern_clauses():
    """'?????,*y*' -> 5 letters AND contains a y (TODO.md line 10)."""
    parsed = parse_query("?????,*y*")

    assert parsed.mode == "structural"
    assert parsed.pattern_clauses == ["?????", "*y*"]


def test_disallow_letters_clause_parses_as_structural():
    parsed = parse_query("-abcd")

    assert parsed.mode == "structural"
    assert parsed.pattern_clauses == ["-abcd"]


def test_restrict_letters_clause_parses_as_structural():
    parsed = parse_query("+abcd")

    assert parsed.mode == "structural"
    assert parsed.pattern_clauses == ["+abcd"]


def test_pattern_colon_meaning_parses_as_combined_mode():
    """'bl*:snow' -> start with bl and have a meaning related to snow (TODO.md line 11)."""
    parsed = parse_query("bl*:snow")

    assert parsed.mode == "combined"
    assert parsed.pattern_clauses == ["bl*"]
    assert parsed.meaning_text == "snow"


def test_comma_clauses_combined_with_meaning():
    parsed = parse_query("?????,*y*:winter sport")

    assert parsed.mode == "combined"
    assert parsed.pattern_clauses == ["?????", "*y*"]
    assert parsed.meaning_text == "winter sport"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_query_syntax.py -v`
Expected: FAIL — the new mode-detection tests fail because `parse_query` currently only ever returns `mode="meaning"`.

- [ ] **Step 3: Write the implementation**

```python
# src/revdict/query_syntax.py
from dataclasses import dataclass, field

_WILDCARD_TRIGGER_CHARS = set("*?#@")


@dataclass
class ParsedQuery:
    mode: str  # "meaning" | "structural" | "combined" | "expand" | "phrase_contains"
    pattern_clauses: list[str] = field(default_factory=list)
    meaning_text: str | None = None
    expand_target: str | None = None
    phrase_word: str | None = None


def _split_pattern_clauses(text: str) -> list[str]:
    return [clause.strip() for clause in text.split(",") if clause.strip()]


def _looks_structural(text: str) -> bool:
    if "//" in text:
        return True
    if text.startswith("-") or text.startswith("+"):
        return True
    return any(char in _WILDCARD_TRIGGER_CHARS for char in text)


def parse_query(raw: str) -> ParsedQuery:
    text = raw.strip()

    if text.lower().startswith("expand:"):
        return ParsedQuery(mode="expand", expand_target=text[len("expand:"):].strip().lower())

    if text.startswith("**") and text.endswith("**") and len(text) > 4:
        return ParsedQuery(mode="phrase_contains", phrase_word=text[2:-2].strip().lower())

    if ":" in text:
        pattern_part, meaning_part = text.split(":", 1)
        pattern_part = pattern_part.strip()
        meaning_part = meaning_part.strip()
        if not pattern_part:
            return ParsedQuery(mode="meaning", meaning_text=meaning_part)
        return ParsedQuery(
            mode="combined",
            pattern_clauses=_split_pattern_clauses(pattern_part),
            meaning_text=meaning_part,
        )

    if _looks_structural(text):
        return ParsedQuery(mode="structural", pattern_clauses=_split_pattern_clauses(text))

    return ParsedQuery(mode="meaning", meaning_text=text)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_query_syntax.py -v`
Expected: PASS (16 passed)

- [ ] **Step 5: Commit**

```bash
git add src/revdict/query_syntax.py tests/test_query_syntax.py
git commit -m "Parse expand/phrase-contains/structural/combined query modes"
```

---

### Task 3: `pattern_matcher.py` — wildcard/regex clause compiler

**Files:**
- Create: `src/revdict/pattern_matcher.py`
- Test: `tests/test_pattern_matcher.py`

**Interfaces:**
- Consumes: raw clause strings as produced by `query_syntax._split_pattern_clauses` (e.g. `"blue*"`, `"bl????rd"`).
- Produces: `compile_pattern(clause: str) -> Callable[[str], bool]` — this task covers only the wildcard/regex branch; Task 4 adds the disallow/restrict/anagram branches to the same function.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pattern_matcher.py
from revdict.pattern_matcher import compile_pattern


def test_prefix_wildcard_matches_words_starting_with_the_literal_prefix():
    matches = compile_pattern("blue*")

    assert matches("bluebird") is True
    assert matches("blueprint") is True
    assert matches("skyblue") is False


def test_suffix_wildcard_matches_words_ending_with_the_literal_suffix():
    matches = compile_pattern("*bird")

    assert matches("bluebird") is True
    assert matches("mockingbird") is True
    assert matches("birdcage") is False


def test_middle_wildcard_matches_words_containing_the_literal_substring():
    matches = compile_pattern("*y*")

    assert matches("happy") is True
    assert matches("yellow") is True
    assert matches("gold") is False


def test_single_letter_wildcard_matches_exact_length_with_fixed_head_and_tail():
    """'bl????rd' -> start with bl, end with rd, 4 letters between (TODO.md line 8)."""
    matches = compile_pattern("bl????rd")

    assert matches("blizzard") is True  # b-l-i-z-z-a-rd: 8 letters, bl + 4 + rd
    assert matches("blackbird") is False  # wrong length
    assert matches("blrd") is False  # too short


def test_all_question_marks_matches_pure_length():
    """'?????' -> 5-letter words (TODO.md line 10)."""
    matches = compile_pattern("?????")

    assert matches("happy") is True
    assert matches("sad") is False
    assert matches("gloomy") is False


def test_consonant_wildcard_matches_only_consonants_and_y_counts_as_consonant():
    matches = compile_pattern("#at")

    assert matches("bat") is True
    assert matches("yat") is True
    assert matches("eat") is False


def test_vowel_wildcard_matches_only_vowels():
    matches = compile_pattern("c@t")

    assert matches("cat") is True
    assert matches("cot") is True
    assert matches("cyt") is False


def test_wildcard_matching_is_case_insensitive():
    matches = compile_pattern("Blue*")

    assert matches("BLUEBIRD") is True
    assert matches("BlueJay") is True


def test_literal_characters_in_a_clause_are_matched_exactly():
    matches = compile_pattern("cat")

    assert matches("cat") is True
    assert matches("cats") is False
    assert matches("scat") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_pattern_matcher.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'revdict.pattern_matcher'`

- [ ] **Step 3: Write the implementation**

```python
# src/revdict/pattern_matcher.py
import re
from typing import Callable

# `y` is treated as a consonant (not a vowel) for the `#`/`@` wildcards --
# TODO.md's own symbol legend doesn't specify either way, so this is a
# documented interpretation call: English orthography favors the consonant
# reading far more often (yellow, yes, yard) than the reverse.
_VOWELS = set("aeiou")
_CONSONANTS = set("abcdefghijklmnopqrstuvwxyz") - _VOWELS

_WILDCARD_TRANSLATION = {
    "*": ".*",
    "?": ".",
    "#": f"[{''.join(sorted(_CONSONANTS))}]",
    "@": f"[{''.join(sorted(_VOWELS))}]",
}


def _translate_wildcard_clause(clause: str) -> str:
    # Full anchoring (^...$) is correct uniformly for every wildcard shape
    # here, not just fixed-length ones: 'blue*' -> '^blue.*$' still reads as
    # "starts with blue" because '.*$' already permits anything through to
    # the end of the string.
    parts = [_WILDCARD_TRANSLATION.get(char, re.escape(char)) for char in clause]
    return "^" + "".join(parts) + "$"


def compile_pattern(clause: str) -> Callable[[str], bool]:
    pattern = re.compile(_translate_wildcard_clause(clause), re.IGNORECASE)
    return lambda word: pattern.match(word) is not None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_pattern_matcher.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add src/revdict/pattern_matcher.py tests/test_pattern_matcher.py
git commit -m "Add wildcard/regex clause compiler for structural word patterns"
```

---

### Task 4: `pattern_matcher.py` — disallow, restrict, anagram clauses, and clause AND-combiner

**Files:**
- Modify: `src/revdict/pattern_matcher.py`
- Test: `tests/test_pattern_matcher.py`

**Interfaces:**
- Consumes: `compile_pattern` from Task 3 (extended, not replaced).
- Produces: `compile_pattern` now also handles `-abcd`/`+abcd`/`//abcd//` clauses; new `compile_clauses(clauses: list[str]) -> Callable[[str], bool]` (AND-combines multiple compiled clauses) — this is what `structural_search.py` (Task 5) calls.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pattern_matcher.py (append)
from revdict.pattern_matcher import compile_clauses


def test_disallow_letters_rejects_words_containing_any_excluded_letter():
    matches = compile_pattern("-xyz")

    assert matches("cat") is True
    assert matches("lazy") is False  # contains z
    assert matches("year") is False  # contains y


def test_disallow_letters_is_case_insensitive():
    matches = compile_pattern("-XYZ")

    assert matches("lazy") is False


def test_restrict_letters_only_allows_words_built_from_the_given_alphabet():
    matches = compile_pattern("+cat")

    assert matches("cat") is True
    assert matches("tact") is True
    assert matches("act") is True
    assert matches("cats") is False  # 's' not in the restricted alphabet


def test_anagram_with_closing_slashes_requires_using_every_letter_exactly_once():
    """'//abcd//' -> unscramble (TODO.md line 16 legend)."""
    matches = compile_pattern("//dear//")

    assert matches("read") is True
    assert matches("dare") is True
    assert matches("dear") is True
    assert matches("dared") is False  # extra letter
    assert matches("dea") is False  # missing letter


def test_anagram_without_closing_slashes_matches_the_same_way():
    """'//fuljyo' -> have the letters "fuljyo" (TODO.md line 9) -- confirmed
    an anagram of "joyful" (sorted letters are identical)."""
    matches = compile_pattern("//fuljyo")

    assert matches("joyful") is True
    assert matches("fully") is False


def test_compile_clauses_ands_every_clause_together():
    """'?????,*y*' -> 5 letters AND contains a y (TODO.md line 10)."""
    matches = compile_clauses(["?????", "*y*"])

    assert matches("happy") is True  # 5 letters, contains y
    assert matches("gold") is False  # no y
    assert matches("mystery") is False  # 7 letters, not 5


def test_compile_clauses_with_a_single_clause_behaves_like_compile_pattern():
    matches = compile_clauses(["blue*"])

    assert matches("bluebird") is True
    assert matches("redbird") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_pattern_matcher.py -v`
Expected: FAIL — `ImportError` for `compile_clauses`, and the disallow/restrict/anagram tests fail because those clauses currently fall through to the literal wildcard compiler (`-xyz` would be treated as literal characters `-`, `x`, `y`, `z`, not "exclude these letters").

- [ ] **Step 3: Write the implementation**

```python
# src/revdict/pattern_matcher.py
import re
from typing import Callable

_VOWELS = set("aeiou")
_CONSONANTS = set("abcdefghijklmnopqrstuvwxyz") - _VOWELS

_WILDCARD_TRANSLATION = {
    "*": ".*",
    "?": ".",
    "#": f"[{''.join(sorted(_CONSONANTS))}]",
    "@": f"[{''.join(sorted(_VOWELS))}]",
}


def _translate_wildcard_clause(clause: str) -> str:
    parts = [_WILDCARD_TRANSLATION.get(char, re.escape(char)) for char in clause]
    return "^" + "".join(parts) + "$"


def _compile_wildcard(clause: str) -> Callable[[str], bool]:
    pattern = re.compile(_translate_wildcard_clause(clause), re.IGNORECASE)
    return lambda word: pattern.match(word) is not None


def _compile_disallow(clause: str) -> Callable[[str], bool]:
    excluded = set(clause[1:].lower())
    return lambda word: excluded.isdisjoint(word.lower())


def _compile_restrict(clause: str) -> Callable[[str], bool]:
    allowed = set(clause[1:].lower())
    return lambda word: set(word.lower()) <= allowed


def _compile_anagram(clause: str) -> Callable[[str], bool]:
    letters = sorted(clause.strip("/").lower())
    return lambda word: sorted(word.lower()) == letters


def compile_pattern(clause: str) -> Callable[[str], bool]:
    if "//" in clause:
        return _compile_anagram(clause)
    if clause.startswith("-"):
        return _compile_disallow(clause)
    if clause.startswith("+"):
        return _compile_restrict(clause)
    return _compile_wildcard(clause)


def compile_clauses(clauses: list[str]) -> Callable[[str], bool]:
    predicates = [compile_pattern(clause) for clause in clauses]
    return lambda word: all(predicate(word) for predicate in predicates)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_pattern_matcher.py -v`
Expected: PASS (16 passed)

- [ ] **Step 5: Commit**

```bash
git add src/revdict/pattern_matcher.py tests/test_pattern_matcher.py
git commit -m "Add disallow/restrict/anagram clause compilers and clause AND-combiner"
```

---

### Task 5: `structural_search.py` — matching headwords and full candidate building

**Files:**
- Create: `src/revdict/structural_search.py`
- Modify: `src/revdict/search.py:161` (rename `_get_classifier` to `get_classifier` — it's no longer search.py-private once structural_search.py needs it too; update both existing call sites in the same file)
- Test: `tests/test_structural_search.py`
- Test: `tests/test_search.py` (update any references to `_get_classifier` if they exist — check with `grep -rn _get_classifier src/ tests/` before editing; there are none today outside `search.py`'s own two call sites, this is a safety check, not an assumed edit)

**Interfaces:**
- Consumes: `ParsedQuery` (Task 1/2), `compile_clauses` (Task 4), `revdict.models.emotion.tag_emotion`, `revdict.models.stress.mark`, `revdict.search.relative_relevance`, `revdict.search.get_classifier` (renamed in this task).
- Produces: `matching_headwords(parsed: ParsedQuery, word_index: dict[str, list[int]]) -> list[str]`, `run_structural(parsed: ParsedQuery, state: dict, top_n: int) -> dict` — the latter returns the same `{"exact_match": None, "candidates": [...]}` shape `search.search()` already returns, consumed unchanged by Task 6.

- [ ] **Step 1: Rename `_get_classifier` to `get_classifier` in `search.py`**

In `src/revdict/search.py`, rename the function at line 161 and its two call sites (inside `search()`'s candidate-building loop and inside `tag_exact_match_senses`'s caller):

```python
def get_classifier(state: dict) -> EmotionClassifier:
    if state["classifier"] is None:
        state["classifier"] = EmotionClassifier()
    return state["classifier"]
```

Update both existing `_get_classifier(state)` call sites in the same file to `get_classifier(state)` (one inside `search()`'s `emotion = tag_emotion(record, classifier_factory=lambda: _get_classifier(state))`, one inside the `tag_exact_match_senses(...)` call at the bottom of `search()`).

- [ ] **Step 2: Run the existing test suite to confirm the rename didn't break anything**

Run: `.venv/bin/pytest tests/test_search.py -v`
Expected: PASS (no test references `_get_classifier` by name today — this is a pure internal rename)

- [ ] **Step 3: Write the failing tests for `matching_headwords`**

```python
# tests/test_structural_search.py
from revdict.query_syntax import ParsedQuery
from revdict.structural_search import matching_headwords


def test_structural_mode_returns_headwords_matching_the_compiled_clauses():
    parsed = ParsedQuery(mode="structural", pattern_clauses=["blue*"])
    word_index = {"bluebird": [0], "blueprint": [1], "redbird": [2]}

    result = matching_headwords(parsed, word_index)

    assert set(result) == {"bluebird", "blueprint"}


def test_expand_mode_matches_multiword_headwords_by_initials():
    """'expand:nasa' -> phrases that spell out n.a.s.a. (TODO.md line 15).
    Real acronym expansion skips small function words (and/of/the/...)
    rather than taking every token's initial literally -- verified by hand:
    "national aeronautics and space administration" only reduces to n-a-s-a
    once "and" is skipped (naively it's n-a-a-s-a)."""
    parsed = ParsedQuery(mode="expand", expand_target="nasa")
    word_index = {
        "national aeronautics and space administration": [0],
        "national association of state agencies": [1],
        "bluebird": [2],
    }

    result = matching_headwords(parsed, word_index)

    assert set(result) == {
        "national aeronautics and space administration",
        "national association of state agencies",
    }


def test_expand_mode_skips_single_word_headwords():
    parsed = ParsedQuery(mode="expand", expand_target="n")
    word_index = {"nice": [0]}

    assert matching_headwords(parsed, word_index) == []


def test_phrase_contains_mode_matches_whole_word_tokens_only():
    """'**winter**' -> phrases that contain the word winter (TODO.md line 14) --
    must match the whole token 'winter', not any headword whose letters
    happen to contain that substring across a word boundary."""
    parsed = ParsedQuery(mode="phrase_contains", phrase_word="winter")
    word_index = {
        "winter sport": [0],
        "harsh winter": [1],
        "wintertime": [2],  # single word containing the substring -- must NOT match
        "midwinter storm": [3],  # 'midwinter' is one token, not 'winter' -- must NOT match
    }

    result = matching_headwords(parsed, word_index)

    assert set(result) == {"winter sport", "harsh winter"}
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_structural_search.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'revdict.structural_search'`

- [ ] **Step 5: Write the implementation of `matching_headwords`**

```python
# src/revdict/structural_search.py
from revdict.pattern_matcher import compile_clauses
from revdict.query_syntax import ParsedQuery

# Real acronym expansion skips small function words rather than taking
# every token's initial literally -- "national aeronautics and space
# administration" only reduces to n-a-s-a once "and" is dropped (verified
# by hand: naively including it gives n-a-a-s-a, which never matches).
_EXPAND_SKIP_WORDS = {"and", "of", "the", "for", "a", "an", "&"}


def matching_headwords(parsed: ParsedQuery, word_index: dict[str, list[int]]) -> list[str]:
    if parsed.mode == "structural":
        predicate = compile_clauses(parsed.pattern_clauses)
        return [word for word in word_index if predicate(word)]

    if parsed.mode == "expand":
        target = parsed.expand_target
        matches = []
        for word in word_index:
            tokens = [token for token in word.split() if token.lower() not in _EXPAND_SKIP_WORDS]
            if len(tokens) < 2:
                continue
            initials = "".join(token[0] for token in tokens if token).lower()
            if initials == target:
                matches.append(word)
        return matches

    if parsed.mode == "phrase_contains":
        target = parsed.phrase_word
        matches = []
        for word in word_index:
            tokens = [token.lower() for token in word.split()]
            if len(tokens) < 2:
                continue
            if target in tokens:
                matches.append(word)
        return matches

    raise ValueError(f"matching_headwords does not support mode {parsed.mode!r}")
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_structural_search.py -v`
Expected: PASS (4 passed)

- [ ] **Step 7: Write the failing tests for `run_structural`**

```python
# tests/test_structural_search.py (append)
from revdict.structural_search import run_structural


def _build_state():
    # emolex carries a specific category ("joy", not just a bare sentiment
    # flag) for both fixture records so tag_emotion's classifier fallback
    # never fires -- see emotion.py's _emolex_has_specific_category. Without
    # this, run_structural's classifier_factory would actually construct a
    # real EmotionClassifier (downloads/loads a transformers pipeline),
    # exactly as test_search.py's own existing fixtures are careful to avoid
    # (see its test_tag_exact_match_senses_tags_each_sense_... first-sense
    # comment and _FakeClassifier usage).
    metadata = [
        {
            "headword": "bluebird",
            "pos": "noun",
            "definition": "an American songbird",
            "examples": [],
            "source": "wordnet",
            "sentiwordnet": None,
            "emolex": ["joy"],
            "synonyms": None,
        },
        {
            "headword": "blueprint",
            "pos": "noun",
            "definition": "a technical drawing",
            "examples": [],
            "source": "wordnet",
            "sentiwordnet": None,
            "emolex": ["joy"],
            "synonyms": None,
        },
    ]
    word_index = {"bluebird": [0], "blueprint": [1]}
    literary_frequency = {"bluebird": 1.5, "blueprint": 3.2}
    return {
        "metadata": metadata,
        "word_index": word_index,
        "literary_frequency": literary_frequency,
        "classifier": None,
    }


def test_run_structural_returns_no_exact_match():
    """Structural search matches a set of words, not one pinned headword --
    exact_match is always None for these modes, distinguishing them from a
    dictionary lookup."""
    parsed = ParsedQuery(mode="structural", pattern_clauses=["blue*"])
    state = _build_state()

    result = run_structural(parsed, state, top_n=10)

    assert result["exact_match"] is None


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
        }


def test_run_structural_ranks_more_frequent_words_first():
    """No embedding-based relevance score exists for structural matches, so
    matches are ranked by literary_frequency (blueprint=3.2 > bluebird=1.5)
    -- reuses the same signal combine_score already applies as a nudge in
    the semantic path, just exposed directly here as the primary order."""
    parsed = ParsedQuery(mode="structural", pattern_clauses=["blue*"])
    state = _build_state()

    result = run_structural(parsed, state, top_n=10)

    assert [c["headword"] for c in result["candidates"]] == ["blueprint", "bluebird"]


def test_run_structural_respects_top_n():
    parsed = ParsedQuery(mode="structural", pattern_clauses=["blue*"])
    state = _build_state()

    result = run_structural(parsed, state, top_n=1)

    assert len(result["candidates"]) == 1
```

- [ ] **Step 8: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_structural_search.py -v`
Expected: FAIL with `ImportError: cannot import name 'run_structural'`

- [ ] **Step 9: Write the implementation of `run_structural`**

```python
# src/revdict/structural_search.py (append)
from revdict.models.emotion import tag_emotion
from revdict.models import stress


def _score_and_sort(headwords: list[str], literary_frequency: dict[str, float]) -> list[tuple[str, float]]:
    scored = [(word, literary_frequency.get(word, 0.0)) for word in headwords]
    return sorted(scored, key=lambda pair: (-pair[1], pair[0]))


def run_structural(parsed: ParsedQuery, state: dict, top_n: int) -> dict:
    # Deferred import: search.py imports structural_search for dispatch
    # (Task 6), so importing search.py at module load time here would be
    # circular. Matches the lazy-import pattern already used elsewhere in
    # this codebase (cli.py's _local_search_fallback, daemon.py's
    # run_server) to defer a heavy/cyclic import until it's actually needed.
    from revdict.search import get_classifier, relative_relevance

    word_index = state["word_index"]
    metadata = state["metadata"]
    literary_frequency = state["literary_frequency"]

    headwords = matching_headwords(parsed, word_index)
    ranked = _score_and_sort(headwords, literary_frequency)[:top_n]
    relevances = relative_relevance([score for _, score in ranked])

    candidates = []
    for (headword, _), relevance in zip(ranked, relevances):
        row_index = word_index[headword][0]
        record = dict(metadata[row_index])
        if record.get("emolex"):
            record["emolex"] = frozenset(record["emolex"])
        emotion = tag_emotion(record, classifier_factory=lambda: get_classifier(state))
        candidates.append(
            {
                "headword": record["headword"],
                "pos": record["pos"],
                "definition": record["definition"],
                "examples": record["examples"],
                "label": emotion["label"],
                "polarity": emotion["polarity"],
                "relevance": relevance,
                "stress": stress.mark(record["headword"], record["pos"]),
                "synonyms": record.get("synonyms"),
            }
        )

    return {"exact_match": None, "candidates": candidates}
```

- [ ] **Step 10: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_structural_search.py -v`
Expected: PASS (8 passed)

- [ ] **Step 11: Commit**

```bash
git add src/revdict/structural_search.py src/revdict/search.py tests/test_structural_search.py
git commit -m "Add structural_search: headword matching and full candidate building"
```

---

### Task 6: Wire structural/expand/phrase_contains dispatch into `search.search()`

**Files:**
- Modify: `src/revdict/search.py:202` (the `search()` function)
- Test: `tests/test_search.py`

**Interfaces:**
- Consumes: `query_syntax.parse_query` (Task 2), `structural_search.run_structural` (Task 5).
- Produces: `search()`'s public signature (`search(query: str, top_n: int = 10) -> dict`) is unchanged; only its internal dispatch changes.

- [ ] **Step 1: Write the failing regression + dispatch tests**

These use `search()`'s real `_load_state()` machinery, so they need the same lightweight monkeypatching style the rest of `tests/test_search.py` would need for an integration test — instead, test the dispatch logic directly by calling `search.search` with a monkeypatched `_load_state`, matching this file's existing pure-function-focused style but adding one integration-level test that exercises the real branch-selection code path.

```python
# tests/test_search.py (append)
import pytest

from revdict import search as search_mod


def _fake_state():
    # emolex=["joy"] (a specific category, not None) so tag_emotion's
    # classifier fallback never fires and these tests never construct a
    # real EmotionClassifier -- see the identical note in
    # tests/test_structural_search.py's _build_state().
    metadata = [
        {
            "headword": "bluebird",
            "pos": "noun",
            "definition": "an American songbird",
            "examples": [],
            "source": "wordnet",
            "sentiwordnet": None,
            "emolex": ["joy"],
            "synonyms": None,
        },
    ]
    return {
        "metadata": metadata,
        "word_index": {"bluebird": [0]},
        "literary_frequency": {"bluebird": 1.0},
        "classifier": None,
    }


def test_search_dispatches_structural_queries_to_run_structural_and_skips_embedding(monkeypatch):
    """'blue*' must never touch the embedder/reranker at all -- asserting
    _load_state's embedder/reranker slots are never accessed proves the
    dispatch genuinely bypasses the semantic pipeline rather than just
    happening to produce the same answer."""
    state = _fake_state()
    monkeypatch.setattr(search_mod, "_load_state", lambda: state)

    result = search_mod.search("blue*", top_n=10)

    assert result["exact_match"] is None
    assert [c["headword"] for c in result["candidates"]] == ["bluebird"]


def test_search_still_handles_a_plain_meaning_query_via_the_existing_path(monkeypatch):
    """Backward compatibility: a query with no special characters at all
    must still reach the existing embed/rerank/exact-match code path,
    proven here by confirming the embedder is actually invoked."""
    state = _fake_state()
    calls = []

    class FakeEmbedder:
        def encode_query(self, query):
            calls.append(query)
            import numpy as np

            return np.array([1.0], dtype="float32")

    class FakeReranker:
        def score(self, query, definitions):
            return [1.0 for _ in definitions]

    state["embedder"] = FakeEmbedder()
    state["reranker"] = FakeReranker()
    import numpy as np

    state["embeddings"] = np.array([[1.0]], dtype="float32")
    state["embedding_norms"] = np.array([1.0])
    monkeypatch.setattr(search_mod, "_load_state", lambda: state)

    search_mod.search("bluebird", top_n=10)

    assert calls == ["bluebird"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_search.py -v -k "dispatch or still_handles"`
Expected: FAIL — `search("blue*", ...)` currently runs the full embed/rerank path (there's no dispatch yet), so the structural test's assumptions about `result["candidates"]` don't hold and/or the fake embedder/reranker stubs are exercised for the structural case too.

- [ ] **Step 3: Add the dispatch to `search()`**

At the top of `search.py`, add the import and insert the dispatch as the first lines of the `search()` function body (everything below is the existing, unmodified body — only the first 4 lines are new):

```python
# src/revdict/search.py (imports section, alongside the existing revdict imports)
from revdict import query_syntax
from revdict import structural_search
```

```python
def search(query: str, top_n: int = 10) -> dict:
    state = _load_state()

    parsed = query_syntax.parse_query(query)
    if parsed.mode in ("structural", "expand", "phrase_contains"):
        return structural_search.run_structural(parsed, state, top_n)

    metadata = state["metadata"]
    # ... existing body continues unchanged from here, EXCEPT every use of
    # the raw `query` parameter below must use `meaning_query` instead:
```

Concretely, in the existing body, add this line right after the dispatch block above (before `retrieval_pool_size = ...`):

```python
    meaning_query = parsed.meaning_text if parsed.meaning_text is not None else query
```

Then replace the two existing uses of the raw `query` parameter later in the function body with `meaning_query`:

```python
    query_vec = state["embedder"].encode_query(meaning_query)
```

```python
    exact_match_raw = dictionary.lookup_exact(meaning_query.strip(), state["word_index"], metadata)
```

And the reranker call:

```python
    rerank_scores = state["reranker"].score(meaning_query, definitions)
```

Every other line of `search()`'s existing body (retrieval, combine_score, dedupe, exclude, relevance, candidate building, `tag_exact_match_senses`) is unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_search.py -v`
Expected: PASS (all existing tests plus the 2 new ones)

- [ ] **Step 5: Run the full existing test suite to confirm no regression**

Run: `.venv/bin/pytest tests/ -v`
Expected: PASS — every pre-existing test in `tests/test_cli.py`, `tests/test_daemon.py`, `tests/test_picker.py`, `tests/test_dictionary.py`, `tests/test_query_env.py` still passes unmodified, since none of them touch `search.search`'s internals directly and the public signature/return shape is unchanged.

- [ ] **Step 6: Commit**

```bash
git add src/revdict/search.py tests/test_search.py
git commit -m "Dispatch structural/expand/phrase-contains queries in search()"
```

---

### Task 7: Combined mode (`pattern:meaning`) — restrict the retrieval pool

**Files:**
- Modify: `src/revdict/search.py`
- Modify: `src/revdict/structural_search.py`
- Test: `tests/test_search.py`
- Test: `tests/test_structural_search.py`

**Interfaces:**
- Consumes: `structural_search.matching_headwords` (Task 5).
- Produces: `structural_search.matching_row_indices(parsed: ParsedQuery, word_index: dict) -> list[int]`. `search()`'s combined-mode branch, using the existing `cosine_top_k` unchanged.

- [ ] **Step 1: Write the failing test for `matching_row_indices`**

```python
# tests/test_structural_search.py (append)
from revdict.structural_search import matching_row_indices


def test_matching_row_indices_maps_matched_headwords_to_their_metadata_rows():
    parsed = ParsedQuery(mode="combined", pattern_clauses=["blue*"], meaning_text="snow")
    word_index = {"bluebird": [0, 3], "blueprint": [1], "redbird": [2]}

    result = matching_row_indices(parsed, word_index)

    assert sorted(result) == [0, 1, 3]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_structural_search.py -v -k matching_row_indices`
Expected: FAIL with `ImportError: cannot import name 'matching_row_indices'`

- [ ] **Step 3: Implement `matching_row_indices`**

```python
# src/revdict/structural_search.py (append)
# Reuses the `compile_clauses` already imported at the top of this file in
# Task 5 -- no new import needed here.


def matching_row_indices(parsed: ParsedQuery, word_index: dict[str, list[int]]) -> list[int]:
    predicate = compile_clauses(parsed.pattern_clauses)
    indices = []
    for word, rows in word_index.items():
        if predicate(word):
            indices.extend(rows)
    return indices
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_structural_search.py -v -k matching_row_indices`
Expected: PASS

- [ ] **Step 5: Write the failing integration test for combined-mode dispatch in `search()`**

```python
# tests/test_search.py (append)


def test_search_combined_mode_restricts_candidates_to_the_pattern_match(monkeypatch):
    """'blue*:snow' must only ever surface headwords matching 'blue*',
    even though the fake reranker below would happily score every row
    equally -- proving the structural filter actually narrows the pool
    before reranking, not just after."""
    # emolex=["joy"] on both records, not None -- see _fake_state()'s note
    # above on why this is required to avoid constructing a real
    # EmotionClassifier inside this test.
    metadata = [
        {
            "headword": "bluebird", "pos": "noun", "definition": "a songbird",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None,
        },
        {
            "headword": "redbird", "pos": "noun", "definition": "a songbird",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None,
        },
    ]
    state = {
        "metadata": metadata,
        "word_index": {"bluebird": [0], "redbird": [1]},
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

    result = search_mod.search("blue*:snow", top_n=10)

    assert [c["headword"] for c in result["candidates"]] == ["bluebird"]
    assert result["exact_match"] is None


def test_search_combined_mode_with_no_structural_matches_returns_no_candidates(monkeypatch):
    """A structural clause that matches nothing (e.g. an anagram with no
    real solutions) must return an empty result, not crash -- this is the
    regression test for the empty-definitions guard around the reranker
    call in search()'s combined-mode branch. Deliberately uses the bare
    _fake_state() fixture with no embedder/reranker/embeddings configured:
    zero structural matches means len(restrict_row_indices) == 0 <=
    retrieval_pool_size, so the code must take the direct small-match path
    and never touch those fields at all -- if it did, this test would fail
    with a KeyError instead of the assertions below, which is exactly the
    proof this guard is load-bearing."""
    state = _fake_state()
    monkeypatch.setattr(search_mod, "_load_state", lambda: state)

    result = search_mod.search("//zzzzqx:snow", top_n=10)

    assert result["candidates"] == []
    assert result["exact_match"] is None
```

- [ ] **Step 6: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_search.py -v -k combined_mode`
Expected: FAIL — `search()` has no `"combined"` branch yet, so this currently falls through to plain meaning-mode using the full raw string `"blue*:snow"` as the query text, returning both headwords.

- [ ] **Step 7: Implement combined-mode dispatch in `search()`**

Add the combined-mode branch right after the existing structural/expand/phrase_contains branch (from Task 6), and thread `restrict_row_indices`/`suppress_exact_match` through the rest of the function:

```python
def search(query: str, top_n: int = 10) -> dict:
    state = _load_state()

    parsed = query_syntax.parse_query(query)
    if parsed.mode in ("structural", "expand", "phrase_contains"):
        return structural_search.run_structural(parsed, state, top_n)

    metadata = state["metadata"]
    retrieval_pool_size = max(75, top_n * 3)

    restrict_row_indices = None
    suppress_exact_match = False
    if parsed.mode == "combined":
        restrict_row_indices = structural_search.matching_row_indices(parsed, state["word_index"])
        suppress_exact_match = True

    meaning_query = parsed.meaning_text if parsed.meaning_text is not None else query

    # query_vec is only computed on the branches that actually need cosine
    # retrieval -- when the structural filter has already narrowed the pool
    # to <= retrieval_pool_size rows, there's nothing to retrieve by
    # embedding similarity, so encoding the query would be pure waste.
    if restrict_row_indices is not None and len(restrict_row_indices) <= retrieval_pool_size:
        retrieved = [(index, 0.0) for index in restrict_row_indices]
    elif restrict_row_indices is not None:
        query_vec = state["embedder"].encode_query(meaning_query)
        subset_matrix = state["embeddings"][restrict_row_indices]
        subset_norms = state["embedding_norms"][restrict_row_indices]
        local_top = cosine_top_k(query_vec, subset_matrix, k=retrieval_pool_size, matrix_norms=subset_norms)
        retrieved = [(restrict_row_indices[local_index], score) for local_index, score in local_top]
    else:
        query_vec = state["embedder"].encode_query(meaning_query)
        retrieved = cosine_top_k(
            query_vec, state["embeddings"], k=retrieval_pool_size, matrix_norms=state["embedding_norms"]
        )

    definitions = [metadata[index]["definition"] for index, _ in retrieved]
    # A structural filter that matches zero headwords (e.g. an anagram with
    # no real solutions) reaches here with an empty `retrieved`/`definitions`
    # -- skip the reranker call entirely rather than relying on
    # CrossEncoder.predict's undocumented behavior on an empty batch.
    rerank_scores = state["reranker"].score(meaning_query, definitions) if definitions else []
    literary_frequency = state["literary_frequency"]
    scored = []
    for i in range(len(retrieved)):
        row_index = retrieved[i][0]
        headword = metadata[row_index]["headword"]
        adjusted = combine_score(rerank_scores[i], headword, literary_frequency)
        scored.append((row_index, adjusted))

    exact_match_raw = None
    if not suppress_exact_match:
        exact_match_raw = dictionary.lookup_exact(meaning_query.strip(), state["word_index"], metadata)
    exact_headword = exact_match_raw["headword"] if exact_match_raw is not None else None

    deduped = dedupe_by_headword(scored, metadata)
    deduped = exclude_headword(deduped, metadata, exact_headword)[:top_n]
    relevances = absolute_relevance([score for _, score in deduped])

    candidates = []
    for (row_index, _), relevance in zip(deduped, relevances):
        record = dict(metadata[row_index])
        if record.get("emolex"):
            record["emolex"] = frozenset(record["emolex"])
        emotion = tag_emotion(record, classifier_factory=lambda: get_classifier(state))
        candidates.append(
            {
                "headword": record["headword"],
                "pos": record["pos"],
                "definition": record["definition"],
                "examples": record["examples"],
                "label": emotion["label"],
                "polarity": emotion["polarity"],
                "relevance": relevance,
                "stress": stress.mark(record["headword"], record["pos"]),
                "synonyms": record.get("synonyms"),
            }
        )

    exact_match = tag_exact_match_senses(
        exact_match_raw, classifier_factory=lambda: get_classifier(state)
    )
    return {"exact_match": exact_match, "candidates": candidates}
```

This replaces the entire existing body of `search()` from `retrieval_pool_size = ...` through the final `return` — every line of logic downstream of retrieval (rerank, combine_score, dedupe, exclude, relevance, candidate building) is byte-identical to before; only how `retrieved` is computed at the top, and the new `suppress_exact_match` guard around the exact-match lookup, are new.

- [ ] **Step 8: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_search.py -v`
Expected: PASS (all tests, including the new combined-mode test)

- [ ] **Step 9: Run the full test suite**

Run: `.venv/bin/pytest tests/ -v`
Expected: PASS — full regression check across every test file.

- [ ] **Step 10: Commit**

```bash
git add src/revdict/search.py src/revdict/structural_search.py tests/test_search.py tests/test_structural_search.py
git commit -m "Implement combined pattern:meaning mode by restricting the retrieval pool"
```

---

### Task 8: End-to-end example coverage and README documentation

**Files:**
- Test: `tests/test_query_syntax.py` (a few more literal-example round-trip tests, if any TODO.md example isn't yet covered end-to-end by earlier tasks — check first)
- Modify: `README.md`

**Interfaces:**
- Consumes: everything from Tasks 1-7.
- Produces: nothing new — this task is verification + documentation only.

- [ ] **Step 1: Cross-check every TODO.md example against the test suite**

Run through TODO.md lines 5-16 one by one and confirm each has direct test coverage already:

| TODO.md example | Covered by |
|---|---|
| `bluebird` | `test_plain_word_with_no_special_characters_parses_as_meaning_mode` (Task 1), `test_search_still_handles_a_plain_meaning_query_via_the_existing_path` (Task 6) |
| `blue*` | `test_prefix_wildcard_parses_as_a_single_structural_clause` (Task 2), `test_prefix_wildcard_matches_words_starting_with_the_literal_prefix` (Task 3), `test_search_dispatches_structural_queries_to_run_structural_and_skips_embedding` (Task 6) |
| `*bird` | `test_suffix_wildcard_parses_as_a_single_structural_clause` (Task 2), `test_suffix_wildcard_matches_words_ending_with_the_literal_suffix` (Task 3) |
| `bl????rd` | `test_letter_position_wildcard_parses_as_a_single_structural_clause` (Task 2), `test_single_letter_wildcard_matches_exact_length_with_fixed_head_and_tail` (Task 3) |
| `//fuljyo` | `test_double_slash_contains_letters_parses_as_a_single_structural_clause` (Task 2), `test_anagram_without_closing_slashes_matches_the_same_way` (Task 4) |
| `?????,*y*` | `test_comma_separated_clauses_split_into_multiple_pattern_clauses` (Task 2), `test_compile_clauses_ands_every_clause_together` (Task 4) |
| `bl*:snow` | `test_pattern_colon_meaning_parses_as_combined_mode` (Task 2), `test_search_combined_mode_restricts_candidates_to_the_pattern_match` (Task 7) |
| `:snow` | `test_colon_prefix_with_empty_pattern_part_is_meaning_mode` (Task 1) |
| `:winter sport` | `test_colon_prefix_meaning_text_can_contain_spaces` (Task 1) |
| `**winter**` | `test_double_star_wrapped_word_parses_as_phrase_contains` (Task 2), `test_phrase_contains_mode_matches_whole_word_tokens_only` (Task 5) |
| `expand:nasa` | `test_expand_prefix_parses_the_target_letters_lowercased` (Task 2), `test_expand_mode_matches_multiword_headwords_by_initials` (Task 5) |
| `-abcd` / `+abcd` | `test_disallow_letters_rejects_words_containing_any_excluded_letter`, `test_restrict_letters_only_allows_words_built_from_the_given_alphabet` (Task 4) |

All 12 examples have direct test coverage from Tasks 1-7 — no gaps, so no new tests are needed in this step.

- [ ] **Step 2: Add a regression test for the leading-dash CLI argv edge case**

`revdict "-abcd"` as a one-shot CLI invocation risks argparse treating `-abcd` as an unrecognized flag rather than a positional query, since `_query_parser()`'s `query` argument is `nargs="*"`. This doesn't affect the live fzf session (its `--query-only`/`--jsonl-query` paths read `argv[1]` directly, bypassing argparse entirely — see `cli.py`'s `main()`), only the one-shot form. Confirm the documented workaround (`revdict -- -abcd`) works:

```python
# tests/test_cli.py (append)


def test_leading_dash_query_requires_the_argparse_separator(capsys):
    """A leading '-' in a one-shot query (the disallow-letters pattern
    syntax, e.g. '-abcd') collides with argparse's own flag parsing --
    'revdict -- -abcd' is the documented workaround (POSIX '--' end-of-
    options marker), not a bug in the query parser itself. The live fzf
    session is unaffected: --query-only/--jsonl-query read argv[1] directly
    and never go through this argparse path."""
    from revdict import cli

    code = cli.main(["--", "-abcd", "--no-interactive"])

    assert code == 0
```

- [ ] **Step 3: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/test_cli.py -v -k leading_dash`
Expected: PASS — argparse's built-in `--` handling already resolves this without any code change; this test documents and locks in the existing behavior rather than fixing a defect.

- [ ] **Step 4: Document the query syntax in README.md**

Add a new section to `README.md` (after whatever the current final section is — read the file first to place it correctly) documenting the syntax for end users:

```markdown
## Query syntax

Beyond plain word lookups and free-text meaning search, `revdict` understands
a small pattern-matching DSL, typed directly into the same prompt (works in
both the live session and one-shot `revdict "..."` queries):

| Query | Matches |
|---|---|
| `bluebird` | Exact word lookup / free-text meaning search (unchanged default) |
| `blue*` | Words starting with "blue" |
| `*bird` | Words ending with "bird" |
| `bl????rd` | Starts with "bl", ends with "rd", 4 letters between |
| `?????` | Any 5-letter word |
| `*y*` | Words containing "y" anywhere |
| `?????,*y*` | Combine clauses with a comma (AND): 5 letters AND contains "y" |
| `//fuljyo` or `//fuljyo//` | Anagram/unscramble: words using exactly these letters |
| `-abcd` | Words that don't contain any of these letters |
| `+abcd` | Words built only from these letters |
| `bl*:snow` | Starts with "bl" AND related in meaning to "snow" |
| `:snow` | Meaning search, explicit form (same as typing `snow` directly) |
| `**winter**` | Multi-word phrases containing the whole word "winter" |
| `expand:nasa` | Phrases whose initials spell "nasa" |

Note: `*`, `?`, `#`, `@`, and a leading `+`/`-` are pattern-syntax triggers,
so a free-text meaning query containing one of those characters (e.g. "a
word for asking a question?") will be parsed as a pattern instead. Prefix
the query with `:` to force meaning search explicitly.
```

- [ ] **Step 5: Run the full test suite one final time**

Run: `.venv/bin/pytest tests/ -v`
Expected: PASS — full suite green, confirming Phase 1 is complete and non-regressive.

- [ ] **Step 6: Commit**

```bash
git add tests/test_cli.py README.md
git commit -m "Document the query syntax DSL and lock in the leading-dash CLI edge case"
```

---

## Self-review notes (for the record)

- **Spec coverage:** all 12 literal TODO.md examples (lines 5-16) are covered — see Task 8's cross-check table. The "Pattern symbols" legend (line 16) is fully implemented: `?`, `*`, `#`, `@`, `-abcd`, `+abcd`, `//abcd//`, `pattern:meaning`.
- **Backward compatibility:** proven explicitly by `test_plain_word_with_no_special_characters_parses_as_meaning_mode` (Task 1) and `test_search_still_handles_a_plain_meaning_query_via_the_existing_path` (Task 6), plus the full existing test suite re-run at the end of Tasks 6, 7, and 8.
- **No placeholders:** every step above shows complete, real code — no "add appropriate handling" language anywhere in this document.
- **Type/name consistency:** `ParsedQuery` (Task 1) is used identically in Tasks 2, 5, 6, 7 with the same field names throughout. `compile_pattern`/`compile_clauses` (Tasks 3-4) are consumed by `structural_search.py` (Task 5) with matching signatures. `get_classifier` (renamed in Task 5, Step 1) is used consistently in `search.py`'s Task 7 rewrite and `structural_search.py`. `matching_headwords`/`matching_row_indices`/`run_structural` (Task 5, 7) are called from `search.py`'s Task 6/7 dispatch with matching signatures.
- **Second-pass fixes applied after an independent review of this plan (before any code was written):** (1) the original `expand`-mode draft took every token's initial literally, which contradicted its own test fixture ("national aeronautics and space administration" → `naasa`, not `nasa`) — fixed by skipping a small function-word set, verified by hand-tracing both fixture words. (2) every fake metadata record across Tasks 5-7 originally set `"emolex": None`, which — traced through `tag_emotion`/`_emolex_has_specific_category` in `models/emotion.py` — would have made every one of these "fast" unit tests silently construct a real `EmotionClassifier` (a transformers pipeline load) on every run; fixed by giving fixtures `"emolex": ["joy"]`, matching the existing house style already visible in `tests/test_search.py`'s `_FakeClassifier`/`classifier_factory=None` handling. (3) minor cleanups: `query_vec` is now computed only on the branches that need cosine retrieval, the reranker call is guarded against an empty definitions list (with a dedicated regression test in Task 7), and the duplicate `compile_clauses` import in Task 7 was removed in favor of reusing Task 5's top-level import.
