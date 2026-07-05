# Stressmark Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the standalone `emphasis`/stressmark project (English syllable stress + vowel reduction highlighting) into `revdict`'s candidate/exact-match display, as a fully optional runtime plugin.

**Architecture:** Package `emphasis` as an installable `uv`-managed library with a new POS-aware single-word entry point (`resolve_word_by_pos`, bypassing its own POS-tagging since revdict already knows the correct sense-specific POS from WordNet). `revdict` gets a small wrapper module that tries to import it and degrades to `None` everywhere if it's absent — `emphasis` is never declared as a revdict dependency. The stress-highlighted result is captured as an ANSI-coded string (JSON-safe for revdict's daemon socket protocol) and reconstructed into a Rich `Text` object only at the two display sites that need one.

**Tech Stack:** Python, `rich` (already a dependency of both projects), `uv`. No new third-party dependencies in either project beyond what each already has.

## Global Constraints

- **This is two separate git repositories.** `emphasis` lives at `/home/shichika/redacted/emphasis` (remote: `git@github.com:nijuyonkadesu/emphasis.git`). `revdict` lives at `/home/shichika/redacted/rev-dictionary` (remote: `git@github.com:nijuyonkadesu/revdict.git`). Every task below states which repo it works in — do not assume both are the same checkout.
- `emphasis`/`stressmark` must **never** be declared in `revdict`'s `pyproject.toml` or `uv.lock`, not even as an optional extra. `revdict`'s own install/test/CI behavior must be identical whether or not `emphasis` exists anywhere on the machine.
- "Importable = configured": there is no separate enable/disable flag in revdict. If `import stressmark...` succeeds, the feature is on; if it fails (for any reason), it's off, silently, everywhere.
- No behavior change to `emphasis`'s existing standalone CLI (`stressmark transcript.txt`, `--format html/json`, `--explain`, `--flag-heteronyms`, `--nuclear-only`) — this integration only adds new entry points.
- `resolve_word_by_pos` must bypass `emphasis`'s own POS-tagging/heteronym-guessing — the caller (revdict) already knows the correct sense-specific POS.
- Commit and push both repos as work lands (both already have `origin` remotes configured and reachable).

---

### Task 1: Package `emphasis` as an installable `uv`-managed library

**Repo:** `/home/shichika/redacted/emphasis`

**Files:**
- Create: `src/stressmark/__init__.py`
- Create: `src/stressmark/cli.py` (moved from `stressmark.py`, imports updated)
- Create: `src/stressmark/engine.py` (moved from `stressmark_engine.py`, no logic changes)
- Create: `src/stressmark/render.py` (moved from `stressmark_render.py`, no logic changes)
- Delete: `stressmark.py`, `stressmark_engine.py`, `stressmark_render.py` (superseded by the `src/stressmark/` versions)
- Create: `pyproject.toml`
- Modify: `.gitignore` (add `.venv/`, `uv.lock` already fine to track)
- Modify: `test_engine.py`, `test_heteronyms.py`, `test_secondary_stress.py` (remove a stale hardcoded `sys.path.insert(0, '/home/claude/stressmark')` left over from a different machine — dead/misleading code found while browsing; harmless today since these scripts also work via the script's-own-directory default, but worth removing since it's simply wrong for anyone else who reads it)

**Interfaces:**
- Consumes: nothing new.
- Produces: an installable `stressmark` package with console-script entry point `stressmark = "stressmark.cli:main"`, importable as `import stressmark.engine`, `import stressmark.render`. Task 2 adds new functions inside these same files. Task 3 (in the *other* repo) imports this package by name.

- [ ] **Step 1: Create the package directory and move the three modules**

```bash
mkdir -p src/stressmark
touch src/stressmark/__init__.py
git mv stressmark.py src/stressmark/cli.py
git mv stressmark_engine.py src/stressmark/engine.py
git mv stressmark_render.py src/stressmark/render.py
```

- [ ] **Step 2: Fix `src/stressmark/cli.py`'s imports for the new package layout**

Current top of the file:
```python
#!/usr/bin/env python3
"""
stressmark -- sentence-aware English stress marker.
...
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stressmark_engine import analyze, HETERONYMS
import stressmark_render as render
```

Replace the `sys.path` hack and the two imports (everything from `import argparse` through `import stressmark_render as render`) with:

```python
#!/usr/bin/env python3
"""
stressmark -- sentence-aware English stress marker.
...
"""
import argparse
import sys

from stressmark.engine import analyze, HETERONYMS
from stressmark import render
```

(Keep the rest of the file — the `main()` function and everything below it — unchanged; it doesn't reference `stressmark_engine`/`stressmark_render` by their old names anywhere else, only via the `analyze`/`HETERONYMS`/`render` names already imported.)

- [ ] **Step 3: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "stressmark"
version = "0.1.0"
description = "Sentence-aware English stress marker (primary/secondary/reduced syllables)"
requires-python = ">=3.11"
dependencies = [
    "nltk>=3.9",
    "pyphen>=0.17",
    "rich>=13.9",
    "g2p-en>=2.1",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[project.scripts]
stressmark = "stressmark.cli:main"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 4: Update `.gitignore`**

Add this line if not already present (check the existing file first — it's long, from a generic Python template):

```bash
grep -qxF '.venv/' .gitignore || echo '.venv/' >> .gitignore
```

- [ ] **Step 5: Remove the stale hardcoded path from the three demo/verification scripts**

Each of `test_engine.py`, `test_heteronyms.py`, `test_secondary_stress.py` currently starts with:

```python
import sys
sys.path.insert(0, '/home/claude/stressmark')
from stressmark_engine import analyze
```

Change each to:

```python
from stressmark.engine import analyze
```

(Just those two/three lines at the top of each file — the rest of each file is unchanged. These scripts remain informal manual-verification scripts, run directly with `python3 test_heteronyms.py` etc., not real pytest tests — that's an existing, pre-integration characteristic of this project, not something this task needs to fix.)

- [ ] **Step 6: Set up the `uv`-managed venv**

```bash
uv lock
rm -rf venv
uv sync --all-extras
```

- [ ] **Step 7: Verify the package imports and the CLI still works standalone**

```bash
.venv/bin/python -c "import stressmark.engine; import stressmark.render; print('OK')"
echo "The quick brown fox jumps over the lazy dog." | .venv/bin/stressmark
```

Expected: "OK" printed, then real stress-marked terminal output for the sample sentence (colored CAPS/underline/dim syllables), no traceback.

- [ ] **Step 8: Verify the three demo scripts still run after the import fix**

```bash
.venv/bin/python test_heteronyms.py
```

Expected: prints the OK/FAIL table for all 20 heteronym cases, ending in a score line (e.g. "19/20 correct (95%)") — matching the README's documented accuracy, confirming the import fix didn't change any behavior.

- [ ] **Step 9: Symlink the CLI onto PATH**

```bash
ln -sf "$(pwd)/.venv/bin/stressmark" ~/.local/bin/stressmark
stressmark --help
```

Expected: prints the argparse usage/help text via the bare `stressmark` command (confirms `~/.local/bin` is on PATH, same as already verified for `revdict` earlier).

- [ ] **Step 10: Commit and push**

```bash
git add -A
git commit -m "$(cat <<'EOF'
Package stressmark as an installable uv-managed library

Restructures into a src/stressmark/ package layout with a proper
pyproject.toml and console-script entry point, migrates from a local
pip venv to uv (matching the sibling revdict project), and removes a
stale hardcoded path left over from a different machine in the
verification scripts. No behavior change to the existing CLI.
EOF
)"
git push
```

---

### Task 2: Add `resolve_word_by_pos` and `render_word` to `emphasis`

**Repo:** `/home/shichika/redacted/emphasis`

**Files:**
- Modify: `src/stressmark/engine.py` (add `resolve_word_by_pos`)
- Modify: `src/stressmark/render.py` (add `render_word`)
- Create: `tests/__init__.py`, `tests/test_integration.py`

**Interfaces:**
- Consumes: `resolve_word(raw, tag, sent_tags_after)` and the `WordResult` class (both already exist in `src/stressmark/engine.py`, unchanged); `_confidence_marker(conf)` (already exists in `src/stressmark/render.py`, unchanged).
- Produces: `stressmark.engine.resolve_word_by_pos(word: str, pos: str) -> WordResult`, `stressmark.render.render_word(result: WordResult) -> rich.text.Text`. Task 3 (in the `revdict` repo) imports and calls both of these.

- [ ] **Step 1: Write the failing tests**

These values were verified directly against the real engine before writing this plan (not guessed):
- `resolve_word_by_pos("record", "noun")` → primary stress on syllable 0 (RE-cord)
- `resolve_word_by_pos("record", "verb")` → primary stress on syllable 1 (re-CORD)
- `resolve_word_by_pos("happy", "adjective")` → syllables `["hap", "py"]`, primary 0
- `render_word(resolve_word_by_pos("happy", "adjective"))` → plain text `"HAPpy"`, with a `bold yellow` span over `HAP` and a `grey62` span over `py`
- `render_word(resolve_word_by_pos("kubernetes", "noun"))` → plain text `"kuBERnetes≈"`, confidence `"predicted"`

```python
# tests/test_integration.py
from stressmark.engine import resolve_word_by_pos
from stressmark.render import render_word


def test_resolve_word_by_pos_resolves_heteronym_as_noun():
    result = resolve_word_by_pos("record", "noun")

    assert result.primary == 0


def test_resolve_word_by_pos_resolves_heteronym_as_verb():
    result = resolve_word_by_pos("record", "verb")

    assert result.primary == 1


def test_resolve_word_by_pos_handles_a_plain_dictionary_word():
    result = resolve_word_by_pos("happy", "adjective")

    assert result.syllables == ["hap", "py"]
    assert result.primary == 0


def test_resolve_word_by_pos_falls_back_to_noun_tag_for_an_unrecognized_pos_string():
    # revdict's POS vocabulary includes things beyond noun/verb/adjective/adverb
    # (e.g. WordNet's "name" for proper nouns, or raw Wiktionary POS strings
    # like "article"/"prefix") -- any of these should fall back to treating
    # the word as a common noun rather than raising or guessing wildly.
    result = resolve_word_by_pos("record", "name")

    assert result.primary == 0  # noun reading


def test_resolve_word_by_pos_predicts_words_absent_from_the_dictionary():
    result = resolve_word_by_pos("kubernetes", "noun")

    assert result.confidence == "predicted"
    assert len(result.syllables) > 1


def test_render_word_uppercases_primary_and_styles_the_rest():
    result = resolve_word_by_pos("happy", "adjective")

    text = render_word(result)

    assert text.plain == "HAPpy"
    spans = {(s.start, s.end): s.style for s in text.spans}
    assert spans[(0, 3)] == "bold yellow"
    assert spans[(3, 5)] == "grey62"


def test_render_word_marks_predicted_words_with_the_confidence_symbol():
    result = resolve_word_by_pos("kubernetes", "noun")

    text = render_word(result)

    assert text.plain == "kuBERnetes≈"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_integration.py -v`
Expected: FAIL with `ImportError: cannot import name 'resolve_word_by_pos' from 'stressmark.engine'`

- [ ] **Step 3: Add `resolve_word_by_pos` to `src/stressmark/engine.py`**

Add this near `resolve_word` (which it wraps):

```python
_POS_VOCAB_TO_TAG = {
    "noun": "NN",
    "verb": "VB",
    "adjective": "JJ",
    "adverb": "RB",
}


def resolve_word_by_pos(word, pos):
    """Resolve a single word's stress pattern given an ALREADY-KNOWN part of
    speech, bypassing this module's own POS-tagging and heteronym-guessing
    entirely. Intended for callers (like revdict) that already know the
    correct sense-specific POS from their own dictionary data -- both
    faster (no POS-tagger model needed for this path) and more accurate
    (no context-free guessing on an isolated word, which is exactly where
    heteronym resolution like record/object needs real context).

    `pos` uses revdict's vocabulary ("noun"/"verb"/"adjective"/"adverb");
    anything else (e.g. WordNet's "name" for proper nouns, or a raw
    Wiktionary POS string) falls back to treating the word as a common
    noun.
    """
    tag = _POS_VOCAB_TO_TAG.get(pos, "NN")
    return resolve_word(word, tag, [])
```

- [ ] **Step 4: Add `render_word` to `src/stressmark/render.py`**

Add this near `render_terminal` (which it's a single-word specialization of):

```python
def render_word(result):
    """Render one WordResult's stress-highlighted syllables as a Rich Text
    object. Unlike render_terminal (which operates on a whole analyzed
    document with a sentence-level nuclear-stress tiering pass), a
    WordResult from resolve_word_by_pos() never has .tier set -- tiering
    only happens in analyze()'s sentence-level pass -- so only
    confidence-based styling applies here, not the tier-based styles
    (nuclear/prominent/pre-nuclear)."""
    from rich.text import Text

    text = Text()
    if result.cls == "reducible":
        text.append(result.raw.lower(), style="dim")
        return text

    sylls = result.syllables
    conf = result.confidence
    for i, s in enumerate(sylls):
        if s == "-":
            text.append("-")
            continue
        if i == result.primary:
            if conf == "predicted":
                style = "bold italic yellow3"
            elif conf == "dict-flagged":
                style = "bold orange3"
            else:
                style = "bold yellow"
            text.append(s.upper(), style=style)
        elif i in result.secondary:
            text.append(s.lower(), style="underline yellow3")
        else:
            text.append(s.lower(), style="grey62")

    marker = _confidence_marker(conf)
    if marker:
        text.append(marker, style="bold red" if conf == "dict-flagged" else "yellow3")

    return text
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_integration.py -v`
Expected: PASS (all 7 tests)

- [ ] **Step 6: Run the demo scripts once more to confirm no regression**

```bash
.venv/bin/python test_heteronyms.py
```

Expected: same "19/20 correct (95%)" result as Task 1's Step 8 (these new functions are additive; `resolve_word` itself is unchanged).

- [ ] **Step 7: Commit and push**

```bash
git add src/stressmark/engine.py src/stressmark/render.py tests/
git commit -m "$(cat <<'EOF'
Add resolve_word_by_pos and render_word for single-word integration

New entry points for callers (like revdict) that already know a word's
correct part of speech and just want its stress-highlighted rendering,
without going through the full sentence-level analyze() pipeline.
EOF
)"
git push
```

---

### Task 3: Add `revdict.models.stress` — the optional-plugin wrapper

**Repo:** `/home/shichika/redacted/rev-dictionary`

**Files:**
- Create: `src/revdict/models/stress.py`
- Test: `tests/models/test_stress.py`

**Interfaces:**
- Consumes: nothing from earlier revdict tasks. At runtime, optionally, `stressmark.engine.resolve_word_by_pos` and `stressmark.render.render_word` (Task 2, a *different* repo/package — imported by name only, never declared as a revdict dependency).
- Produces: `revdict.models.stress.is_available() -> bool`, `revdict.models.stress.mark(word: str, pos: str) -> str | None`. Task 4 consumes both.

`mark()` returns a captured ANSI-coded **string**, not a Rich `Text` object — this keeps it JSON-safe for revdict's daemon socket protocol (`daemon.py`'s `_handle_request` does `json.dumps(result)` on the whole search result; a `Text` object isn't JSON-serializable). Callers that want a `Text` object back reconstruct one with `rich.text.Text.from_ansi(result)` — verified empirically before writing this plan: capturing via `Console(file=StringIO(), force_terminal=True, color_system="truecolor")` and reconstructing via `Text.from_ansi()` round-trips both the plain text and the exact style spans correctly.

- [ ] **Step 1: Write the failing tests**

```python
# tests/models/test_stress.py
from revdict.models import stress


def test_is_available_true_when_engine_and_render_modules_present(monkeypatch):
    monkeypatch.setattr(stress, "_engine", object())
    monkeypatch.setattr(stress, "_render", object())

    assert stress.is_available() is True


def test_is_available_false_when_modules_absent(monkeypatch):
    monkeypatch.setattr(stress, "_engine", None)
    monkeypatch.setattr(stress, "_render", None)

    assert stress.is_available() is False


def test_mark_returns_none_when_not_available(monkeypatch):
    monkeypatch.setattr(stress, "_engine", None)
    monkeypatch.setattr(stress, "_render", None)

    assert stress.mark("happy", "adjective") is None


def test_mark_calls_engine_and_render_and_returns_a_captured_ansi_string(monkeypatch):
    calls = {}

    class FakeEngine:
        def resolve_word_by_pos(self, word, pos):
            calls["word"] = word
            calls["pos"] = pos
            return "fake-word-result"

    class FakeRender:
        def render_word(self, result):
            calls["rendered_from"] = result
            from rich.text import Text

            return Text("HAPpy", style="bold yellow")

    monkeypatch.setattr(stress, "_engine", FakeEngine())
    monkeypatch.setattr(stress, "_render", FakeRender())

    result = stress.mark("happy", "adjective")

    assert calls == {"word": "happy", "pos": "adjective", "rendered_from": "fake-word-result"}
    assert isinstance(result, str)
    assert "HAPpy" in result  # the captured ANSI string contains the plain text


def test_mark_returns_none_when_engine_raises(monkeypatch):
    class FailingEngine:
        def resolve_word_by_pos(self, word, pos):
            raise ValueError("boom")

    monkeypatch.setattr(stress, "_engine", FailingEngine())
    monkeypatch.setattr(stress, "_render", object())

    assert stress.mark("happy", "adjective") is None


def test_mark_result_round_trips_through_text_from_ansi(monkeypatch):
    """The whole point of returning an ANSI string instead of a Text object:
    confirm a caller can reconstruct an equivalent Text object from it."""
    from rich.text import Text

    class FakeEngine:
        def resolve_word_by_pos(self, word, pos):
            return "fake-word-result"

    class FakeRender:
        def render_word(self, result):
            text = Text("HAP", style="bold yellow")
            text.append("py", style="grey62")
            return text

    monkeypatch.setattr(stress, "_engine", FakeEngine())
    monkeypatch.setattr(stress, "_render", FakeRender())

    result = stress.mark("happy", "adjective")
    reconstructed = Text.from_ansi(result)

    assert reconstructed.plain == "HAPpy"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/models/test_stress.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'revdict.models.stress'`

- [ ] **Step 3: Write the implementation**

```python
# src/revdict/models/stress.py
try:
    import stressmark.engine as _engine
    import stressmark.render as _render
except ImportError:
    _engine = None
    _render = None


def is_available() -> bool:
    return _engine is not None and _render is not None


def mark(word: str, pos: str) -> str | None:
    """Returns a captured ANSI-coded string of the word's stress-highlighted
    syllable breakdown, or None if stressmark isn't installed or fails for
    this specific word (never raises). Returns a plain string rather than a
    Rich Text object so this stays JSON-safe for the daemon's socket
    protocol -- reconstruct a Text object with
    `rich.text.Text.from_ansi(result)` if you need one."""
    if not is_available():
        return None
    try:
        from io import StringIO

        from rich.console import Console

        result = _engine.resolve_word_by_pos(word, pos)
        text = _render.render_word(result)
        buffer = StringIO()
        console = Console(file=buffer, force_terminal=True, width=200, color_system="truecolor")
        console.print(text, end="")
        return buffer.getvalue()
    except Exception:
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/models/test_stress.py -v`
Expected: PASS (all 6 tests) — note these tests never import real `stressmark`; they monkeypatch `stress._engine`/`stress._render` directly, so they pass identically whether or not `emphasis` happens to be installed on the machine running them.

- [ ] **Step 5: Confirm the module degrades correctly with the real (likely absent) import**

```bash
.venv/bin/python -c "from revdict.models import stress; print('available:', stress.is_available()); print('mark result:', stress.mark('happy', 'adjective'))"
```

Expected: since `emphasis` is not installed in revdict's own venv yet at this point in the plan, this should print `available: False` and `mark result: None`, with no traceback — confirming the real (not mocked) ImportError path works.

- [ ] **Step 6: Run the full test suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: all tests pass (existing suite + the 6 new ones)

- [ ] **Step 7: Commit and push**

```bash
git add src/revdict/models/stress.py tests/models/test_stress.py
git commit -m "$(cat <<'EOF'
Add optional stressmark integration wrapper (revdict.models.stress)

Fully optional runtime plugin: stressmark is never declared as a
revdict dependency, so this degrades to no-op everywhere if it isn't
installed. Returns a captured ANSI string (not a Rich Text object) so
the result stays JSON-safe for the daemon's socket protocol.
EOF
)"
git push
```

---

### Task 4: Wire `stress.mark()` into `search.py`

**Repo:** `/home/shichika/redacted/rev-dictionary`

**Files:**
- Modify: `src/revdict/search.py`
- Test: `tests/test_search.py`

**Interfaces:**
- Consumes: `revdict.models.stress.mark(word, pos) -> str | None` (Task 3).
- Produces: both `search()`'s candidate dicts and `tag_exact_match_senses()`'s per-sense dicts gain one more key, `"stress": str | None`. Task 5 (`picker.py`/`cli.py`) consumes this key.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_search.py -- add this test (keep all existing tests in the file unchanged)
def test_search_candidates_and_exact_match_senses_include_a_stress_key(monkeypatch):
    """search() must always include a "stress" key on every candidate and
    every exact-match sense -- None when stressmark isn't installed/fails,
    a string when it succeeds -- so callers never need a .get() with a
    default; the key is always present."""
    import revdict.search as search_mod

    monkeypatch.setattr(search_mod.stress, "mark", lambda word, pos: f"STRESS[{word}/{pos}]")

    exact_match_raw = {
        "headword": "happy",
        "senses": [
            {
                "pos": "adjective",
                "definition": "feeling pleasure",
                "examples": [],
                "source": "wordnet",
                "sentiwordnet": None,
                "emolex": None,
                "synonyms": None,
            }
        ],
    }

    # classifier_factory=None (not a lambda) -- the fixture's record has no
    # emolex/sentiwordnet data, so tag_emotion needs to know no classifier
    # is available at all; passing a callable here (even one returning
    # None) would make tag_emotion call it and then crash calling
    # .classify() on the None it got back.
    tagged = search_mod.tag_exact_match_senses(exact_match_raw, classifier_factory=None)

    assert tagged["senses"][0]["stress"] == "STRESS[happy/adjective]"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_search.py::test_search_candidates_and_exact_match_senses_include_a_stress_key -v`
Expected: FAIL with `AttributeError: module 'revdict.search' has no attribute 'stress'` (the test references `search_mod.stress` to monkeypatch it, which doesn't exist until Step 3 adds the import)

- [ ] **Step 3: Wire `stress.mark()` into `search.py`**

Add the import near the top of `src/revdict/search.py` (alongside the existing `from revdict.models.emotion import ...` line):

```python
from revdict.models import stress
```

In `tag_exact_match_senses`, add `"stress"` to the per-sense dict being built (find the `tagged_senses.append({...})` block and add one line inside it):

```python
        tagged_senses.append(
            {
                "pos": sense["pos"],
                "definition": sense["definition"],
                "examples": sense["examples"],
                "source": sense["source"],
                "synonyms": sense.get("synonyms"),
                "label": emotion["label"],
                "polarity": emotion["polarity"],
                "stress": stress.mark(exact_match_raw["headword"], sense["pos"]),
            }
        )
```

In `search()`, add `"stress"` to the candidate dict being built (find the `candidates.append({...})` block and add one line inside it):

```python
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
            }
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_search.py::test_search_candidates_and_exact_match_senses_include_a_stress_key -v`
Expected: PASS

- [ ] **Step 5: Run the full test suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: all tests pass. Since `stress.mark()` returns `None` when `stressmark` isn't installed (still the case at this point in the plan), every OTHER existing test in `test_search.py` that builds candidates/exact-match senses will now see `"stress": None` in the resulting dicts too — this shouldn't break any existing assertion, since none of them assert an exact/exhaustive dict equality that would be broken by one more key (they check specific keys/values, not full dict equality) — but if you find one that does, add `"stress": None` to its expected dict rather than changing the production code.

- [ ] **Step 6: Commit and push**

```bash
git add src/revdict/search.py tests/test_search.py
git commit -m "$(cat <<'EOF'
Add stress-marking to search candidates and exact-match senses

Every candidate and exact-match sense now carries a "stress" key (an
optional ANSI-coded string from the stressmark integration, None when
unavailable) -- wired the same way emotion tagging already is.
EOF
)"
git push
```

---

### Task 5: Display stress-marking in the fzf preview and the static table

**Repo:** `/home/shichika/redacted/rev-dictionary`

**Files:**
- Modify: `src/revdict/picker.py`
- Modify: `src/revdict/cli.py`
- Test: `tests/test_picker.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: the `"stress"` key on candidate/sense dicts (Task 4) — accessed defensively via `.get("stress")` throughout, since callers/fixtures that predate this key (or a real query where `stress.mark()` returned `None`) won't have a truthy value there.
- Produces: no new public functions — this task only changes rendering inside existing functions.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_picker.py` (the existing `_EXACT_MATCH_FIXTURE`/`_CANDIDATE_FIXTURE` at the top of the file are missing a `"stress"` key entirely, same as a real result would be when `stress.mark()` returns `None` — so the "no stress info" behavior is exercised by the EXISTING tests running unmodified; add these two new tests for the "stress info present" case):

```python
def test_render_exact_preview_shows_stress_info_when_present():
    fixture = {
        "headword": "happy",
        "senses": [
            {
                "pos": "adjective",
                "definition": "feeling great pleasure",
                "examples": [],
                "source": "wordnet",
                "synonyms": None,
                "label": "joy",
                "polarity": "positive",
                "stress": "HAPpy",
            }
        ],
    }

    preview = _render_exact_preview(fixture)

    assert "Stress: HAPpy" in preview


def test_render_candidate_preview_omits_stress_line_when_absent():
    from revdict.picker import _render_candidate_preview

    candidate = dict(_CANDIDATE_FIXTURE[0])
    candidate["stress"] = None

    preview = _render_candidate_preview(candidate)

    assert "Stress:" not in preview
```

Add to `tests/test_cli.py` (extends the existing exact-match fixture test):

```python
def test_run_query_prints_stress_column_when_present(monkeypatch, capsys):
    fake_result = {
        "exact_match": {
            "headword": "happy",
            "senses": [
                {
                    "pos": "adjective",
                    "definition": "feeling great pleasure",
                    "examples": [],
                    "source": "wordnet",
                    "synonyms": None,
                    "label": "joy",
                    "polarity": "positive",
                    "stress": "HAPpy",
                }
            ],
        },
        "candidates": [],
    }
    monkeypatch.setattr(cli, "_get_search_result", lambda query, top_n: fake_result)

    code = cli._run_query("happy", top_n=10, interactive=False)

    captured = capsys.readouterr()
    assert code == 0
    assert "HAPpy" in captured.out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_picker.py::test_render_exact_preview_shows_stress_info_when_present tests/test_picker.py::test_render_candidate_preview_omits_stress_line_when_absent tests/test_cli.py::test_run_query_prints_stress_column_when_present -v`
Expected: FAIL — the first two `assert` on text that isn't produced yet; the third fails because `Text.from_ansi` hasn't been applied yet so the plain string "HAPpy" isn't in the rendered table (it's fine if this one's failure mode is a plain `AssertionError` rather than an exception, as long as it fails before Step 3's changes).

- [ ] **Step 3: Update `src/revdict/picker.py`'s preview renderers**

```python
def _render_exact_preview(exact_match: dict) -> str:
    lines = [f"Exact match — {exact_match['headword']}", ""]
    for sense in exact_match["senses"]:
        lines.append(f"({sense['pos']}) {sense['definition']}")
        if sense.get("stress"):
            lines.append(f"Stress: {sense['stress']}")
        lines.append(f"Emotion: {sense['label']} · {sense['polarity']}")
        synonyms = sense.get("synonyms")
        if synonyms:
            lines.append(f"Synonyms: {', '.join(synonyms)}")
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
    ]
    if candidate.get("stress"):
        lines.append(f"Stress: {candidate['stress']}")
    lines.append(f"Emotion: {candidate['label']} · {candidate['polarity']}")
    lines.append(f"Match confidence: {candidate['relevance']}%")
    if candidate["examples"]:
        lines.append("")
        for example in candidate["examples"]:
            lines.append(f'"{example}"')
    return "\n".join(lines)
```

(These replace the current bodies of both functions — the only change in each is the new conditional `Stress:` line; everything else is unchanged.)

- [ ] **Step 4: Update `src/revdict/cli.py`'s static table rendering**

Add the import at the top of the file (alongside the existing `from rich.table import Table` line):

```python
from rich.text import Text
```

Replace `_print_static_results` with:

```python
def _print_static_results(result: dict) -> None:
    if result["exact_match"] is not None:
        table = Table(title=f"Exact match — {result['exact_match']['headword']}")
        table.add_column("POS")
        table.add_column("Definition")
        table.add_column("Stress")
        table.add_column("Emotion")
        table.add_column("Synonyms")
        for sense in result["exact_match"]["senses"]:
            synonyms = sense.get("synonyms")
            stress_text = Text.from_ansi(sense["stress"]) if sense.get("stress") else ""
            table.add_row(
                sense["pos"],
                sense["definition"],
                stress_text,
                f"{sense['label']} · {sense['polarity']}",
                ", ".join(synonyms) if synonyms else "",
            )
        console.print(table)

    table = Table(title="Related words you might mean")
    table.add_column("#")
    table.add_column("Word")
    table.add_column("Definition")
    table.add_column("Stress")
    table.add_column("Emotion")
    table.add_column("Relevance")
    for position, candidate in enumerate(result["candidates"], start=1):
        stress_text = Text.from_ansi(candidate["stress"]) if candidate.get("stress") else ""
        table.add_row(
            str(position),
            candidate["headword"],
            candidate["definition"],
            stress_text,
            f"{candidate['label']} · {candidate['polarity']}",
            f"{candidate['relevance']}%",
        )
    console.print(table)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_picker.py tests/test_cli.py -v`
Expected: PASS (all tests in both files)

- [ ] **Step 6: Run the full test suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: all tests pass

- [ ] **Step 7: Commit and push**

```bash
git add src/revdict/picker.py src/revdict/cli.py tests/test_picker.py tests/test_cli.py
git commit -m "$(cat <<'EOF'
Display stress-marking in the fzf preview and the static table

The fzf preview pane and --no-interactive table both gain a Stress
line/column, populated from the "stress" key search() now attaches to
every candidate/sense. Silently omitted wherever it's None (stressmark
not installed, or it failed for that specific word).
EOF
)"
git push
```

---

### Task 6: Manual end-to-end validation

No new unit tests — this validates the real optional-plugin boundary in
both directions: installed (feature works, real colors) and not installed
(zero trace, identical behavior to before this integration).

**Files:** None created.

- [ ] **Step 1: Confirm the boundary with stressmark NOT installed (current default state)**

```bash
cd /home/shichika/redacted/rev-dictionary
.venv/bin/python -c "from revdict.models import stress; print(stress.is_available())"
```

Expected: `False`

```bash
.venv/bin/revdict daemon stop 2>&1 || true
.venv/bin/revdict "happy" --no-interactive -n 3
```

Expected: identical output shape to before this integration, except the new "Stress" column is present but empty for every row (no crash, no visible difference otherwise).

- [ ] **Step 2: Install stressmark into revdict's venv and confirm it's picked up**

```bash
.venv/bin/uv pip install -e /home/shichika/redacted/emphasis
.venv/bin/python -c "from revdict.models import stress; print(stress.is_available())"
```

Expected: `True`

- [ ] **Step 3: Confirm real stress-marked output end-to-end (static table)**

```bash
.venv/bin/revdict daemon stop 2>&1 || true
.venv/bin/revdict "happy" --no-interactive -n 5
```

Expected: the "Stress" column now shows real syllable-highlighted text (e.g. a bold-yellow-highlighted "HAP" followed by "py") for both the exact-match table and the candidates table.

- [ ] **Step 4: Confirm real stress-marked output in the fzf preview pane**

```bash
.venv/bin/revdict "happy"
```

Expected (if a real TTY is available to actually test this): fzf's preview pane shows a "Stress:" line with colored syllable highlighting for the highlighted candidate. If no real TTY is available in this environment, confirm at minimum that the command doesn't crash and falls back to the static table exactly as validated in Step 3.

- [ ] **Step 5: Confirm a specific heteronym resolves correctly through the real integration**

```bash
.venv/bin/revdict daemon stop 2>&1 || true
.venv/bin/revdict "record" --no-interactive -n 1
```

Expected: the exact-match table's "Stress" column shows DIFFERENT stress patterns for "record"'s noun senses vs. verb senses (RE-cord for noun, re-CORD for verb) — confirming `resolve_word_by_pos` is genuinely using each sense's own POS, not one guess applied to all senses.

- [ ] **Step 6: Uninstall stressmark and confirm revdict returns to identical pre-integration behavior**

```bash
.venv/bin/uv pip uninstall stressmark
.venv/bin/python -c "from revdict.models import stress; print(stress.is_available())"
.venv/bin/revdict daemon stop 2>&1 || true
.venv/bin/revdict "happy" --no-interactive -n 3
```

Expected: `False` printed; the subsequent query works identically to Step 1 (empty Stress column, no error) — proving the optional-plugin boundary is real, not just theoretical.

- [ ] **Step 7: Clean up and commit the validation note**

```bash
.venv/bin/revdict daemon stop 2>&1 || true
git commit --allow-empty -m "Validate stressmark integration end-to-end (installed and uninstalled)"
git push
```

If you want to keep using the feature day-to-day after this validation, re-install it:

```bash
.venv/bin/uv pip install -e /home/shichika/redacted/emphasis
```
