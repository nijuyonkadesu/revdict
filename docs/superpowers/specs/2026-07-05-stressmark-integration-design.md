# Integrate `emphasis`/stressmark into revdict — Design

Date: 2026-07-05
Status: approved

## Problem

A separate local project, `emphasis` (at `../emphasis` relative to this repo,
CLI name `stressmark`), analyzes English text and highlights primary stress,
secondary stress, and vowel reduction per syllable — using CMUdict, a G2P
fallback for unknown words, and POS-based heteronym resolution (record vs.
record). It's a general-purpose tool for whole articles/transcripts, built
and tested independently of revdict.

The user wants revdict's word/candidate display to also show this
stress-marked breakdown — surfacing the phonetic-emphasis information that
was explicitly out of scope for revdict's original spec, by reusing the
already-built `emphasis` project instead of re-implementing it.

## Design

### Packaging `emphasis` as an installable library

`emphasis` gets restructured as a proper installable Python package:
- `pyproject.toml` (uv-managed, mirroring revdict's own setup) with a
  `stressmark = "stressmark.cli:main"` console-script entry point — this is
  what gives it a `~/.local/bin` binding (via `uv sync` + the same
  symlink-into-`~/.local/bin` pattern already used for `revdict`), so
  `stressmark transcript.txt` keeps working exactly as documented in its
  README, standalone, unaffected by this integration.
- Its existing modules (`stressmark.py` → `stressmark/cli.py`,
  `stressmark_engine.py` → `stressmark/engine.py`,
  `stressmark_render.py` → `stressmark/render.py`) move into a
  `src/stressmark/` package layout, matching revdict's own convention. No
  behavior change to the existing CLI (`--format html/json`, `--explain`,
  `--flag-heteronyms`, `--nuclear-only` all keep working as before).

### New integration entry points in `emphasis`

- `stressmark.engine.resolve_word_by_pos(word: str, pos: str) -> WordResult`
  — translates revdict's POS vocabulary (`"noun"/"verb"/"adjective"/"adverb"`)
  into the Penn Treebank tags the existing `resolve_word(raw, tag,
  sent_tags_after)` already expects, and calls it directly with
  `sent_tags_after=[]`. This **skips `emphasis`'s own POS-tagging and
  heteronym-guessing** for this path — revdict already knows the
  sense-specific part of speech from WordNet, so re-deriving it from an
  isolated single word (with no sentence context) would be both slower
  (needs the POS-tagger model loaded) and less accurate (context-free
  guessing on exactly the kind of ambiguous words — record, object — this
  matters most for).
- `stressmark.render.render_word(result: WordResult) -> rich.text.Text` —
  renders one word's stress-highlighting as a Rich `Text` object (extracted
  from the existing per-token rendering logic in `render_terminal`, scoped
  down to a single word). Reusable both in-process (revdict's static table
  can embed the `Text` object directly, since both projects use Rich) and
  captured to an ANSI string (revdict's fzf preview file, via a
  `Console(file=StringIO(), force_terminal=True)`).

### Integration is a fully optional runtime plugin in revdict — not a dependency

`emphasis`/`stressmark` is **never declared in revdict's `pyproject.toml` or
`uv.lock`**, not even as an optional extra — someone cloning just `revdict`
sees no trace of it, and `uv sync` behaves identically with or without
`emphasis` existing anywhere on the machine.

New module `src/revdict/models/stress.py` in revdict:
```python
try:
    import stressmark.engine as _engine
    import stressmark.render as _render
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


def mark(word: str, pos: str):
    """Returns a rendered stress result, or None if stressmark isn't
    installed or fails for this specific word (never raises)."""
    if not AVAILABLE:
        return None
    try:
        result = _engine.resolve_word_by_pos(word, pos)
        return _render.render_word(result)
    except Exception:
        return None
```
"Importable = configured": if a user wants this feature, they manually
install `emphasis` into revdict's own venv (e.g.
`.venv/bin/uv pip install -e /path/to/emphasis`) — that single act both
installs and enables it. No separate config flag. Everyone else's revdict
install, tests, and CI are completely unaffected.

### Data flow

- `search.py`'s per-candidate tagging loop (where `tag_emotion()` already
  runs per candidate) gains one more optional step: `stress.mark(headword,
  pos)`. `None` (not available, or this word failed) means "no stress info
  for this candidate" — never blocks or errors the query.
- Exact-match senses get the same treatment per-sense (a word can have
  different POS across senses, e.g. "record" noun vs. verb — each sense's
  own POS is passed through, giving correct heteronym resolution for free).
- `picker.py`'s preview-file renderer appends the captured ANSI text block
  when stress info exists for that candidate/sense.
- `cli.py`'s static (`--no-interactive`) tables gain one more column, fed
  the `Text` object directly — no ANSI round-trip needed there.
- The compact one-line-per-candidate fzf list row is **not** changed (per
  the earlier scope decision) — stress info only shows in the preview pane
  and the static table.

### Error handling

- `emphasis` not installed → feature fully absent everywhere, no warning,
  no error.
- `emphasis` installed but a specific word fails inside it (tokenization
  edge case, G2P failure, etc.) → that one candidate/sense just shows no
  stress info; other candidates and the query overall are unaffected.
- `emphasis`'s own first-run NLTK data download (CMUdict) happens lazily on
  first real use inside revdict's warm daemon, the same way revdict's own
  WordNet/SentiWordNet downloads already work. The POS-tagger model is
  **not** needed for revdict's integration path (since `resolve_word_by_pos`
  bypasses tagging), so it's one less download than `emphasis`'s own
  full-`analyze()` path requires.

### Testing

- In `emphasis`: unit tests for `resolve_word_by_pos`'s POS-vocabulary
  translation (revdict vocabulary → Penn Treebank tags) and for
  `render_word`, following the existing test style in `test_engine.py`.
- In `revdict`: `stress.py`'s availability-check and result-adapting logic
  gets unit tests using a fake/injected stressmark-like module (monkeypatched
  in place of a real `stressmark` import), so revdict's test suite never
  depends on `emphasis` actually being installed — consistent with how the
  rest of revdict's test suite avoids real model/dependency calls.
- Manual validation (not automated): with `emphasis` actually installed into
  revdict's venv, run a real query and confirm stress-marked output appears
  correctly in both the fzf preview and the static table; then uninstall it
  and confirm revdict's queries still work identically with no trace of the
  feature (proving the optional-plugin boundary is real, not just
  theoretical).

## Out of scope

- Any change to `emphasis`'s own whole-article/transcript CLI behavior,
  formats, or accuracy characteristics — this integration only adds new,
  additive entry points.
- Stress-marking full descriptive-phrase queries (only candidate/exact-match
  headwords get marked, per the earlier scope decision).
- An explicit enable/disable config flag beyond "is it installed" — a
  deliberate simplicity choice; revisit only if the user later wants to have
  it installed but temporarily disabled.
