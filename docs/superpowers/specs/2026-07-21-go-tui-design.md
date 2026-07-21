# Go TUI (Phase 6) — Design

**Status:** Approved by user 2026-07-21, ready for implementation planning.

## Goal

Phase 6 of the OneLook-feature-parity roadmap (`docs/superpowers/plans/2026-07-19-onelook-feature-parity-roadmap.md`): a standalone, installable Go terminal UI that exposes every feature built in Phases 1-5 (query syntax, sort modes, category filter, phonetic filters) through one clean, low-overhead, keyboard-driven interface — a live multi-facet filter/sort/category experience that fzf's single-prompt model can't support, per the roadmap's own architecture decision.

## Non-goals

- Not a replacement for the existing default `revdict "query"` fzf experience — this is a separate, opt-in tool.
- Not a rewrite or restructuring of any Phase 1-5 backend logic — the TUI is a pure consumer of `revdict`'s existing (plus one new) CLI-level JSON interface.
- Not extending or changing `revdict.nvim`'s existing `--jsonl-query` contract in any way.

## Architecture

**Two independent halves of one repository**, per the roadmap's already-locked-in decision:

- `revdict` (Python, unchanged except one new CLI flag) stays a pure query/data layer.
- `tui/` (new, Go) is a standalone terminal application that shells out to the `revdict` binary and parses its JSON output — it never imports or links against the Python code.

### The Python-side prerequisite: a new `--tui-query` flag

`revdict --jsonl-query "$prompt"` (used today by `revdict.nvim` and the live fzf session) accepts only a bare query string — no `--sort`/`--category`/phonetic-filter parameters, even though the underlying `_get_search_result()` helper (`src/revdict/cli.py`) already accepts all of them. Extending `--jsonl-query`'s exact contract is out because `revdict.nvim` is a separate repo depending on that exact invocation today.

Instead, add a new flag, `--tui-query`, taking a single JSON-encoded argument mirroring the daemon's own wire-protocol field names:

```
revdict --tui-query '{"query": "toilet", "top_n": 30, "sort": "most_formal", "category": null, "syllables": null, "primary_vowel": null, "rhymes_with": null, "sounds_like": null, "meter": null}'
```

Implementation: parse the JSON, call `_get_search_result()` with the same kwargs `--jsonl-query`'s handler already uses, emit the same JSONL row shape `_run_jsonl_query` produces today (`headword, pos, definition, stress, label, polarity, synonyms, examples, relevance, is_exact`) — no new fields needed on the row shape itself, since the TUI's filter/sort state lives client-side, not per-row.

### The Go side: `tui/`

- `tui/go.mod` — module path `github.com/nijuyonkadesu/revdict/tui`, so `go install github.com/nijuyonkadesu/revdict/tui/cmd/revdict-tui@latest` works once pushed, per the user's explicit choice of `go install` as the distribution mechanism (not a bare script, not embedded in the Python package).
- `tui/cmd/revdict-tui/main.go` — entrypoint.
- Internal packages (exact breakdown decided at plan-writing time, not here): a query-client (builds the JSON request, execs `revdict --tui-query`, parses JSONL response, cancellable via `context.Context`), a bubbletea model/update/view set for the main search+results+preview screen, a separate model for the filter/sort/category panel overlay, and a help-overlay view.
- Stack: `bubbletea` (event loop), `lipgloss` (styling/layout), `bubbles` (text input, viewport for wrapping preview, list component for results) — Charm's standard toolkit, matching the roadmap's own tech-stack note.

## Layout & interaction model

**Default view** (search always focused, typing never blocked):

```
┌─────────────────────────────────────────────┐
│ > feeling of intense annoyance_              │
├─────────────────────────────────────────────┤
│ sort:relevance  cat:all                      │  <- 1-line active-filter status
├───────────────────────┬───────────────────────┤
│ ★ annoyance    (n)    │ annoyance              │
│   irritation   (n)    │ ------------------     │
│   vexation     (n)    │ a strong feeling of    │
│   pique        (n)    │ being bothered...       │
│   exasperation (n)    │ ex: "her constant..."  │
└───────────────────────┴───────────────────────┘
 F1 help · Tab filters · Ctrl-R sort · Esc clear · Enter copy
```

- Left pane: results list (headword, pos, short gloss).
- Right pane: preview (full definition, examples, synonyms, stress) for the highlighted result — wraps on resize, visible by default, toggleable (`F2`) when terminal is narrow or the user wants more list space. Mirrors the existing fzf live session's `<50-cols-stacks-below` threshold behavior for narrow terminals.
- One-line status bar between the search box and the results always shows the currently active sort/category/phonetic filters at a glance, even when the panel is closed.

**Filter/sort/category panel** (`Tab` to open, `Esc` to close):

```
┌─────────────────────────────────────────────┐
│  Sort:     ( ) relevance  (*) most_formal    │
│            ( ) oldest     ( ) most_lyrical...│
│  Category: (*) all  ( ) noun  ( ) idiom_slang│
│  Syllables: [   ]   Primary vowel: [    ]    │
│  Rhymes with: [        ]  Meter: [   ]       │
│                                    [Esc: back]│
└─────────────────────────────────────────────┘
```

All 11 sort modes, all 7 categories, and all 5 phonetic filter fields are directly editable here. Closing the panel re-runs the current query with whatever's now active (through the same debounced pipeline as typing — no special-cased "instant" path).

## Keybindings (full list — also shown in the `F1` help overlay)

Every Phase 1 query-syntax trigger character (`*`, `?`, `#`, `@`, `//`, leading `+`/`-`, `:`) must remain typeable as literal query text, so no bare printable character is ever repurposed as a hotkey — every action key is a modifier combo or a special/named key.

**Default view:**
| Key | Action |
|---|---|
| any printable char | types into the query box; live search fires (debounced) |
| `Up`/`Down` | move highlighted result |
| `Enter` | copy highlighted candidate to clipboard (OSC 52 over SSH/tmux, same mechanism as today's fzf session); status flashes "Copied: \<word\>"; this is Enter's **only** meaning anywhere in the TUI |
| `Esc` | clear the query if non-empty; if the query is already empty, a second `Esc` quits |
| `Ctrl-C` | immediate hard quit, no confirmation |
| `Tab` | open the filter/sort/category panel |
| `Ctrl-R` | quick-cycle sort mode in place (relevance → alpha → ... → most_lyrical → relevance), no panel trip needed |
| `F2` | toggle the preview pane |
| `F1` | help overlay: this full keybinding list plus a compact legend of the Phase 1 query-syntax forms (`blue*`, `*bird`, `//anagram`, `bl????rd`, etc.) |

**Inside the filter panel:**
| Key | Action |
|---|---|
| `Tab`/`Shift-Tab` | move between fields |
| `Up`/`Down` | move within the sort/category radio lists |
| any printable char | fills the focused text field (syllables/primary-vowel/rhymes-with/sounds-like/meter), live-validated against the same rules the CLI already enforces |
| `Esc` | close the panel, return to search, re-run with whatever's now active — the only way out; `Enter` has no special meaning here |

## Data flow & debounce

Each keystroke or filter change (after a 0.1s debounce — kept identical to `revdict.nvim`'s existing, empirically-validated value, since the underlying cost driver, a Python interpreter start plus daemon round-trip, is unchanged regardless of which language calls it) triggers a fresh `revdict --tui-query '<json>'` subprocess call. The call is spawned asynchronously via bubbletea's `Cmd` mechanism so the render loop never blocks; an in-flight call is cancelled (`context.Context`) if a newer keystroke/filter-change supersedes it before it returns — the same "cancel the stale one" principle `revdict.nvim`'s Telescope job-finder already relies on, expressed in Go's own idiom instead of Neovim's job-kill-by-PID mechanism.

Filter-panel changes go through the identical debounced pipeline as query text changes — no special-cased instant path, so toggling a category and typing feel consistent.

## Error handling

- `revdict` not found on `PATH` at startup → a clear error before entering the TUI loop at all, not a panic mid-session.
- A `--tui-query` subprocess call fails (non-zero exit — e.g. daemon down and cold-start fails, or an invalid filter combination) → the error surfaces as a single-line message in the results pane (reusing the same `revdict: error: ...` text the CLI already prints for a `ValueError`, e.g. an unresolvable `--rhymes-with` target), not a crash. The TUI stays alive; the next query/filter-change just tries again.
- Terminal resize mid-render → handled via `bubbletea`'s built-in `tea.WindowSizeMsg`; all pane widths and preview-wrap width recompute on every resize event.

## Testing

- Go: unit tests (standard `testing` package) for JSON request-building from filter state, JSONL response parsing, the sort-mode-cycle order, and debounce/cancellation logic — a fake query-executor interface stands in for the real subprocess in unit tests.
- One real end-to-end smoke check (manual/scripted, not part of the blocking test suite — mirrors this project's existing precedent of manual tmux validation for `revdict.nvim`): build the binary, drive it against a real `revdict` install, confirm a real query returns real results and a filter change actually changes them.
- Python: a handful of new `pytest` cases for `--tui-query`'s JSON parsing/dispatch in `test_cli.py`, mirroring the existing `--jsonl-query` test style.

## Open items deliberately left to the implementation plan, not this design

- Exact internal Go package breakdown/file layout within `tui/`.
- Exact `lipgloss` styling choices (colors, borders) — cosmetic, not architectural.
- Whether `go install`-ing `tui/cmd/revdict-tui` requires any Go workspace (`go.work`) consideration given `tui/go.mod` lives inside a repo whose root isn't itself a Go module — a real detail, but a plan-writing-time concern, not a design-time one.
