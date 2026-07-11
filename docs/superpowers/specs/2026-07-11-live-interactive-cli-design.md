# Live Interactive CLI — Design

Date: 2026-07-11
Status: approved

## Intent

From `TODO.md`: running `revdict` with no arguments should open a
persistent, `psql`-like interactive session — live word suggestions
updating as you type a description, not a one-shot "type a query, get a
static result, process exits" flow. Exiting the session should return to
the calling shell directly; it must never require re-invoking `revdict` to
search again. This spec covers `TODO.md` items 1–2 (the live session and
its responsive layout). Item 4 (a neovim/Telescope plugin) is an explicitly
separate, later sub-project per the user's own sequencing ("the last task
once revdict is made close to perfect") and is out of scope here.

## Prerequisite finding: query latency

Before this feature was feasible at all, per-query latency had to be
measured. It was ~1.0–1.2s per search — far too slow for a live-typing
feel. Root cause: `cosine_top_k` was recomputing
`np.linalg.norm(matrix, axis=1)` over the full ~800K-row embedding matrix
on every single call, instead of once. Fixed (committed `80b3675`,
already shipped independently of this feature): the norms are now
precomputed once in `_load_state()` and passed through. Real, measured
result: per-query latency dropped to ~150–250ms. This is fast enough for
**debounced** live search — not literal zero-delay per-keystroke
re-search, which no reasonable implementation of real semantic search
(embedding + cross-encoder reranking) can deliver. The design below is
built around that honest constraint.

## Architecture: one persistent fzf session, not a custom REPL

Bare `revdict` (no query argument) launches a single, long-lived `fzf`
process for the entire session. You never leave it except by quitting
outright — there is no "select an item, process exits, re-invoke revdict"
cycle. `revdict "some query"` (with an explicit argument) is completely
unchanged: today's one-shot behavior, same code path (`run_picker()` in
`picker.py`), untouched.

fzf was chosen over building a custom REPL (e.g. with `prompt_toolkit`)
because it already natively provides every mechanism this feature needs:
live reload-on-type, a history file with rebindable navigation keys, and
size-conditional layout switching. Reimplementing any of that from scratch
would be strictly worse for no benefit, and this project already depends
on fzf for its existing one-shot picker.

### Live reload as you type

```
--disabled --bind 'change:reload:sleep 0.1; revdict --query-only {q}'
```

`--disabled` turns off fzf's own local fuzzy-filter (which would otherwise
filter a static list — the opposite of what's wanted: the *query itself*,
not fzf's filter, drives what gets searched). After each keystroke, fzf
waits out the `sleep 0.1`, then — if no newer keystroke has arrived —
re-invokes `revdict --query-only "{q}"`, a new fast entry point (see
below) that hits the same daemon-backed `search()` used everywhere else
and prints the reloaded candidate lines. This is the standard, documented
fzf pattern for live-reload-as-you-type (used for e.g. ripgrep-backed live
grep); the `sleep` is fzf's community-standard debounce mechanism since
fzf has no native debounce timer on the `change` event itself. The `0.1`s
value above is illustrative — the real debounce duration is an
implementation-time empirical choice (balancing "feels laggy" against
"fires a search on every single keystroke"), same as the layout threshold
below.

### Key rebindings

fzf's defaults don't match the desired session semantics, so several keys
are rebound:

| Key | Default action | Rebound to | Why |
|---|---|---|---|
| Esc | `abort` (quits) | `clear-query` | Esc should clear what you've typed and let you keep searching, not end the session. |
| Ctrl-C, Ctrl-D | `abort` / `delete-char/eof` | `abort` (kept/confirmed) | These are the actual "quit revdict" keys. |
| Enter | `accept` (selects highlighted item, exits) | `execute-silent(echo {q} >> HISTORY_FILE)+clear-query` | Enter finalizes the *typed query* into history and clears the box for the next search — it does not "select" a candidate and end the session. There is no separate select-and-exit action in this design: the preview pane already shows full detail (definition, synonyms, emotion, stress) for whatever candidate is highlighted, live, as you arrow through the list — no extra keypress needed to see it. |
| Up / Down | `up`/`down` (move list cursor) | `transform-query:tail -n 1 HISTORY_FILE` / `clear-query` | Up recalls the most recently committed query; Down clears the box. |
| Ctrl-P / Ctrl-N | (not bound to list movement by default in this scheme) | `up` / `down` | List-cursor movement moves here since plain arrows are now history. |

`HISTORY_FILE` is a new file under `CACHE_DIR` (e.g.
`~/.cache/rev_dictionary/query_history`), self-managed entirely by the
`enter` and `up` bindings above — fzf's own `--history=` flag is
deliberately NOT used. That flag only loads history at process start and
writes it back at exit, so it cannot recall a query committed by `enter`
earlier in the *same* running session (confirmed empirically against the
real fzf binary — see the implementation plan's fix note for detail).
`transform-query` reads the file live and, confirmed empirically, also
re-triggers the `change:reload` binding, so the result list refreshes to
match the recalled query automatically. This does trade away multi-step
history *cycling* (repeated presses of Up always show the same
most-recently-committed query, not progressively older ones) — a
deliberate, small scope reduction to avoid a stateful position-tracking
mechanism for a "maybe" feature the user themselves hedged on.

### The `--query-only` entry point

A new, fast CLI path that the `change:reload` binding shells out to on
every debounced keystroke. It must skip all the normal CLI ceremony
(argument parsing beyond the query text, table rendering, prompts) and do
only: hit the daemon's existing `search()`, format the results as the same
tab-delimited lines `format_candidate_line()` already produces for
one-shot mode, and print them to stdout for fzf to consume as its new
list. Preview text for the live list works the same way one-shot mode's
preview already does (per-candidate `.txt` files fzf's `--preview` reads
via `cat`), just regenerated on each reload instead of once up front —
exact tempdir/staleness handling (avoiding a preview read racing a
still-being-written reload) is an implementation-plan-level detail, not
resolved further here.

### Responsive layout

Solved natively by fzf's `--preview-window` size-threshold syntax — no
custom terminal-resize handling needed:

```
--preview-window 'right,50%,<100(up,50%)'
```

Side-by-side by default; automatically switches to stacked (preview above
the list) when the terminal drops below 100 columns. The exact threshold
(100 here is illustrative) needs empirical tuning against real phone- and
laptop-terminal widths during implementation.

## Error handling: fzf missing

One-shot mode already degrades gracefully to a static Rich table when fzf
isn't installed (`run_picker()` returns `None`). Live mode has no
meaningful equivalent — the entire feature *is* fzf's reload/history/layout
machinery. Rather than build a second, parallel fallback REPL loop for
what's already a fast-shrinking edge case (fzf is already load-bearing for
the existing one-shot experience), bare `revdict` prints a clear message
when fzf is missing — "live mode requires fzf; install it, or use
`revdict \"your query\"` for one-shot search" — and exits. No new fallback
code path is built.

## Testing

Same shape as the existing `picker.py` test coverage: unit-test the pieces
that don't require a live terminal — the `--query-only` output formatting,
the fzf bind-string construction, history-file read/write logic. The
actual live *feel* (debounce timing, key rebindings working correctly in
a real terminal) is manually verified, consistent with how `run_picker()`
is verified today — no existing test drives a real interactive fzf TTY
session, and this feature doesn't change that pattern.

## Out of scope

- The neovim/Telescope plugin (`TODO.md` item 4) — separate, later spec.
- Building a non-fzf fallback interactive loop.
- Tuning the exact debounce duration and column-width threshold — real
  values are an implementation-time empirical decision, not a design-time
  one.
