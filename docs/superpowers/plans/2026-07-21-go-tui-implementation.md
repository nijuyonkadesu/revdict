# Phase 6: Go TUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Go terminal UI (`revdict-tui`, installable via `go install`) exposing every Phase 1-5 feature (query syntax, sort modes, category filter, phonetic filters) through one clean, keyboard-driven interface, per the user-approved design in `docs/superpowers/specs/2026-07-21-go-tui-design.md`.

**Architecture:** Two independent halves in one repo. `revdict` (Python) gains one new CLI flag, `--tui-query`, accepting a single JSON-encoded argument and emitting the same JSONL row shape `--jsonl-query` already produces. `tui/` (new, Go, its own `tui/go.mod`) is a `bubbletea`/`lipgloss`/`bubbles` terminal app that shells out to the `revdict` binary via `--tui-query` and never imports or links against the Python code.

**Tech Stack:** Go 1.24 (confirmed installed: `go version go1.24.2`), `github.com/charmbracelet/bubbletea` (event loop), `github.com/charmbracelet/bubbles` (`textinput`, `list`, `viewport` components), `github.com/charmbracelet/lipgloss` (styling/layout). Every framework call in this plan was verified by compiling and running a real throwaway program against these exact libraries before this plan was written (versions resolved: bubbletea v1.3.10, bubbles v1.0.0, lipgloss v1.1.0) — not transcribed from memory.

## Global Constraints

- **Keybindings (exact, from the design spec) — no bare printable character is ever a hotkey**, since every Phase 1 query-syntax trigger character (`*`, `?`, `#`, `@`, `//`, leading `+`/`-`, `:`) must stay typeable in the search box:
  - Default screen: any printable char types into the query box (debounced live search); `Up`/`Down` move the highlighted result; `Enter` copies the highlighted candidate to clipboard and is its **only** meaning anywhere in the app; `Esc` clears the query if non-empty, and quits if the query is already empty; `Ctrl-C` quits immediately; `Tab` opens the filter panel; `Ctrl-R` quick-cycles the sort mode in place; `F2` toggles the preview pane; `F1` opens the help overlay.
  - Filter panel screen: `Tab`/`Shift-Tab` move between the 7 fields (sort radio, category radio, syllables, primary vowel, rhymes-with, sounds-like, meter); `Up`/`Down` move within the sort/category radio lists; printable chars fill the focused text field; `Esc` closes the panel and re-runs the query — `Enter` has no special meaning in the panel.
- **`--tui-query` JSON request shape** (Go → Python), field names matching the daemon's existing wire protocol exactly: `{"query": string, "top_n": int, "sort": string, "category": string, "syllables": int|null, "primary_vowel": string|null, "rhymes_with": string|null, "sounds_like": string|null, "meter": string|null}`. `sort`/`category` are always sent as explicit strings (`"relevance"`/`"all"` are valid, meaningful "no filter" values already handled identically to `None` by `sort.py`/`category.py` — confirmed by reading `apply_sort`'s `if not sort_mode or sort_mode == "relevance"` guard and `matches_category`'s `if not category or category == "all"` guard). The 5 phonetic fields are `null`/absent when unset — `syllables` MUST distinguish "unset" from an explicit `0` (Phase 4/5's own review history recorded a real bug from conflating these), so it needs a nullable type on the Go side, never a bare `int` defaulting to `0`.
- **Response row shape**, identical to what `--jsonl-query` already emits: `{"headword", "pos", "definition", "stress" (nullable), "label", "polarity", "synonyms" (array), "examples" (array), "relevance" (int), "is_exact" (bool)}`.
- **0.1s debounce**, matching `revdict.nvim`'s existing empirically-validated value (`../revdict.nvim/lua/revdict/finder.lua`) — the cost driver (Python interpreter start + daemon round-trip, ~150-250ms) is identical regardless of which language calls it.
- **Preview-pane wrapping**: `bubbles/viewport` does NOT wrap content itself (confirmed by reading its source — it only splits on newlines and tracks a `longestLineWidth` for horizontal scroll). Wrapping must happen by rendering the text through `lipgloss.NewStyle().Width(w).Render(text)` *before* calling `viewport.SetContent(...)` — confirmed working via a real spike program.
- **Go module path:** `github.com/nijuyonkadesu/revdict/tui`, with the installable command at `github.com/nijuyonkadesu/revdict/tui/cmd/revdict-tui`. No `go.work` file — this is the only Go module in the repository, so a workspace file provides no benefit.
- **`go install .../tui/cmd/revdict-tui@latest` only resolves against a git tag of the form `tui/vX.Y.Z`** (subdirectory-prefixed), not a bare `vX.Y.Z` — a nested-module convention. This can't be verified until after the code is pushed and tagged; document it clearly rather than trying to test it mid-plan.
- **Non-goals** (explicitly out of scope): not replacing the default fzf-based `revdict "query"` experience; not changing `--jsonl-query`'s existing contract in any way (a separate repo, `revdict.nvim`, depends on its exact current invocation); not touching `revdict.nvim` itself.
- Every Go task ends with `go build ./...` and `go test ./...` run from inside `tui/`, both passing, as an explicit verification step — this is how framework usage gets confirmed correct, not by re-reading this plan's prose.
- Full Python test suite baseline before this phase: 338 passed + 2 known pre-existing `FORCE_COLOR`-environment-artifact failures in `test_cli.py` (`test_main_error_message_is_not_mangled_by_rich_markup`, `test_main_routes_daemon_status`) — never a regression, confirmed unrelated to any code in this project's history.

---

### Task 1: `--tui-query` CLI flag (Python)

**Files:**
- Modify: `src/revdict/cli.py` (add `_build_result_rows` helper, refactor `_run_jsonl_query` to use it, add `_run_tui_query`, add dispatch branch in `main()`)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: the existing `_get_search_result(query, top_n, sort_mode=None, category=None, syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None) -> dict` (`cli.py:217-244`, unchanged) and `LIVE_SESSION_TOP_N` (`cli.py:23`, unchanged).
- Produces: `revdict --tui-query '<json>'` prints one JSON object per line to stdout, exit code 0 on success, exit code 1 with a `revdict: error: ...` message on invalid JSON or an unresolvable filter combination (both already handled by `main()`'s existing top-level `except ValueError` — `json.JSONDecodeError` IS a `ValueError` subclass, so no new exception handling is needed anywhere in this task).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_cli.py` (near the existing `test_jsonl_query_*` tests, around line 550):

```python
def test_tui_query_prints_one_json_object_per_candidate(monkeypatch, capsys):
    fake_result = {
        "exact_match": None,
        "candidates": [
            {
                "headword": "joyful", "pos": "adjective",
                "definition": "feeling great happiness",
                "examples": [], "label": "joy", "polarity": "positive",
                "relevance": 90,
            }
        ],
    }
    captured_kwargs = {}

    def fake_get_search_result(query, top_n, sort_mode=None, category=None, syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None):
        captured_kwargs.update(
            query=query, top_n=top_n, sort_mode=sort_mode, category=category,
            syllables=syllables, primary_vowel=primary_vowel, rhymes_with=rhymes_with,
            sounds_like=sounds_like, meter=meter,
        )
        return fake_result

    monkeypatch.setattr(cli, "_get_search_result", fake_get_search_result)

    payload = json.dumps({"query": "happy", "sort": "most_formal", "category": "noun"})
    code = cli.main(["--tui-query", payload])

    captured = capsys.readouterr()
    assert code == 0
    lines = captured.out.strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["headword"] == "joyful"
    assert row["is_exact"] is False
    assert captured_kwargs["query"] == "happy"
    assert captured_kwargs["sort_mode"] == "most_formal"
    assert captured_kwargs["category"] == "noun"


def test_tui_query_defaults_top_n_when_omitted(monkeypatch, capsys):
    captured_kwargs = {}

    def fake_get_search_result(query, top_n, **kwargs):
        captured_kwargs["top_n"] = top_n
        return {"exact_match": None, "candidates": []}

    monkeypatch.setattr(cli, "_get_search_result", fake_get_search_result)

    code = cli.main(["--tui-query", json.dumps({"query": "happy"})])

    assert code == 0
    assert captured_kwargs["top_n"] == cli.LIVE_SESSION_TOP_N


def test_tui_query_flattens_exact_match_first_sense_as_first_row(monkeypatch, capsys):
    fake_result = {
        "exact_match": {
            "headword": "happy",
            "senses": [
                {
                    "pos": "adjective", "definition": "feeling or showing pleasure",
                    "stress": "\x1b[1mHAP\x1b[0mpy", "label": "joy", "polarity": "positive",
                    "synonyms": ["glad", "cheerful"], "examples": ["a happy childhood"],
                }
            ],
        },
        "candidates": [],
    }
    monkeypatch.setattr(
        cli, "_get_search_result",
        lambda query, top_n, **kwargs: fake_result,
    )

    code = cli.main(["--tui-query", json.dumps({"query": "happy"})])

    captured = capsys.readouterr()
    lines = captured.out.strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert code == 0
    assert row["is_exact"] is True
    assert row["relevance"] == 100
    assert row["synonyms"] == ["glad", "cheerful"]


def test_tui_query_with_blank_query_prints_nothing(monkeypatch, capsys):
    code = cli.main(["--tui-query", json.dumps({"query": ""})])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out == ""


def test_tui_query_with_empty_payload_prints_nothing(capsys):
    code = cli.main(["--tui-query", ""])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out == ""


def test_tui_query_with_invalid_json_prints_a_clean_error(capsys):
    code = cli.main(["--tui-query", "{not valid json"])
    captured = capsys.readouterr()
    assert code == 1
    assert "revdict: error:" in captured.out


def test_tui_query_propagates_search_value_error_as_a_clean_message(monkeypatch, capsys):
    def fake_get_search_result(query, top_n, **kwargs):
        raise ValueError("Unknown sort mode: 'bogus'")

    monkeypatch.setattr(cli, "_get_search_result", fake_get_search_result)

    code = cli.main(["--tui-query", json.dumps({"query": "happy", "sort": "bogus"})])

    captured = capsys.readouterr()
    assert code == 1
    assert "revdict: error: Unknown sort mode" in captured.out


def test_jsonl_query_and_tui_query_produce_identical_rows_for_the_same_result(monkeypatch, capsys):
    """Both flags must share the exact same row-building logic (DRY) --
    this test locks in that they can never silently drift apart."""
    fake_result = {
        "exact_match": None,
        "candidates": [
            {
                "headword": "joyful", "pos": "adjective",
                "definition": "feeling great happiness",
                "examples": [], "label": "joy", "polarity": "positive",
                "relevance": 90,
            }
        ],
    }
    monkeypatch.setattr(
        cli, "_get_search_result",
        lambda query, top_n, **kwargs: fake_result,
    )

    code_jsonl = cli.main(["--jsonl-query", "happy"])
    jsonl_row = json.loads(capsys.readouterr().out.strip())

    code_tui = cli.main(["--tui-query", json.dumps({"query": "happy"})])
    tui_row = json.loads(capsys.readouterr().out.strip())

    assert code_jsonl == 0
    assert code_tui == 0
    assert jsonl_row == tui_row
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/shichika/redacted/rev-dictionary && source .venv/bin/activate && pytest tests/test_cli.py -k tui_query -v`
Expected: FAIL — `--tui-query` isn't a recognized dispatch yet, so `cli.main(["--tui-query", ...])` falls through to normal argparse and errors on an unrecognized flag.

- [ ] **Step 3: Extract the shared row-building helper and add `_run_tui_query`**

In `src/revdict/cli.py`, replace the existing `_run_jsonl_query` function (currently at line 326) with:

```python
def _build_result_rows(result: dict) -> list[dict]:
    rows = []
    if result["exact_match"] is not None:
        first_sense = result["exact_match"]["senses"][0]
        rows.append(
            {
                "headword": result["exact_match"]["headword"],
                "pos": first_sense["pos"],
                "definition": first_sense["definition"],
                "stress": first_sense.get("stress"),
                "label": first_sense["label"],
                "polarity": first_sense["polarity"],
                "synonyms": first_sense.get("synonyms") or [],
                "examples": first_sense["examples"],
                "relevance": 100,
                "is_exact": True,
            }
        )
    for candidate in result["candidates"]:
        rows.append(
            {
                "headword": candidate["headword"],
                "pos": candidate["pos"],
                "definition": candidate["definition"],
                "stress": candidate.get("stress"),
                "label": candidate["label"],
                "polarity": candidate["polarity"],
                "synonyms": candidate.get("synonyms") or [],
                "examples": candidate["examples"],
                "relevance": candidate["relevance"],
                "is_exact": False,
            }
        )
    return rows


def _run_jsonl_query(query: str) -> int:
    if not query.strip():
        return 0

    result = _get_search_result(query, LIVE_SESSION_TOP_N)
    for row in _build_result_rows(result):
        print(json.dumps(row))
    return 0


def _run_tui_query(payload: str) -> int:
    if not payload.strip():
        return 0

    request = json.loads(payload)
    query = request.get("query", "")
    if not query.strip():
        return 0

    result = _get_search_result(
        query,
        request.get("top_n", LIVE_SESSION_TOP_N),
        sort_mode=request.get("sort"),
        category=request.get("category"),
        syllables=request.get("syllables"),
        primary_vowel=request.get("primary_vowel"),
        rhymes_with=request.get("rhymes_with"),
        sounds_like=request.get("sounds_like"),
        meter=request.get("meter"),
    )
    for row in _build_result_rows(result):
        print(json.dumps(row))
    return 0
```

`json.JSONDecodeError` (raised by `json.loads` on malformed input) is a `ValueError` subclass, and `_get_search_result`'s own `ValueError` (e.g. an unknown sort mode) also propagates naturally — both are caught by `main()`'s existing `except ValueError` handler (`cli.py:498-504`), so `_run_tui_query` needs no exception handling of its own.

- [ ] **Step 4: Add the dispatch branch**

In `src/revdict/cli.py`'s `main()`, add this branch immediately after the existing `if argv and argv[0] == "--jsonl-query":` block (around line 452-454):

```python
        if argv and argv[0] == "--tui-query":
            payload = argv[1] if len(argv) > 1 else ""
            return _run_tui_query(payload)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_cli.py -k tui_query -v`
Expected: all PASS.

- [ ] **Step 6: Run the full test suite**

Run: `pytest -q`
Expected: 345 passed (338 baseline + 7 new tests) + the same 2 known pre-existing failures. No regressions — in particular, the existing `test_jsonl_query_*` tests must still pass unchanged, confirming the `_build_result_rows` extraction didn't alter `_run_jsonl_query`'s observable behavior.

- [ ] **Step 7: Commit**

```bash
git add src/revdict/cli.py tests/test_cli.py
git commit -m "Add --tui-query CLI flag for the Go TUI"
```

---

### Task 2: Go module skeleton

**Files:**
- Create: `tui/go.mod`
- Create: `tui/cmd/revdict-tui/main.go`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: a working, empty-but-real Go module at `tui/`, module path `github.com/nijuyonkadesu/revdict/tui`, that builds and runs — proves the toolchain/module-path mechanics before any real feature code is added on top.

- [ ] **Step 1: Initialize the Go module**

```bash
mkdir -p /home/shichika/redacted/rev-dictionary/tui/cmd/revdict-tui
cd /home/shichika/redacted/rev-dictionary/tui
go mod init github.com/nijuyonkadesu/revdict/tui
```

This creates `tui/go.mod` with whatever `go` directive your installed toolchain writes (confirmed on the reference machine: `go 1.24.2`) — don't hand-edit this file's `go` line, let the real toolchain populate it.

- [ ] **Step 2: Write a minimal real entrypoint**

Create `tui/cmd/revdict-tui/main.go`:

```go
package main

import (
	"fmt"
	"os"
	"os/exec"
)

func main() {
	if _, err := exec.LookPath("revdict"); err != nil {
		fmt.Fprintln(os.Stderr, "revdict-tui: 'revdict' not found on PATH -- install it first, see the repo README")
		os.Exit(1)
	}
	fmt.Println("revdict-tui: skeleton OK, revdict found on PATH")
}
```

This is deliberately minimal — it only proves the module builds, installs, and can find its one real dependency (the `revdict` binary) before any bubbletea code is written. Real UI logic is added in later tasks.

- [ ] **Step 3: Verify it builds and runs**

```bash
cd /home/shichika/redacted/rev-dictionary/tui
go build -o /tmp/revdict-tui-skeleton ./cmd/revdict-tui
/tmp/revdict-tui-skeleton
```

Expected: prints `revdict-tui: skeleton OK, revdict found on PATH` (assuming `revdict` is on `PATH` in this environment, which it is per this project's existing setup) and exits 0.

- [ ] **Step 4: Commit**

```bash
cd /home/shichika/redacted/rev-dictionary
git add tui/go.mod tui/cmd/revdict-tui/main.go
git commit -m "Initialize the tui/ Go module skeleton"
```

---

### Task 3: Query-client package

**Files:**
- Create: `tui/internal/queryclient/queryclient.go`
- Test: `tui/internal/queryclient/queryclient_test.go`

**Interfaces:**
- Consumes: the `--tui-query` flag from Task 1 (only at real-usage time — this task's tests use a fake executor, never a real subprocess).
- Produces: `queryclient.Request` (struct), `queryclient.ResultRow` (struct), `queryclient.Client` with `New() *Client` (real subprocess executor) and `NewWithExecutor(Executor) *Client` (for tests), and `(*Client) Query(ctx context.Context, req Request) ([]ResultRow, error)`. Task 4/5 (the bubbletea model) call `queryclient.New()` once at startup and `.Query(ctx, req)` per debounced search.

- [ ] **Step 1: Write the failing tests**

Create `tui/internal/queryclient/queryclient_test.go`:

```go
package queryclient

import (
	"context"
	"errors"
	"strings"
	"testing"
)

type fakeExecutor struct {
	output  []byte
	err     error
	gotArgs []string
}

func (f *fakeExecutor) Run(ctx context.Context, args ...string) ([]byte, error) {
	f.gotArgs = args
	return f.output, f.err
}

func TestQueryBuildsCorrectJSONRequest(t *testing.T) {
	fake := &fakeExecutor{output: []byte("")}
	c := NewWithExecutor(fake)
	req := Request{Query: "happy", TopN: 30, Sort: "most_formal", Category: "all"}

	_, err := c.Query(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(fake.gotArgs) != 2 || fake.gotArgs[0] != "--tui-query" {
		t.Fatalf("expected [--tui-query, <json>], got %v", fake.gotArgs)
	}
	if !strings.Contains(fake.gotArgs[1], `"query":"happy"`) {
		t.Fatalf("expected query in JSON payload, got %s", fake.gotArgs[1])
	}
	if !strings.Contains(fake.gotArgs[1], `"sort":"most_formal"`) {
		t.Fatalf("expected sort in JSON payload, got %s", fake.gotArgs[1])
	}
}

func TestQueryOmitsUnsetPhoneticFields(t *testing.T) {
	fake := &fakeExecutor{output: []byte("")}
	c := NewWithExecutor(fake)
	req := Request{Query: "happy", TopN: 30, Sort: "relevance", Category: "all"}

	_, err := c.Query(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if strings.Contains(fake.gotArgs[1], "rhymes_with") {
		t.Fatalf("expected rhymes_with omitted when unset, got %s", fake.gotArgs[1])
	}
}

func TestQueryDistinguishesUnsetSyllablesFromExplicitZero(t *testing.T) {
	fake := &fakeExecutor{output: []byte("")}
	c := NewWithExecutor(fake)
	zero := 0
	req := Request{Query: "happy", TopN: 30, Sort: "relevance", Category: "all", Syllables: &zero}

	_, err := c.Query(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(fake.gotArgs[1], `"syllables":0`) {
		t.Fatalf("expected explicit syllables:0 in payload, got %s", fake.gotArgs[1])
	}
}

func TestQueryParsesJSONLResponseRows(t *testing.T) {
	output := `{"headword":"joyful","pos":"adjective","definition":"feeling great happiness","stress":null,"label":"joy","polarity":"positive","synonyms":["glad"],"examples":[],"relevance":90,"is_exact":false}
`
	fake := &fakeExecutor{output: []byte(output)}
	c := NewWithExecutor(fake)

	rows, err := c.Query(context.Background(), Request{Query: "happy", TopN: 30, Sort: "relevance", Category: "all"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(rows) != 1 {
		t.Fatalf("expected 1 row, got %d", len(rows))
	}
	if rows[0].Headword != "joyful" || rows[0].Relevance != 90 {
		t.Fatalf("unexpected row: %+v", rows[0])
	}
	if rows[0].Stress != nil {
		t.Fatalf("expected nil stress, got %v", *rows[0].Stress)
	}
	if len(rows[0].Synonyms) != 1 || rows[0].Synonyms[0] != "glad" {
		t.Fatalf("unexpected synonyms: %v", rows[0].Synonyms)
	}
}

func TestQueryReturnsEmptyRowsForBlankOutput(t *testing.T) {
	fake := &fakeExecutor{output: []byte("")}
	c := NewWithExecutor(fake)

	rows, err := c.Query(context.Background(), Request{Query: "", TopN: 30, Sort: "relevance", Category: "all"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(rows) != 0 {
		t.Fatalf("expected 0 rows, got %d", len(rows))
	}
}

func TestQueryPropagatesExecutorError(t *testing.T) {
	fake := &fakeExecutor{err: errors.New("revdict: error: Unknown sort mode: 'bogus'")}
	c := NewWithExecutor(fake)

	_, err := c.Query(context.Background(), Request{Query: "happy", TopN: 30, Sort: "bogus", Category: "all"})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
}
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /home/shichika/redacted/rev-dictionary/tui
go test ./internal/queryclient/... -v
```
Expected: FAIL to compile — `Request`, `ResultRow`, `NewWithExecutor`, `Client` don't exist yet.

- [ ] **Step 3: Implement the package**

Create `tui/internal/queryclient/queryclient.go`:

```go
package queryclient

import (
	"context"
	"encoding/json"
	"os/exec"
	"strings"
)

// Request mirrors --tui-query's expected JSON shape, which itself mirrors
// the revdict daemon's existing wire-protocol field names exactly.
type Request struct {
	Query        string `json:"query"`
	TopN         int    `json:"top_n"`
	Sort         string `json:"sort"`
	Category     string `json:"category"`
	Syllables    *int   `json:"syllables,omitempty"`
	PrimaryVowel string `json:"primary_vowel,omitempty"`
	RhymesWith   string `json:"rhymes_with,omitempty"`
	SoundsLike   string `json:"sounds_like,omitempty"`
	Meter        string `json:"meter,omitempty"`
}

// ResultRow mirrors one JSONL row --tui-query emits (identical to what
// --jsonl-query already emits).
type ResultRow struct {
	Headword   string   `json:"headword"`
	POS        string   `json:"pos"`
	Definition string   `json:"definition"`
	Stress     *string  `json:"stress"`
	Label      string   `json:"label"`
	Polarity   string   `json:"polarity"`
	Synonyms   []string `json:"synonyms"`
	Examples   []string `json:"examples"`
	Relevance  int      `json:"relevance"`
	IsExact    bool     `json:"is_exact"`
}

// Executor runs `revdict` with the given args and returns its stdout.
// Swapped for a fake in tests so no real subprocess is spawned.
type Executor interface {
	Run(ctx context.Context, args ...string) ([]byte, error)
}

type execExecutor struct{}

func (execExecutor) Run(ctx context.Context, args ...string) ([]byte, error) {
	return exec.CommandContext(ctx, "revdict", args...).Output()
}

type Client struct {
	executor Executor
}

// New returns a Client that shells out to the real `revdict` binary on PATH.
func New() *Client {
	return &Client{executor: execExecutor{}}
}

// NewWithExecutor returns a Client backed by a caller-supplied Executor --
// used by tests to avoid spawning a real subprocess.
func NewWithExecutor(e Executor) *Client {
	return &Client{executor: e}
}

// Query builds the --tui-query JSON payload, runs it, and parses the JSONL
// response into rows. Cancelling ctx (e.g. because a newer keystroke
// superseded this request) aborts the in-flight subprocess.
func (c *Client) Query(ctx context.Context, req Request) ([]ResultRow, error) {
	payload, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	out, err := c.executor.Run(ctx, "--tui-query", string(payload))
	if err != nil {
		return nil, err
	}

	var rows []ResultRow
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if line == "" {
			continue
		}
		var row ResultRow
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			return nil, err
		}
		rows = append(rows, row)
	}
	return rows, nil
}
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
go test ./internal/queryclient/... -v
```
Expected: all PASS.

- [ ] **Step 5: Run the full Go test suite and build**

```bash
go build ./... && go test ./...
```
Expected: builds clean, all tests pass.

- [ ] **Step 6: Commit**

```bash
cd /home/shichika/redacted/rev-dictionary
git add tui/internal/queryclient/queryclient.go tui/internal/queryclient/queryclient_test.go tui/go.mod tui/go.sum
git commit -m "Add the query-client package"
```

(`go.sum` appears here for the first time since this task is the first to actually need the JSON stdlib's transitive nothing — in practice `go.sum` may not change yet since `encoding/json`/`os/exec`/`context`/`strings` are all stdlib with zero third-party dependencies. If `go.sum` doesn't change, don't force-add it.)

---

### Task 4: Main screen (static rendering)

**Files:**
- Create: `tui/internal/ui/model.go`
- Test: `tui/internal/ui/model_test.go`
- Modify: `tui/go.mod`, `tui/go.sum` (adds bubbletea/bubbles/lipgloss dependencies)

**Interfaces:**
- Consumes: `queryclient.ResultRow` (Task 3) as the row shape the results list and preview pane render.
- Produces: `ui.Model` (implements `tea.Model`: `Init() tea.Cmd`, `Update(tea.Msg) (tea.Model, tea.Cmd)`, `View() string`), `ui.NewModel(rows []queryclient.ResultRow) Model` for testing with fixed data. Task 5 wires this up to real, live, debounced queries in place of the fixed `rows` this task uses.

This task renders the default screen (search box, results list, preview pane, status bar) and handles navigation (`Up`/`Down`), `Enter` (copy — the clipboard mechanism itself is stubbed as a no-op function value in this task, wired to the real OSC52/system-clipboard call in Task 7 alongside the help overlay, since clipboard access has no bearing on this task's own rendering/navigation logic), `Esc` (clear/quit), `Ctrl-C` (quit), and `F2` (toggle preview) against a **fixed, hardcoded set of rows** — no live querying yet, that's Task 5.

- [ ] **Step 1: Add the bubbletea/bubbles/lipgloss dependencies**

```bash
cd /home/shichika/redacted/rev-dictionary/tui
go get github.com/charmbracelet/bubbletea@v1.3.10
go get github.com/charmbracelet/bubbles@v1.0.0
go get github.com/charmbracelet/lipgloss@v1.1.0
go mod tidy
```

(These exact versions are what a real spike confirmed compiling and running correctly before this plan was written; `go mod tidy` will also pull in the several transitive dependencies these three packages need — that's expected, not a mistake.)

- [ ] **Step 2: Write the failing tests**

Create `tui/internal/ui/model_test.go`:

```go
package ui

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/nijuyonkadesu/revdict/tui/internal/queryclient"
)

func testRows() []queryclient.ResultRow {
	return []queryclient.ResultRow{
		{Headword: "annoyance", POS: "noun", Definition: "a feeling of being bothered", Relevance: 92},
		{Headword: "irritation", POS: "noun", Definition: "a feeling of anger about something", Relevance: 88},
	}
}

func TestNewModelStartsWithSearchFocused(t *testing.T) {
	m := NewModel(testRows())
	if !m.input.Focused() {
		t.Fatal("expected search input to be focused on start")
	}
}

func TestTypingAppendsToQuery(t *testing.T) {
	m := NewModel(testRows())
	mm, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'h'}})
	m = mm.(Model)
	if m.input.Value() != "h" {
		t.Fatalf("expected query 'h', got %q", m.input.Value())
	}
}

func TestDownMovesSelectionToNextResult(t *testing.T) {
	m := NewModel(testRows())
	m.selected = 0
	mm, _ := m.Update(tea.KeyMsg{Type: tea.KeyDown})
	m = mm.(Model)
	if m.selected != 1 {
		t.Fatalf("expected selected=1, got %d", m.selected)
	}
}

func TestDownAtLastResultStaysPut(t *testing.T) {
	m := NewModel(testRows())
	m.selected = 1
	mm, _ := m.Update(tea.KeyMsg{Type: tea.KeyDown})
	m = mm.(Model)
	if m.selected != 1 {
		t.Fatalf("expected selected to stay at 1, got %d", m.selected)
	}
}

func TestEscClearsNonEmptyQuery(t *testing.T) {
	m := NewModel(testRows())
	m.input.SetValue("something")
	mm, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEsc})
	m = mm.(Model)
	if m.input.Value() != "" {
		t.Fatalf("expected query cleared, got %q", m.input.Value())
	}
	if cmd != nil {
		msg := cmd()
		if _, isQuit := msg.(tea.QuitMsg); isQuit {
			t.Fatal("expected Esc on a non-empty query not to quit")
		}
	}
}

func TestSecondEscOnAlreadyEmptyQueryQuits(t *testing.T) {
	m := NewModel(testRows())
	m.input.SetValue("")
	_, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEsc})
	if cmd == nil {
		t.Fatal("expected a quit command when Esc pressed on an already-empty query")
	}
	if _, isQuit := cmd().(tea.QuitMsg); !isQuit {
		t.Fatal("expected tea.QuitMsg")
	}
}

func TestCtrlCAlwaysQuits(t *testing.T) {
	m := NewModel(testRows())
	m.input.SetValue("something")
	_, cmd := m.Update(tea.KeyMsg{Type: tea.KeyCtrlC})
	if cmd == nil {
		t.Fatal("expected a quit command")
	}
	if _, isQuit := cmd().(tea.QuitMsg); !isQuit {
		t.Fatal("expected tea.QuitMsg")
	}
}

func TestF2TogglesPreviewVisibility(t *testing.T) {
	m := NewModel(testRows())
	initial := m.previewVisible
	mm, _ := m.Update(tea.KeyMsg{Type: tea.KeyF2})
	m = mm.(Model)
	if m.previewVisible == initial {
		t.Fatal("expected previewVisible to flip")
	}
}

func TestViewIncludesHighlightedHeadwordAndWrappedPreview(t *testing.T) {
	m := NewModel(testRows())
	mm, _ := m.Update(tea.WindowSizeMsg{Width: 80, Height: 24})
	m = mm.(Model)
	out := m.View()
	if !strings.Contains(out, "annoyance") {
		t.Fatalf("expected view to contain the first result's headword, got: %s", out)
	}
}
```

- [ ] **Step 3: Run the tests to verify they fail**

```bash
go test ./internal/ui/... -v
```
Expected: FAIL to compile — `Model`, `NewModel`, `m.selected`, `m.previewVisible`, `m.input` don't exist yet.

- [ ] **Step 4: Implement the model**

Create `tui/internal/ui/model.go`:

```go
package ui

import (
	"fmt"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	"github.com/charmbracelet/lipgloss"
	"github.com/nijuyonkadesu/revdict/tui/internal/queryclient"
)

// copyFunc is called with the selected headword on Enter and reports
// whether the copy succeeded. Task 7 wires this to the real clipboard
// mechanism (clipboard.Copy, which can fail e.g. with no clipboard
// utility available); tests and this task's own construction path use a
// no-op default so this file has no clipboard dependency of its own.
type copyFunc func(headword string) error

type Model struct {
	input          textinput.Model
	preview        viewport.Model
	rows           []queryclient.ResultRow
	selected       int
	previewVisible bool
	width, height  int
	statusMessage  string
	onCopy         copyFunc
}

func NewModel(rows []queryclient.ResultRow) Model {
	ti := textinput.New()
	ti.Focus()
	vp := viewport.New(0, 0)
	return Model{
		input:          ti,
		preview:        vp,
		rows:           rows,
		previewVisible: true,
		onCopy:         func(string) error { return nil },
	}
}

func (m Model) Init() tea.Cmd {
	return m.input.Focus()
}

func (m Model) selectedRow() (queryclient.ResultRow, bool) {
	if len(m.rows) == 0 || m.selected < 0 || m.selected >= len(m.rows) {
		return queryclient.ResultRow{}, false
	}
	return m.rows[m.selected], true
}

func (m *Model) refreshPreview() {
	row, ok := m.selectedRow()
	if !ok {
		m.preview.SetContent("")
		return
	}
	previewWidth := m.width / 2
	if previewWidth < 1 {
		previewWidth = 1
	}
	text := fmt.Sprintf("%s\n\n%s", row.Headword, row.Definition)
	wrapped := lipgloss.NewStyle().Width(previewWidth).Render(text)
	m.preview.SetContent(wrapped)
}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
		m.preview.Width = msg.Width / 2
		m.preview.Height = msg.Height - 4
		m.refreshPreview()
		return m, nil

	case tea.KeyMsg:
		switch msg.Type {
		case tea.KeyCtrlC:
			return m, tea.Quit

		case tea.KeyEsc:
			if m.input.Value() == "" {
				return m, tea.Quit
			}
			m.input.SetValue("")
			return m, nil

		case tea.KeyEnter:
			if row, ok := m.selectedRow(); ok {
				if err := m.onCopy(row.Headword); err != nil {
					m.statusMessage = "Copy failed: " + err.Error()
				} else {
					m.statusMessage = "Copied: " + row.Headword
				}
			}
			return m, nil

		case tea.KeyUp:
			if m.selected > 0 {
				m.selected--
				m.refreshPreview()
			}
			return m, nil

		case tea.KeyDown:
			if m.selected < len(m.rows)-1 {
				m.selected++
				m.refreshPreview()
			}
			return m, nil

		case tea.KeyF2:
			m.previewVisible = !m.previewVisible
			return m, nil
		}

		var cmd tea.Cmd
		m.input, cmd = m.input.Update(msg)
		return m, cmd
	}

	return m, nil
}

func (m Model) View() string {
	var b []string
	for i, row := range m.rows {
		marker := "  "
		if i == m.selected {
			marker = "> "
		}
		b = append(b, fmt.Sprintf("%s%s (%s)", marker, row.Headword, row.POS))
	}
	resultsView := lipgloss.JoinVertical(lipgloss.Left, b...)

	var body string
	if m.previewVisible {
		listWidth := m.width / 2
		left := lipgloss.NewStyle().Width(listWidth).Render(resultsView)
		right := lipgloss.NewStyle().Width(m.width - listWidth).Render(m.preview.View())
		body = lipgloss.JoinHorizontal(lipgloss.Top, left, right)
	} else {
		body = resultsView
	}

	status := m.statusMessage
	return lipgloss.JoinVertical(lipgloss.Left, m.input.View(), status, body)
}
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
go test ./internal/ui/... -v
```
Expected: all PASS.

- [ ] **Step 6: Build and run the full Go test suite**

```bash
go build ./... && go test ./...
```
Expected: builds clean, all tests pass.

- [ ] **Step 7: Commit**

```bash
cd /home/shichika/redacted/rev-dictionary
git add tui/internal/ui/model.go tui/internal/ui/model_test.go tui/go.mod tui/go.sum
git commit -m "Add the main screen (static rendering, fixed test data)"
```

---

### Task 5: Debounced live-query wiring

**Files:**
- Modify: `tui/internal/ui/model.go` (add debounce + query dispatch)
- Test: `tui/internal/ui/model_test.go`

**Interfaces:**
- Consumes: `queryclient.Client`/`queryclient.Request`/`queryclient.ResultRow` (Task 3).
- Produces: `NewLiveModel(client *queryclient.Client) Model` — the real constructor `cmd/revdict-tui/main.go` (Task 8) will use; `NewModel` (Task 4, fixed test data) remains for tests that don't need real querying.

Every keystroke that changes the query schedules a debounced re-query (0.1s, per Global Constraints); a newer keystroke arriving before the debounce fires must supersede the older one, and a query already in flight when a newer one starts must be cancelled via `context.Context`.

- [ ] **Step 1: Write the failing tests**

Add `"context"` to `tui/internal/ui/model_test.go`'s existing import block (`queryclient` is already imported there, from Task 4's `testRows()` helper). Then add to the same file:

```go
type fakeExecutor struct {
	calls [][]string
}

func (f *fakeExecutor) Run(ctx context.Context, args ...string) ([]byte, error) {
	f.calls = append(f.calls, args)
	return []byte(`{"headword":"annoyance","pos":"noun","definition":"a feeling","stress":null,"label":"joy","polarity":"positive","synonyms":[],"examples":[],"relevance":92,"is_exact":false}` + "\n"), nil
}

func TestTypingSchedulesADebouncedQuery(t *testing.T) {
	fake := &fakeExecutor{}
	client := queryclient.NewWithExecutor(fake)
	m := NewLiveModel(client)

	mm, cmd := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'h'}})
	m = mm.(Model)
	if cmd == nil {
		t.Fatal("expected a debounce command to be scheduled")
	}

	msg := cmd()
	debounce, ok := msg.(debounceFiredMsg)
	if !ok {
		t.Fatalf("expected debounceFiredMsg, got %T", msg)
	}
	if debounce.query != "h" {
		t.Fatalf("expected debounce for query 'h', got %q", debounce.query)
	}
}

func TestStaleDebounceIsIgnoredIfQueryChangedSince(t *testing.T) {
	fake := &fakeExecutor{}
	client := queryclient.NewWithExecutor(fake)
	m := NewLiveModel(client)
	m.input.SetValue("current")

	mm, cmd := m.Update(debounceFiredMsg{query: "stale"})
	m = mm.(Model)
	if cmd != nil {
		t.Fatal("expected no query dispatched for a stale debounce message")
	}
}

func TestFreshDebounceDispatchesAQuery(t *testing.T) {
	fake := &fakeExecutor{}
	client := queryclient.NewWithExecutor(fake)
	m := NewLiveModel(client)
	m.input.SetValue("annoyance")

	mm, cmd := m.Update(debounceFiredMsg{query: "annoyance"})
	m = mm.(Model)
	if cmd == nil {
		t.Fatal("expected a query command to be dispatched")
	}
	msg := cmd()
	result, ok := msg.(queryResultMsg)
	if !ok {
		t.Fatalf("expected queryResultMsg, got %T", msg)
	}
	if len(result.rows) != 1 || result.rows[0].Headword != "annoyance" {
		t.Fatalf("unexpected rows: %v", result.rows)
	}
}

func TestQueryResultMsgReplacesRows(t *testing.T) {
	fake := &fakeExecutor{}
	client := queryclient.NewWithExecutor(fake)
	m := NewLiveModel(client)

	mm, _ := m.Update(queryResultMsg{rows: []queryclient.ResultRow{{Headword: "new-word"}}})
	m = mm.(Model)
	if len(m.rows) != 1 || m.rows[0].Headword != "new-word" {
		t.Fatalf("expected rows replaced with query result, got %v", m.rows)
	}
}
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
go test ./internal/ui/... -v
```
Expected: FAIL to compile — `NewLiveModel`, `debounceFiredMsg`, `queryResultMsg` don't exist yet.

- [ ] **Step 3: Implement debounce + query dispatch**

In `tui/internal/ui/model.go`, add these imports: `"context"`, `"time"`. Add this new state to the `Model` struct (alongside the existing fields):

```go
	client        *queryclient.Client
	filters       FilterState
	cancelInFlight context.CancelFunc
```

Add `FilterState` (used here as the always-`relevance`/`all`-with-no-phonetics default for this task; Task 6 makes it mutable via the panel):

```go
// FilterState holds the currently-active sort/category/phonetic filters.
// The zero value is NOT valid -- always construct via NewFilterState.
type FilterState struct {
	Sort         string
	Category     string
	Syllables    *int
	PrimaryVowel string
	RhymesWith   string
	SoundsLike   string
	Meter        string
}

func NewFilterState() FilterState {
	return FilterState{Sort: "relevance", Category: "all"}
}

func (f FilterState) toRequest(query string) queryclient.Request {
	return queryclient.Request{
		Query: query, TopN: 30, Sort: f.Sort, Category: f.Category,
		Syllables: f.Syllables, PrimaryVowel: f.PrimaryVowel,
		RhymesWith: f.RhymesWith, SoundsLike: f.SoundsLike, Meter: f.Meter,
	}
}
```

Add the debounce/query messages and commands:

```go
type debounceFiredMsg struct{ query string }
type queryResultMsg struct{ rows []queryclient.ResultRow }
type queryErrorMsg struct{ err error }

const debounceDelay = 100 * time.Millisecond

func debounceCmd(query string) tea.Cmd {
	return tea.Tick(debounceDelay, func(time.Time) tea.Msg {
		return debounceFiredMsg{query: query}
	})
}

func runQueryCmd(ctx context.Context, client *queryclient.Client, req queryclient.Request) tea.Cmd {
	return func() tea.Msg {
		rows, err := client.Query(ctx, req)
		if err != nil {
			return queryErrorMsg{err: err}
		}
		return queryResultMsg{rows: rows}
	}
}
```

Add `NewLiveModel`:

```go
func NewLiveModel(client *queryclient.Client) Model {
	m := NewModel(nil)
	m.client = client
	m.filters = NewFilterState()
	return m
}
```

In `Update`, after the existing `case tea.KeyMsg:` block's final `var cmd tea.Cmd; m.input, cmd = m.input.Update(msg); return m, cmd` (the fallthrough path for ordinary typing), change it to also schedule a debounce:

```go
		var inputCmd tea.Cmd
		m.input, inputCmd = m.input.Update(msg)
		return m, tea.Batch(inputCmd, debounceCmd(m.input.Value()))
```

Add two new top-level `case` branches to `Update`'s switch (alongside `tea.WindowSizeMsg`/`tea.KeyMsg`):

```go
	case debounceFiredMsg:
		if msg.query != m.input.Value() {
			return m, nil
		}
		if m.client == nil {
			return m, nil
		}
		ctx, cancel := context.WithCancel(context.Background())
		if m.cancelInFlight != nil {
			m.cancelInFlight()
		}
		m.cancelInFlight = cancel
		return m, runQueryCmd(ctx, m.client, m.filters.toRequest(msg.query))

	case queryResultMsg:
		m.rows = msg.rows
		m.selected = 0
		m.refreshPreview()
		return m, nil

	case queryErrorMsg:
		m.statusMessage = msg.err.Error()
		return m, nil
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
go test ./internal/ui/... -v
```
Expected: all PASS, including Task 4's earlier tests (unaffected).

- [ ] **Step 5: Build and run the full Go test suite**

```bash
go build ./... && go test ./...
```
Expected: builds clean, all tests pass.

- [ ] **Step 6: Commit**

```bash
cd /home/shichika/redacted/rev-dictionary
git add tui/internal/ui/model.go tui/internal/ui/model_test.go
git commit -m "Wire the main screen to debounced, cancellable live queries"
```

---

### Task 6: Filter/sort/category panel overlay

**Files:**
- Create: `tui/internal/ui/panel.go`
- Test: `tui/internal/ui/panel_test.go`
- Modify: `tui/internal/ui/model.go` (add `Tab` dispatch to open the panel, panel-mode routing in `Update`/`View`)
- Modify: `tui/internal/ui/model_test.go`

**Interfaces:**
- Consumes: `FilterState` (Task 5).
- Produces: `panelState` (internal to the `ui` package) with `Update(tea.KeyMsg) panelState` and `View() string`; `Model` gains a `screen` field (`screenSearch`/`screenPanel` — `screenHelp` added in Task 7) that routes `Update`/`View` to the right sub-behavior.

- [ ] **Step 1: Write the failing tests**

Create `tui/internal/ui/panel_test.go`:

`sortModes` and `categories` are package-level variables that `panel.go` (Step 3, below) defines -- this test file only *references* them, it must not redeclare them (both files are `package ui`, so a second `var sortModes = ...` here would be a duplicate-declaration compile error once Step 3's `panel.go` exists).

```go
package ui

import (
	"testing"

	tea "github.com/charmbracelet/bubbletea"
)

func TestNewPanelStartsOnSortField(t *testing.T) {
	p := newPanelState(NewFilterState())
	if p.focusedField != fieldSort {
		t.Fatalf("expected initial focus on sort field, got %d", p.focusedField)
	}
}

func TestTabAdvancesThroughAllSevenFields(t *testing.T) {
	p := newPanelState(NewFilterState())
	seen := []int{p.focusedField}
	for i := 0; i < 6; i++ {
		p = p.handleKey(tea.KeyMsg{Type: tea.KeyTab})
		seen = append(seen, p.focusedField)
	}
	want := []int{fieldSort, fieldCategory, fieldSyllables, fieldPrimaryVowel, fieldRhymesWith, fieldSoundsLike, fieldMeter}
	for i, w := range want {
		if seen[i] != w {
			t.Fatalf("field order mismatch at %d: got %d, want %d", i, seen[i], w)
		}
	}
	// Tab wraps back to the first field.
	p = p.handleKey(tea.KeyMsg{Type: tea.KeyTab})
	if p.focusedField != fieldSort {
		t.Fatalf("expected wraparound to fieldSort, got %d", p.focusedField)
	}
}

func TestDownMovesSortSelection(t *testing.T) {
	p := newPanelState(NewFilterState())
	p = p.handleKey(tea.KeyMsg{Type: tea.KeyDown})
	if p.sortSelected != 1 {
		t.Fatalf("expected sortSelected=1, got %d", p.sortSelected)
	}
	if p.toFilterState().Sort != sortModes[1] {
		t.Fatalf("expected filter state sort=%s, got %s", sortModes[1], p.toFilterState().Sort)
	}
}

func TestCategoryFieldNavigatesIndependentlyOfSort(t *testing.T) {
	p := newPanelState(NewFilterState())
	p.focusedField = fieldCategory
	p = p.handleKey(tea.KeyMsg{Type: tea.KeyDown})
	if p.categorySelected != 1 {
		t.Fatalf("expected categorySelected=1, got %d", p.categorySelected)
	}
	if p.toFilterState().Category != categories[1] {
		t.Fatalf("expected filter state category=%s, got %s", categories[1], p.toFilterState().Category)
	}
}

func TestSyllablesFieldOnlyAcceptsDigits(t *testing.T) {
	p := newPanelState(NewFilterState())
	p.focusedField = fieldSyllables
	p = p.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'2'}})
	p = p.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'x'}})
	if p.syllablesText != "2" {
		t.Fatalf("expected non-digit rejected, syllablesText=%q", p.syllablesText)
	}
	fs := p.toFilterState()
	if fs.Syllables == nil || *fs.Syllables != 2 {
		t.Fatalf("expected Syllables=2, got %v", fs.Syllables)
	}
}

func TestMeterFieldOnlyAcceptsSlashAndX(t *testing.T) {
	p := newPanelState(NewFilterState())
	p.focusedField = fieldMeter
	p = p.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'/'}})
	p = p.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'q'}})
	p = p.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'x'}})
	if p.meterText != "/x" {
		t.Fatalf("expected invalid char rejected, meterText=%q", p.meterText)
	}
}

func TestTabInMainScreenOpensThePanel(t *testing.T) {
	m := NewModel(nil)
	mm, _ := m.Update(tea.KeyMsg{Type: tea.KeyTab})
	m = mm.(Model)
	if m.screen != screenPanel {
		t.Fatalf("expected screen=screenPanel, got %v", m.screen)
	}
}

func TestEscInPanelClosesItAndReturnsToSearch(t *testing.T) {
	m := NewModel(nil)
	mm, _ := m.Update(tea.KeyMsg{Type: tea.KeyTab})
	m = mm.(Model)
	mm, _ = m.Update(tea.KeyMsg{Type: tea.KeyEsc})
	m = mm.(Model)
	if m.screen != screenSearch {
		t.Fatalf("expected screen=screenSearch after Esc, got %v", m.screen)
	}
}
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
go test ./internal/ui/... -v
```
Expected: FAIL to compile — `newPanelState`, `fieldSort`, `screenPanel`, etc. don't exist yet.

- [ ] **Step 3: Implement the panel**

Create `tui/internal/ui/panel.go`:

```go
package ui

import (
	"strconv"
	"strings"
	"unicode"

	tea "github.com/charmbracelet/bubbletea"
)

var sortModes = []string{
	"relevance", "alpha", "alpha_desc", "shortest", "longest",
	"most_common", "least_common", "most_formal", "oldest", "most_modern", "most_lyrical",
}

var categories = []string{"all", "noun", "adjective", "verb", "adverb", "idiom_slang", "old"}

var arpabetVowels = []string{
	"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
	"IH", "IY", "OW", "OY", "UH", "UW",
}

const (
	fieldSort = iota
	fieldCategory
	fieldSyllables
	fieldPrimaryVowel
	fieldRhymesWith
	fieldSoundsLike
	fieldMeter
	fieldCount
)

type panelState struct {
	focusedField      int
	sortSelected      int
	categorySelected  int
	syllablesText     string
	primaryVowelText  string
	rhymesWithText    string
	soundsLikeText    string
	meterText         string
}

func newPanelState(initial FilterState) panelState {
	p := panelState{}
	for i, s := range sortModes {
		if s == initial.Sort {
			p.sortSelected = i
		}
	}
	for i, c := range categories {
		if c == initial.Category {
			p.categorySelected = i
		}
	}
	if initial.Syllables != nil {
		p.syllablesText = strconv.Itoa(*initial.Syllables)
	}
	p.primaryVowelText = initial.PrimaryVowel
	p.rhymesWithText = initial.RhymesWith
	p.soundsLikeText = initial.SoundsLike
	p.meterText = initial.Meter
	return p
}

func acceptRuneForField(field int, r rune) bool {
	switch field {
	case fieldSyllables:
		return unicode.IsDigit(r)
	case fieldPrimaryVowel:
		return unicode.IsLetter(r)
	case fieldRhymesWith, fieldSoundsLike:
		return unicode.IsLetter(r) || r == '-' || r == '\''
	case fieldMeter:
		return r == '/' || r == 'x'
	}
	return false
}

func (p panelState) handleKey(msg tea.KeyMsg) panelState {
	switch msg.Type {
	case tea.KeyTab:
		p.focusedField = (p.focusedField + 1) % fieldCount
		return p
	case tea.KeyShiftTab:
		p.focusedField = (p.focusedField - 1 + fieldCount) % fieldCount
		return p
	case tea.KeyUp:
		if p.focusedField == fieldSort && p.sortSelected > 0 {
			p.sortSelected--
		} else if p.focusedField == fieldCategory && p.categorySelected > 0 {
			p.categorySelected--
		}
		return p
	case tea.KeyDown:
		if p.focusedField == fieldSort && p.sortSelected < len(sortModes)-1 {
			p.sortSelected++
		} else if p.focusedField == fieldCategory && p.categorySelected < len(categories)-1 {
			p.categorySelected++
		}
		return p
	case tea.KeyBackspace:
		switch p.focusedField {
		case fieldSyllables:
			p.syllablesText = trimLastRune(p.syllablesText)
		case fieldPrimaryVowel:
			p.primaryVowelText = trimLastRune(p.primaryVowelText)
		case fieldRhymesWith:
			p.rhymesWithText = trimLastRune(p.rhymesWithText)
		case fieldSoundsLike:
			p.soundsLikeText = trimLastRune(p.soundsLikeText)
		case fieldMeter:
			p.meterText = trimLastRune(p.meterText)
		}
		return p
	case tea.KeyRunes:
		for _, r := range msg.Runes {
			if !acceptRuneForField(p.focusedField, r) {
				continue
			}
			switch p.focusedField {
			case fieldSyllables:
				p.syllablesText += string(r)
			case fieldPrimaryVowel:
				p.primaryVowelText += string(unicode.ToUpper(r))
			case fieldRhymesWith:
				p.rhymesWithText += string(r)
			case fieldSoundsLike:
				p.soundsLikeText += string(r)
			case fieldMeter:
				p.meterText += string(r)
			}
		}
		return p
	}
	return p
}

func trimLastRune(s string) string {
	runes := []rune(s)
	if len(runes) == 0 {
		return s
	}
	return string(runes[:len(runes)-1])
}

func (p panelState) toFilterState() FilterState {
	fs := FilterState{
		Sort: sortModes[p.sortSelected], Category: categories[p.categorySelected],
		PrimaryVowel: p.primaryVowelText, RhymesWith: p.rhymesWithText,
		SoundsLike: p.soundsLikeText, Meter: p.meterText,
	}
	if p.syllablesText != "" {
		if n, err := strconv.Atoi(p.syllablesText); err == nil {
			fs.Syllables = &n
		}
	}
	return fs
}

func (p panelState) View() string {
	var b strings.Builder
	b.WriteString("Sort:     " + radioLine(sortModes, p.sortSelected) + "\n")
	b.WriteString("Category: " + radioLine(categories, p.categorySelected) + "\n")
	b.WriteString("Syllables: [" + p.syllablesText + "]  Primary vowel: [" + p.primaryVowelText + "]\n")
	b.WriteString("Rhymes with: [" + p.rhymesWithText + "]  Sounds like: [" + p.soundsLikeText + "]\n")
	b.WriteString("Meter: [" + p.meterText + "]\n")
	return b.String()
}

func radioLine(options []string, selected int) string {
	var b strings.Builder
	for i, opt := range options {
		marker := "( )"
		if i == selected {
			marker = "(*)"
		}
		b.WriteString(marker + " " + opt + "  ")
	}
	return b.String()
}
```

- [ ] **Step 4: Wire the panel into `Model`**

In `tui/internal/ui/model.go`, add a `screen` type and field:

```go
type screenID int

const (
	screenSearch screenID = iota
	screenPanel
)
```

Add `screen screenID` and `panel panelState` to the `Model` struct. In `Update`'s `case tea.KeyMsg:` block, add panel-mode routing as the very first check (before the existing `switch msg.Type` for the search screen):

```go
	case tea.KeyMsg:
		if m.screen == screenPanel {
			if msg.Type == tea.KeyEsc {
				m.filters = m.panel.toFilterState()
				m.screen = screenSearch
				return m, debounceCmd(m.input.Value())
			}
			m.panel = m.panel.handleKey(msg)
			return m, nil
		}

		switch msg.Type {
		case tea.KeyTab:
			m.panel = newPanelState(m.filters)
			m.screen = screenPanel
			return m, nil
		case tea.KeyCtrlC:
			return m, tea.Quit
		// ... (rest of the existing cases from Task 4/5 unchanged)
```

In `View`, add panel-mode routing at the top:

```go
func (m Model) View() string {
	if m.screen == screenPanel {
		return m.panel.View()
	}
	// ... (rest of the existing View unchanged)
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
go test ./internal/ui/... -v
```
Expected: all PASS.

- [ ] **Step 6: Build and run the full Go test suite**

```bash
go build ./... && go test ./...
```
Expected: builds clean, all tests pass.

- [ ] **Step 7: Commit**

```bash
cd /home/shichika/redacted/rev-dictionary
git add tui/internal/ui/panel.go tui/internal/ui/panel_test.go tui/internal/ui/model.go tui/internal/ui/model_test.go
git commit -m "Add the filter/sort/category panel overlay"
```

---

### Task 7: Help overlay, quick-cycle-sort, and real clipboard wiring

**Files:**
- Create: `tui/internal/ui/help.go`
- Create: `tui/internal/clipboard/clipboard.go`
- Test: `tui/internal/ui/model_test.go`
- Modify: `tui/internal/ui/model.go` (add `F1`/help routing, `Ctrl-R` quick-cycle, wire `onCopy` to the real clipboard)

**Interfaces:**
- Consumes: `sortModes` (Task 6).
- Produces: `clipboard.Copy(text string) error` (Task 8's real constructor wires this into `Model.onCopy`); `screenHelp` added to the `screenID` enum; help text content.

- [ ] **Step 1: Write the failing tests**

Add `"errors"` to `tui/internal/ui/model_test.go`'s existing import block (needed by `TestEnterSurfacesACopyFailureInStatusMessage` below). Then add to the same file:

```go
func TestF1OpensHelpScreen(t *testing.T) {
	m := NewModel(nil)
	mm, _ := m.Update(tea.KeyMsg{Type: tea.KeyF1})
	m = mm.(Model)
	if m.screen != screenHelp {
		t.Fatalf("expected screen=screenHelp, got %v", m.screen)
	}
}

func TestEscClosesHelpScreen(t *testing.T) {
	m := NewModel(nil)
	mm, _ := m.Update(tea.KeyMsg{Type: tea.KeyF1})
	m = mm.(Model)
	mm, _ = m.Update(tea.KeyMsg{Type: tea.KeyEsc})
	m = mm.(Model)
	if m.screen != screenSearch {
		t.Fatalf("expected screen=screenSearch after Esc, got %v", m.screen)
	}
}

func TestCtrlRCyclesSortMode(t *testing.T) {
	m := NewModel(nil)
	m.filters = NewFilterState()
	if m.filters.Sort != "relevance" {
		t.Fatalf("expected initial sort=relevance, got %s", m.filters.Sort)
	}
	mm, _ := m.Update(tea.KeyMsg{Type: tea.KeyCtrlR})
	m = mm.(Model)
	if m.filters.Sort != "alpha" {
		t.Fatalf("expected sort cycled to alpha, got %s", m.filters.Sort)
	}
}

func TestCtrlRWrapsAroundAfterLastSortMode(t *testing.T) {
	m := NewModel(nil)
	m.filters = FilterState{Sort: "most_lyrical", Category: "all"}
	mm, _ := m.Update(tea.KeyMsg{Type: tea.KeyCtrlR})
	m = mm.(Model)
	if m.filters.Sort != "relevance" {
		t.Fatalf("expected wraparound to relevance, got %s", m.filters.Sort)
	}
}

func TestEnterCallsOnCopyWithSelectedHeadword(t *testing.T) {
	m := NewModel([]queryclient.ResultRow{{Headword: "annoyance"}})
	var copied string
	m.onCopy = func(h string) error { copied = h; return nil }
	m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	if copied != "annoyance" {
		t.Fatalf("expected onCopy called with 'annoyance', got %q", copied)
	}
}

func TestEnterSurfacesACopyFailureInStatusMessage(t *testing.T) {
	m := NewModel([]queryclient.ResultRow{{Headword: "annoyance"}})
	m.onCopy = func(h string) error { return errors.New("no clipboard utility found") }
	mm, _ := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = mm.(Model)
	if !strings.Contains(m.statusMessage, "no clipboard utility found") {
		t.Fatalf("expected copy failure in status message, got %q", m.statusMessage)
	}
}
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
go test ./internal/ui/... -v
```
Expected: FAIL — `screenHelp` doesn't exist, `Ctrl-R` isn't wired yet (`TestEnterCallsOnCopyWithSelectedHeadword` should already pass since Task 4 built `onCopy` — confirm it does, don't reintroduce it if so).

- [ ] **Step 3: Implement the help overlay and sort-cycling**

Create `tui/internal/ui/help.go`:

```go
package ui

const helpText = `revdict-tui -- keyboard shortcuts

Search screen:
  (typing)      type into the query box (live search, debounced)
  Up / Down     move the highlighted result
  Enter         copy the highlighted candidate to the clipboard
  Esc           clear the query; press again on an empty query to quit
  Ctrl-C        quit immediately
  Tab           open the filter/sort/category panel
  Ctrl-R        quick-cycle the sort mode
  F2            toggle the preview pane
  F1            this help screen

Filter panel:
  Tab / Shift-Tab   move between fields
  Up / Down         move within the sort/category lists
  (typing)          fill the focused text field
  Esc               close the panel and re-run the query

Query syntax (typed directly into the search box):
  blue*         starts with "blue"
  *bird         ends with "bird"
  bl????rd      starts with "bl", ends with "rd", 4 letters between
  ?????         any 5-letter word
  //fuljyo      anagram/unscramble
  -abcd         excludes these letters
  +abcd         built only from these letters
  bl*:snow      starts with "bl" AND related in meaning to "snow"
  **winter**    multi-word phrases containing the whole word "winter"
  expand:nasa   phrases whose initials spell "nasa"
`
```

In `tui/internal/ui/model.go`, add `screenHelp` to the `screenID` enum:

```go
const (
	screenSearch screenID = iota
	screenPanel
	screenHelp
)
```

In `Update`'s `case tea.KeyMsg:` block, add help-mode routing right after the existing panel-mode routing (and before the `switch msg.Type` for the search screen):

```go
		if m.screen == screenHelp {
			if msg.Type == tea.KeyEsc {
				m.screen = screenSearch
			}
			return m, nil
		}
```

In the search-screen `switch msg.Type`, add two cases (alongside the existing `tea.KeyTab`/`tea.KeyCtrlC`/etc.):

```go
		case tea.KeyF1:
			m.screen = screenHelp
			return m, nil

		case tea.KeyCtrlR:
			m.filters.Sort = nextSortMode(m.filters.Sort)
			return m, debounceCmd(m.input.Value())
```

Add `nextSortMode` to `tui/internal/ui/panel.go` (it belongs alongside `sortModes`):

```go
func nextSortMode(current string) string {
	for i, mode := range sortModes {
		if mode == current {
			return sortModes[(i+1)%len(sortModes)]
		}
	}
	return sortModes[0]
}
```

In `View`, add help-mode routing at the top (alongside the existing panel-mode routing):

```go
	if m.screen == screenHelp {
		return helpText
	}
```

- [ ] **Step 4: Implement the real clipboard package**

Create `tui/internal/clipboard/clipboard.go`:

```go
package clipboard

import (
	"encoding/base64"
	"fmt"
	"os"
)

// Copy writes text to the terminal's clipboard via the OSC 52 escape
// sequence -- this works over SSH/tmux (reaching the local machine's
// clipboard, not the remote host's), the same mechanism this project's
// existing fzf-based picker already uses (see picker.py's clipboard
// handling and the README's "Clipboard copy on Enter" section).
func Copy(text string) error {
	encoded := base64.StdEncoding.EncodeToString([]byte(text))
	_, err := fmt.Fprintf(os.Stderr, "\x1b]52;c;%s\x07", encoded)
	return err
}
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
go test ./internal/ui/... -v
```
Expected: all PASS.

- [ ] **Step 6: Build and run the full Go test suite**

```bash
go build ./... && go test ./...
```
Expected: builds clean, all tests pass.

- [ ] **Step 7: Commit**

```bash
cd /home/shichika/redacted/rev-dictionary
git add tui/internal/ui/help.go tui/internal/ui/panel.go tui/internal/ui/model.go tui/internal/ui/model_test.go tui/internal/clipboard/clipboard.go
git commit -m "Add the help overlay, sort-mode quick-cycle, and clipboard copy"
```

---

### Task 8: Real entrypoint, README, and distribution docs

**Files:**
- Modify: `tui/cmd/revdict-tui/main.go` (replace the Task 2 skeleton with the real wiring)
- Modify: `README.md` (new "Advanced TUI" section)

**Interfaces:**
- Consumes: `ui.NewLiveModel`, `queryclient.New`, `clipboard.Copy` (all prior tasks).
- Produces: the finished, installable `revdict-tui` binary. Nothing downstream in this phase.

- [ ] **Step 1: Replace the skeleton entrypoint**

Replace `tui/cmd/revdict-tui/main.go` in full:

```go
package main

import (
	"fmt"
	"os"
	"os/exec"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/nijuyonkadesu/revdict/tui/internal/clipboard"
	"github.com/nijuyonkadesu/revdict/tui/internal/queryclient"
	"github.com/nijuyonkadesu/revdict/tui/internal/ui"
)

func main() {
	if _, err := exec.LookPath("revdict"); err != nil {
		fmt.Fprintln(os.Stderr, "revdict-tui: 'revdict' not found on PATH -- install it first, see the repo README")
		os.Exit(1)
	}

	model := ui.NewLiveModel(queryclient.New())
	model.SetCopyFunc(clipboard.Copy)

	p := tea.NewProgram(model)
	if _, err := p.Run(); err != nil {
		fmt.Fprintln(os.Stderr, "revdict-tui:", err)
		os.Exit(1)
	}
}
```

This calls a new exported method, `SetCopyFunc`, rather than reaching into `Model.onCopy` directly (an unexported field of a value type returned by `NewLiveModel` cannot be set from another package). Add this method to `tui/internal/ui/model.go`:

```go
// SetCopyFunc overrides the clipboard behavior invoked on Enter. Exported
// so cmd/revdict-tui can wire in the real clipboard.Copy without this
// package needing to import the clipboard package itself (keeping
// dependency direction one-way: cmd -> ui -> queryclient, never ui ->
// clipboard).
func (m *Model) SetCopyFunc(f func(string) error) {
	m.onCopy = f
}
```

- [ ] **Step 2: Verify it builds**

```bash
cd /home/shichika/redacted/rev-dictionary/tui
go build ./... && go test ./...
go build -o /tmp/revdict-tui ./cmd/revdict-tui
```
Expected: builds clean, all tests pass, `/tmp/revdict-tui` exists.

- [ ] **Step 3: Manual smoke check (not part of the automated test suite)**

```bash
/tmp/revdict-tui
```
Manually confirm: the search box accepts typed input, results appear (assuming `revdict`'s index is built and the daemon is reachable or can cold-start), `Up`/`Down` move the selection, `Enter` on a selection doesn't crash, `Tab` opens the filter panel and `Esc` closes it, `F1` opens help and `Esc` closes it, `Ctrl-C` quits. Resize the terminal while it's running and confirm the layout adapts rather than corrupting.

- [ ] **Step 4: Document it in the README**

Add a new section to `README.md`, after the existing "Phonetic filters" section and before "Sort order" (or at the end — pick whichever location reads best given the file's current section order at commit time; re-read the file's current heading order before inserting):

`````markdown
## Advanced TUI

For a live, keyboard-driven interface exposing every filter/sort/category
option at once (rather than one-shot CLI flags), install the standalone
`revdict-tui` binary:

```bash
go install github.com/nijuyonkadesu/revdict/tui/cmd/revdict-tui@latest
```

This requires Go 1.24+ and a tagged release of this repo (`go install`
against a subdirectory module only resolves `@latest` once a git tag of
the form `tui/vX.Y.Z` exists — until then, install a specific commit with
`go install github.com/nijuyonkadesu/revdict/tui/cmd/revdict-tui@<commit-sha>`).

`revdict-tui` shells out to the `revdict` binary already on your `PATH` --
it needs the same index built (`revdict build-index`) as the regular CLI,
and benefits from the same background daemon for fast repeat queries.

Keyboard shortcuts (also shown in-app via `F1`):

| Key | Action |
|---|---|
| (typing) | live search, debounced |
| `Up`/`Down` | move the highlighted result |
| `Enter` | copy the highlighted candidate to the clipboard |
| `Esc` | clear the query; press again on an empty query to quit |
| `Ctrl-C` | quit immediately |
| `Tab` | open the filter/sort/category panel |
| `Ctrl-R` | quick-cycle the sort mode |
| `F2` | toggle the preview pane |
| `F1` | help |
`````

- [ ] **Step 5: Commit**

```bash
cd /home/shichika/redacted/rev-dictionary
git add tui/cmd/revdict-tui/main.go tui/internal/ui/model.go README.md
git commit -m "Wire the real revdict-tui entrypoint and document installation"
```

---

## Post-plan note for the final whole-branch reviewer

Per this project's established Phase 3/4/5 precedent, the final whole-branch review should include a real-usage check, not just passing unit tests: build the real binary (`go build -o /tmp/revdict-tui ./tui/cmd/revdict-tui`) and drive it interactively against a real, built `revdict` index — confirm a real query returns real results, a filter change in the panel actually changes them, resizing the terminal doesn't corrupt the layout, and `Ctrl-C`/double-`Esc` both cleanly exit. `go install .../tui/cmd/revdict-tui@latest` itself cannot be verified until after this branch is pushed AND a `tui/vX.Y.Z` tag exists — note this explicitly rather than trying to test it mid-review; a `go build`/local-path `go install` check is the correct pre-push substitute.
