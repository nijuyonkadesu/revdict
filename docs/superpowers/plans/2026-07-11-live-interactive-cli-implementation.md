# Live Interactive CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bare `revdict` (no query argument, real terminal) launches a persistent, live-updating fzf session instead of today's one-shot "type one query, see one result, process exits" flow.

**Architecture:** A single long-lived `fzf` process drives the whole session, using fzf's native `change:reload` binding (debounced with `sleep`) to re-run the real daemon-backed search after each pause in typing, plus rebound keys (Esc clears the query, Enter commits it to a history file instead of selecting/exiting, Ctrl-C/Ctrl-D quit) and a size-conditional `--preview-window` for responsive layout. A new fast `revdict --query-only QUERY` entry point is what the reload binding shells out to.

**Tech Stack:** Python (existing `argparse`/`subprocess` patterns already used in `cli.py`/`picker.py`/`daemon.py`), fzf (already a soft dependency).

## Global Constraints

- `revdict "some query"` (one-shot mode, explicit query argument) is **completely unchanged** — same code path (`run_picker()`), same tests, same behavior.
- Bare `revdict` when stdout is **not** a tty keeps today's existing behavior unchanged (read one line, one-shot search, static table) — only the **tty** bare-invocation case changes.
- If fzf is missing, live mode prints `"Live mode requires fzf. Install it, or use revdict \"your query\" for one-shot search."` and exits with code 1 — no fallback interactive loop is built (per the approved spec).
- Debounce duration and the responsive-layout column threshold are implementation-time tuning constants, not fixed design requirements — this plan picks concrete starting values (`0.1`s, `100` columns) that Task 5's manual validation may adjust.
- Preview rendering for the live list reuses the exact same per-candidate preview text (`_render_exact_preview`/`_render_candidate_preview`) already used by one-shot mode — no new preview format.

---

### Task 1: Extract a shared candidate-file-writing helper in `picker.py`

**Files:**
- Modify: `src/revdict/paths.py`
- Modify: `src/revdict/picker.py`
- Test: `tests/test_picker.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `revdict.picker.write_candidate_files(tmp_path: Path, candidates: list[dict], exact_match: dict | None) -> list[str]`. Task 2 and Task 4 both call this — it's the single place that turns a search result into fzf-ready lines and preview files, used by both one-shot mode and the new live-reload mode.

- [ ] **Step 1: Add the history file path constant**

In `src/revdict/paths.py`, add this line after `DAEMON_LOG_PATH`:

```python
QUERY_HISTORY_PATH = CACHE_DIR / "query_history"
```

- [ ] **Step 2: Write the failing test for the extracted helper**

Add to `tests/test_picker.py` (near the top, after the existing imports — add `write_candidate_files` to the import list from `revdict.picker`, and add `from pathlib import Path` and `import tempfile` at the top of the file):

```python
def test_write_candidate_files_returns_one_line_per_candidate_plus_exact_match():
    with tempfile.TemporaryDirectory() as tmp:
        lines = write_candidate_files(Path(tmp), _CANDIDATE_FIXTURE, _EXACT_MATCH_FIXTURE)

        assert len(lines) == 2  # exact match + 1 candidate
        assert lines[0].startswith("★")
        assert (Path(tmp) / "0.txt").exists()
        assert (Path(tmp) / "1.txt").exists()


def test_write_candidate_files_with_no_exact_match_writes_only_candidates():
    with tempfile.TemporaryDirectory() as tmp:
        lines = write_candidate_files(Path(tmp), _CANDIDATE_FIXTURE, None)

        assert len(lines) == 1
        assert not lines[0].startswith("★")
        assert (Path(tmp) / "0.txt").exists()
        assert not (Path(tmp) / "1.txt").exists()
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_picker.py -v -k write_candidate_files`
Expected: FAIL with `ImportError: cannot import name 'write_candidate_files' from 'revdict.picker'`

- [ ] **Step 4: Extract the helper and refactor `run_picker` to use it**

In `src/revdict/picker.py`, replace the body of `run_picker` from the `with tempfile.TemporaryDirectory() as tmp:` line through the `input_text = "\n".join(lines) + "\n"` line (i.e. everything that builds `lines` and writes preview files) with a call to the new helper. The full updated file section:

```python
def write_candidate_files(
    tmp_path: Path, candidates: list[dict], exact_match: dict | None
) -> list[str]:
    """Writes one preview .txt file per row to tmp_path and returns the
    matching tab-delimited fzf input lines. Shared by run_picker's one-shot
    session (writes once, invokes fzf once) and the live session's
    change:reload path (invoked repeatedly against the same tmp_path as the
    query changes)."""
    lines = []
    index = 0

    if exact_match is not None:
        first_sense = exact_match["senses"][0]
        (tmp_path / f"{index}.txt").write_text(
            _render_exact_preview(exact_match), encoding="utf-8"
        )
        lines.append(
            format_candidate_line(
                exact_match["headword"],
                first_sense["pos"],
                first_sense["definition"],
                first_sense["label"],
                first_sense["polarity"],
                100,
                index=index,
                is_exact=True,
            )
        )
        index += 1

    for candidate in candidates:
        (tmp_path / f"{index}.txt").write_text(
            _render_candidate_preview(candidate), encoding="utf-8"
        )
        lines.append(
            format_candidate_line(
                candidate["headword"],
                candidate["pos"],
                candidate["definition"],
                candidate["label"],
                candidate["polarity"],
                candidate["relevance"],
                index=index,
            )
        )
        index += 1

    return lines


def run_picker(candidates: list[dict], exact_match: dict | None) -> str | None:
    if shutil.which("fzf") is None:
        return None

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        lines = write_candidate_files(tmp_path, candidates, exact_match)

        input_text = "\n".join(lines) + "\n"
        result = subprocess.run(
            [
                "fzf",
                "--delimiter",
                "\t",
                "--with-nth=1,2,3,4",
                "--preview",
                f"cat {tmp_path}/{{5}}.txt",
                "--preview-window",
                "right:60%:wrap",
                "--bind",
                "?:toggle-preview",
            ],
            input=input_text,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            if result.returncode in _CANCELLED_RETURN_CODES:
                return None
            raise PickerError(result.returncode, result.stderr)

        selection_index = parse_selection(result.stdout)
        if selection_index is None:
            return None
        if exact_match is not None:
            if selection_index == 0:
                return exact_match["headword"]
            return candidates[selection_index - 1]["headword"]
        return candidates[selection_index]["headword"]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_picker.py -v`
Expected: all tests pass, including every pre-existing `run_picker` test (this step is a pure refactor — no behavior change).

- [ ] **Step 6: Commit**

```bash
git add src/revdict/paths.py src/revdict/picker.py tests/test_picker.py
git commit -m "Extract write_candidate_files from run_picker for reuse by live mode"
```

---

### Task 2: Add the `revdict --query-only` fast entry point

**Files:**
- Modify: `src/revdict/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `revdict.picker.write_candidate_files(tmp_path, candidates, exact_match) -> list[str]` (Task 1).
- Produces: a new `main()` branch handling `argv[0] == "--query-only"`. Task 4's live session shells out to this via `{sys.executable} -u -m revdict.cli --query-only {q}`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli.py`:

```python
def test_query_only_prints_candidate_lines_into_the_given_preview_dir(monkeypatch, capsys, tmp_path):
    fake_result = {
        "exact_match": None,
        "candidates": [
            {
                "headword": "joyful",
                "pos": "adjective",
                "definition": "feeling great happiness",
                "examples": [],
                "label": "joy",
                "polarity": "positive",
                "relevance": 90,
            }
        ],
    }
    monkeypatch.setattr(cli, "_get_search_result", lambda query, top_n: fake_result)
    monkeypatch.setenv("REVDICT_LIVE_PREVIEW_DIR", str(tmp_path))

    code = cli.main(["--query-only", "happy"])

    captured = capsys.readouterr()
    assert code == 0
    assert "joyful" in captured.out
    assert (tmp_path / "0.txt").exists()


def test_query_only_with_blank_query_prints_nothing(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("REVDICT_LIVE_PREVIEW_DIR", str(tmp_path))

    code = cli.main(["--query-only", ""])

    captured = capsys.readouterr()
    assert code == 0
    assert captured.out == ""
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_cli.py -v -k query_only`
Expected: FAIL — `--query-only` currently falls through to the normal query parser and errors or behaves unexpectedly (not a clean "0 with no output" or "0 with candidate lines").

- [ ] **Step 3: Implement the `--query-only` branch**

In `src/revdict/cli.py`, add these imports at the top (alongside the existing `from revdict.picker import PickerError, run_picker` line):

```python
import os
from pathlib import Path

from revdict.picker import PickerError, run_picker, write_candidate_files
```

Add this constant near the top of the file, after `console = Console()`:

```python
LIVE_SESSION_TOP_N = 15
```

Add this new function after `_run_query`:

```python
def _run_query_only(query: str) -> int:
    if not query.strip():
        return 0

    preview_dir = Path(os.environ["REVDICT_LIVE_PREVIEW_DIR"])
    result = _get_search_result(query, LIVE_SESSION_TOP_N)
    lines = write_candidate_files(preview_dir, result["candidates"], result["exact_match"])
    for line in lines:
        print(line)
    return 0
```

In `main()`, add a new branch right after the existing `if argv and argv[0] == "daemon":` block (before the `if not argv:` check):

```python
        if argv and argv[0] == "--query-only":
            query = argv[1] if len(argv) > 1 else ""
            return _run_query_only(query)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_cli.py -v`
Expected: all tests pass, including the two new ones and every pre-existing test.

- [ ] **Step 5: Commit**

```bash
git add src/revdict/cli.py tests/test_cli.py
git commit -m "Add revdict --query-only, the fast entry point live-reload shells out to"
```

---

### Task 3: Build the fzf argument list for the live session

**Files:**
- Modify: `src/revdict/picker.py`
- Test: `tests/test_picker.py`

**Interfaces:**
- Consumes: `revdict.paths.QUERY_HISTORY_PATH` (Task 1).
- Produces: `revdict.picker.build_live_session_args(preview_dir: Path, history_path: Path, python_executable: str, debounce_seconds: float = 0.1, layout_threshold_columns: int = 100) -> list[str]`. Task 4 consumes this to build the actual subprocess call. Kept as a pure function (no subprocess call, no side effects) specifically so the exact argument list is unit-testable without a real terminal.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_picker.py`:

```python
def test_build_live_session_args_includes_disabled_and_reload_bindings():
    args = picker.build_live_session_args(
        preview_dir=Path("/tmp/preview"),
        history_path=Path("/tmp/history"),
        python_executable="/usr/bin/python3",
    )

    assert "--disabled" in args
    joined = " ".join(args)
    assert "change:reload:sleep 0.1" in joined
    assert "/usr/bin/python3 -u -m revdict.cli --query-only {q}" in joined
    assert "start:reload:sleep 0.1" in joined


def test_build_live_session_args_includes_history_file():
    args = picker.build_live_session_args(
        preview_dir=Path("/tmp/preview"),
        history_path=Path("/tmp/history"),
        python_executable="/usr/bin/python3",
    )

    assert "--history=/tmp/history" in args


def test_build_live_session_args_rebinds_esc_enter_and_arrow_keys():
    args = picker.build_live_session_args(
        preview_dir=Path("/tmp/preview"),
        history_path=Path("/tmp/history"),
        python_executable="/usr/bin/python3",
    )

    joined = " ".join(args)
    assert "esc:clear-query" in joined
    assert "enter:execute-silent(echo {q} >> /tmp/history)+clear-query" in joined
    assert "up:prev-history" in joined
    assert "down:next-history" in joined
    assert "ctrl-p:up" in joined
    assert "ctrl-n:down" in joined


def test_build_live_session_args_uses_the_given_debounce_and_threshold():
    args = picker.build_live_session_args(
        preview_dir=Path("/tmp/preview"),
        history_path=Path("/tmp/history"),
        python_executable="/usr/bin/python3",
        debounce_seconds=0.25,
        layout_threshold_columns=80,
    )

    joined = " ".join(args)
    assert "sleep 0.25" in joined
    assert "<80(up,50%)" in joined


def test_build_live_session_args_preview_command_reads_from_the_preview_dir():
    args = picker.build_live_session_args(
        preview_dir=Path("/tmp/preview"),
        history_path=Path("/tmp/history"),
        python_executable="/usr/bin/python3",
    )

    joined = " ".join(args)
    assert "cat /tmp/preview/{5}.txt" in joined
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_picker.py -v -k build_live_session_args`
Expected: FAIL with `AttributeError: module 'revdict.picker' has no attribute 'build_live_session_args'`

- [ ] **Step 3: Implement `build_live_session_args`**

Add to `src/revdict/picker.py`, after `run_picker`:

```python
def build_live_session_args(
    preview_dir: Path,
    history_path: Path,
    python_executable: str,
    debounce_seconds: float = 0.1,
    layout_threshold_columns: int = 100,
) -> list[str]:
    """Builds the fzf argument list for the persistent live-typing session
    (see docs/superpowers/specs/2026-07-11-live-interactive-cli-design.md).
    Pure and side-effect-free so the exact bindings are unit-testable
    without a real terminal -- run_live_session is the thin wrapper that
    actually invokes this as a subprocess."""
    reload_command = (
        f"sleep {debounce_seconds}; "
        f"{python_executable} -u -m revdict.cli --query-only {{q}}"
    )
    return [
        "fzf",
        "--disabled",
        "--delimiter",
        "\t",
        "--with-nth=1,2,3,4",
        "--history",
        str(history_path),
        "--preview",
        f"cat {preview_dir}/{{5}}.txt",
        "--preview-window",
        f"right,50%,wrap,<{layout_threshold_columns}(up,50%)",
        "--bind",
        f"start:reload:{reload_command}",
        "--bind",
        f"change:reload:{reload_command}",
        "--bind",
        "esc:clear-query",
        "--bind",
        f"enter:execute-silent(echo {{q}} >> {history_path})+clear-query",
        "--bind",
        "ctrl-c:abort",
        "--bind",
        "ctrl-d:abort",
        "--bind",
        "up:prev-history",
        "--bind",
        "down:next-history",
        "--bind",
        "ctrl-p:up",
        "--bind",
        "ctrl-n:down",
        "--bind",
        "?:toggle-preview",
    ]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_picker.py -v`
Expected: all tests pass, including the five new ones and every pre-existing test.

- [ ] **Step 5: Commit**

```bash
git add src/revdict/picker.py tests/test_picker.py
git commit -m "Add build_live_session_args, the fzf binding construction for live mode"
```

---

### Task 4: Wire the live session into `main()`

**Files:**
- Modify: `src/revdict/picker.py`
- Modify: `src/revdict/cli.py`
- Test: `tests/test_picker.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `revdict.picker.build_live_session_args(...)` (Task 3), `revdict.paths.QUERY_HISTORY_PATH` (Task 1).
- Produces: `revdict.picker.run_live_session() -> None`, called from `cli.main()`'s bare-invocation tty branch.

- [ ] **Step 1: Write the failing test for `run_live_session`**

Add to `tests/test_picker.py`:

```python
def test_run_live_session_returns_none_when_fzf_binary_is_missing(monkeypatch):
    monkeypatch.setattr(picker.shutil, "which", lambda name: None)

    result = picker.run_live_session()

    assert result is None


def test_run_live_session_invokes_fzf_with_the_built_args_and_cleans_up(monkeypatch):
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["env"] = kwargs.get("env")
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(picker.shutil, "which", lambda name: "/usr/bin/fzf")
    monkeypatch.setattr(picker.subprocess, "run", fake_run)

    created_dirs = []
    real_mkdtemp = picker.tempfile.mkdtemp

    def tracking_mkdtemp(*a, **k):
        path = real_mkdtemp(*a, **k)
        created_dirs.append(path)
        return path

    monkeypatch.setattr(picker.tempfile, "mkdtemp", tracking_mkdtemp)

    picker.run_live_session()

    assert captured["args"][0] == "fzf"
    assert "--disabled" in captured["args"]
    assert captured["env"]["REVDICT_LIVE_PREVIEW_DIR"] == created_dirs[0]
    assert not picker.Path(created_dirs[0]).exists()  # cleaned up after fzf exits
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_picker.py -v -k run_live_session`
Expected: FAIL with `AttributeError: module 'revdict.picker' has no attribute 'run_live_session'`

- [ ] **Step 3: Implement `run_live_session`**

Add to `src/revdict/picker.py`, after `build_live_session_args`. First add `import os` and `import sys` near the top of the file alongside the existing `import shutil` / `import subprocess` / `import tempfile` imports, and add `from revdict.paths import QUERY_HISTORY_PATH` to the imports:

```python
def run_live_session() -> None:
    if shutil.which("fzf") is None:
        return None

    QUERY_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUERY_HISTORY_PATH.touch(exist_ok=True)

    preview_dir = tempfile.mkdtemp(prefix="revdict-live-")
    try:
        args = build_live_session_args(
            preview_dir=Path(preview_dir),
            history_path=QUERY_HISTORY_PATH,
            python_executable=sys.executable,
        )
        subprocess.run(
            args,
            env={**os.environ, "REVDICT_LIVE_PREVIEW_DIR": preview_dir},
        )
    finally:
        shutil.rmtree(preview_dir, ignore_errors=True)
    return None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_picker.py -v`
Expected: all tests pass.

- [ ] **Step 5: Write the failing test for `main()`'s dispatch**

Add to `tests/test_cli.py`:

```python
def test_main_with_no_args_and_a_tty_launches_the_live_session(monkeypatch):
    monkeypatch.setattr(cli, "_index_exists", lambda: True)

    called = {"ran": False}
    monkeypatch.setattr(cli.picker, "run_live_session", lambda: called.__setitem__("ran", True))

    class _TtyStdout:
        def isatty(self):
            return True

    monkeypatch.setattr(cli.sys, "stdout", _TtyStdout())

    code = cli.main([])

    assert code == 0
    assert called["ran"] is True


def test_main_with_no_args_and_a_tty_but_missing_fzf_prints_a_clear_message(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_index_exists", lambda: True)
    monkeypatch.setattr(cli, "_fzf_missing", lambda: True)

    # Patch only isatty on the REAL sys.stdout object -- replacing sys.stdout
    # wholesale (as the sibling test above does, which is safe there because
    # that path never calls console.print/console.input) breaks both Rich's
    # Console (which resolves sys.stdout dynamically on every print call) and
    # capsys's own capture mechanism, since capsys itself works by replacing
    # sys.stdout.
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)

    code = cli.main([])

    captured = capsys.readouterr()
    assert code == 1
    assert "fzf" in captured.out.lower()
```

- [ ] **Step 6: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_cli.py -v -k "launches_the_live_session or missing_fzf_prints"`
Expected: FAIL — `main([])` with a tty stdout currently still runs the old `console.input()` one-shot flow, not the live session.

- [ ] **Step 7: Wire `main()`'s bare-invocation branch**

In `src/revdict/cli.py`, add `from revdict import picker` near the top (alongside the existing `from revdict.picker import PickerError, run_picker, write_candidate_files` line — keep both imports, one for the module and one for the specific names already used elsewhere in the file).

Replace the existing bare-invocation block in `main()`:

```python
        if not argv:
            if not _index_exists():
                _print_no_index_error()
                return 1
            query = console.input("[bold]> [/bold]")
            return _run_query(query, top_n=30, interactive=sys.stdout.isatty())
```

with:

```python
        if not argv:
            if not _index_exists():
                _print_no_index_error()
                return 1
            if sys.stdout.isatty():
                if _fzf_missing():
                    console.print(
                        "[yellow]Live mode requires fzf. Install it, or use "
                        "revdict \"your query\" for one-shot search.[/yellow]"
                    )
                    return 1
                picker.run_live_session()
                return 0
            query = console.input("[bold]> [/bold]")
            return _run_query(query, top_n=30, interactive=False)
```

Note this preserves the existing non-tty behavior exactly (down to the `console.input()` prompt and `_run_query` call) — only the `interactive=sys.stdout.isatty()` argument becomes the literal `interactive=False`, which is equivalent in the surviving code path since it's only reached when `sys.stdout.isatty()` is already known to be `False`.

- [ ] **Step 8: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all tests pass, including `test_main_with_no_args_checks_isatty_before_going_interactive` (the pre-existing non-tty test) unchanged.

- [ ] **Step 9: Commit**

```bash
git add src/revdict/picker.py src/revdict/cli.py tests/test_picker.py tests/test_cli.py
git commit -m "Launch the live interactive session for bare revdict on a real terminal"
```

---

### Task 5: Manual end-to-end validation

No new automated tests — this validates the real, live-terminal behavior that no unit test exercises (per the spec's Testing section: the live *feel* is manually verified, matching how `run_picker()` has always been validated).

**Files:** None created.

- [ ] **Step 1: Confirm the full suite passes and restart the daemon**

```bash
cd /home/shichika/redacted/rev-dictionary
.venv/bin/python -m pytest tests/ -q
revdict daemon stop 2>&1 || true
```

Expected: all tests pass; daemon stop message or "not running" (either is fine).

- [ ] **Step 2: Launch bare `revdict` in a real terminal and verify the live-typing feel**

```bash
revdict
```

Expected: fzf opens with an empty list. Type `happy` slowly, one letter at a time — after each short pause, the list should update to real search results (not instant on every keystroke, but not sluggish either — if it feels laggy, lower `debounce_seconds` in `run_live_session`'s call to `build_live_session_args`; if it fires too eagerly mid-word, raise it).

- [ ] **Step 3: Verify the rebound keys**

With the session still open: press **Esc** — the typed query should clear, but fzf should remain open (not exit). Type a query and press **Enter** — the query box should clear and fzf should remain open; check `~/.cache/rev_dictionary/query_history` afterward and confirm the query text was appended. Press **Up** — the previous query should reappear in the box (history recall). Press **Ctrl-C** — fzf should exit and control should return directly to the shell prompt, with no need to re-run `revdict`.

- [ ] **Step 4: Verify responsive layout**

```bash
revdict
```

With a wide terminal (>100 columns), confirm the preview pane appears to the right of the list. Resize the terminal narrower than 100 columns (or open a narrow terminal, e.g. on a phone SSH client) and re-launch; confirm the preview pane now appears above/below the list instead. Adjust `layout_threshold_columns` in `run_live_session`'s call to `build_live_session_args` if 100 doesn't feel like the right cutoff for real phone-vs-laptop widths.

- [ ] **Step 5: Verify one-shot mode is unaffected**

```bash
revdict "happy" --no-interactive -n 5
```

Expected: identical output shape to before this plan — a static table, process exits immediately. This confirms Task 1's refactor didn't change one-shot behavior.

- [ ] **Step 6: Commit the validation note**

```bash
git commit --allow-empty -m "Validate the live interactive CLI session against a real terminal"
git push
```

If any step in this task didn't match its expected outcome, do not treat this feature as complete — investigate and fix before moving on.
