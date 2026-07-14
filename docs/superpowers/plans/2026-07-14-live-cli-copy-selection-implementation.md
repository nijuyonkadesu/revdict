# Live CLI Copy-Selection-on-Enter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pressing Enter in `revdict`'s live interactive session copies the currently-highlighted candidate's headword to the clipboard, in addition to its existing history-commit-and-clear behavior — unconditionally, no flag or opt-in.

**Architecture:** A new additive CLI entry point, `revdict --copy-selection "MARKED_TEXT"`, strips the display marker from the given text and copies the resulting headword via OSC 52 (written directly to `/dev/tty`) when running inside tmux or over SSH, or via an auto-detected system clipboard tool otherwise. `build_live_session_args`'s existing `enter` binding gets a third chained `execute-silent` action invoking this new entry point with fzf's `{1}` placeholder.

**Tech Stack:** Python (`base64`, `subprocess`, `shutil`, `os` — all standard library, no new dependency), fzf's `--bind` chaining.

## Global Constraints

- **Unconditional, no flag or opt-in.** No new parameter on `build_live_session_args`; every live session gets this behavior from the moment it ships.
- **Enter's existing behavior is unchanged**, not replaced: history-commit and clear-query still happen exactly as before; the copy is a third chained action.
- **Remote detection:** `bool($TMUX or $SSH_TTY or $SSH_CONNECTION or $SSH_CLIENT)` → OSC 52. Otherwise → system clipboard tool.
- **OSC 52 must be written directly to `/dev/tty`, not stdout** — empirically confirmed during design that `execute-silent`'s child process does not otherwise reach the pty tmux monitors for OSC 52 (verified against the real fzf/tmux binaries; see the design spec's Mechanism section for the exact evidence).
- **No DCS-passthrough wrapping** — considered and tested during design, not used (a plain OSC 52 sequence is sufficient given `set-clipboard on`, and DCS-wrapping's success can't be locally verified since it deliberately bypasses tmux's own observable buffer).
- **System clipboard tool priority order:** `wl-copy`, then `xclip -selection clipboard`, then `xsel --clipboard --input`, then `pbcopy` — try each in that order, use the first one found on `PATH`, stop at the first one that succeeds.
- **Marker-stripping:** `format_candidate_line`'s field 1 (`src/revdict/picker.py`) is always exactly `f"{marker} {headword}"` where `marker` is `"★"` or `" "` — both one character — so the field is always a 2-character marker+separator prefix. `text[2:]` (Python strings are Unicode-aware) correctly strips this regardless of which marker was used.
- **Silent failure:** if neither mechanism is usable, `--copy-selection` exits quietly (no stdout, no crash) — consistent with `execute-silent`'s own silent-by-design nature and the fact that Enter's other two chained actions must succeed regardless.
- **The two copy mechanisms' actual I/O (`/dev/tty` writes, shelling out to a real clipboard tool) are not meaningfully unit-testable** and are validated manually in Task 3, via the same tmux-driven technique already established for this project. The logic around them (marker-stripping, environment detection, OSC 52 sequence construction, clipboard-tool priority/fallback) is fully unit-testable and must be tested.

Full design rationale, including the empirical evidence for the `/dev/tty` requirement and the plain-vs-DCS-wrapped OSC 52 decision, lives in `docs/superpowers/specs/2026-07-14-live-cli-copy-selection-design.md` — read it if a task's "why" isn't obvious from this plan alone.

---

## Task 1: `--copy-selection` CLI entry point

**Files:**
- Modify: `src/revdict/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: nothing from other tasks (this task is self-contained).
- Produces: `revdict --copy-selection "MARKED_TEXT"` as a working CLI invocation, dispatched from `main()`. Task 2 shells out to this exact invocation from `build_live_session_args`.

- [ ] **Step 1: Write the failing tests**

Add `import base64` and `import subprocess` to the top of `src/revdict/cli.py`'s import block (it currently starts with `import argparse`, `import json`, `import os`, `import shutil`, `import sys` — add the two new imports alphabetically among them):

```python
import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
```

Add these tests to `tests/test_cli.py` (append near the existing `test_jsonl_query_*` tests):

```python
def test_strip_candidate_marker_removes_the_star_marker_and_separator():
    assert cli._strip_candidate_marker("★ joy") == "joy"


def test_strip_candidate_marker_removes_the_space_marker_and_separator():
    assert cli._strip_candidate_marker("  joy") == "joy"


def test_strip_candidate_marker_handles_multi_word_headwords():
    assert cli._strip_candidate_marker("  in someone's eyes") == "in someone's eyes"


def test_strip_candidate_marker_returns_empty_string_for_blank_input():
    assert cli._strip_candidate_marker("") == ""


def test_is_remote_session_true_when_tmux_is_set(monkeypatch):
    monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,12345,0")
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_CLIENT", raising=False)

    assert cli._is_remote_session() is True


def test_is_remote_session_true_when_ssh_tty_is_set(monkeypatch):
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setenv("SSH_TTY", "/dev/pts/3")
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_CLIENT", raising=False)

    assert cli._is_remote_session() is True


def test_is_remote_session_true_when_ssh_connection_is_set(monkeypatch):
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.setenv("SSH_CONNECTION", "10.0.0.1 22 10.0.0.2 22")
    monkeypatch.delenv("SSH_CLIENT", raising=False)

    assert cli._is_remote_session() is True


def test_is_remote_session_true_when_ssh_client_is_set(monkeypatch):
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.setenv("SSH_CLIENT", "10.0.0.1 22 22")

    assert cli._is_remote_session() is True


def test_is_remote_session_false_when_nothing_is_set(monkeypatch):
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_CLIENT", raising=False)

    assert cli._is_remote_session() is False


def test_build_osc52_sequence_base64_encodes_and_wraps_correctly():
    result = cli._build_osc52_sequence("joy")

    assert result == "\x1b]52;c;am95\x07"


def test_copy_via_system_clipboard_uses_the_first_available_tool(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cli.shutil, "which", lambda name: "/usr/bin/xclip" if name == "xclip" else None
    )

    def fake_run(command, input, check, timeout):
        calls.append((command, input))

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    cli._copy_via_system_clipboard("joy")

    assert len(calls) == 1
    assert calls[0][0] == ["xclip", "-selection", "clipboard"]
    assert calls[0][1] == b"joy"


def test_copy_via_system_clipboard_prefers_wl_copy_when_multiple_tools_exist(monkeypatch):
    calls = []
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/usr/bin/{name}")

    def fake_run(command, input, check, timeout):
        calls.append(command)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    cli._copy_via_system_clipboard("joy")

    assert calls == [["wl-copy"]]


def test_copy_via_system_clipboard_does_nothing_when_no_tool_is_available(monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    calls = []
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: calls.append(a))

    cli._copy_via_system_clipboard("joy")

    assert calls == []


def test_run_copy_selection_uses_osc52_for_a_remote_session(monkeypatch):
    monkeypatch.setattr(cli, "_is_remote_session", lambda: True)
    osc52_calls = []
    clipboard_calls = []
    monkeypatch.setattr(cli, "_copy_via_osc52", lambda text: osc52_calls.append(text))
    monkeypatch.setattr(cli, "_copy_via_system_clipboard", lambda text: clipboard_calls.append(text))

    code = cli._run_copy_selection("★ joy")

    assert code == 0
    assert osc52_calls == ["joy"]
    assert clipboard_calls == []


def test_run_copy_selection_uses_system_clipboard_for_a_local_session(monkeypatch):
    monkeypatch.setattr(cli, "_is_remote_session", lambda: False)
    osc52_calls = []
    clipboard_calls = []
    monkeypatch.setattr(cli, "_copy_via_osc52", lambda text: osc52_calls.append(text))
    monkeypatch.setattr(cli, "_copy_via_system_clipboard", lambda text: clipboard_calls.append(text))

    code = cli._run_copy_selection("  joy")

    assert code == 0
    assert clipboard_calls == ["joy"]
    assert osc52_calls == []


def test_run_copy_selection_does_nothing_for_blank_input(monkeypatch):
    osc52_calls = []
    clipboard_calls = []
    monkeypatch.setattr(cli, "_copy_via_osc52", lambda text: osc52_calls.append(text))
    monkeypatch.setattr(cli, "_copy_via_system_clipboard", lambda text: clipboard_calls.append(text))

    code = cli._run_copy_selection("")

    assert code == 0
    assert osc52_calls == []
    assert clipboard_calls == []


def test_main_dispatches_copy_selection(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "_run_copy_selection", lambda marked: calls.append(marked) or 0)

    code = cli.main(["--copy-selection", "★ joy"])

    assert code == 0
    assert calls == ["★ joy"]


def test_main_dispatches_copy_selection_with_empty_string_when_no_argument_given(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "_run_copy_selection", lambda marked: calls.append(marked) or 0)

    code = cli.main(["--copy-selection"])

    assert code == 0
    assert calls == [""]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/shichika/redacted/rev-dictionary && .venv/bin/python -m pytest tests/test_cli.py -v -k "strip_candidate_marker or is_remote_session or osc52 or copy_via_system_clipboard or run_copy_selection or dispatches_copy_selection"`

Expected: all new tests FAIL with `AttributeError` (the functions don't exist yet) or similar.

- [ ] **Step 3: Implement the helper functions and dispatch wiring**

Add these functions to `src/revdict/cli.py`, immediately after the existing `_run_jsonl_query` function (which currently ends around line 255 with `return 0`):

```python
_CLIPBOARD_TOOL_CANDIDATES = [
    ["wl-copy"],
    ["xclip", "-selection", "clipboard"],
    ["xsel", "--clipboard", "--input"],
    ["pbcopy"],
]


def _strip_candidate_marker(marked_headword: str) -> str:
    """format_candidate_line's field 1 (picker.py) is always exactly
    f"{marker} {headword}" where marker is "★" (exact match) or " "
    (regular candidate) -- both one character, so the field is always a
    2-character marker+separator prefix. Python strings are Unicode-
    aware, so this correctly strips the prefix regardless of which
    marker was used, with no special multi-byte handling needed."""
    if len(marked_headword) < 2:
        return marked_headword.strip()
    return marked_headword[2:].strip()


def _is_remote_session() -> bool:
    return bool(
        os.environ.get("TMUX")
        or os.environ.get("SSH_TTY")
        or os.environ.get("SSH_CONNECTION")
        or os.environ.get("SSH_CLIENT")
    )


def _build_osc52_sequence(text: str) -> str:
    """Builds the OSC 52 escape sequence that sets the terminal's
    clipboard to `text`. Pure and testable without a real tty --
    _copy_via_osc52 is the thin wrapper that actually writes this to
    /dev/tty (confirmed necessary during design: writing to stdout
    instead does not reach the pty tmux monitors for OSC 52)."""
    encoded = base64.b64encode(text.encode()).decode()
    return f"\x1b]52;c;{encoded}\x07"


def _copy_via_osc52(text: str) -> None:
    try:
        with open("/dev/tty", "w") as tty:
            tty.write(_build_osc52_sequence(text))
    except OSError:
        pass


def _copy_via_system_clipboard(text: str) -> None:
    for command in _CLIPBOARD_TOOL_CANDIDATES:
        if shutil.which(command[0]) is None:
            continue
        try:
            subprocess.run(command, input=text.encode(), check=True, timeout=2)
        except (subprocess.SubprocessError, OSError):
            continue
        return


def _run_copy_selection(marked_headword: str) -> int:
    headword = _strip_candidate_marker(marked_headword)
    if not headword:
        return 0
    if _is_remote_session():
        _copy_via_osc52(headword)
    else:
        _copy_via_system_clipboard(headword)
    return 0
```

In `main()`, find this existing block:

```python
        if argv and argv[0] == "--jsonl-query":
            query = argv[1] if len(argv) > 1 else ""
            return _run_jsonl_query(query)
```

Add the new branch immediately after it (still inside the same `try:`, before `if not argv:`):

```python
        if argv and argv[0] == "--jsonl-query":
            query = argv[1] if len(argv) > 1 else ""
            return _run_jsonl_query(query)

        if argv and argv[0] == "--copy-selection":
            marked_headword = argv[1] if len(argv) > 1 else ""
            return _run_copy_selection(marked_headword)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/shichika/redacted/rev-dictionary && .venv/bin/python -m pytest tests/test_cli.py -v -k "strip_candidate_marker or is_remote_session or osc52 or copy_via_system_clipboard or run_copy_selection or dispatches_copy_selection"`

Expected: all pass (18 tests).

- [ ] **Step 5: Run the full test suite**

Run: `cd /home/shichika/redacted/rev-dictionary && .venv/bin/python -m pytest tests/ -q`

Expected: all tests pass (138 passed before this change; should be 156 after — 138 + 18 new — 0 failed).

- [ ] **Step 6: Commit**

```bash
cd /home/shichika/redacted/rev-dictionary
git add src/revdict/cli.py tests/test_cli.py
git commit -m "Add --copy-selection: OSC 52 or system-clipboard copy for the live CLI

Additive only -- no existing entry point or behavior changed.
Environment detection (tmux/SSH -> OSC 52, else auto-detected system
clipboard tool) and OSC 52 sequence construction are fully unit-tested;
the two mechanisms' actual I/O (writing to /dev/tty, shelling out to a
real clipboard tool) is validated manually in a later task, since
neither is meaningfully testable without a real terminal."
```

---

## Task 2: Wire `--copy-selection` into the live session's Enter binding

**Files:**
- Modify: `src/revdict/picker.py:190-251` (`build_live_session_args`)
- Test: `tests/test_picker.py`

**Interfaces:**
- Consumes: `revdict --copy-selection "MARKED_TEXT"` from Task 1 — invoked exactly as `{python_executable} -u -m revdict.cli --copy-selection {1}`, matching the existing `reload_command`'s style of invoking `revdict.cli` via `{python_executable} -u -m`.
- Produces: nothing further downstream — this is the last code task; Task 3 is manual validation only.

- [ ] **Step 1: Write the failing test**

Add this test to `tests/test_picker.py`, near the existing `test_build_live_session_args_rebinds_esc_enter_and_arrow_keys` test:

```python
def test_build_live_session_args_enter_also_copies_the_selection():
    args = picker.build_live_session_args(
        preview_dir=Path("/tmp/preview"),
        history_path=Path("/tmp/history"),
        python_executable="/usr/bin/python3",
    )

    joined = " ".join(args)
    assert (
        "enter:execute-silent(echo {q} >> /tmp/history)+clear-query"
        "+execute-silent(/usr/bin/python3 -u -m revdict.cli --copy-selection {1})"
        in joined
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/shichika/redacted/rev-dictionary && .venv/bin/python -m pytest tests/test_picker.py -v -k copies_the_selection`

Expected: FAIL — the current `enter` binding doesn't have the third chained action yet.

- [ ] **Step 3: Modify `build_live_session_args`**

In `src/revdict/picker.py`, find this exact line inside `build_live_session_args`'s returned list:

```python
        f"enter:execute-silent(echo {{q}} >> {history_path})+clear-query",
```

Replace it with:

```python
        f"enter:execute-silent(echo {{q}} >> {history_path})+clear-query"
        f"+execute-silent({python_executable} -u -m revdict.cli --copy-selection {{1}})",
```

(This is a single `--bind` argument value, split across two lines for readability — the `"--bind",` line immediately above it is unchanged, and the two string literals concatenate into one Python string, matching how `reload_command` above it in the same function is already built from concatenated f-strings.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/shichika/redacted/rev-dictionary && .venv/bin/python -m pytest tests/test_picker.py -v -k copies_the_selection`

Expected: PASS.

- [ ] **Step 5: Run the full test suite**

Run: `cd /home/shichika/redacted/rev-dictionary && .venv/bin/python -m pytest tests/ -q`

Expected: all tests pass (156 passed after Task 1; should stay 156 + 1 new = 157, 0 failed).

- [ ] **Step 6: Commit**

```bash
cd /home/shichika/redacted/rev-dictionary
git add src/revdict/picker.py tests/test_picker.py
git commit -m "Chain a clipboard copy onto the live session's Enter binding

Enter's existing behavior (commit query to history, clear the box) is
unchanged -- the copy is a third chained execute-silent action, using
fzf's {1} placeholder to pass the currently-highlighted candidate's
marker+headword field to --copy-selection."
```

---

## Task 3: Manual validation and README update

Same reasoning as this project's other live-CLI validation tasks: the two copy mechanisms' actual I/O (a real OSC 52 write reaching a real terminal's clipboard, a real clipboard tool actually running) cannot be honestly validated by a unit test — it needs a real fzf/tmux session, the same technique already used and proven for this project's earlier live-CLI work.

**Files:**
- Modify: `README.md` (add a short section on clipboard behavior, per explicit user request during design)

- [ ] **Step 1: Drive a real live session inside tmux and confirm the copy works**

```bash
cd /home/shichika/redacted/rev-dictionary
tmux kill-session -t revdictcopytest 2>/dev/null || true
tmux new-session -d -s revdictcopytest -x 200 -y 45
tmux send-keys -t revdictcopytest "revdict" Enter
sleep 1
tmux send-keys -t revdictcopytest "feeling of deep happiness"
sleep 1.5
tmux capture-pane -t revdictcopytest -p | tail -20
```

Expected: the live picker opens and shows real ranked candidates for the query, matching the already-shipped live-typing behavior (unaffected by this change).

- [ ] **Step 2: Press Enter on a highlighted candidate and confirm tmux's own clipboard buffer was set**

```bash
tmux send-keys -t revdictcopytest Enter
sleep 1
tmux list-buffers | head -1
```

Expected: the newest tmux buffer (top of the list) contains the plain headword text of whatever candidate was highlighted when Enter was pressed (no `★`/leading-space marker, no other row fields) — confirming the OSC 52 write reached tmux and tmux's `set-clipboard` correctly intercepted it. Also confirm via `tmux capture-pane -p | tail -5` that the live session is still running normally (query box cleared, per Enter's existing unchanged behavior) — the copy must not have disrupted anything else.

- [ ] **Step 3: Confirm the history-commit behavior is genuinely unchanged**

```bash
tmux send-keys -t revdictcopytest Up
sleep 1
tmux capture-pane -t revdictcopytest -p | tail -5
```

Expected: the previously-committed query ("feeling of deep happiness") is recalled into the query box, exactly matching the already-validated same-session history-recall behavior from the original live-CLI work — confirming this task's change didn't regress it.

- [ ] **Step 4: Clean up**

```bash
tmux send-keys -t revdictcopytest C-c
sleep 0.5
tmux kill-session -t revdictcopytest 2>/dev/null || true
```

- [ ] **Step 5: Add a README section on clipboard behavior**

Read `README.md` first to find the right insertion point (likely near any existing description of the live session's key bindings, or as a new short section). Add a section explaining: Enter copies the highlighted candidate to the clipboard in addition to committing the query to history; over SSH and/or inside tmux this goes through OSC 52 (reaching the clipboard of the device you're physically using, not the remote host's own clipboard) provided your terminal emulator and tmux's `set-clipboard` support it; otherwise it uses whichever of `wl-copy`/`xclip`/`xsel`/`pbcopy` is available locally. Write the exact section content based on what's actually in the README at the time this step runs — read the file first, match its existing tone and heading style, and don't duplicate a section that might already partially exist.

- [ ] **Step 6: Commit**

If Steps 1-3 all passed as described, commit the README update:

```bash
cd /home/shichika/redacted/rev-dictionary
git add README.md
git commit -m "README: document clipboard-copy-on-Enter and its tmux/SSH behavior"
```

If anything in Steps 1-3 failed, do not commit — identify which of Task 1 or Task 2's files needs a fix, apply it, re-run that task's automated tests, and repeat the relevant manual step here before considering Task 3 done.
