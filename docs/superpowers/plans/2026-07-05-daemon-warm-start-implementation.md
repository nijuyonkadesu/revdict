# Background Daemon (Warm-Start Queries) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a background daemon that keeps the index and models warm across queries, so repeated `revdict word` invocations skip the multi-second cold-start of loading a 2GB index and constructing three ML models from scratch.

**Architecture:** A new `revdict daemon start` process loads the index once and serves queries over a Unix domain socket, reusing the existing `search.search()` function unchanged. `revdict word` becomes daemon-aware: try the socket first, transparently auto-spawn a detached daemon if none is running, and fall back to today's direct in-process search if the daemon can't be reached in time. The client-side heavy imports (`revdict.search`, `revdict.data.build_index`) become lazy so the common case (daemon already running) never loads `torch`/`transformers` in the client process at all.

**Tech Stack:** Python stdlib only — `socket` (`AF_UNIX`), `json`, `subprocess`, `signal`, `os`, `pathlib`. No new dependencies.

## Global Constraints

- No new third-party dependencies — everything here is Python stdlib.
- The daemon (`run_server()`) must never need network access — it always sets `HF_HUB_OFFLINE=1` unconditionally (unlike the old cli.py logic, which had to special-case `build-index`; the daemon is never involved in `build-index` at all, so no such special-casing is needed here).
- `cli.py`'s imports of `revdict.search` and `revdict.data.build_index` must be lazy (inside the functions that need them, not at module top), since both transitively import `sentence_transformers`/`torch` — the whole point of this feature is that the common case (daemon already running) never pays that cost client-side.
- No idle-timeout auto-shutdown — this was an explicit user decision. The daemon runs until `revdict daemon stop` or a reboot.
- Socket path: `~/.cache/rev_dictionary/daemon.sock`. PID file: `~/.cache/rev_dictionary/daemon.pid`. Log file (for the auto-spawned detached process's stdout/stderr): `~/.cache/rev_dictionary/daemon.log`.
- A stale socket/PID file (left behind by a crashed daemon, e.g. `kill -9`) must be treated identically to "no daemon running" — cleaned up and a fresh daemon spawned, not treated as an error.

---

### Task 1: Daemon paths and shared env-config helper

**Files:**
- Modify: `src/revdict/paths.py`
- Create: `src/revdict/query_env.py`
- Test: `tests/test_query_env.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `revdict.paths.DAEMON_SOCKET_PATH: Path`, `revdict.paths.DAEMON_PID_PATH: Path`, `revdict.paths.DAEMON_LOG_PATH: Path`; `revdict.query_env.configure_offline_quiet_env() -> None`. Task 2/3's `daemon.py` and Task 4's `cli.py` fallback branch both import these.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_query_env.py
import os

from revdict.query_env import configure_offline_quiet_env


def test_configure_offline_quiet_env_sets_all_three_vars_without_overwriting_existing(
    monkeypatch,
):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)
    monkeypatch.setenv("TRANSFORMERS_VERBOSITY", "debug")

    configure_offline_quiet_env()

    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    # Pre-existing value is respected, not clobbered (setdefault semantics).
    assert os.environ["TRANSFORMERS_VERBOSITY"] == "debug"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_query_env.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'revdict.query_env'`

- [ ] **Step 3: Write the implementation**

```python
# src/revdict/query_env.py
import os


def configure_offline_quiet_env() -> None:
    """Must be called before importing anything that touches
    huggingface_hub/transformers (revdict.search, or anything that imports
    revdict.models.*) -- those libraries snapshot these env vars into
    module-level constants the moment they're first imported, so setting
    them any later than that has no effect."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
```

- [ ] **Step 4: Add the three path constants**

```python
# src/revdict/paths.py -- full file after this change
from pathlib import Path

CACHE_DIR = Path.home() / ".cache" / "rev_dictionary"
INDEX_DIR = CACHE_DIR / "index"
RAW_WIKTIONARY_PATH = CACHE_DIR / "raw-wiktextract-data.jsonl.gz"
DAEMON_SOCKET_PATH = CACHE_DIR / "daemon.sock"
DAEMON_PID_PATH = CACHE_DIR / "daemon.pid"
DAEMON_LOG_PATH = CACHE_DIR / "daemon.log"
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_query_env.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/revdict/paths.py src/revdict/query_env.py tests/test_query_env.py
git commit -m "Add daemon path constants and shared offline-env helper"
```

---

### Task 2: Daemon client-side + protocol logic

**Files:**
- Create: `src/revdict/daemon.py`
- Test: `tests/test_daemon.py`

**Interfaces:**
- Consumes: `revdict.paths.DAEMON_SOCKET_PATH`, `DAEMON_PID_PATH`, `DAEMON_LOG_PATH` (Task 1).
- Produces (all in `src/revdict/daemon.py`): `send_query(query: str, top_n: int, timeout: float = 30.0) -> dict | None`, `ensure_daemon_running(startup_timeout: float = 10.0) -> bool`, `stop_daemon() -> bool`, `daemon_status() -> str`, `_handle_request(request_text: str, search_fn) -> str` (module-private, reused by Task 3's `run_server()` in the same file). Task 3 also reuses the private helpers `_read_pid`, `_process_is_alive`, `_remove_stale_files` defined here. Task 4's `cli.py` consumes `send_query`, `ensure_daemon_running`, `stop_daemon`, `daemon_status`.

This module's top-level imports must stay lightweight (stdlib only: `json`, `os`, `signal`, `socket`, `subprocess`, `sys`, `time`) — `revdict.search` is only ever imported lazily, inside `run_server()` (Task 3), never at this file's module scope. This is what lets `cli.py` import `revdict.daemon` cheaply for the client-side functions without pulling in `torch`/`transformers`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_daemon.py
import json
import os
import socket
import subprocess
import threading
import time

from revdict import daemon


def _run_echo_server(socket_path, response_payload, ready_event):
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    server.listen(1)
    ready_event.set()
    conn, _ = server.accept()
    with conn:
        chunks = []
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
        conn.sendall(json.dumps(response_payload).encode("utf-8"))
    server.close()


def test_send_query_round_trips_a_real_request_over_a_unix_socket(tmp_path, monkeypatch):
    socket_path = tmp_path / "daemon.sock"
    monkeypatch.setattr(daemon, "DAEMON_SOCKET_PATH", socket_path)
    response_payload = {"exact_match": None, "candidates": [{"headword": "joyful"}]}

    ready_event = threading.Event()
    server_thread = threading.Thread(
        target=_run_echo_server, args=(socket_path, response_payload, ready_event)
    )
    server_thread.start()
    ready_event.wait(timeout=2)

    result = daemon.send_query("happy", 10, timeout=2.0)

    server_thread.join(timeout=2)
    assert result == response_payload


def test_send_query_returns_none_when_socket_file_does_not_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(daemon, "DAEMON_SOCKET_PATH", tmp_path / "does-not-exist.sock")

    assert daemon.send_query("happy", 10, timeout=0.5) is None


def test_send_query_returns_none_when_server_reports_an_error(tmp_path, monkeypatch):
    socket_path = tmp_path / "daemon.sock"
    monkeypatch.setattr(daemon, "DAEMON_SOCKET_PATH", socket_path)

    ready_event = threading.Event()
    server_thread = threading.Thread(
        target=_run_echo_server, args=(socket_path, {"error": "boom"}, ready_event)
    )
    server_thread.start()
    ready_event.wait(timeout=2)

    result = daemon.send_query("happy", 10, timeout=2.0)

    server_thread.join(timeout=2)
    assert result is None


def test_handle_request_calls_search_fn_with_parsed_args_and_returns_json_result():
    calls = {}

    def fake_search(query, top_n):
        calls["query"] = query
        calls["top_n"] = top_n
        return {"exact_match": None, "candidates": []}

    request_text = json.dumps({"query": "happy", "top_n": 10})

    response_text = daemon._handle_request(request_text, fake_search)

    assert calls == {"query": "happy", "top_n": 10}
    assert json.loads(response_text) == {"exact_match": None, "candidates": []}


def test_handle_request_returns_error_payload_when_search_fn_raises():
    def failing_search(query, top_n):
        raise RuntimeError("index not loaded")

    request_text = json.dumps({"query": "happy", "top_n": 10})

    response_text = daemon._handle_request(request_text, failing_search)

    payload = json.loads(response_text)
    assert "index not loaded" in payload["error"]


def test_handle_request_returns_error_payload_on_malformed_json():
    response_text = daemon._handle_request("not valid json", lambda query, top_n: {})

    payload = json.loads(response_text)
    assert "error" in payload


def test_ensure_daemon_running_returns_true_immediately_when_already_up(
    tmp_path, monkeypatch
):
    socket_path = tmp_path / "daemon.sock"
    pid_path = tmp_path / "daemon.pid"
    monkeypatch.setattr(daemon, "DAEMON_SOCKET_PATH", socket_path)
    monkeypatch.setattr(daemon, "DAEMON_PID_PATH", pid_path)
    pid_path.write_text(str(os.getpid()))  # this test process is guaranteed alive

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    server.listen(1)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("subprocess.Popen should not be called when already running")

    monkeypatch.setattr(subprocess, "Popen", fail_if_called)

    try:
        assert daemon.ensure_daemon_running(startup_timeout=1.0) is True
    finally:
        server.close()


def test_ensure_daemon_running_spawns_and_waits_for_a_fresh_daemon(tmp_path, monkeypatch):
    socket_path = tmp_path / "daemon.sock"
    pid_path = tmp_path / "daemon.pid"
    log_path = tmp_path / "daemon.log"
    monkeypatch.setattr(daemon, "DAEMON_SOCKET_PATH", socket_path)
    monkeypatch.setattr(daemon, "DAEMON_PID_PATH", pid_path)
    monkeypatch.setattr(daemon, "DAEMON_LOG_PATH", log_path)

    def fake_popen(*args, **kwargs):
        def _start_late_server():
            time.sleep(0.2)
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(str(socket_path))
            server.listen(1)
            server.accept()  # keep it alive long enough for the probe connect

        threading.Thread(target=_start_late_server, daemon=True).start()

        class _FakeProcess:
            pass

        return _FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    assert daemon.ensure_daemon_running(startup_timeout=3.0) is True


def test_ensure_daemon_running_returns_false_if_daemon_never_becomes_ready(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(daemon, "DAEMON_SOCKET_PATH", tmp_path / "daemon.sock")
    monkeypatch.setattr(daemon, "DAEMON_PID_PATH", tmp_path / "daemon.pid")
    monkeypatch.setattr(daemon, "DAEMON_LOG_PATH", tmp_path / "daemon.log")
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: object())

    assert daemon.ensure_daemon_running(startup_timeout=0.5) is False


def test_stop_daemon_returns_false_when_nothing_is_running(tmp_path, monkeypatch):
    monkeypatch.setattr(daemon, "DAEMON_PID_PATH", tmp_path / "daemon.pid")
    monkeypatch.setattr(daemon, "DAEMON_SOCKET_PATH", tmp_path / "daemon.sock")

    assert daemon.stop_daemon() is False


def test_stop_daemon_terminates_a_real_process_and_cleans_up_files(tmp_path, monkeypatch):
    pid_path = tmp_path / "daemon.pid"
    socket_path = tmp_path / "daemon.sock"
    monkeypatch.setattr(daemon, "DAEMON_PID_PATH", pid_path)
    monkeypatch.setattr(daemon, "DAEMON_SOCKET_PATH", socket_path)

    stand_in = subprocess.Popen(["sleep", "30"])
    pid_path.write_text(str(stand_in.pid))
    socket_path.write_text("")  # dummy file standing in for a real socket

    try:
        assert daemon.stop_daemon() is True
        assert not pid_path.exists()
        assert not socket_path.exists()
        stand_in.wait(timeout=5)
        assert stand_in.returncode is not None
    finally:
        if stand_in.poll() is None:
            stand_in.kill()


def test_daemon_status_reports_not_running_with_no_pid_file(tmp_path, monkeypatch):
    monkeypatch.setattr(daemon, "DAEMON_PID_PATH", tmp_path / "daemon.pid")
    monkeypatch.setattr(daemon, "DAEMON_SOCKET_PATH", tmp_path / "daemon.sock")

    assert "not running" in daemon.daemon_status()


def test_daemon_status_reports_running_with_a_live_pid_and_socket(tmp_path, monkeypatch):
    pid_path = tmp_path / "daemon.pid"
    socket_path = tmp_path / "daemon.sock"
    monkeypatch.setattr(daemon, "DAEMON_PID_PATH", pid_path)
    monkeypatch.setattr(daemon, "DAEMON_SOCKET_PATH", socket_path)
    pid_path.write_text(str(os.getpid()))
    socket_path.write_text("")

    status = daemon.daemon_status()

    assert "running" in status
    assert "not running" not in status
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_daemon.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'revdict.daemon'`

- [ ] **Step 3: Write the implementation**

```python
# src/revdict/daemon.py
import json
import os
import signal
import socket
import subprocess
import sys
import time

from revdict.paths import DAEMON_LOG_PATH, DAEMON_PID_PATH, DAEMON_SOCKET_PATH


def _read_pid() -> int | None:
    if not DAEMON_PID_PATH.exists():
        return None
    try:
        return int(DAEMON_PID_PATH.read_text().strip())
    except (ValueError, OSError):
        return None


def _process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _remove_stale_files() -> None:
    for path in (DAEMON_SOCKET_PATH, DAEMON_PID_PATH):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def send_query(query: str, top_n: int, timeout: float = 30.0) -> dict | None:
    if not DAEMON_SOCKET_PATH.exists():
        return None
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(str(DAEMON_SOCKET_PATH))
            request = json.dumps({"query": query, "top_n": top_n})
            sock.sendall(request.encode("utf-8"))
            sock.shutdown(socket.SHUT_WR)
            chunks = []
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
            response_text = b"".join(chunks).decode("utf-8")
    except OSError:
        return None

    if not response_text.strip():
        return None
    payload = json.loads(response_text)
    if "error" in payload:
        return None
    return payload


def ensure_daemon_running(startup_timeout: float = 10.0) -> bool:
    pid = _read_pid()
    if pid is not None and _process_is_alive(pid) and DAEMON_SOCKET_PATH.exists():
        return True
    _remove_stale_files()

    DAEMON_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DAEMON_LOG_PATH, "a") as log_file:
        subprocess.Popen(
            [sys.executable, "-m", "revdict.cli", "daemon", "start"],
            stdout=log_file,
            stderr=log_file,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

    deadline = time.time() + startup_timeout
    while time.time() < deadline:
        if DAEMON_SOCKET_PATH.exists():
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                    probe.settimeout(0.5)
                    probe.connect(str(DAEMON_SOCKET_PATH))
                return True
            except OSError:
                pass
        time.sleep(0.1)
    return False


def stop_daemon() -> bool:
    pid = _read_pid()
    if pid is None or not _process_is_alive(pid):
        _remove_stale_files()
        return False
    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + 5.0
    while time.time() < deadline and _process_is_alive(pid):
        time.sleep(0.1)
    _remove_stale_files()
    return True


def daemon_status() -> str:
    pid = _read_pid()
    if pid is not None and _process_is_alive(pid) and DAEMON_SOCKET_PATH.exists():
        return f"revdict daemon is running (pid {pid})."
    return "revdict daemon is not running."


def _handle_request(request_text: str, search_fn) -> str:
    try:
        request = json.loads(request_text)
        result = search_fn(request["query"], top_n=request["top_n"])
    except Exception as error:
        return json.dumps({"error": str(error)})
    return json.dumps(result)
```

Note: `test_stop_daemon_terminates_a_real_process_and_cleans_up_files` uses `subprocess.Popen(["sleep", "30"])` as a stand-in for a real daemon process (just needs *some* real, killable process — doesn't need to be an actual revdict daemon for this test's purpose). This requires the `sleep` binary to exist, which it does on any Linux dev machine including this one.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_daemon.py -v`
Expected: PASS (all 12 tests)

- [ ] **Step 5: Commit**

```bash
git add src/revdict/daemon.py tests/test_daemon.py
git commit -m "Add daemon client-side lifecycle and protocol logic"
```

---

### Task 3: Daemon server loop (`run_server`)

**Files:**
- Modify: `src/revdict/daemon.py`

**Interfaces:**
- Consumes: `revdict.query_env.configure_offline_quiet_env` (Task 1), `revdict.search.search` (lazily imported — do not add this to the module's top-level imports), the private helpers `_remove_stale_files`, `_handle_request` (Task 2, same file).
- Produces: `run_server() -> None`. Task 4's `cli.py` calls this for the `revdict daemon start` subcommand.

This function is not unit-tested — like `search()`'s real model-loading path and `build()`'s real corpus-building path elsewhere in this codebase, it requires a real built index and real models to exercise meaningfully. It's validated manually in Task 5. Only its already-tested building block (`_handle_request`, Task 2) carries automated coverage.

- [ ] **Step 1: Write the implementation directly (no TDD — see rationale above)**

Append to `src/revdict/daemon.py`:

```python
def run_server() -> None:
    from revdict.query_env import configure_offline_quiet_env

    configure_offline_quiet_env()
    from revdict import search as search_mod

    DAEMON_SOCKET_PATH.parent.mkdir(parents=True, exist_ok=True)
    _remove_stale_files()

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(DAEMON_SOCKET_PATH))
    server.listen(5)
    DAEMON_PID_PATH.write_text(str(os.getpid()))

    def _cleanup_and_exit(signum, frame):
        server.close()
        _remove_stale_files()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _cleanup_and_exit)
    signal.signal(signal.SIGINT, _cleanup_and_exit)

    print(f"revdict daemon listening on {DAEMON_SOCKET_PATH} (pid {os.getpid()})")

    try:
        while True:
            conn, _ = server.accept()
            try:
                with conn:
                    chunks = []
                    while True:
                        chunk = conn.recv(65536)
                        if not chunk:
                            break
                        chunks.append(chunk)
                    request_text = b"".join(chunks).decode("utf-8")
                    response_text = _handle_request(request_text, search_mod.search)
                    conn.sendall(response_text.encode("utf-8"))
            except Exception as error:
                print(f"revdict daemon: error handling a request: {error}")
    finally:
        _remove_stale_files()
```

The full file's import block at the top should now read:

```python
import json
import os
import signal
import socket
import subprocess
import sys
import time

from revdict.paths import DAEMON_LOG_PATH, DAEMON_PID_PATH, DAEMON_SOCKET_PATH
```

(unchanged from Task 2 — confirm no new top-level imports were added; `configure_offline_quiet_env` and `search_mod` must only appear as the two lazy imports inside `run_server()`'s body.)

- [ ] **Step 2: Confirm the module still imports cheaply**

Run: `.venv/bin/python -c "import time; start = time.time(); from revdict import daemon; print(f'{time.time() - start:.3f}s')"`
Expected: prints a small number (well under 1 second) — confirms importing `revdict.daemon` does not transitively import `torch`/`sentence_transformers` (which would take noticeably longer and print HF/transformers import-time output).

- [ ] **Step 3: Run the full test suite to confirm no regressions**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: all tests pass (Task 2's 12 daemon tests + everything from before, unaffected)

- [ ] **Step 4: Commit**

```bash
git add src/revdict/daemon.py
git commit -m "Add daemon server loop (run_server)"
```

---

### Task 4: CLI integration — daemon subcommands + lazy fallback

**Files:**
- Modify: `src/revdict/cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: `revdict.daemon.send_query`, `ensure_daemon_running`, `stop_daemon`, `daemon_status`, `run_server` (Tasks 2-3); lazily, `revdict.search.search` and `revdict.data.build_index.build` (existing modules, now imported lazily instead of at module top).
- Produces: updated `main()` routing `revdict daemon start|stop|status`; `_get_search_result(query: str, top_n: int) -> dict` replacing the old direct `search_mod.search(...)` call inside `_run_query`; `_build_index(skip_confirm: bool) -> None` wrapping the lazy `build` import, which also checks `daemon.daemon_status()` after `build()` completes and prints a reminder if a daemon is still running with the now-stale index (the spec's build-index/daemon coordination requirement); `_daemon_start()`, `_daemon_stop()`, `_daemon_status()` wrapping the corresponding `daemon` module functions.

This task removes the old module-top env-var configuration block entirely (that responsibility now lives in `query_env.py`, invoked lazily by `daemon.py`'s `run_server()` and by `_get_search_result`'s fallback branch) and removes the top-level `from revdict import search as search_mod` / `from revdict.data.build_index import build` imports, replacing them with lazy imports inside small wrapper functions — both because those imports are what pulls in `torch`/`transformers`, and because a *lazy* `from ... import` statement inside a function creates a fresh local binding each call, which would NOT pick up `monkeypatch.setattr(cli, "build", ...)`-style test patching; wrapping each in its own module-level function (`_build_index`, `_get_search_result`) keeps them independently mockable in tests, matching the existing pattern already used for `_index_exists`/`_fzf_missing` in this file.

- [ ] **Step 1: Write the failing tests (full replacement of `tests/test_cli.py`)**

The two existing tests exercising the old module-load-time env-var behavior (`test_module_load_sets_offline_and_quiet_env_vars_for_a_query_invocation`, `test_module_load_does_not_force_offline_mode_for_build_index_invocation`) are removed — `cli.py` no longer configures env vars at module load at all, since it no longer eagerly imports anything that needs them; that responsibility moved to `query_env.py`/`daemon.py` (already tested in Tasks 1-2). All five tests that patched `cli.search_mod.search` are updated to patch `cli._get_search_result` instead. `test_main_routes_the_build_index_subcommand` is updated to patch `cli._build_index` instead of `cli.build`. Three new tests cover the daemon subcommands and `_get_search_result`'s daemon-then-fallback behavior.

```python
# tests/test_cli.py -- full file after this change
import sys

from revdict import cli
from revdict.picker import PickerError


def test_main_prints_error_and_returns_1_when_index_missing(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_index_exists", lambda: False)

    code = cli.main(["happy"])

    captured = capsys.readouterr()
    assert code == 1
    assert "build-index" in captured.out


def test_main_routes_the_build_index_subcommand(monkeypatch):
    called = {}

    def fake_build_index(skip_confirm):
        called["skip_confirm"] = skip_confirm

    monkeypatch.setattr(cli, "_build_index", fake_build_index)

    code = cli.main(["build-index", "--yes"])

    assert code == 0
    assert called["skip_confirm"] is True


def test_build_index_warns_when_a_daemon_is_still_running_afterward(monkeypatch, capsys):
    import revdict.data.build_index as build_index_module

    monkeypatch.setattr(build_index_module, "build", lambda skip_confirm: None)
    monkeypatch.setattr(cli.daemon, "daemon_status", lambda: "revdict daemon is running (pid 1).")

    cli._build_index(skip_confirm=True)

    captured = capsys.readouterr()
    assert "daemon stop" in captured.out


def test_build_index_says_nothing_when_no_daemon_is_running(monkeypatch, capsys):
    import revdict.data.build_index as build_index_module

    monkeypatch.setattr(build_index_module, "build", lambda skip_confirm: None)
    monkeypatch.setattr(cli.daemon, "daemon_status", lambda: "revdict daemon is not running.")

    cli._build_index(skip_confirm=True)

    captured = capsys.readouterr()
    assert captured.out == ""


def test_main_routes_daemon_start_to_run_server(monkeypatch):
    called = {"ran": False}
    monkeypatch.setattr(cli, "_daemon_start", lambda: called.__setitem__("ran", True))

    code = cli.main(["daemon", "start"])

    assert code == 0
    assert called["ran"] is True


def test_main_routes_daemon_stop_and_reports_when_nothing_was_running(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_daemon_stop", lambda: False)

    code = cli.main(["daemon", "stop"])

    captured = capsys.readouterr()
    assert code == 0
    assert "not running" in captured.out.lower()


def test_main_routes_daemon_stop_and_reports_success(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_daemon_stop", lambda: True)

    code = cli.main(["daemon", "stop"])

    captured = capsys.readouterr()
    assert code == 0
    assert "stopped" in captured.out.lower()


def test_main_routes_daemon_status(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_daemon_status", lambda: "revdict daemon is running (pid 123).")

    code = cli.main(["daemon", "status"])

    captured = capsys.readouterr()
    assert code == 0
    assert "pid 123" in captured.out


def test_main_daemon_subcommand_with_unknown_or_missing_action_prints_usage(capsys):
    code = cli.main(["daemon"])

    captured = capsys.readouterr()
    assert code == 1
    assert "start" in captured.out and "stop" in captured.out and "status" in captured.out


def test_get_search_result_uses_daemon_when_it_answers(monkeypatch):
    monkeypatch.setattr(
        cli.daemon, "send_query", lambda query, top_n: {"exact_match": None, "candidates": []}
    )

    def fail_if_called():
        raise AssertionError("ensure_daemon_running should not be called if send_query answers")

    monkeypatch.setattr(cli.daemon, "ensure_daemon_running", fail_if_called)

    result = cli._get_search_result("happy", 10)

    assert result == {"exact_match": None, "candidates": []}


def test_get_search_result_starts_daemon_and_retries_when_first_attempt_fails(monkeypatch):
    attempts = {"count": 0}

    def fake_send_query(query, top_n):
        attempts["count"] += 1
        if attempts["count"] == 1:
            return None
        return {"exact_match": None, "candidates": [{"headword": "joyful"}]}

    monkeypatch.setattr(cli.daemon, "send_query", fake_send_query)
    monkeypatch.setattr(cli.daemon, "ensure_daemon_running", lambda: True)

    result = cli._get_search_result("happy", 10)

    assert attempts["count"] == 2
    assert result == {"exact_match": None, "candidates": [{"headword": "joyful"}]}


def test_get_search_result_falls_back_to_local_search_when_daemon_unavailable(monkeypatch):
    monkeypatch.setattr(cli.daemon, "send_query", lambda query, top_n: None)
    monkeypatch.setattr(cli.daemon, "ensure_daemon_running", lambda: False)

    fake_result = {"exact_match": None, "candidates": [{"headword": "fallback-used"}]}
    monkeypatch.setattr(cli, "_local_search_fallback", lambda query, top_n: fake_result)

    result = cli._get_search_result("happy", 10)

    assert result == fake_result


def test_run_query_warns_and_returns_0_on_blank_query(capsys):
    code = cli._run_query("   ", top_n=10, interactive=False)

    captured = capsys.readouterr()
    assert code == 0
    assert "word or phrase" in captured.out


def test_run_query_prints_exact_match_emotion_and_synonyms_when_present(monkeypatch, capsys):
    """Fix 1 + Fix 2: the exact-match table must show an emotion badge per
    sense (the headline feature that was previously silently dropped for the
    exact match) and synonyms when present, skipping the synonyms line
    cleanly when they're absent."""
    fake_result = {
        "exact_match": {
            "headword": "happy",
            "senses": [
                {
                    "pos": "adjective",
                    "definition": "feeling great pleasure",
                    "examples": [],
                    "source": "wordnet",
                    "synonyms": ["glad", "content"],
                    "label": "joy",
                    "polarity": "positive",
                },
                {
                    "pos": "adjective",
                    "definition": "willing to do something",
                    "examples": [],
                    "source": "wiktionary",
                    "synonyms": None,
                    "label": "neutral",
                    "polarity": "neutral",
                },
            ],
        },
        "candidates": [],
    }
    monkeypatch.setattr(cli, "_get_search_result", lambda query, top_n: fake_result)

    code = cli._run_query("happy", top_n=10, interactive=False)

    captured = capsys.readouterr()
    assert code == 0
    assert "joy · positive" in captured.out
    assert "neutral · neutral" in captured.out
    assert "glad" in captured.out and "content" in captured.out
    assert "Synonyms: \n" not in captured.out


def test_run_query_prints_static_results_when_not_interactive(monkeypatch, capsys):
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

    code = cli._run_query("happy", top_n=10, interactive=False)

    captured = capsys.readouterr()
    assert code == 0
    assert "joyful" in captured.out


_FAKE_INTERACTIVE_RESULT = {
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


def test_run_query_falls_back_to_static_results_when_fzf_missing(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_get_search_result", lambda query, top_n: _FAKE_INTERACTIVE_RESULT)
    monkeypatch.setattr(cli, "run_picker", lambda candidates, exact_match: None)
    monkeypatch.setattr(cli, "_fzf_missing", lambda: True)

    code = cli._run_query("happy", top_n=10, interactive=True)

    captured = capsys.readouterr()
    assert code == 0
    assert "joyful" in captured.out


def test_run_query_returns_quietly_when_user_cancels_the_picker(monkeypatch, capsys):
    """fzf present, user just pressed Esc/Ctrl-C (run_picker -> None) --
    this is a deliberate cancellation, not an error, so nothing should be
    printed and there should be no static-table fallback."""
    monkeypatch.setattr(cli, "_get_search_result", lambda query, top_n: _FAKE_INTERACTIVE_RESULT)
    monkeypatch.setattr(cli, "run_picker", lambda candidates, exact_match: None)
    monkeypatch.setattr(cli, "_fzf_missing", lambda: False)

    code = cli._run_query("happy", top_n=10, interactive=True)

    captured = capsys.readouterr()
    assert code == 0
    assert captured.out == ""


def test_run_query_falls_back_to_static_results_and_warns_on_picker_runtime_error(
    monkeypatch, capsys
):
    """Root requirement of Fix 3: a genuine fzf runtime failure (fzf present
    but erroring, e.g. no controlling terminal) must never produce zero
    output -- it must fall back to the static table and mention the error."""
    monkeypatch.setattr(cli, "_get_search_result", lambda query, top_n: _FAKE_INTERACTIVE_RESULT)

    def fake_run_picker(candidates, exact_match):
        raise PickerError(2, "inappropriate ioctl for device")

    monkeypatch.setattr(cli, "run_picker", fake_run_picker)
    monkeypatch.setattr(cli, "_fzf_missing", lambda: False)

    code = cli._run_query("happy", top_n=10, interactive=True)

    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.strip() != ""
    assert "joyful" in captured.out
    assert "ioctl" in captured.out or "fzf" in captured.out.lower()


def test_main_with_no_args_checks_isatty_before_going_interactive(monkeypatch):
    """The argv-parsing path already guards interactive with
    `sys.stdout.isatty()`; the no-arg path previously set interactive=True
    unconditionally. Both paths must apply the same guard."""
    monkeypatch.setattr(cli, "_index_exists", lambda: True)
    monkeypatch.setattr(cli.console, "input", lambda prompt: "happy")

    calls = {}

    def fake_run_query(query, top_n, interactive):
        calls["interactive"] = interactive
        return 0

    monkeypatch.setattr(cli, "_run_query", fake_run_query)

    class _NonTtyStdout:
        def isatty(self):
            return False

    monkeypatch.setattr(cli.sys, "stdout", _NonTtyStdout())

    code = cli.main([])

    assert code == 0
    assert calls["interactive"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_cli.py -v`
Expected: FAIL — `cli._build_index`, `cli._daemon_start`, `cli._daemon_stop`, `cli._daemon_status`, `cli._get_search_result`, `cli._local_search_fallback`, `cli.daemon` don't exist yet (`AttributeError`), and `cli.main(["daemon", ...])` isn't routed yet.

- [ ] **Step 3: Write the implementation (full replacement of `src/revdict/cli.py`)**

```python
# src/revdict/cli.py -- full file after this change
import shutil
import sys

from rich.console import Console
from rich.table import Table

from revdict import daemon
from revdict.paths import INDEX_DIR
from revdict.picker import PickerError, run_picker

console = Console()


def _index_exists() -> bool:
    return (INDEX_DIR / "embeddings.npy").exists()


def _fzf_missing() -> bool:
    return shutil.which("fzf") is None


def _print_no_index_error() -> None:
    console.print("[bold red]No index found.[/bold red] Run: [bold]revdict build-index[/bold]")


def _print_static_results(result: dict) -> None:
    if result["exact_match"] is not None:
        table = Table(title=f"Exact match — {result['exact_match']['headword']}")
        table.add_column("POS")
        table.add_column("Definition")
        table.add_column("Emotion")
        table.add_column("Synonyms")
        for sense in result["exact_match"]["senses"]:
            synonyms = sense.get("synonyms")
            table.add_row(
                sense["pos"],
                sense["definition"],
                f"{sense['label']} · {sense['polarity']}",
                ", ".join(synonyms) if synonyms else "",
            )
        console.print(table)

    table = Table(title="Related words you might mean")
    table.add_column("#")
    table.add_column("Word")
    table.add_column("Definition")
    table.add_column("Emotion")
    table.add_column("Relevance")
    for position, candidate in enumerate(result["candidates"], start=1):
        table.add_row(
            str(position),
            candidate["headword"],
            candidate["definition"],
            f"{candidate['label']} · {candidate['polarity']}",
            f"{candidate['relevance']}%",
        )
    console.print(table)


def _local_search_fallback(query: str, top_n: int) -> dict:
    from revdict.query_env import configure_offline_quiet_env

    configure_offline_quiet_env()
    from revdict import search as search_mod

    return search_mod.search(query, top_n=top_n)


def _get_search_result(query: str, top_n: int) -> dict:
    result = daemon.send_query(query, top_n)
    if result is not None:
        return result
    if daemon.ensure_daemon_running():
        result = daemon.send_query(query, top_n)
        if result is not None:
            return result
    return _local_search_fallback(query, top_n)


def _build_index(skip_confirm: bool) -> None:
    from revdict.data.build_index import build

    build(skip_confirm=skip_confirm)

    if "is running" in daemon.daemon_status():
        console.print(
            "[yellow]A revdict daemon is still running with the old index loaded — "
            "run `revdict daemon stop` so your next query picks up the refreshed "
            "data.[/yellow]"
        )


def _daemon_start() -> None:
    daemon.run_server()


def _daemon_stop() -> bool:
    return daemon.stop_daemon()


def _daemon_status() -> str:
    return daemon.daemon_status()


def _run_query(query: str, top_n: int, interactive: bool) -> int:
    if not query.strip():
        console.print("[yellow]Please enter a word or phrase.[/yellow]")
        return 0

    result = _get_search_result(query, top_n)

    if interactive:
        try:
            selected = run_picker(result["candidates"], result["exact_match"])
        except PickerError as error:
            console.print(
                f"[yellow]fzf exited unexpectedly (code {error.returncode}): "
                f"{error.stderr.strip() or 'no error output'}[/yellow]"
            )
            _print_static_results(result)
            return 0
        if selected is None and _fzf_missing():
            _print_static_results(result)
            return 0
        if selected:
            print(selected)
        return 0

    _print_static_results(result)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    if argv and argv[0] == "build-index":
        _build_index(skip_confirm="--yes" in argv)
        return 0

    if argv and argv[0] == "daemon":
        action = argv[1] if len(argv) > 1 else None
        if action == "start":
            _daemon_start()
            return 0
        if action == "stop":
            if _daemon_stop():
                console.print("Daemon stopped.")
            else:
                console.print("[yellow]No daemon was running.[/yellow]")
            return 0
        if action == "status":
            console.print(_daemon_status())
            return 0
        console.print("[red]Usage: revdict daemon start|stop|status[/red]")
        return 1

    if not argv:
        if not _index_exists():
            _print_no_index_error()
            return 1
        query = console.input("[bold]> [/bold]")
        return _run_query(query, top_n=30, interactive=sys.stdout.isatty())

    no_interactive = "--no-interactive" in argv
    args = [arg for arg in argv if arg != "--no-interactive"]

    top_n = 30
    if "-n" in args:
        position = args.index("-n")
        top_n = int(args[position + 1])
        args = args[:position] + args[position + 2 :]

    query = " ".join(args)

    if not _index_exists():
        _print_no_index_error()
        return 1

    interactive = not no_interactive and sys.stdout.isatty()
    return _run_query(query, top_n, interactive)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_cli.py -v`
Expected: PASS (all tests, including the new daemon-routing and `_get_search_result` tests)

- [ ] **Step 5: Confirm the client-side import stays cheap**

Run: `.venv/bin/python -c "import time; start = time.time(); from revdict import cli; print(f'{time.time() - start:.3f}s')"`
Expected: a small number, comparable to Task 3's `revdict.daemon` import-time check — confirms `cli.py`'s module scope no longer transitively imports `torch`/`sentence_transformers`.

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: all tests pass

- [ ] **Step 7: Commit**

```bash
git add src/revdict/cli.py tests/test_cli.py
git commit -m "Wire cli.py to the daemon: daemon subcommands, lazy fallback imports"
```

---

### Task 5: Manual end-to-end validation

No new unit tests — this validates the real daemon lifecycle against the real built index, the way Task 14 of the original plan validated the base pipeline.

**Files:**
- None created.

- [ ] **Step 1: Confirm no daemon is currently running, then run a cold query and time it**

```bash
.venv/bin/revdict daemon status
```

Expected: "revdict daemon is not running." (if a daemon happens to already be running from earlier manual testing, stop it first: `.venv/bin/revdict daemon stop`)

```bash
time .venv/bin/revdict "happy" --no-interactive -n 5
```

Expected: prints results as before; this also transparently spawned a daemon in the background since none was running (this run pays the cold-start cost, same as before this feature).

- [ ] **Step 2: Confirm the daemon actually started**

```bash
.venv/bin/revdict daemon status
```

Expected: "revdict daemon is running (pid N)."

```bash
cat ~/.cache/rev_dictionary/daemon.log
```

Expected: shows "revdict daemon listening on .../daemon.sock (pid N)" with no errors/tracebacks.

- [ ] **Step 3: Confirm a second query is fast and uses the warm daemon**

```bash
time .venv/bin/revdict "joy" --no-interactive -n 5
```

Expected: noticeably faster than Step 1's cold run (no "Loading weights" progress bars, no HF Hub warning — those only ever appear in the daemon's own log now, from its one-time startup, not on every query) and correct results.

- [ ] **Step 4: Confirm the interactive fzf picker still works through the daemon**

```bash
.venv/bin/revdict "happy"
```

Expected: fzf opens quickly (no visible cold-start delay), same picker behavior as before this feature (live preview, `?` to toggle, arrow-key navigation, Enter to select).

- [ ] **Step 5: Stop the daemon and confirm the next query transparently restarts one**

```bash
.venv/bin/revdict daemon stop
```

Expected: "Daemon stopped."

```bash
.venv/bin/revdict daemon status
```

Expected: "revdict daemon is not running."

```bash
time .venv/bin/revdict "cheerful" --no-interactive -n 3
```

Expected: correct results; this run cold-starts again (spawns a fresh daemon), confirming the auto-spawn path works repeatably, not just once.

- [ ] **Step 6: Confirm `revdict build-index` still works unaffected and doesn't route through the daemon**

Not required to run the full ~25 minute rebuild again — just confirm the command still dispatches correctly and doesn't error before reaching the benchmark step:

```bash
timeout 90 .venv/bin/revdict build-index 2>&1 | head -20
```

Expected: prints "Loading WordNet + SentiWordNet...", proceeds through Wiktionary loading/merging without error, reaches the benchmark line — confirms `_build_index`'s lazy import still works correctly. Let the `timeout 90` cut it off before the confirmation prompt (don't answer `y` — no need to actually rebuild).

- [ ] **Step 7: Stop the daemon left over from Step 5 to leave a clean environment**

```bash
.venv/bin/revdict daemon stop
```

- [ ] **Step 8: Record any issues, then commit the validation note**

If any step didn't match its expected output, note it and do not proceed to treat the feature as complete — fix the underlying issue and re-validate the affected steps.

```bash
git commit --allow-empty -m "Validate daemon warm-start end-to-end manually"
```

---

### Task 6: Add a concise README

Added per an explicit user request during implementation, orthogonal to the daemon feature itself but grouped into this plan's execution at the user's direction. Written to describe the project's install/usage in terms of the `uv`-based workflow that Task 7 (next) makes literally true — by the time Task 7 finishes, every command shown here works exactly as written.

**Files:**
- Create: `README.md`

**Interfaces:** None — this is documentation, not code.

- [ ] **Step 1: Write `README.md`**

```markdown
# revdict

A local, offline reverse-dictionary CLI. Give it a word and it shows the
standard definition; give it a phrase describing a meaning and it suggests
matching words — every result tagged with an emotion/connotation badge.
Runs entirely on-device (WordNet, Wiktionary, and small local ML models),
no API keys, no per-query network calls.

## Requirements

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- [`fzf`](https://github.com/junegunn/fzf) for the interactive picker (optional — falls back to a plain printed list if absent)

## Install

```bash
uv sync --all-extras
```

This creates `.venv/` and installs everything, including a CPU-only PyTorch
build (no CUDA download, regardless of platform).

## First-time setup

Build the local search index once (downloads WordNet, a Wiktionary extract,
and a few small ML models; takes on the order of 30 minutes depending on
your machine):

```bash
.venv/bin/revdict build-index
```

Re-run this any time you want to refresh the underlying data.

## Usage

```bash
# Interactive picker (fzf) — arrow keys + live preview, ? to toggle preview,
# Enter to print the selected word
.venv/bin/revdict "happy"
.venv/bin/revdict "feeling of intense annoyance"

# One-shot, plain-text output (no fzf) — good for scripting
.venv/bin/revdict "happy" --no-interactive

# Show more/fewer candidates (default 30)
.venv/bin/revdict "happy" --no-interactive -n 10
```

The first query in a while starts a background daemon that keeps the index
and models warm in memory, so subsequent queries are fast:

```bash
revdict daemon status   # is it running?
revdict daemon stop     # stop it (e.g. before rebuilding the index)
```

If you rebuild the index while a daemon is running, it keeps serving the old
data until you stop it — `build-index` will remind you if this applies.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "Add concise README with install and usage instructions"
```

---

### Task 7: Migrate dependency management to uv (with a committed lock file)

Added per an explicit user request during implementation. Confirmed
beforehand (see the plan's own investigation, not repeated here) that this
migration touches only the Python packaging/venv layer — it does not
require re-downloading the Wiktionary dump, re-running WordNet/model
downloads, or rebuilding the search index, all of which live under
`~/.cache/rev_dictionary/` and the Hugging Face cache, entirely independent
of which tool manages `.venv`. A real `uv lock` resolution was verified to
correctly select `torch==2.12.1+cpu` from the CPU-only PyTorch index with
zero `nvidia-*` packages pulled in, confirming this exact configuration
works before writing it into the real project.

**Files:**
- Modify: `pyproject.toml`
- Create: `uv.lock` (generated by `uv lock`, committed to version control)

**Interfaces:** None — this is a tooling/dependency-management change, not application code. No existing module's public interface changes.

- [ ] **Step 1: Update `pyproject.toml`**

Add `torch` as an explicit dependency (previously it was deliberately left out of `dependencies` and installed manually via a separate `pip install torch --index-url ...` step before `pip install -e ".[dev]"`, to avoid pip pulling the default CUDA-bundled wheel — `uv`'s per-package index configuration below replaces that workaround with a declarative, lock-fileable equivalent), and add the two new `[tool.uv...]` sections:

```toml
# pyproject.toml -- full file after this change
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "revdict"
version = "0.1.0"
description = "Local offline reverse-dictionary CLI"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.9",
    "nltk>=3.9",
    "sentence-transformers>=5.0",
    "transformers>=5.0",
    "nrclex>=4.1",
    "numpy>=1.26",
    "rich>=13.9",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[project.scripts]
revdict = "revdict.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

- [ ] **Step 2: Generate the lock file**

```bash
uv lock
```

Expected: creates `uv.lock`; prints something like "Resolved N packages" with no errors.

- [ ] **Step 3: Verify the lock file resolved the CPU-only torch build with no CUDA packages**

```bash
grep 'name = "torch"' uv.lock
grep -i "nvidia" uv.lock
```

Expected: the `torch` lines show `source = { registry = "https://download.pytorch.org/whl/cpu" }` and a `+cpu` version suffix (e.g. `2.12.1+cpu`) for non-macOS platforms; the `nvidia` grep prints nothing (no matches).

- [ ] **Step 4: Replace the pip-created venv with a uv-managed one**

The existing `.venv/` was created by plain `python -m venv` + `pip install` in an earlier task; remove it and let `uv sync` create a fresh one matching the lock file exactly (avoids any stale packages left over from the old install method):

```bash
rm -rf .venv
uv sync --all-extras
```

Expected: creates a new `.venv/`, installs all dependencies per `uv.lock` including the CPU-only torch build, no errors.

- [ ] **Step 5: Verify torch is the CPU-only build in the real venv**

```bash
.venv/bin/python -c "import torch; print(torch.__version__)"
```

Expected: prints a version ending in `+cpu` (e.g. `2.12.1+cpu`), confirming no CUDA build was installed.

- [ ] **Step 6: Run the full test suite in the new venv**

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: all tests pass (same count as before this task — this migration changes nothing about application code, only how dependencies are installed).

- [ ] **Step 7: Smoke-test the actual CLI still works**

```bash
.venv/bin/revdict "happy" --no-interactive -n 3
```

Expected: prints real results (the search index built in earlier tasks is untouched by this migration, so this should work immediately without rebuilding).

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "Migrate dependency management to uv, commit lock file"
```

Note: `.venv/` remains gitignored (unchanged) — only `pyproject.toml`'s changes and the new `uv.lock` are tracked, per the user's explicit request to commit the lock file.
