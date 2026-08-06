# src/revdict/daemon.py
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
import fcntl

from revdict.paths import (
    DAEMON_LOCK_PATH,
    DAEMON_LOG_PATH,
    DAEMON_PID_PATH,
    DAEMON_SOCKET_PATH,
    DAEMON_START_LOCK_PATH,
)
from revdict.progress import ProgressReporter


PROGRESS_PROTOCOL = "progress-v1"


def _query_payload(
    query: str,
    top_n: int,
    sort_mode: str | None = None,
    category: str | None = None,
    syllables: int | None = None,
    primary_vowel: str | None = None,
    rhymes_with: str | None = None,
    sounds_like: str | None = None,
    meter: str | None = None,
) -> dict:
    return {
        "query": query,
        "top_n": top_n,
        "sort": sort_mode,
        "category": category,
        "syllables": syllables,
        "primary_vowel": primary_vowel,
        "rhymes_with": rhymes_with,
        "sounds_like": sounds_like,
        "meter": meter,
    }


def _search_kwargs(request: dict) -> dict:
    return {
        "top_n": request["top_n"],
        "sort_mode": request.get("sort"),
        "category": request.get("category"),
        "syllables": request.get("syllables"),
        "primary_vowel": request.get("primary_vowel"),
        "rhymes_with": request.get("rhymes_with"),
        "sounds_like": request.get("sounds_like"),
        "meter": request.get("meter"),
    }


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


def _acquire_lock(path) -> int | None:
    """Take a nonblocking, process-lifetime advisory lock at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        return None
    return descriptor


def _release_lock(descriptor: int) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _acquire_server_lock() -> int | None:
    """Acquire the lifetime lock before an expensive daemon model load.

    A UNIX socket is unavailable while the server imports its search index, so
    it cannot by itself prevent sibling launchers from loading that index in
    parallel. ``flock`` is released automatically if the owner dies.
    """
    return _acquire_lock(DAEMON_LOCK_PATH)


def _wait_for_socket(startup_timeout: float) -> bool:
    deadline = time.time() + startup_timeout
    while time.time() < deadline:
        if _socket_is_reachable():
            return True
        time.sleep(0.1)
    return False


def _socket_is_reachable() -> bool:
    if not DAEMON_SOCKET_PATH.exists():
        return False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.5)
            probe.connect(str(DAEMON_SOCKET_PATH))
        return True
    except OSError:
        return False


def send_query(
    query: str,
    top_n: int,
    sort_mode: str | None = None,
    category: str | None = None,
    syllables: int | None = None,
    primary_vowel: str | None = None,
    rhymes_with: str | None = None,
    sounds_like: str | None = None,
    meter: str | None = None,
    timeout: float = 30.0,
) -> dict | None:
    if not DAEMON_SOCKET_PATH.exists():
        return None
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(str(DAEMON_SOCKET_PATH))
            request = json.dumps(
                _query_payload(
                    query, top_n, sort_mode, category, syllables, primary_vowel,
                    rhymes_with, sounds_like, meter,
                )
            )
            sock.sendall(request.encode("utf-8"))
            sock.shutdown(socket.SHUT_WR)
            chunks = []
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
            response_text = b"".join(chunks).decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    if not response_text.strip():
        return None
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or "error" in payload:
        return None
    return payload


def send_progress_query(
    query: str,
    top_n: int,
    on_progress: Callable[[dict], None],
    sort_mode: str | None = None,
    category: str | None = None,
    syllables: int | None = None,
    primary_vowel: str | None = None,
    rhymes_with: str | None = None,
    sounds_like: str | None = None,
    meter: str | None = None,
    timeout: float = 30.0,
) -> dict | None:
    """Request the opt-in JSONL progress protocol without changing legacy clients."""
    if not DAEMON_SOCKET_PATH.exists():
        return None
    payload = _query_payload(
        query, top_n, sort_mode, category, syllables, primary_vowel,
        rhymes_with, sounds_like, meter,
    )
    payload["protocol"] = PROGRESS_PROTOCOL
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(str(DAEMON_SOCKET_PATH))
            sock.sendall(json.dumps(payload).encode("utf-8"))
            sock.shutdown(socket.SHUT_WR)
            with sock.makefile("r", encoding="utf-8") as stream:
                for line in stream:
                    event = json.loads(line)
                    if not isinstance(event, dict):
                        return None
                    if event.get("type") == "stage":
                        on_progress(event)
                    elif event.get("type") == "result":
                        result = event.get("result")
                        return result if isinstance(result, dict) else None
                    elif event.get("type") == "error":
                        return None
                    else:
                        return None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return None


def supports_progress(timeout: float = 1.0) -> bool | None:
    """Return true/false for an answer, or None when a live daemon is busy."""
    if not DAEMON_SOCKET_PATH.exists():
        return None
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(str(DAEMON_SOCKET_PATH))
            sock.sendall(json.dumps({"op": "capabilities"}).encode("utf-8"))
            sock.shutdown(socket.SHUT_WR)
            response = b""
            while chunk := sock.recv(65536):
                response += chunk
        payload = json.loads(response.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return isinstance(payload, dict) and PROGRESS_PROTOCOL in payload.get("protocols", [])


def spawn_daemon(startup_timeout: float = 20.0) -> bool:
    """Launch the daemon in the background, returning True once its socket is live.

    If the daemon is already running this is a no-op.  Callers that need
    mutual exclusion (so multiple clients don't race to spawn) should
    serialize through ``ensure_daemon_running()`` instead.
    """
    if _socket_is_reachable():
        return True
    _remove_stale_files()

    DAEMON_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DAEMON_LOG_PATH, "a") as log_file:
        subprocess.Popen(
            [sys.executable, "-u", "-m", "revdict.cli", "daemon", "start"],
            stdout=log_file,
            stderr=log_file,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env={**os.environ, "REVDICT_DAEMON_CHILD": "1"},
        )
    return _wait_for_socket(startup_timeout)


def ensure_daemon_running(startup_timeout: float = 20.0) -> bool:
    if _socket_is_reachable():
        return True
    start_lock = _acquire_lock(DAEMON_START_LOCK_PATH)
    if start_lock is None:
        return _wait_for_socket(startup_timeout)
    try:
        return spawn_daemon(startup_timeout)
    finally:
        _release_lock(start_lock)


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


def is_daemon_running() -> bool:
    pid = _read_pid()
    return pid is not None and _process_is_alive(pid) and DAEMON_SOCKET_PATH.exists()


def daemon_status() -> str:
    if is_daemon_running():
        pid = _read_pid()
        mem = _read_memory(pid)
        mem_str = f", {mem:.2f} GB" if mem is not None else ""
        return f"revdict daemon is running (pid {pid}{mem_str})."
    return "revdict daemon is not running."


def _read_memory(pid: int) -> float | None:
    """Read the daemon's resident set size from /proc, in gibibytes."""
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return None
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            try:
                kb = int(line.split()[1])
                return kb / 1048576
            except (IndexError, ValueError):
                return None
    return None


def _handle_request(request_text: str, search_fn) -> str:
    try:
        request = json.loads(request_text)
        result = search_fn(request["query"], **_search_kwargs(request))
    except Exception as error:
        return json.dumps({"error": str(error)})
    return json.dumps(result)


def _handle_progress_request(request_text: str, search_fn, emit: Callable[[dict], None]) -> None:
    """Serve one opt-in streaming search; legacy requests remain single JSON."""
    try:
        request = json.loads(request_text)
        result = search_fn(
            request["query"],
            **_search_kwargs(request),
            progress=ProgressReporter(emit),
        )
    except Exception as error:
        emit({"type": "error", "message": str(error)})
        return
    emit({"type": "result", "result": result})


def run_server() -> None:
    # The lock is deliberately taken *before* importing the search module:
    # that import can load a multi-gigabyte index while no socket exists yet.
    # A socket-only race guard lets every concurrent launcher pay that cost.
    lock_descriptor = _acquire_server_lock()
    if lock_descriptor is None:
        return
    try:
        if _socket_is_reachable():
            return
        from revdict.query_env import configure_offline_quiet_env

        configure_offline_quiet_env()
        from revdict import search as search_mod

        DAEMON_SOCKET_PATH.parent.mkdir(parents=True, exist_ok=True)
        _remove_stale_files()

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(str(DAEMON_SOCKET_PATH))
        except OSError:
            # Retain the socket check for manually started old daemons that
            # predate the lock, or for a filesystem-level bind race.
            if _socket_is_reachable():
                server.close()
                return
            raise

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
                        if not request_text.strip():
                            continue
                        request = json.loads(request_text)
                        if request.get("op") == "capabilities":
                            conn.sendall(json.dumps({"protocols": [PROGRESS_PROTOCOL]}).encode("utf-8"))
                        elif request.get("protocol") == PROGRESS_PROTOCOL:
                            def emit(event: dict) -> None:
                                conn.sendall((json.dumps(event) + "\n").encode("utf-8"))

                            _handle_progress_request(request_text, search_mod.search, emit)
                        else:
                            response_text = _handle_request(request_text, search_mod.search)
                            conn.sendall(response_text.encode("utf-8"))
                except Exception as error:
                    print(f"revdict daemon: error handling a request: {error}")
        finally:
            _remove_stale_files()
    finally:
        _release_lock(lock_descriptor)
