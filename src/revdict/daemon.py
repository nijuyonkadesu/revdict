# src/revdict/daemon.py
import bisect
import heapq
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import fcntl

from revdict.paths import (
    DAEMON_LOCK_PATH,
    DAEMON_LOG_PATH,
    DAEMON_PID_PATH,
    DAEMON_SOCKET_PATH,
    DAEMON_START_LOCK_PATH,
    INDEX_DIR,
)
from revdict.progress import ProgressReporter


PROGRESS_PROTOCOL = "progress-v1"


def _autocomplete_rank_key(word: str, literary_frequency: dict[str, float]) -> tuple:
    """Prefer attested/common completions, then concise single headwords."""
    return (
        -literary_frequency.get(word, 0.0),
        " " in word,
        len(word),
        word,
    )


def _autocomplete_suggest(
    prefix: str,
    limit: int,
    words: list[str],
    words_by_length: dict[int, list[str]],
    literary_frequency: dict[str, float],
) -> list[str]:
    """Return prefix-first suggestions without letting rare phrases crowd them out."""
    from rapidfuzz import fuzz, process

    if not prefix or limit <= 0:
        return []

    start = bisect.bisect_left(words, prefix)
    end = bisect.bisect_right(words, prefix + "\U0010ffff")
    prefix_matches = (word for word in words[start:end] if word != prefix)
    matches = heapq.nsmallest(
        limit,
        prefix_matches,
        key=lambda word: _autocomplete_rank_key(word, literary_frequency),
    )
    if len(matches) >= limit:
        return matches

    target_len = len(prefix)
    candidates: list[str] = []
    for delta in range(4):
        candidates.extend(words_by_length.get(target_len - delta, []))
        if delta > 0:
            candidates.extend(words_by_length.get(target_len + delta, []))
    if candidates:
        remaining = limit - len(matches)
        fuzzy = process.extract(
            prefix,
            candidates,
            scorer=fuzz.ratio,
            limit=max(remaining * 5, remaining),
            score_cutoff=70,
        )
        fuzzy.sort(
            key=lambda item: (
                -item[1],
                _autocomplete_rank_key(item[0], literary_frequency),
            )
        )
        for word, _score, _idx in fuzzy:
            if word != prefix and word not in matches:
                matches.append(word)
                if len(matches) >= limit:
                    break
    return matches


@dataclass(frozen=True)
class DaemonRecord:
    pid: int
    identity: str
    phase: str


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


def _process_start_identity(pid: int) -> str | None:
    """Return Linux's per-process start tick, which survives PID reuse checks."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    closing_paren = stat.rfind(")")
    if closing_paren < 0:
        return None
    fields_after_name = stat[closing_paren + 1 :].split()
    try:
        if fields_after_name[0] == "Z":
            return None
        return fields_after_name[19]
    except IndexError:
        return None


def _write_daemon_record(descriptor: int, phase: str) -> None:
    if phase not in {"starting", "ready"}:
        raise ValueError(f"invalid daemon phase: {phase}")
    identity = _process_start_identity(os.getpid())
    if identity is None:
        raise RuntimeError("cannot determine daemon process identity")
    payload = json.dumps(
        {"pid": os.getpid(), "identity": identity, "phase": phase},
        separators=(",", ":"),
    ).encode("utf-8")
    os.lseek(descriptor, 0, os.SEEK_SET)
    os.ftruncate(descriptor, 0)
    offset = 0
    while offset < len(payload):
        offset += os.write(descriptor, payload[offset:])
    os.fsync(descriptor)


def _clear_daemon_record(descriptor: int) -> None:
    os.ftruncate(descriptor, 0)
    os.fsync(descriptor)


def _read_daemon_record() -> DaemonRecord | None:
    try:
        payload = json.loads(DAEMON_LOCK_PATH.read_text())
        pid = payload["pid"]
        identity = payload["identity"]
        phase = payload["phase"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
        return None
    if (
        not isinstance(pid, int)
        or isinstance(pid, bool)
        or not isinstance(identity, str)
        or phase not in {"starting", "ready"}
    ):
        return None
    return DaemonRecord(pid=pid, identity=identity, phase=phase)


def _record_identifies_live_process(record: DaemonRecord) -> bool:
    identity = _process_start_identity(record.pid)
    return identity is not None and identity == record.identity


def _remove_stale_files() -> None:
    for path in (DAEMON_SOCKET_PATH, DAEMON_PID_PATH):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _acquire_lock(path, *, blocking: bool = False) -> int | None:
    """Take a process-lifetime advisory lock at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        operation = fcntl.LOCK_EX
        if not blocking:
            operation |= fcntl.LOCK_NB
        fcntl.flock(descriptor, operation)
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


def _server_lock_is_held() -> bool:
    descriptor = _acquire_server_lock()
    if descriptor is None:
        return True
    _release_lock(descriptor)
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


def send_autocomplete(prefix: str, limit: int = 20, timeout: float = 1.0) -> list[str]:
    if not DAEMON_SOCKET_PATH.exists():
        return []
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(str(DAEMON_SOCKET_PATH))
            sock.sendall(json.dumps({"op": "autocomplete", "prefix": prefix, "limit": limit}).encode("utf-8"))
            sock.shutdown(socket.SHUT_WR)
            chunks = []
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
            response = json.loads(b"".join(chunks).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    return response.get("suggestions", [])


def _spawn_daemon(start_lock_descriptor: int) -> subprocess.Popen:
    """Start a detached child that inherits the launcher's startup claim."""
    DAEMON_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DAEMON_LOG_PATH, "a") as log_file:
        return subprocess.Popen(
            [sys.executable, "-u", "-m", "revdict.cli", "daemon", "start"],
            stdout=log_file,
            stderr=log_file,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env={
                **os.environ,
                "REVDICT_DAEMON_CHILD": "1",
                "REVDICT_START_LOCK_FD": str(start_lock_descriptor),
            },
            pass_fds=(start_lock_descriptor,),
        )


def _wait_for_existing_daemon() -> bool:
    """Wait while the actual lock owner is starting, without a time deadline."""
    while True:
        if _socket_is_reachable():
            return True
        if not _server_lock_is_held():
            return False
        record = _read_daemon_record()
        if (
            record is not None
            and record.phase == "ready"
            and _record_identifies_live_process(record)
        ):
            # A ready owner with no reachable socket is unhealthy. It still
            # owns the daemon lock, so spawning another copy would be unsafe.
            return False
        time.sleep(0.1)


def _wait_for_spawned_daemon(process: subprocess.Popen) -> bool:
    """Wait for readiness or a concrete child failure, never elapsed time."""
    while True:
        if _socket_is_reachable():
            return True
        if process.poll() is not None:
            if _socket_is_reachable():
                return True
            if _server_lock_is_held():
                return _wait_for_existing_daemon()
            return False
        if _server_lock_is_held():
            record = _read_daemon_record()
            if (
                record is not None
                and record.phase == "ready"
                and _record_identifies_live_process(record)
            ):
                return False
        time.sleep(0.1)


def ensure_daemon_running() -> bool:
    """Use the one serialized path to find, adopt, or launch the daemon."""
    if _socket_is_reachable():
        return True

    start_lock = _acquire_lock(DAEMON_START_LOCK_PATH, blocking=True)
    assert start_lock is not None
    start_lock_was_inherited = False
    try:
        if _socket_is_reachable():
            return True
        cleanup_lock = _acquire_server_lock()
        if cleanup_lock is None:
            return _wait_for_existing_daemon()

        # Both conditions are required for cleanup: this coordinator owns the
        # startup lock, and no daemon owns the lifetime lock.
        try:
            prior_record = _read_daemon_record()
            while (
                prior_record is not None
                and _record_identifies_live_process(prior_record)
            ):
                time.sleep(0.1)
            _clear_daemon_record(cleanup_lock)
            _remove_stale_files()
        finally:
            _release_lock(cleanup_lock)
        start_lock_was_inherited = True
        try:
            process = _spawn_daemon(start_lock)
        except OSError:
            return False
        return _wait_for_spawned_daemon(process)
    finally:
        if start_lock_was_inherited:
            # Do not issue LOCK_UN on the shared open-file description. If this
            # coordinator is interrupted, the child keeps the startup claim
            # until it has acquired and published the lifetime lock.
            os.close(start_lock)
        else:
            _release_lock(start_lock)


def spawn_daemon() -> bool:
    """Compatibility wrapper; all launches go through the coordinator."""
    return ensure_daemon_running()


def stop_daemon() -> bool:
    if not _server_lock_is_held():
        return False
    record = _read_daemon_record()
    if record is None or not _record_identifies_live_process(record):
        return False
    try:
        os.kill(record.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return False
    while _record_identifies_live_process(record):
        time.sleep(0.1)
    return True


def is_daemon_running() -> bool:
    if not _server_lock_is_held():
        return False
    record = _read_daemon_record()
    return record is not None and _record_identifies_live_process(record)


def daemon_status() -> str:
    if not _server_lock_is_held():
        return "revdict daemon is not running."

    record = _read_daemon_record()
    if record is None or not _record_identifies_live_process(record):
        return "revdict daemon is starting (lock owner details pending)."

    mem = _read_memory(record.pid)
    mem_str = f", {mem:.2f} GB" if mem is not None else ""
    if record.phase == "starting":
        return f"revdict daemon is starting (pid {record.pid}{mem_str})."
    if _socket_is_reachable():
        return f"revdict daemon is running (pid {record.pid}{mem_str})."
    return (
        f"revdict daemon is unhealthy (pid {record.pid}{mem_str}): "
        "the process owns the daemon lock but its socket is unreachable."
    )


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
    inherited_start_lock = None
    inherited_value = os.environ.get("REVDICT_START_LOCK_FD")
    if inherited_value is not None:
        try:
            inherited_start_lock = int(inherited_value)
        except ValueError:
            pass

    lock_descriptor = _acquire_server_lock()
    if lock_descriptor is None:
        if inherited_start_lock is not None:
            os.close(inherited_start_lock)
        return
    server = None
    owns_socket = False
    try:
        try:
            _write_daemon_record(lock_descriptor, "starting")
        finally:
            if inherited_start_lock is not None:
                # The lifetime lock is now the authoritative claim. Closing,
                # rather than unlocking, preserves a still-live parent copy.
                os.close(inherited_start_lock)

        def _exit_on_signal(signum, frame):
            if server is not None:
                server.close()
            raise SystemExit(0)

        signal.signal(signal.SIGTERM, _exit_on_signal)
        signal.signal(signal.SIGINT, _exit_on_signal)

        if _socket_is_reachable():
            return
        from revdict.query_env import configure_offline_quiet_env

        configure_offline_quiet_env()
        from revdict import search as search_mod
        from revdict import dictionary as dict_mod
        from revdict.index_bundle import resolve_active_index_dir

        active_index_dir = resolve_active_index_dir(INDEX_DIR)
        _autocomplete_words = sorted(dict_mod.load_word_index(active_index_dir).keys())
        _autocomplete_by_length: dict[int, list[str]] = {}
        for word in _autocomplete_words:
            _autocomplete_by_length.setdefault(len(word), []).append(word)
        _autocomplete_frequency = search_mod._load_literary_frequency(active_index_dir)

        DAEMON_SOCKET_PATH.parent.mkdir(parents=True, exist_ok=True)

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

        owns_socket = True
        server.listen(5)
        DAEMON_PID_PATH.write_text(str(os.getpid()))
        _write_daemon_record(lock_descriptor, "ready")

        print(f"revdict daemon listening on {DAEMON_SOCKET_PATH} (pid {os.getpid()})")

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
                    elif request.get("op") == "autocomplete":
                        prefix = request.get("prefix", "").lower()
                        limit = request.get("limit", 20)
                        suggestions = _autocomplete_suggest(
                            prefix,
                            limit,
                            _autocomplete_words,
                            _autocomplete_by_length,
                            _autocomplete_frequency,
                        )
                        conn.sendall(json.dumps({"suggestions": suggestions}).encode("utf-8"))
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
        if server is not None:
            server.close()
        if owns_socket:
            try:
                DAEMON_SOCKET_PATH.unlink()
            except FileNotFoundError:
                pass
        try:
            if _read_pid() == os.getpid():
                DAEMON_PID_PATH.unlink()
        except FileNotFoundError:
            pass
        _release_lock(lock_descriptor)
