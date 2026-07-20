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
                {
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


def ensure_daemon_running(startup_timeout: float = 20.0) -> bool:
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


def is_daemon_running() -> bool:
    pid = _read_pid()
    return pid is not None and _process_is_alive(pid) and DAEMON_SOCKET_PATH.exists()


def daemon_status() -> str:
    if is_daemon_running():
        pid = _read_pid()
        return f"revdict daemon is running (pid {pid})."
    return "revdict daemon is not running."


def _handle_request(request_text: str, search_fn) -> str:
    try:
        request = json.loads(request_text)
        result = search_fn(
            request["query"],
            top_n=request["top_n"],
            sort_mode=request.get("sort"),
            category=request.get("category"),
            syllables=request.get("syllables"),
            primary_vowel=request.get("primary_vowel"),
            rhymes_with=request.get("rhymes_with"),
            sounds_like=request.get("sounds_like"),
            meter=request.get("meter"),
        )
    except Exception as error:
        return json.dumps({"error": str(error)})
    return json.dumps(result)


def run_server() -> None:
    # Cheap early exit before paying for the model load: if a live daemon
    # already owns the socket, we lost the spawn race -- bail immediately
    # rather than wastefully loading a ~2GB index we'd just discard.
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
        # Lost a narrow race against a sibling daemon that bound the path
        # between our reachability check and this bind. If it's now live,
        # exit gracefully (the spec's bind-race-loser behavior); otherwise
        # this is a genuine, unexpected bind failure -- propagate it.
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
                    response_text = _handle_request(request_text, search_mod.search)
                    conn.sendall(response_text.encode("utf-8"))
            except Exception as error:
                print(f"revdict daemon: error handling a request: {error}")
    finally:
        _remove_stale_files()
