# tests/test_daemon.py
import json
import os
import signal
import socket
import subprocess
import threading
import time

import pytest

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


def test_send_progress_query_receives_stage_events_before_its_result(tmp_path, monkeypatch):
    """The native UI needs live stage data, while result payloads stay unchanged."""
    socket_path = tmp_path / "daemon.sock"
    monkeypatch.setattr(daemon, "DAEMON_SOCKET_PATH", socket_path)
    ready_event = threading.Event()
    received = {}

    def streaming_server():
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(socket_path))
        server.listen(1)
        ready_event.set()
        conn, _ = server.accept()
        with conn:
            request = b""
            while chunk := conn.recv(65536):
                request += chunk
            received.update(json.loads(request.decode("utf-8")))
            conn.sendall(
                b'{"type":"stage","id":"ready","state":"active","ordinal":1,"total":9,"label":"Ready"}\n'
                b'{"type":"result","result":{"exact_match":null,"candidates":[]}}\n'
            )
        server.close()

    thread = threading.Thread(target=streaming_server)
    thread.start()
    assert ready_event.wait(timeout=2)
    events = []

    result = daemon.send_progress_query("happy", 50, events.append, timeout=2.0)

    thread.join(timeout=2)
    assert received["protocol"] == "progress-v1"
    assert events == [{"type": "stage", "id": "ready", "state": "active", "ordinal": 1, "total": 9, "label": "Ready"}]
    assert result == {"exact_match": None, "candidates": []}


def test_send_query_returns_none_when_socket_file_does_not_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(daemon, "DAEMON_SOCKET_PATH", tmp_path / "does-not-exist.sock")

    assert daemon.send_query("happy", 10, timeout=0.5) is None


def test_progress_capability_is_unknown_when_a_live_socket_does_not_answer(tmp_path, monkeypatch):
    """A busy current daemon must not be mistaken for legacy and killed."""
    socket_path = tmp_path / "daemon.sock"
    socket_path.write_text("placeholder")
    monkeypatch.setattr(daemon, "DAEMON_SOCKET_PATH", socket_path)

    assert daemon.supports_progress(timeout=0.01) is None


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


def test_send_query_returns_none_on_malformed_json_response(tmp_path, monkeypatch):
    socket_path = tmp_path / "daemon.sock"
    monkeypatch.setattr(daemon, "DAEMON_SOCKET_PATH", socket_path)

    def _run_garbage_server(socket_path, ready_event):
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(socket_path))
        server.listen(1)
        ready_event.set()
        conn, _ = server.accept()
        with conn:
            while True:
                chunk = conn.recv(65536)
                if not chunk:
                    break
            conn.sendall(b"not valid json {{{")
        server.close()

    ready_event = threading.Event()
    server_thread = threading.Thread(
        target=_run_garbage_server, args=(socket_path, ready_event)
    )
    server_thread.start()
    ready_event.wait(timeout=2)

    result = daemon.send_query("happy", 10, timeout=2.0)

    server_thread.join(timeout=2)
    assert result is None


def test_handle_request_calls_search_fn_with_parsed_args_and_returns_json_result():
    calls = {}

    def fake_search(query, top_n, sort_mode, category, syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None):
        calls["query"] = query
        calls["top_n"] = top_n
        calls["sort_mode"] = sort_mode
        calls["category"] = category
        return {"exact_match": None, "candidates": []}

    request_text = json.dumps({"query": "happy", "top_n": 10})

    response_text = daemon._handle_request(request_text, fake_search)

    assert calls == {"query": "happy", "top_n": 10, "sort_mode": None, "category": None}
    assert json.loads(response_text) == {"exact_match": None, "candidates": []}


def test_handle_request_returns_error_payload_when_search_fn_raises():
    def failing_search(query, top_n, sort_mode, category, syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None):
        raise RuntimeError("index not loaded")

    request_text = json.dumps({"query": "happy", "top_n": 10})

    response_text = daemon._handle_request(request_text, failing_search)

    payload = json.loads(response_text)
    assert "index not loaded" in payload["error"]


def test_handle_request_returns_error_payload_on_malformed_json():
    response_text = daemon._handle_request("not valid json", lambda query, top_n: {})

    payload = json.loads(response_text)
    assert "error" in payload


def test_handle_progress_request_forwards_search_events_and_finishes_with_result():
    """A streaming daemon response must preserve search's actual stage ordering."""
    events = []

    def fake_search(query, top_n, sort_mode, category, progress, **_kwargs):
        assert (query, top_n, sort_mode, category) == ("happy", 50, None, None)
        progress.active("ready")
        progress.completed("ready")
        return {"exact_match": None, "candidates": [{"headword": "joyful"}]}

    daemon._handle_progress_request(
        json.dumps({"query": "happy", "top_n": 50, "protocol": "progress-v1"}),
        fake_search,
        events.append,
    )

    assert events[-1] == {
        "type": "result",
        "result": {"exact_match": None, "candidates": [{"headword": "joyful"}]},
    }
    assert [event["state"] for event in events[:-1]] == ["active", "completed"]


def test_socket_is_reachable_returns_false_when_socket_file_does_not_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(daemon, "DAEMON_SOCKET_PATH", tmp_path / "does-not-exist.sock")

    assert daemon._socket_is_reachable() is False


def test_socket_is_reachable_returns_true_for_a_real_listening_socket(tmp_path, monkeypatch):
    socket_path = tmp_path / "daemon.sock"
    monkeypatch.setattr(daemon, "DAEMON_SOCKET_PATH", socket_path)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    server.listen(1)

    try:
        assert daemon._socket_is_reachable() is True
    finally:
        server.close()


def _hold_daemon_lock(phase="starting"):
    descriptor = daemon._acquire_server_lock()
    assert descriptor is not None
    daemon._write_daemon_record(descriptor, phase)
    return descriptor


def test_daemon_record_round_trips_pid_identity_and_phase():
    descriptor = _hold_daemon_lock("starting")
    try:
        record = daemon._read_daemon_record()
        assert record.pid == os.getpid()
        assert record.identity == daemon._process_start_identity(os.getpid())
        assert record.phase == "starting"
    finally:
        daemon._release_lock(descriptor)


def test_status_uses_held_lifetime_lock_to_report_starting(monkeypatch):
    descriptor = _hold_daemon_lock("starting")
    monkeypatch.setattr(daemon, "_socket_is_reachable", lambda: False)
    try:
        status = daemon.daemon_status()
        assert "starting" in status
        assert str(os.getpid()) in status
        assert "not running" not in status
    finally:
        daemon._release_lock(descriptor)


def test_status_tolerates_corrupt_lock_record(monkeypatch):
    descriptor = daemon._acquire_server_lock()
    assert descriptor is not None
    os.write(descriptor, b"\xff")
    monkeypatch.setattr(daemon, "_socket_is_reachable", lambda: False)
    try:
        assert "starting" in daemon.daemon_status()
    finally:
        daemon._release_lock(descriptor)


def test_status_reports_ready_owner_without_socket_as_unhealthy(monkeypatch):
    descriptor = _hold_daemon_lock("ready")
    monkeypatch.setattr(daemon, "_socket_is_reachable", lambda: False)
    try:
        status = daemon.daemon_status()
        assert "unhealthy" in status
        assert str(os.getpid()) in status
        assert "not running" not in status
    finally:
        daemon._release_lock(descriptor)


def test_status_reports_ready_lock_owner_with_reachable_socket_as_running(monkeypatch):
    descriptor = _hold_daemon_lock("ready")
    monkeypatch.setattr(daemon, "_socket_is_reachable", lambda: True)
    try:
        status = daemon.daemon_status()
        assert "running" in status
        assert "not running" not in status
    finally:
        daemon._release_lock(descriptor)


def test_status_ignores_stale_record_when_lifetime_lock_is_free():
    descriptor = _hold_daemon_lock("ready")
    daemon._release_lock(descriptor)

    assert "not running" in daemon.daemon_status()
    assert daemon.is_daemon_running() is False


def test_is_daemon_running_recognizes_a_live_starting_owner():
    descriptor = _hold_daemon_lock("starting")
    try:
        assert daemon.is_daemon_running() is True
    finally:
        daemon._release_lock(descriptor)


def test_run_server_bails_before_loading_when_another_process_holds_lifetime_lock(
    monkeypatch,
):
    monkeypatch.setattr(daemon, "_acquire_server_lock", lambda: None)

    daemon.run_server()

    assert not daemon.DAEMON_SOCKET_PATH.exists()
    assert not daemon.DAEMON_PID_PATH.exists()


def test_run_server_publishes_starting_before_expensive_initialization(monkeypatch):
    from revdict import query_env

    observed = {}
    monkeypatch.setattr(daemon, "_socket_is_reachable", lambda: False)
    monkeypatch.setattr(daemon.signal, "signal", lambda *_args: None)

    def fail_during_initialization():
        record = daemon._read_daemon_record()
        observed["record"] = record
        observed["lock_held"] = daemon._server_lock_is_held()
        raise RuntimeError("startup failed")

    monkeypatch.setattr(
        query_env, "configure_offline_quiet_env", fail_during_initialization
    )

    with pytest.raises(RuntimeError, match="startup failed"):
        daemon.run_server()

    assert observed["record"].pid == os.getpid()
    assert observed["record"].phase == "starting"
    assert observed["lock_held"] is True
    assert daemon._server_lock_is_held() is False
    assert not daemon.DAEMON_PID_PATH.exists()


def test_run_server_releases_inherited_start_lock_only_after_lifetime_claim(
    monkeypatch,
):
    from revdict import query_env

    inherited = daemon._acquire_lock(daemon.DAEMON_START_LOCK_PATH)
    assert inherited is not None
    monkeypatch.setenv("REVDICT_START_LOCK_FD", str(inherited))
    monkeypatch.setattr(daemon, "_socket_is_reachable", lambda: False)
    monkeypatch.setattr(daemon.signal, "signal", lambda *_args: None)

    def fail_after_handoff():
        assert daemon._server_lock_is_held() is True
        with pytest.raises(OSError):
            os.fstat(inherited)
        raise RuntimeError("stop after handoff")

    monkeypatch.setattr(query_env, "configure_offline_quiet_env", fail_after_handoff)

    with pytest.raises(RuntimeError, match="stop after handoff"):
        daemon.run_server()


def test_spawned_daemon_is_detached_and_uses_child_entrypoint(monkeypatch):
    captured = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(daemon.subprocess, "Popen", fake_popen)

    start_lock = daemon._acquire_lock(daemon.DAEMON_START_LOCK_PATH)
    assert start_lock is not None
    try:
        daemon._spawn_daemon(start_lock)
    finally:
        daemon._release_lock(start_lock)

    assert captured["command"] == [
        daemon.sys.executable,
        "-u",
        "-m",
        "revdict.cli",
        "daemon",
        "start",
    ]
    assert captured["stdin"] is daemon.subprocess.DEVNULL
    assert captured["start_new_session"] is True
    assert captured["env"]["REVDICT_DAEMON_CHILD"] == "1"
    assert captured["env"]["REVDICT_START_LOCK_FD"] == str(start_lock)
    assert captured["pass_fds"] == (start_lock,)


def test_ensure_daemon_running_returns_immediately_for_reachable_socket(monkeypatch):
    monkeypatch.setattr(daemon, "_socket_is_reachable", lambda: True)
    monkeypatch.setattr(
        daemon,
        "_spawn_daemon",
        lambda _start_lock: (_ for _ in ()).throw(AssertionError("must not spawn")),
    )

    assert daemon.ensure_daemon_running() is True


def test_ensure_daemon_running_waits_for_live_starting_owner_without_timeout(
    monkeypatch,
):
    descriptor = _hold_daemon_lock("starting")
    ready = threading.Event()
    monkeypatch.setattr(daemon, "_socket_is_reachable", ready.is_set)
    monkeypatch.setattr(
        daemon,
        "_spawn_daemon",
        lambda _start_lock: (_ for _ in ()).throw(AssertionError("must not spawn")),
    )
    monkeypatch.setattr(
        daemon.time,
        "time",
        lambda: (_ for _ in ()).throw(AssertionError("startup must not use a deadline")),
    )

    def finish_starting():
        time.sleep(0.15)
        daemon._write_daemon_record(descriptor, "ready")
        ready.set()

    thread = threading.Thread(target=finish_starting)
    thread.start()
    try:
        assert daemon.ensure_daemon_running() is True
    finally:
        thread.join(timeout=2)
        daemon._release_lock(descriptor)


def test_new_lock_owner_is_not_mistaken_for_stale_ready_record(monkeypatch):
    descriptor = daemon._acquire_server_lock()
    assert descriptor is not None
    os.write(
        descriptor,
        json.dumps({"pid": 999999, "identity": "old", "phase": "ready"}).encode(),
    )
    ready = threading.Event()
    monkeypatch.setattr(daemon, "_socket_is_reachable", ready.is_set)
    monkeypatch.setattr(
        daemon,
        "_spawn_daemon",
        lambda _start_lock: (_ for _ in ()).throw(AssertionError("must not spawn")),
    )

    def publish_new_owner():
        time.sleep(0.15)
        daemon._write_daemon_record(descriptor, "ready")
        ready.set()

    thread = threading.Thread(target=publish_new_owner)
    thread.start()
    try:
        assert daemon.ensure_daemon_running() is True
    finally:
        thread.join(timeout=2)
        daemon._release_lock(descriptor)


def test_ensure_daemon_running_reports_failure_when_spawned_child_exits(monkeypatch):
    monkeypatch.setattr(daemon, "_socket_is_reachable", lambda: False)

    class FailedProcess:
        def poll(self):
            return 1

    monkeypatch.setattr(daemon, "_spawn_daemon", lambda _start_lock: FailedProcess())

    assert daemon.ensure_daemon_running() is False


def test_interrupted_spawn_does_not_unlock_child_startup_claim(monkeypatch):
    inherited_copy = {"descriptor": None}

    def interrupt_after_inheriting(start_lock):
        inherited_copy["descriptor"] = os.dup(start_lock)
        raise KeyboardInterrupt

    monkeypatch.setattr(daemon, "_socket_is_reachable", lambda: False)
    monkeypatch.setattr(daemon, "_spawn_daemon", interrupt_after_inheriting)

    with pytest.raises(KeyboardInterrupt):
        daemon.ensure_daemon_running()

    contender = daemon._acquire_lock(daemon.DAEMON_START_LOCK_PATH)
    try:
        assert contender is None
    finally:
        if contender is not None:
            daemon._release_lock(contender)
        os.close(inherited_copy["descriptor"])


def test_losing_spawn_adopts_the_actual_lifetime_lock_owner(monkeypatch):
    ready = threading.Event()
    owner = {"descriptor": None}
    monkeypatch.setattr(daemon, "_socket_is_reachable", ready.is_set)

    class LosingProcess:
        def poll(self):
            return 1

    def spawn_while_another_owner_wins(_start_lock):
        owner["descriptor"] = daemon._acquire_server_lock()
        assert owner["descriptor"] is not None
        daemon._write_daemon_record(owner["descriptor"], "starting")
        threading.Timer(0.15, ready.set).start()
        return LosingProcess()

    monkeypatch.setattr(daemon, "_spawn_daemon", spawn_while_another_owner_wins)

    try:
        assert daemon.ensure_daemon_running() is True
    finally:
        if owner["descriptor"] is not None:
            daemon._release_lock(owner["descriptor"])


def test_ensure_daemon_running_does_not_respawn_unhealthy_lock_owner(monkeypatch):
    descriptor = _hold_daemon_lock("ready")
    monkeypatch.setattr(daemon, "_socket_is_reachable", lambda: False)
    monkeypatch.setattr(
        daemon,
        "_spawn_daemon",
        lambda _start_lock: (_ for _ in ()).throw(AssertionError("must not spawn")),
    )
    try:
        assert daemon.ensure_daemon_running() is False
    finally:
        daemon._release_lock(descriptor)


def test_concurrent_ensure_calls_share_one_spawn(monkeypatch):
    ready = threading.Event()
    spawn_count = 0
    spawn_count_lock = threading.Lock()
    monkeypatch.setattr(daemon, "_socket_is_reachable", ready.is_set)

    class LiveProcess:
        def poll(self):
            return None

    def spawn_once(_start_lock):
        nonlocal spawn_count
        with spawn_count_lock:
            spawn_count += 1
        threading.Timer(0.15, ready.set).start()
        return LiveProcess()

    monkeypatch.setattr(daemon, "_spawn_daemon", spawn_once)
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(daemon.ensure_daemon_running()))
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert results == [True, True]
    assert spawn_count == 1


def test_stale_sidecars_are_removed_only_while_startup_lock_is_held(monkeypatch):
    daemon.DAEMON_SOCKET_PATH.parent.mkdir(parents=True)
    daemon.DAEMON_SOCKET_PATH.write_text("stale")
    daemon.DAEMON_PID_PATH.write_text("123")
    checked = []

    def checked_cleanup():
        contender = daemon._acquire_lock(daemon.DAEMON_START_LOCK_PATH)
        assert contender is None
        server_lock = daemon._acquire_server_lock()
        assert server_lock is None
        checked.append(True)
        daemon.DAEMON_SOCKET_PATH.unlink()
        daemon.DAEMON_PID_PATH.unlink()

    class FailedProcess:
        def poll(self):
            return 1

    monkeypatch.setattr(daemon, "_socket_is_reachable", lambda: False)
    monkeypatch.setattr(daemon, "_remove_stale_files", checked_cleanup)
    monkeypatch.setattr(daemon, "_spawn_daemon", lambda _start_lock: FailedProcess())

    assert daemon.ensure_daemon_running() is False
    assert checked == [True]


def test_cleanup_invalidates_stale_lifetime_record_before_spawning(monkeypatch):
    descriptor = _hold_daemon_lock("ready")
    daemon._release_lock(descriptor)
    prior_process_exited = threading.Event()
    monkeypatch.setattr(
        daemon,
        "_record_identifies_live_process",
        lambda _record: not prior_process_exited.is_set(),
    )

    class FailedProcess:
        def poll(self):
            return 1

    def spawn_after_teardown(_start_lock):
        assert prior_process_exited.is_set()
        return FailedProcess()

    monkeypatch.setattr(daemon, "_socket_is_reachable", lambda: False)
    monkeypatch.setattr(daemon, "_spawn_daemon", spawn_after_teardown)
    threading.Timer(0.15, prior_process_exited.set).start()

    assert daemon.ensure_daemon_running() is False
    assert daemon._read_daemon_record() is None


def test_stop_daemon_returns_false_when_lifetime_lock_is_free():
    assert daemon.stop_daemon() is False


def test_stop_daemon_signals_pid_from_validated_lock_record(monkeypatch):
    descriptor = _hold_daemon_lock("ready")
    signals = []
    lock_released = threading.Event()
    process_exited = threading.Event()

    def signal_and_release(pid, sig):
        signals.append((pid, sig))

        def release_owner():
            daemon._release_lock(descriptor)
            lock_released.set()
            time.sleep(0.15)
            process_exited.set()

        threading.Timer(0.15, release_owner).start()

    monkeypatch.setattr(daemon.os, "kill", signal_and_release)
    monkeypatch.setattr(
        daemon,
        "_record_identifies_live_process",
        lambda _record: not process_exited.is_set(),
    )
    try:
        assert daemon.stop_daemon() is True
        assert signals == [(os.getpid(), signal.SIGTERM)]
        assert lock_released.is_set()
        assert process_exited.is_set()
    finally:
        lock_released.wait(timeout=1)
        if not lock_released.is_set():
            daemon._release_lock(descriptor)


def test_zombie_pid_is_not_a_live_daemon_identity():
    child = subprocess.Popen([daemon.sys.executable, "-c", "pass"])
    try:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            stat = daemon.Path(f"/proc/{child.pid}/stat").read_text()
            if stat[stat.rfind(")") + 1 :].split()[0] == "Z":
                break
            time.sleep(0.01)
        else:
            pytest.fail("stand-in process did not become a zombie")

        assert daemon._process_start_identity(child.pid) is None
    finally:
        child.wait(timeout=2)


def test_stop_daemon_terminates_real_validated_lock_owner():
    script = r"""
import fcntl
import json
import os
from pathlib import Path
import sys
import time

lock_path = Path(sys.argv[1])
lock_path.parent.mkdir(parents=True, exist_ok=True)
with lock_path.open("w") as lock_file:
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    stat = Path(f"/proc/{os.getpid()}/stat").read_text()
    identity = stat[stat.rfind(")") + 1:].split()[19]
    json.dump(
        {"pid": os.getpid(), "identity": identity, "phase": "ready"},
        lock_file,
    )
    lock_file.flush()
    os.fsync(lock_file.fileno())
    print("ready", flush=True)
    time.sleep(30)
"""
    stand_in = subprocess.Popen(
        [daemon.sys.executable, "-c", script, str(daemon.DAEMON_LOCK_PATH)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert stand_in.stdout.readline().strip() == "ready"
        assert daemon.stop_daemon() is True
        stand_in.wait(timeout=5)
        assert stand_in.returncode == -signal.SIGTERM
        assert daemon._server_lock_is_held() is False
    finally:
        if stand_in.poll() is None:
            stand_in.kill()


def _run_capturing_server(socket_path, received, ready_event):
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
        received["request"] = json.loads(b"".join(chunks).decode("utf-8"))
        conn.sendall(json.dumps({"exact_match": None, "candidates": []}).encode("utf-8"))
    server.close()


def test_send_query_includes_sort_mode_in_the_request_payload(tmp_path, monkeypatch):
    """The wire-protocol extension: a non-default sort_mode must actually
    reach the server in the request JSON, not get silently dropped."""
    socket_path = tmp_path / "daemon.sock"
    monkeypatch.setattr(daemon, "DAEMON_SOCKET_PATH", socket_path)
    received = {}

    ready_event = threading.Event()
    server_thread = threading.Thread(
        target=_run_capturing_server, args=(socket_path, received, ready_event)
    )
    server_thread.start()
    ready_event.wait(timeout=2)

    daemon.send_query("happy", 10, sort_mode="alpha", timeout=2.0)

    server_thread.join(timeout=2)
    assert received["request"] == {
        "query": "happy", "top_n": 10, "sort": "alpha", "category": None,
        "syllables": None, "primary_vowel": None, "rhymes_with": None,
        "sounds_like": None, "meter": None,
    }


def test_send_query_defaults_sort_mode_to_none_when_omitted(tmp_path, monkeypatch):
    """Backward compatibility for the CLIENT side: an existing call site
    that doesn't pass sort_mode at all must still send a well-formed
    request (with "sort": null), matching what an updated server expects."""
    socket_path = tmp_path / "daemon.sock"
    monkeypatch.setattr(daemon, "DAEMON_SOCKET_PATH", socket_path)
    received = {}

    ready_event = threading.Event()
    server_thread = threading.Thread(
        target=_run_capturing_server, args=(socket_path, received, ready_event)
    )
    server_thread.start()
    ready_event.wait(timeout=2)

    daemon.send_query("happy", 10, timeout=2.0)

    server_thread.join(timeout=2)
    assert received["request"] == {
        "query": "happy", "top_n": 10, "sort": None, "category": None,
        "syllables": None, "primary_vowel": None, "rhymes_with": None,
        "sounds_like": None, "meter": None,
    }


def test_handle_request_passes_sort_mode_through_to_search_fn():
    calls = {}

    def fake_search(query, top_n, sort_mode, category, syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None):
        calls["sort_mode"] = sort_mode
        return {"exact_match": None, "candidates": []}

    request_text = json.dumps({"query": "happy", "top_n": 10, "sort": "alpha"})

    daemon._handle_request(request_text, fake_search)

    assert calls == {"sort_mode": "alpha"}


def test_handle_request_defaults_sort_mode_to_none_for_requests_without_it():
    """Backward compatibility for the SERVER side: an OLD client's request
    (no "sort" key at all, not even null) must still work, with sort_mode
    defaulting to None."""
    calls = {}

    def fake_search(query, top_n, sort_mode, category, syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None):
        calls["sort_mode"] = sort_mode
        return {"exact_match": None, "candidates": []}

    request_text = json.dumps({"query": "happy", "top_n": 10})

    daemon._handle_request(request_text, fake_search)

    assert calls == {"sort_mode": None}


def test_send_query_includes_category_in_the_request_payload(tmp_path, monkeypatch):
    """The wire-protocol extension: a non-default category must actually
    reach the server in the request JSON, not get silently dropped."""
    socket_path = tmp_path / "daemon.sock"
    monkeypatch.setattr(daemon, "DAEMON_SOCKET_PATH", socket_path)
    received = {}

    ready_event = threading.Event()
    server_thread = threading.Thread(
        target=_run_capturing_server, args=(socket_path, received, ready_event)
    )
    server_thread.start()
    ready_event.wait(timeout=2)

    daemon.send_query("happy", 10, category="noun", timeout=2.0)

    server_thread.join(timeout=2)
    assert received["request"] == {
        "query": "happy", "top_n": 10, "sort": None, "category": "noun",
        "syllables": None, "primary_vowel": None, "rhymes_with": None,
        "sounds_like": None, "meter": None,
    }


def test_send_query_defaults_category_to_none_when_omitted(tmp_path, monkeypatch):
    """Backward compatibility for the CLIENT side: an existing call site
    that doesn't pass category at all must still send a well-formed
    request (with "category": null), matching what an updated server
    expects."""
    socket_path = tmp_path / "daemon.sock"
    monkeypatch.setattr(daemon, "DAEMON_SOCKET_PATH", socket_path)
    received = {}

    ready_event = threading.Event()
    server_thread = threading.Thread(
        target=_run_capturing_server, args=(socket_path, received, ready_event)
    )
    server_thread.start()
    ready_event.wait(timeout=2)

    daemon.send_query("happy", 10, timeout=2.0)

    server_thread.join(timeout=2)
    assert received["request"] == {
        "query": "happy", "top_n": 10, "sort": None, "category": None,
        "syllables": None, "primary_vowel": None, "rhymes_with": None,
        "sounds_like": None, "meter": None,
    }


def test_handle_request_passes_category_through_to_search_fn():
    calls = {}

    def fake_search(query, top_n, sort_mode, category, syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None):
        calls["category"] = category
        return {"exact_match": None, "candidates": []}

    request_text = json.dumps({"query": "happy", "top_n": 10, "category": "noun"})

    daemon._handle_request(request_text, fake_search)

    assert calls == {"category": "noun"}


def test_handle_request_defaults_category_to_none_for_requests_without_it():
    """Backward compatibility for the SERVER side: an OLD client's request
    (no "category" key at all, not even null) must still work, with
    category defaulting to None."""
    calls = {}

    def fake_search(query, top_n, sort_mode, category, syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None):
        calls["category"] = category
        return {"exact_match": None, "candidates": []}

    request_text = json.dumps({"query": "happy", "top_n": 10})

    daemon._handle_request(request_text, fake_search)

    assert calls == {"category": None}


def test_send_query_includes_all_five_phonetic_fields_in_the_request_payload(tmp_path, monkeypatch):
    socket_path = tmp_path / "daemon.sock"
    monkeypatch.setattr(daemon, "DAEMON_SOCKET_PATH", socket_path)
    received = {}

    ready_event = threading.Event()
    server_thread = threading.Thread(
        target=_run_capturing_server, args=(socket_path, received, ready_event)
    )
    server_thread.start()
    ready_event.wait(timeout=2)

    daemon.send_query(
        "happy", 10, syllables=2, primary_vowel="AE", rhymes_with="cat",
        sounds_like="bat", meter="/x", timeout=2.0,
    )

    server_thread.join(timeout=2)
    assert received["request"] == {
        "query": "happy", "top_n": 10, "sort": None, "category": None,
        "syllables": 2, "primary_vowel": "AE", "rhymes_with": "cat",
        "sounds_like": "bat", "meter": "/x",
    }


def test_send_query_defaults_all_five_phonetic_fields_to_none_when_omitted(tmp_path, monkeypatch):
    socket_path = tmp_path / "daemon.sock"
    monkeypatch.setattr(daemon, "DAEMON_SOCKET_PATH", socket_path)
    received = {}

    ready_event = threading.Event()
    server_thread = threading.Thread(
        target=_run_capturing_server, args=(socket_path, received, ready_event)
    )
    server_thread.start()
    ready_event.wait(timeout=2)

    daemon.send_query("happy", 10, timeout=2.0)

    server_thread.join(timeout=2)
    assert received["request"]["syllables"] is None
    assert received["request"]["primary_vowel"] is None
    assert received["request"]["rhymes_with"] is None
    assert received["request"]["sounds_like"] is None
    assert received["request"]["meter"] is None


def test_handle_request_passes_all_five_phonetic_fields_through_to_search_fn():
    calls = {}

    def fake_search(query, top_n, sort_mode, category, syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None):
        calls.update(
            syllables=syllables, primary_vowel=primary_vowel, rhymes_with=rhymes_with,
            sounds_like=sounds_like, meter=meter,
        )
        return {"exact_match": None, "candidates": []}

    request_text = json.dumps(
        {"query": "happy", "top_n": 10, "syllables": 2, "primary_vowel": "AE", "rhymes_with": "cat", "sounds_like": "bat", "meter": "/x"}
    )

    daemon._handle_request(request_text, fake_search)

    assert calls == {"syllables": 2, "primary_vowel": "AE", "rhymes_with": "cat", "sounds_like": "bat", "meter": "/x"}


def test_handle_request_defaults_all_five_phonetic_fields_to_none_for_requests_without_them():
    calls = {}

    def fake_search(query, top_n, sort_mode, category, syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None):
        calls.update(
            syllables=syllables, primary_vowel=primary_vowel, rhymes_with=rhymes_with,
            sounds_like=sounds_like, meter=meter,
        )
        return {"exact_match": None, "candidates": []}

    request_text = json.dumps({"query": "happy", "top_n": 10})

    daemon._handle_request(request_text, fake_search)

    assert calls == {"syllables": None, "primary_vowel": None, "rhymes_with": None, "sounds_like": None, "meter": None}
