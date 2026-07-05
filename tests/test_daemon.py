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
