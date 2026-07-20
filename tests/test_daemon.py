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


def test_run_server_bails_immediately_when_a_live_daemon_already_owns_the_socket(
    tmp_path, monkeypatch
):
    """The core regression test for the bind-race bug: run_server() must not
    attempt to steal/rebind the socket (which would orphan the live owner)
    when one is already reachable -- it should return immediately, before
    ever importing revdict.search or touching the PID/socket files."""
    socket_path = tmp_path / "daemon.sock"
    pid_path = tmp_path / "daemon.pid"
    monkeypatch.setattr(daemon, "DAEMON_SOCKET_PATH", socket_path)
    monkeypatch.setattr(daemon, "DAEMON_PID_PATH", pid_path)
    pid_path.write_text("999999")  # a real "owner" PID, should be left untouched

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    server.listen(1)

    try:
        daemon.run_server()  # must return quickly, not hang or raise
        # The "losing" run_server() must not have touched the live owner's files.
        assert pid_path.read_text() == "999999"
        assert socket_path.exists()
    finally:
        server.close()


def test_is_daemon_running_true_only_with_live_pid_and_existing_socket(tmp_path, monkeypatch):
    pid_path = tmp_path / "daemon.pid"
    socket_path = tmp_path / "daemon.sock"
    monkeypatch.setattr(daemon, "DAEMON_PID_PATH", pid_path)
    monkeypatch.setattr(daemon, "DAEMON_SOCKET_PATH", socket_path)

    assert daemon.is_daemon_running() is False  # no pid file at all

    pid_path.write_text(str(os.getpid()))
    assert daemon.is_daemon_running() is False  # pid alive but no socket file

    socket_path.write_text("")
    assert daemon.is_daemon_running() is True  # both present


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
