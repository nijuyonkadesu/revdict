import pytest

from revdict import daemon


@pytest.fixture(autouse=True)
def isolated_daemon_runtime(tmp_path, monkeypatch):
    """Keep every test away from the user's live daemon state."""
    runtime_dir = tmp_path / "daemon-runtime"
    monkeypatch.setattr(daemon, "DAEMON_SOCKET_PATH", runtime_dir / "daemon.sock")
    monkeypatch.setattr(daemon, "DAEMON_PID_PATH", runtime_dir / "daemon.pid")
    monkeypatch.setattr(daemon, "DAEMON_LOCK_PATH", runtime_dir / "daemon.lock")
    monkeypatch.setattr(
        daemon, "DAEMON_START_LOCK_PATH", runtime_dir / "daemon.start.lock"
    )
    monkeypatch.setattr(daemon, "DAEMON_LOG_PATH", runtime_dir / "daemon.log")
