import pytest

from revdict import daemon
from revdict import search
from revdict.models import emotion


@pytest.fixture(scope="session")
def isolated_emotion_cache(tmp_path_factory):
    return tmp_path_factory.mktemp("emotion-runtime") / "predictions.sqlite3"


@pytest.fixture(autouse=True)
def isolated_daemon_runtime(tmp_path, monkeypatch, isolated_emotion_cache):
    """Keep every test away from the user's live daemon state."""
    runtime_dir = tmp_path / "daemon-runtime"
    monkeypatch.setattr(daemon, "DAEMON_SOCKET_PATH", runtime_dir / "daemon.sock")
    monkeypatch.setattr(daemon, "DAEMON_PID_PATH", runtime_dir / "daemon.pid")
    monkeypatch.setattr(daemon, "DAEMON_LOCK_PATH", runtime_dir / "daemon.lock")
    monkeypatch.setattr(
        daemon, "DAEMON_START_LOCK_PATH", runtime_dir / "daemon.start.lock"
    )
    monkeypatch.setattr(daemon, "DAEMON_LOG_PATH", runtime_dir / "daemon.log")
    monkeypatch.setattr(
        emotion,
        "EMOTION_CACHE_PATH",
        isolated_emotion_cache,
    )

    class DeterministicEmotionClassifier:
        def classify(self, _text):
            return {
                "label": "neutral",
                "polarity": "neutral",
                "confidence": 0.0,
                "scores": {"neutral": 0.0},
            }

        def classify_many(self, texts):
            return [self.classify(text) for text in texts]

    monkeypatch.setattr(search, "EmotionClassifier", DeterministicEmotionClassifier)
