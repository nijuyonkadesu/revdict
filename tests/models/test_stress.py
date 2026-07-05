# tests/models/test_stress.py
from revdict.models import stress


def test_is_available_true_when_engine_and_render_modules_present(monkeypatch):
    monkeypatch.setattr(stress, "_engine", object())
    monkeypatch.setattr(stress, "_render", object())

    assert stress.is_available() is True


def test_is_available_false_when_modules_absent(monkeypatch):
    monkeypatch.setattr(stress, "_engine", None)
    monkeypatch.setattr(stress, "_render", None)

    assert stress.is_available() is False


def test_mark_returns_none_when_not_available(monkeypatch):
    monkeypatch.setattr(stress, "_engine", None)
    monkeypatch.setattr(stress, "_render", None)

    assert stress.mark("happy", "adjective") is None


def test_mark_calls_engine_and_render_and_returns_a_captured_ansi_string(monkeypatch):
    calls = {}

    class FakeEngine:
        def resolve_word_by_pos(self, word, pos):
            calls["word"] = word
            calls["pos"] = pos
            return "fake-word-result"

    class FakeRender:
        def render_word(self, result):
            calls["rendered_from"] = result
            from rich.text import Text

            return Text("HAPpy", style="bold yellow")

    monkeypatch.setattr(stress, "_engine", FakeEngine())
    monkeypatch.setattr(stress, "_render", FakeRender())

    result = stress.mark("happy", "adjective")

    assert calls == {"word": "happy", "pos": "adjective", "rendered_from": "fake-word-result"}
    assert isinstance(result, str)
    assert "HAPpy" in result  # the captured ANSI string contains the plain text


def test_mark_returns_none_when_engine_raises(monkeypatch):
    class FailingEngine:
        def resolve_word_by_pos(self, word, pos):
            raise ValueError("boom")

    monkeypatch.setattr(stress, "_engine", FailingEngine())
    monkeypatch.setattr(stress, "_render", object())

    assert stress.mark("happy", "adjective") is None


def test_mark_result_round_trips_through_text_from_ansi(monkeypatch):
    """The whole point of returning an ANSI string instead of a Text object:
    confirm a caller can reconstruct an equivalent Text object from it."""
    from rich.text import Text

    class FakeEngine:
        def resolve_word_by_pos(self, word, pos):
            return "fake-word-result"

    class FakeRender:
        def render_word(self, result):
            text = Text("HAP", style="bold yellow")
            text.append("py", style="grey62")
            return text

    monkeypatch.setattr(stress, "_engine", FakeEngine())
    monkeypatch.setattr(stress, "_render", FakeRender())

    result = stress.mark("happy", "adjective")
    reconstructed = Text.from_ansi(result)

    assert reconstructed.plain == "HAPpy"
