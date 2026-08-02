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


def test_mark_uses_terminal_renderer_with_a_nuclear_tier(monkeypatch):
    from rich.text import Text

    calls = {}

    class WordResult:
        tier = None

    class FakeEngine:
        def resolve_word_by_pos(self, word, pos):
            calls["word"] = word
            calls["pos"] = pos
            return WordResult()

    class FakeRender:
        def render_terminal(self, raw_tokens, results, console):
            calls["tokens"] = raw_tokens
            calls["tier"] = results[0].tier
            console.print(Text("PEARL", style="bold reverse yellow"), end="")

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(stress, "_engine", FakeEngine())
    monkeypatch.setattr(stress, "_render", FakeRender())

    result = stress.mark("pearlstone", "noun")

    assert calls == {"word": "pearlstone", "pos": "noun", "tokens": [(True, "pearlstone")], "tier": "nuclear"}
    assert "\x1b[7;33mPEARL" in result or "\x1b[1;7;33mPEARL" in result


def test_mark_honors_no_color_while_preserving_stress_attributes(monkeypatch):
    """NO_COLOR removes pigment but preserves non-colour stress semantics."""
    from rich.text import Text

    class FakeEngine:
        def resolve_word_by_pos(self, word, pos):
            return "fake-word-result"

    class FakeRender:
        def render_word(self, result):
            return Text("TAI", style="bold yellow")

    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(stress, "_engine", FakeEngine())
    monkeypatch.setattr(stress, "_render", FakeRender())

    assert stress.mark("tailor", "noun") == "\x1b[1mTAI\x1b[0m"


def test_mark_can_preserve_pigment_for_daemon_transport_when_no_color_is_set(monkeypatch):
    """The server must not let its environment erase a client's stress hue."""
    from rich.text import Text

    class FakeEngine:
        def resolve_word_by_pos(self, word, pos):
            return "fake-word-result"

    class FakeRender:
        def render_word(self, result):
            return Text("TAI", style="bold yellow")

    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(stress, "_engine", FakeEngine())
    monkeypatch.setattr(stress, "_render", FakeRender())

    rendered = stress.mark("tailor", "noun", preserve_color=True)

    assert "\x1b[" in rendered
    assert "TAI" in rendered
    assert "33" in rendered or "38;5;" in rendered


def test_mark_preserves_reverse_video_when_no_color_is_set(monkeypatch):
    from rich.text import Text

    class FakeEngine:
        def resolve_word_by_pos(self, word, pos):
            return "fake-word-result"

    class FakeRender:
        def render_word(self, result):
            return Text("stone", style="reverse yellow")

    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(stress, "_engine", FakeEngine())
    monkeypatch.setattr(stress, "_render", FakeRender())

    assert stress.mark("pearlstone", "noun") == "\x1b[7mstone\x1b[0m"


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
