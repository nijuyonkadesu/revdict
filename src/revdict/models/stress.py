# src/revdict/models/stress.py
import os

try:
    import stressmark.engine as _engine
    import stressmark.render as _render
except ImportError:
    _engine = None
    _render = None


def is_available() -> bool:
    return _engine is not None and _render is not None


def mark(word: str, pos: str) -> str | None:
    """Returns a captured ANSI-coded string of the word's stress-highlighted
    syllable breakdown, or None if stressmark isn't installed or fails for
    this specific word (never raises). Returns a plain string rather than a
    Rich Text object so this stays JSON-safe for the daemon's socket
    protocol -- reconstruct a Text object with
    `rich.text.Text.from_ansi(result)` if you need one.  `NO_COLOR` returns
    plain text by convention; otherwise the terminal capability selects 256
    or truecolor ANSI output."""
    if not is_available():
        return None
    try:
        from io import StringIO

        from rich.console import Console

        result = _engine.resolve_word_by_pos(word, pos)
        text = _render.render_word(result)
        buffer = StringIO()
        no_color = "NO_COLOR" in os.environ
        truecolor = os.environ.get("COLORTERM", "").casefold() in {"truecolor", "24bit"}
        console = Console(
            file=buffer,
            force_terminal=True,
            no_color=no_color,
            width=200,
            color_system=None if no_color else ("truecolor" if truecolor else "256"),
        )
        console.print(text, end="")
        return buffer.getvalue()
    except Exception:
        return None
