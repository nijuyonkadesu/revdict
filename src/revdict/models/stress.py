# src/revdict/models/stress.py
import os
import re

try:
    import stressmark.engine as _engine
    import stressmark.render as _render
except ImportError:
    _engine = None
    _render = None


_SGR = re.compile(r"\x1b\[([0-9;]*)m")


def _strip_ansi_colours(value: str) -> str:
    """Remove SGR pigment while retaining bold, underline, and reverse spans."""
    def replace(match: re.Match[str]) -> str:
        codes = [int(code) for code in match.group(1).split(";") if code] or [0]
        preserved: list[int] = []
        index = 0
        while index < len(codes):
            code = codes[index]
            if code in {38, 48, 58} and index + 1 < len(codes):
                mode = codes[index + 1]
                index += 3 if mode == 5 else 5 if mode == 2 else 2
                continue
            if code in {39, 49, 59} or 30 <= code <= 37 or 40 <= code <= 47 or 90 <= code <= 97 or 100 <= code <= 107:
                index += 1
                continue
            preserved.append(code)
            index += 1
        return f"\x1b[{';'.join(str(code) for code in preserved) or '0'}m"

    return _SGR.sub(replace, value)


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
            no_color=False,
            width=200,
            color_system="truecolor" if truecolor else "256",
        )
        console.print(text, end="")
        rendered = buffer.getvalue()
        return _strip_ansi_colours(rendered) if no_color else rendered
    except Exception:
        return None
