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
    """Return one word as the nuclear unit of its own intonation phrase.

    The terminal stressmark renderer provides the authoritative nuclear,
    prominent, and secondary semantics. A single dictionary headword has no
    sentence context, so its primary stress is deliberately the nuclear span.
    The result remains JSON-safe ANSI for the daemon protocol; NO_COLOR
    removes pigment while preserving non-colour attributes.
    """
    if not is_available():
        return None
    try:
        from io import StringIO

        from rich.console import Console

        result = _engine.resolve_word_by_pos(word, pos)
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
        terminal_renderer = getattr(_render, "render_terminal", None)
        if terminal_renderer is None:
            console.print(_render.render_word(result), end="")
        else:
            result.tier = "nuclear"
            terminal_renderer([(True, word)], [result], console=console)
        rendered = buffer.getvalue()
        return _strip_ansi_colours(rendered) if no_color else rendered
    except Exception:
        return None
