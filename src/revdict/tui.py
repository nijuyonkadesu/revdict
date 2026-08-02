"""The native, daemon-backed terminal interface for revdict."""

from __future__ import annotations

import asyncio
from io import StringIO
import os
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from typing import Any

from prompt_toolkit.formatted_text import ANSI, to_formatted_text
from prompt_toolkit.application.current import get_app
from prompt_toolkit.layout import HSplit, VSplit, Window
from prompt_toolkit.layout.controls import UIContent, UIControl
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from prompt_toolkit.data_structures import Point
from prompt_toolkit.output.color_depth import ColorDepth
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.widgets import Label, RadioList
from rich.console import Console
from rich.markdown import Markdown

from revdict.category import CATEGORIES
from revdict import chat as chat_module
from revdict.progress import STAGES
from revdict.sort import SORT_MODES


ARPABET_VOWELS = frozenset(
    {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
)
LIVE_TUI_TOP_N = 50
COMPACT_PANE_MAX_COLUMNS = 99
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


@dataclass(frozen=True)
class TerminalTheme:
    """Semantic styles expressed through the user's terminal ANSI palette."""

    no_color: bool
    truecolor: bool
    colorfgbg: str | None

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
        *,
        truecolor_requested: bool | None = None,
    ) -> TerminalTheme:
        environment = os.environ if environment is None else environment
        no_color = "NO_COLOR" in environment
        truecolor = (
            not no_color
            and truecolor_requested is not False
            and environment.get("COLORTERM", "").casefold() in {"truecolor", "24bit"}
        )
        return cls(no_color, truecolor, environment.get("COLORFGBG"))

    @property
    def color_depth(self) -> ColorDepth:
        if self.no_color:
            return ColorDepth.DEPTH_1_BIT
        return ColorDepth.DEPTH_24_BIT if self.truecolor else ColorDepth.DEPTH_8_BIT

    @property
    def styles(self) -> dict[str, str]:
        if self.no_color:
            return {
                "query": "bold", "muted": "dim", "border": "", "section.title": "bold",
                "result.headword": "bold", "result.pos": "dim", "result.selected": "reverse dim",
                "result.sentiment.positive": "", "result.sentiment.negative": "", "result.confidence": "",
                "stress": "bold", "stress.no_color": "bold",
                "scrollbar.thumb": "", "scrollbar.track": "dim",
                "button": "dim", "button.focused": "bold underline", "button.arrow": "", "button.text": "",
            }
        return {
            "query": "bold", "muted": "dim", "border": "ansiblue", "section.title": "ansiblue bold",
            "result.headword": "ansicyan bold", "result.pos": "ansiblue dim", "result.selected": "reverse dim",
            "result.sentiment.positive": "ansigreen", "result.sentiment.negative": "ansired", "result.confidence": "ansimagenta",
            "stress": "#ffcc00 bold", "stress.no_color": "bold",
            "scrollbar.thumb": "ansiblue", "scrollbar.track": "dim",
            "button": "dim", "button.focused": "ansimagenta bold", "button.arrow": "", "button.text": "",
        }


class ProportionalScrollbarMargin(ScrollbarMargin):
    """A compact, readable scrollbar with no arrow-button chrome."""

    def __init__(self) -> None:
        super().__init__(display_arrows=False)

    def create_margin(self, window_render_info, width: int, height: int):
        content_height = window_render_info.content_height
        track_height = min(height, window_render_info.window_height)
        if content_height <= 0 or track_height <= 0:
            return []
        visible = len(window_render_info.displayed_lines)
        thumb_height = min(track_height, max(1, round(track_height * visible / content_height)))
        scrollable = max(1, content_height - window_render_info.window_height)
        top = round((track_height - thumb_height) * window_render_info.vertical_scroll / scrollable)
        fragments = []
        for row in range(track_height):
            is_thumb = top <= row < top + thumb_height
            fragments.append(("class:scrollbar.thumb" if is_thumb else "class:scrollbar.track", "█" if is_thumb else "░"))
            if row + 1 < track_height:
                fragments.append(("", "\n"))
        return fragments


class HairlineSection:
    """A titled section with one rule, replacing heavyweight box chrome."""

    def __init__(self, body, title: str, *, width=None, height=None) -> None:
        self.title = title
        title_rule = VSplit(
            [
                Window(width=1, height=1, char="─", style="class:border"),
                Label(f" {title} ", style="class:section.title", dont_extend_width=True),
                Window(height=1, char="─", style="class:border"),
            ],
            height=1,
        )
        self.container = HSplit([title_rule, body], padding=0, width=width, height=height)

    def __pt_container__(self):
        return self.container


class ValidationError(ValueError):
    """A control value cannot be submitted to the search backend."""


@dataclass(frozen=True)
class ControlSpec:
    label: str
    help_text: str
    kind: str = "text"
    choices: tuple[tuple[str | None, str], ...] = ()


SORT_CHOICES = tuple(
    zip(
        (None,) + SORT_MODES[1:],
        ("Relevance", "A–Z", "Z–A", "Shortest", "Longest", "Most common", "Least common", "Most formal", "Oldest", "Most modern", "Most lyrical"),
        strict=True,
    )
)
CATEGORY_CHOICES = (
    (None, "All words"), ("noun", "Nouns"), ("adjective", "Adjectives"),
    ("verb", "Verbs"), ("adverb", "Adverbs"), ("idiom_slang", "Idioms and slang"),
    ("old", "Historic and archaic"),
)


@dataclass
class SearchControls:
    """The TUI's single source of truth for each supported filter."""

    sort_mode: str | None = field(default=None, metadata={"spec": ControlSpec("Sort", "How candidates are ordered.", "radio", SORT_CHOICES)})
    category: str | None = field(default=None, metadata={"spec": ControlSpec("Category", "Limit candidates by part of speech or register.", "radio", CATEGORY_CHOICES)})
    syllables: int | None = field(default=None, metadata={"spec": ControlSpec("Syllables", "Whole number, zero or greater.")})
    primary_vowel: str | None = field(default=None, metadata={"spec": ControlSpec("Primary vowel", "ARPAbet vowel: " + ", ".join(sorted(ARPABET_VOWELS)))})
    rhymes_with: str | None = field(default=None, metadata={"spec": ControlSpec("Rhymes with", "A word whose final rhyme should match.")})
    sounds_like: str | None = field(default=None, metadata={"spec": ControlSpec("Sounds like", "A word whose pronunciation should match.")})
    meter: str | None = field(default=None, metadata={"spec": ControlSpec("Meter", "Use / for stressed and x for unstressed syllables.")})

    def validate(self) -> None:
        if self.sort_mode is not None and self.sort_mode not in SORT_MODES:
            raise ValidationError("Unknown sort mode.")
        if self.category is not None and self.category not in CATEGORIES:
            raise ValidationError("Unknown category.")
        if self.syllables is not None and self.syllables < 0:
            raise ValidationError("Syllables must be zero or greater.")
        if self.primary_vowel is not None and self.primary_vowel not in ARPABET_VOWELS:
            raise ValidationError("Primary vowel must be an ARPAbet vowel.")
        if self.meter is not None and any(character not in "/x" for character in self.meter):
            raise ValidationError("Meter may contain only / and x.")
        if self.rhymes_with is not None and not self.rhymes_with.strip():
            raise ValidationError("Rhymes-with target cannot be blank.")
        if self.sounds_like is not None and not self.sounds_like.strip():
            raise ValidationError("Sounds-like target cannot be blank.")

    def as_search_kwargs(self) -> dict[str, Any]:
        self.validate()
        return {item.name: getattr(self, item.name) for item in fields(self)}


@dataclass(frozen=True)
class UiAction:
    key: str
    description: str
    button_label: str | None = None


@dataclass(frozen=True)
class ChatSessionKey:
    """An exact dictionary sense, independent of the selected chat provider."""

    headword: str
    part_of_speech: str
    definition: str

    @classmethod
    def from_context(cls, context: chat_module.LexicalContext) -> ChatSessionKey:
        return cls(context.headword, context.part_of_speech, context.definition)


@dataclass
class ChatSession:
    bootstrap: str
    history: list[tuple[str, str]]
    transcript: str = ""
    streamed_answer: str = ""
    draft: str = ""


class ChatSessionRequestError(RuntimeError):
    """Associates an asynchronous provider failure with its initiating session."""

    def __init__(self, key: ChatSessionKey, error: Exception) -> None:
        self.key = key
        super().__init__(str(error))


ACTIONS = (
    UiAction("F1", "Toggle generated help", "Help"), UiAction("F2", "Open or close filters", "Filters"),
    UiAction("F3", "Toggle preview", "Preview"), UiAction("F4", "Toggle writing chat", "Chat"),
    UiAction("F5", "Open or save chat provider settings", "Setup"),
    UiAction("Enter (chat)", "Send chat message"), UiAction("Ctrl-R", "Cycle sort order"),
    UiAction("Ctrl-N / Ctrl-P", "Select next / previous result"), UiAction("Enter (query)", "Copy selected headword"),
    UiAction("Esc", "Clear query, then quit"), UiAction("Ctrl-C", "Quit"),
)
QUERY_HELP = (
    ("blue* / *bird / bl????rd", "prefix, suffix, and wildcard patterns"),
    ("//letters", "anagram"), ("-abcd / +abcd", "exclude letters / allowed letters"),
    ("pattern:meaning / :meaning", "combined or explicit meaning search"),
    ("**word** / expand:nasa", "phrase contains / initials expansion"),
)


def build_help_text() -> str:
    lines = ["Query syntax:"]
    lines.extend(f"  {query:<29} {description}" for query, description in QUERY_HELP)
    lines.extend(["", "Filters:"])
    for item in fields(SearchControls):
        spec: ControlSpec = item.metadata["spec"]
        choices = " (" + ", ".join(label for _, label in spec.choices) + ")" if spec.choices else ""
        lines.append(f"  {spec.label}: {spec.help_text}{choices}")
    lines.extend(["", "Keys:"])
    lines.extend(f"  {action.key:<15} {action.description}" for action in ACTIONS)
    return "\n".join(lines)


def _strip_ansi(value: str) -> str:
    return _ANSI_ESCAPE.sub("", value)


def format_candidate_preview(candidate: dict) -> str:
    """Return a plain-text version for tests, copy paths, and non-ANSI fallbacks."""
    lines = [f"{candidate['headword']} ({candidate['pos']})", "", candidate["definition"]]
    synonyms = candidate.get("synonyms") or []
    if synonyms:
        lines.extend(["", "Synonyms: " + ", ".join(synonyms)])
    lines.extend([f"Emotion: {candidate['label']} · {candidate['polarity']}", f"Match confidence: {candidate['relevance']}%"])
    if candidate.get("stress"):
        lines.append("Stress: " + _strip_ansi(candidate["stress"]).rstrip())
    examples = candidate.get("examples") or []
    if examples:
        lines.extend(["", 'Example: "' + examples[0] + '"'])
    return "\n".join(lines)


def _stress_fragments(value: str, *, no_color: bool) -> list[tuple[str, str]]:
    """Keep stress attributes while pinning the stress-marker hue itself."""
    fragments: list[tuple[str, str]] = []
    for style, text, *_ in to_formatted_text(ANSI(value.rstrip())):
        attributes = " ".join(attribute for attribute in ("bold", "italic", "underline", "dim") if attribute in style)
        is_stress_marker = "ansiyellow" in style or "#d7d700" in style
        if is_stress_marker:
            mapped_style = "class:stress.no_color" if no_color else "class:stress"
        elif "#" in style or "ansi" in style:
            mapped_style = "class:muted"
        else:
            mapped_style = attributes
        fragments.append((mapped_style, text))
    return fragments


def candidate_preview_fragments(candidate: dict, *, no_color: bool | None = None):
    """Render semantic preview fields without painting over the terminal theme."""
    no_color = "NO_COLOR" in os.environ if no_color is None else no_color
    fragments = [
        ("class:result.headword", candidate["headword"]),
        ("class:result.pos", f" ({candidate['pos']})"),
        ("", f"\n\n{candidate['definition']}"),
    ]
    synonyms = candidate.get("synonyms") or []
    if synonyms:
        fragments.append(("", "\n\nSynonyms: " + ", ".join(synonyms)))
    polarity = candidate.get("polarity", "neutral")
    sentiment_style = f"class:result.sentiment.{polarity}" if polarity in {"positive", "negative"} else ""
    fragments.extend([
        ("", "\nEmotion: "),
        (sentiment_style, f"{candidate['label']} · {polarity}"),
        ("", "\nMatch confidence: "),
        ("class:result.confidence", f"{candidate['relevance']}%"),
    ])
    if candidate.get("stress"):
        fragments.extend([("", "\nStress: "), *_stress_fragments(candidate["stress"], no_color=no_color)])
    examples = candidate.get("examples") or []
    if examples:
        fragments.append(("", '\n\nExample: "' + examples[0] + '"'))
    return fragments


def markdown_fragments(markdown: str, width: int = 120):
    """Render Markdown through Rich, retaining terminal attributes but no colours."""
    console = Console(
        file=StringIO(),
        force_terminal=True,
        color_system=None,
        width=max(1, width),
    )
    fragments: list[tuple[str, str]] = []

    def append(style: str, text: str) -> None:
        if not text:
            return
        if fragments and fragments[-1][0] == style:
            fragments[-1] = (style, fragments[-1][1] + text)
        else:
            fragments.append((style, text))

    def trim_line_padding() -> None:
        while fragments and fragments[-1][1].endswith(" "):
            style, text = fragments[-1]
            text = text.rstrip(" ")
            if text:
                fragments[-1] = (style, text)
                return
            fragments.pop()

    for segment in console.render(Markdown(markdown)):
        style = segment.style
        attributes = [
            name
            for name in ("bold", "italic", "underline", "reverse", "dim", "strike")
            if style is not None and getattr(style, name, False)
        ]
        style = " ".join(attributes)
        for part in segment.text.splitlines(keepends=True):
            if part.endswith("\n"):
                append(style, part[:-1])
                trim_line_padding()
                append("", "\n")
            else:
                append(style, part)
    while fragments and fragments[-1][1].endswith("\n"):
        style, text = fragments[-1]
        text = text.rstrip("\n")
        if text:
            fragments[-1] = (style, text)
            break
        fragments.pop()
    trim_line_padding()
    return fragments


def word_wrap_fragments(fragments, width: int):
    """Wrap formatted text at words while retaining every fragment's style."""
    width = max(1, width)
    lines: list[list[tuple[str, str]]] = []
    line: list[tuple[str, str]] = []
    word: list[tuple[str, str]] = []
    used = 0
    word_width = 0
    pending_space = False

    def append_fragment(target, style: str, text: str) -> None:
        if target and target[-1][0] == style:
            target[-1] = (style, target[-1][1] + text)
        else:
            target.append((style, text))

    def flush_word() -> None:
        nonlocal line, word, used, word_width, pending_space
        if not word:
            return
        separator = 1 if pending_space and line else 0
        if line and used + separator + word_width > width:
            lines.append(line)
            line = []
            used = 0
            separator = 0
        if separator:
            append_fragment(line, "", " ")
            used += 1
        line.extend(word)
        used += word_width
        word = []
        word_width = 0
        pending_space = False

    for style, text, *_ in fragments:
        for character in text:
            if character == "\n":
                flush_word()
                lines.append(line)
                line = []
                used = 0
                pending_space = False
            elif character.isspace():
                flush_word()
                pending_space = bool(line)
            else:
                append_fragment(word, style, character)
                word_width += get_cwidth(character)
    flush_word()
    if line or not lines:
        lines.append(line)
    return lines


def format_progress_line(states: dict[str, str], details: dict[str, str]) -> str:
    """Summarize real daemon events without a timer-driven redraw loop."""
    finished = sum(states.get(stage.id) in {"completed", "skipped"} for stage in STAGES)
    active = next((stage for stage in STAGES if states.get(stage.id) == "active"), None)
    current = active or STAGES[min(finished, len(STAGES) - 1)]
    detail = details.get(current.id)
    line = f"Searching {finished * 100 // len(STAGES)}% · {STAGES.index(current) + 1}/{len(STAGES)} · {current.label}"
    return f"{line} — {detail}" if detail else line


def _wrap_result_fragments(rows: list[dict], selected_index: int, width: int):
    """Wrap result rows by tokens, preserving styles and never splitting a word."""
    all_lines: list[list[tuple[str, str]]] = []
    line_rows: list[int] = []
    width = max(1, width)
    for index, row in enumerate(rows):
        selected = index == selected_index
        selected_style = "class:result.selected" if selected else ""
        header = [
            (selected_style, "❯ " if selected else "  "),
            (f"{selected_style} class:result.headword", row["headword"]),
            (f"{selected_style} class:result.pos", f"  ({row['pos']})  "),
        ]
        tokens = header + [(selected_style, word + (" " if number < len(row["definition"].split()) - 1 else "")) for number, word in enumerate(row["definition"].split())]
        line: list[tuple[str, str]] = []
        used = 0
        for style, token in tokens:
            token_width = get_cwidth(token)
            if line and used + token_width > width and token.strip():
                all_lines.append(line)
                line_rows.append(index)
                line = [(selected_style, "  ")]
                used = 2
                token = token.lstrip()
                token_width = get_cwidth(token)
            line.append((style, token))
            used += token_width
        all_lines.append(line or [(selected_style, "")])
        line_rows.append(index)
    return all_lines or [[("", "Type a meaning or word to search.")]], line_rows or [-1]


class ResultsControl(UIControl):
    def __init__(
        self,
        rows: Callable[[], list[dict]],
        selected: Callable[[], int],
        on_move: Callable[[int], None],
        on_select: Callable[[int], None],
    ) -> None:
        self._rows, self._selected = rows, selected
        self._on_move, self._on_select = on_move, on_select
        self._line_rows: list[int] = []

    def create_content(self, width: int, height: int | None) -> UIContent:
        lines, line_rows = _wrap_result_fragments(self._rows(), self._selected(), width)
        self._line_rows = line_rows
        selected_line = next((line for line, row in enumerate(line_rows) if row == self._selected()), 0)
        return UIContent(
            get_line=lambda i: lines[i],
            line_count=len(lines),
            cursor_position=Point(x=0, y=selected_line),
            show_cursor=False,
        )

    def mouse_handler(self, mouse_event: MouseEvent):
        if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
            self._on_move(1)
            return None
        if mouse_event.event_type == MouseEventType.SCROLL_UP:
            self._on_move(-1)
            return None
        if mouse_event.event_type == MouseEventType.MOUSE_DOWN and mouse_event.button == MouseButton.LEFT:
            if mouse_event.position.y < len(self._line_rows) and self._line_rows[mouse_event.position.y] >= 0:
                self._on_select(self._line_rows[mouse_event.position.y])
                return None
        return NotImplemented


class WordWrappedControl(UIControl):
    """A scrollable formatted-text control that never breaks words mid-token."""

    def __init__(self) -> None:
        self.fragments = []
        self._cursor_line = 0

    def reset_scroll(self) -> None:
        self._cursor_line = 0

    def set_cursor_line(self, line: int) -> None:
        self._cursor_line = max(0, line)

    def create_content(self, width: int, height: int | None) -> UIContent:
        lines = word_wrap_fragments(self.fragments, width)
        self._cursor_line = min(self._cursor_line, len(lines) - 1)
        return UIContent(
            get_line=lambda i: lines[i],
            line_count=len(lines),
            cursor_position=Point(x=0, y=self._cursor_line),
            show_cursor=False,
        )

    def mouse_handler(self, mouse_event: MouseEvent):
        return NotImplemented


class MarkdownControl(WordWrappedControl):
    """A Rich Markdown document adapted to prompt-toolkit's scrollable text API."""

    def __init__(self) -> None:
        super().__init__()
        self.markdown = ""

    def create_content(self, width: int, height: int | None) -> UIContent:
        self.fragments = markdown_fragments(self.markdown, width)
        return super().create_content(width, height)


class MouseScrollableWindow(Window):
    """A Window whose wheel handler persists a viewport change and repaints."""

    def _mouse_handler(self, mouse_event: MouseEvent):
        if mouse_event.event_type not in {MouseEventType.SCROLL_DOWN, MouseEventType.SCROLL_UP}:
            return super()._mouse_handler(mouse_event)
        info = self.render_info
        if info is None:
            return NotImplemented
        step = 1 if mouse_event.event_type == MouseEventType.SCROLL_DOWN else -1
        maximum = max(0, info.content_height - info.window_height)
        self.vertical_scroll = max(0, min(maximum, self.vertical_scroll + step))
        set_cursor_line = getattr(self.content, "set_cursor_line", None)
        if set_cursor_line is not None:
            set_cursor_line(self.vertical_scroll)
        get_app().invalidate()
        return None


class TrackingRadioList(RadioList):
    def __init__(self, *args, on_change: Callable[[], None], **kwargs) -> None:
        self._on_change = on_change
        super().__init__(*args, **kwargs)

    def _handle_enter(self) -> None:
        before = self.current_value
        super()._handle_enter()
        if self.current_value != before:
            self._on_change()


class DebouncedSearchController:
    """One long-lived worker coalesces input without subprocess or thread churn."""

    def __init__(self, execute: Callable[..., dict], on_result: Callable[[dict], None], on_error: Callable[[Exception], None], *, on_progress: Callable[[dict], None] | None = None, debounce_seconds: float = 0.2, callback_scheduler: Callable[[Callable[[], None]], None] | None = None) -> None:
        self._execute, self._on_result, self._on_error, self._on_progress = execute, on_result, on_error, on_progress
        self._debounce_seconds = debounce_seconds
        self._schedule_callback = callback_scheduler or (lambda callback: callback())
        self._condition = threading.Condition()
        self._pending: tuple[int, str, SearchControls, float] | None = None
        self._generation = 0
        self._closed = False
        self._worker = threading.Thread(target=self._work, name="revdict-search", daemon=True)
        self._worker.start()

    def request(self, query: str, controls: SearchControls) -> None:
        with self._condition:
            if self._closed:
                return
            self._generation += 1
            self._pending = (self._generation, query, controls, time.monotonic() + self._debounce_seconds)
            self._condition.notify()

    def clear(self) -> None:
        with self._condition:
            if not self._closed:
                self._generation += 1
                self._pending = None
                self._condition.notify()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._pending = None
            self._condition.notify()
        self._worker.join(timeout=0.2)

    def _work(self) -> None:
        while True:
            with self._condition:
                while not self._closed and self._pending is None:
                    self._condition.wait()
                if self._closed:
                    return
                generation, query, controls, due = self._pending
                delay = due - time.monotonic()
                if delay > 0:
                    self._condition.wait(delay)
                    continue
                self._pending = None
            try:
                result = self._execute(query, on_progress=lambda event: self._publish_progress(generation, event), **controls.as_search_kwargs())
            except Exception as error:
                self._schedule_callback(lambda error=error: self._publish_error(generation, error))
            else:
                self._schedule_callback(lambda: self._publish_result(generation, result))

    def _current(self, generation: int) -> bool:
        with self._condition:
            return not self._closed and generation == self._generation

    def _publish_result(self, generation: int, result: dict) -> None:
        if self._current(generation):
            self._on_result(result)

    def _publish_error(self, generation: int, error: Exception) -> None:
        if self._current(generation):
            self._on_error(error)

    def _publish_progress(self, generation: int, event: dict) -> None:
        if self._on_progress is not None:
            self._schedule_callback(lambda: self._current(generation) and self._on_progress(event))


class NativeTui:
    """A colour-neutral prompt-toolkit interface over the daemon-backed API."""

    def __init__(self, search_executor: Callable[..., dict], *, truecolor: bool | None = None) -> None:
        from prompt_toolkit.application import Application
        from prompt_toolkit.filters import Condition
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import ConditionalContainer, DynamicContainer, Layout
        from prompt_toolkit.layout.dimension import Dimension
        from prompt_toolkit.styles import Style
        from prompt_toolkit.widgets import Button, TextArea

        self.theme = TerminalTheme.from_environment(truecolor_requested=truecolor)
        self._search_executor, self._show_controls, self._show_help, self._show_preview = search_executor, False, False, True
        self._show_chat = False
        self._show_chat_settings = False
        self._chat_visible_before_settings = False
        self._rows: list[dict] = []
        self._selected_index = 0
        self._controls = SearchControls()
        self._stage_states = {stage.id: "pending" for stage in STAGES}
        self._stage_details: dict[str, str] = {}
        self._progress_visibility_token = 0
        self._progress_clear_handle = None
        try:
            self._chat_settings = chat_module.load_settings()
            self._chat_settings_error = None
        except chat_module.ChatConfigurationError as error:
            self._chat_settings = chat_module.ChatSettings.defaults()
            self._chat_settings_error = str(error)
        self._chat_sessions: dict[ChatSessionKey, ChatSession] = {}
        self._active_chat_session_key: ChatSessionKey | None = None
        self._chat_client = chat_module.ChatClient()
        self._chat_loading_settings = False
        self._chat_form_provider = self._chat_settings.active_provider
        self._chat_transcript_text = ""
        self._chat_spinner_active = False
        self._chat_spinner_frame = 0
        self._chat_chunk_lock = threading.Lock()
        self._chat_pending_chunks: list[tuple[ChatSessionKey, str]] = []
        self._chat_chunk_flush_scheduled = False
        self.query = TextArea(prompt="> ", multiline=False, height=1, style="class:query", focus_on_click=True)
        self.sort_field = TrackingRadioList(SORT_CHOICES, default=None, select_on_focus=True, on_change=self._schedule_search)
        self.category_field = TrackingRadioList(CATEGORY_CHOICES, default=None, select_on_focus=True, on_change=self._schedule_search)
        self.syllables_field = TextArea(prompt="Syllables: ", multiline=False, height=1)
        self.vowel_field = TextArea(prompt="Primary vowel: ", multiline=False, height=1)
        self.rhymes_field = TextArea(prompt="Rhymes with: ", multiline=False, height=1)
        self.sounds_field = TextArea(prompt="Sounds like: ", multiline=False, height=1)
        self.meter_field = TextArea(prompt="Meter: ", multiline=False, height=1)
        self.chat_header = Label(text="")
        self.chat_transcript_control = MarkdownControl()
        self.chat_transcript = MouseScrollableWindow(
            content=self.chat_transcript_control,
            wrap_lines=False,
            right_margins=[ProportionalScrollbarMargin()],
            always_hide_cursor=True,
        )
        self.chat_input = TextArea(prompt="You: ", multiline=True, height=Dimension(min=1, preferred=3, max=6), wrap_lines=True)
        self.chat_input_label = Label(text="Message · Enter sends", style="class:muted")
        self.chat_provider_field = TrackingRadioList(
            tuple((provider, provider.title()) for provider in chat_module.SUPPORTED_PROVIDERS),
            default=self._chat_settings.active_provider,
            select_on_focus=True,
            on_change=self._change_chat_provider,
        )
        active_chat_provider = self._active_chat_provider()
        self.chat_model_field = TextArea(prompt="Model: ", text=active_chat_provider.model, multiline=False, height=1)
        self.chat_endpoint_field = TextArea(prompt="Endpoint: ", text=active_chat_provider.base_url, multiline=False, height=1)
        self.chat_api_key_field = TextArea(prompt="API key: ", text=active_chat_provider.api_key or "", password=True, multiline=False, height=1)
        self.chat_known_models = Label(text="")
        self.results_control = ResultsControl(
            lambda: self._rows,
            lambda: self._selected_index,
            self._move_selection_from_mouse,
            self._select_result_from_mouse,
        )
        self.results = Window(content=self.results_control, wrap_lines=False, right_margins=[ProportionalScrollbarMargin()], always_hide_cursor=True)
        self.preview_control = WordWrappedControl()
        self.preview = MouseScrollableWindow(content=self.preview_control, wrap_lines=False, right_margins=[ProportionalScrollbarMargin()], always_hide_cursor=True)
        self.progress = Label(text="", dont_extend_height=True, wrap_lines=False)
        self.progress_container = ConditionalContainer(self.progress, Condition(lambda: bool(self.progress.text)))
        self.chat_progress = Label(text="", style="class:muted", dont_extend_height=True, wrap_lines=False)
        self.function_key_buttons = tuple(
            Button(
                f"{action.key} {action.button_label}",
                handler=lambda key=action.key: self._invoke_function_key(key),
                width=get_cwidth(f"{action.key} {action.button_label}"),
                left_symbol="",
                right_symbol="",
            )
            for action in ACTIONS
            if action.key.startswith("F")
        )
        self.function_key_bar = VSplit(self.function_key_buttons, padding=1, height=1)
        self.active_filters = Label(text="sort: relevance  category: all", style="class:muted")
        self.status = Label(text="", style="class:muted")
        self.status_container = ConditionalContainer(self.status, Condition(lambda: bool(self.status.text)))
        self.controls_section = HairlineSection(
            HSplit([self.sort_field, self.category_field, self.syllables_field, self.vowel_field, self.rhymes_field, self.sounds_field, self.meter_field], padding=0),
            "Search controls — arrows choose Sort and Category",
        )
        self.results_section = HairlineSection(self.results, "Results", width=Dimension(weight=3))
        self.preview_section = HairlineSection(self.preview, "Preview", width=Dimension(weight=2))
        self.side_by_side_panes = VSplit(
            [
                self.results_section,
                ConditionalContainer(self.preview_section, Condition(lambda: self._show_preview)),
            ],
            padding=1,
            height=Dimension(weight=1),
        )
        self.stacked_panes = HSplit(
            [
                HairlineSection(self.results, "Results", height=Dimension(weight=3)),
                ConditionalContainer(
                    HairlineSection(self.preview, "Preview", height=Dimension(weight=2)),
                    Condition(lambda: self._show_preview),
                ),
            ],
            padding=1,
            height=Dimension(weight=1),
        )
        panes = DynamicContainer(self._panes_for_current_terminal_width)
        self.chat_known_models_container = ConditionalContainer(
            self.chat_known_models,
            Condition(lambda: self._chat_settings.active_provider == "gemini"),
        )
        self.chat_settings_frame = HairlineSection(
            HSplit([self.chat_provider_field, self.chat_model_field, self.chat_endpoint_field, self.chat_api_key_field, self.chat_known_models_container], padding=0),
            "Chat provider settings — ↑/↓ provider · Tab fields · F5 saves locally",
        )
        chat_conversation = ConditionalContainer(
            HSplit([
                self.chat_header,
                self.chat_transcript,
                self.chat_input_label,
                self.chat_input,
            ], padding=1),
            Condition(lambda: not self._show_chat_settings),
        )
        self.chat_section = HairlineSection(
            HSplit([
                chat_conversation,
                ConditionalContainer(self.chat_settings_frame, Condition(lambda: self._show_chat_settings)),
            ], padding=1),
            "Writing assistant",
            height=Dimension(weight=1),
        )
        self.query_section = HairlineSection(self.query, "revdict")
        self.help_section = HairlineSection(Label(text=build_help_text()), "Help")
        self.root = HSplit([
            self.query_section, self.active_filters,
            ConditionalContainer(panes, Condition(lambda: not (self._show_chat or self._show_chat_settings))),
            ConditionalContainer(self.chat_section, Condition(lambda: self._show_chat or self._show_chat_settings)),
            ConditionalContainer(self.controls_section, Condition(lambda: self._show_controls)),
            ConditionalContainer(self.help_section, Condition(lambda: self._show_help)),
            self.status_container,
            self.progress_container,
            ConditionalContainer(self.chat_progress, Condition(lambda: self._chat_spinner_active)),
            self.function_key_bar,
        ], padding=0)
        bindings = KeyBindings()
        @bindings.add("f1")
        def toggle_help(event): self._invoke_function_key("F1")
        @bindings.add("f2")
        def toggle_controls(event): self._invoke_function_key("F2")
        @bindings.add("f3")
        def toggle_preview(event): self._invoke_function_key("F3")
        @bindings.add("f4")
        def toggle_chat(event): self._invoke_function_key("F4")
        @bindings.add("f5")
        def toggle_chat_settings(event): self._invoke_function_key("F5")
        @bindings.add("tab", filter=Condition(lambda: self._show_chat_settings))
        def next_chat_setting(event): self._focus_chat_setting(1)
        @bindings.add("s-tab", filter=Condition(lambda: self._show_chat_settings))
        def previous_chat_setting(event): self._focus_chat_setting(-1)
        @bindings.add("c-r")
        def cycle_sort(event):
            values = [value for value, _ in SORT_CHOICES]; current = self.sort_field.current_value
            self.sort_field.current_value = values[(values.index(current) + 1) % len(values)]
            self._schedule_search(); event.app.invalidate()
        @bindings.add("c-n")
        def next_result(event): self._move_selection(1); event.app.invalidate()
        @bindings.add("c-p")
        def previous_result(event): self._move_selection(-1); event.app.invalidate()
        @bindings.add("enter", filter=Condition(lambda: self.application.layout.current_window == self.query.window))
        def copy_result(event): self._copy_selected(); event.app.invalidate()
        @bindings.add("enter", filter=Condition(lambda: self.application.layout.current_window == self.chat_input.window))
        def send_chat(event): self._send_chat()
        @bindings.add("escape", eager=True)
        def clear_or_exit(event):
            self._clear_or_exit()
        @bindings.add("c-c")
        def quit_ui(event): self.close()
        self.application = Application(
            layout=Layout(self.root, focused_element=self.query),
            key_bindings=bindings,
            style=Style.from_dict(self.theme.styles),
            full_screen=True,
            mouse_support=True,
            color_depth=self.theme.color_depth,
        )
        # A bare Escape is otherwise delayed for 500ms while prompt-toolkit
        # waits to see whether it begins an Alt/terminal escape sequence.
        self.application.ttimeoutlen = 0.01
        self._controller = DebouncedSearchController(
            self._search_executor,
            self._receive_result,
            self._receive_error,
            on_progress=self._receive_progress,
            debounce_seconds=0.2,
            callback_scheduler=self._schedule_on_ui_thread,
        )
        self._chat_controller = chat_module.ChatController(
            self._execute_chat,
            lambda result: self._schedule_on_ui_thread(lambda: self._receive_chat_answer(result)),
            lambda error: self._schedule_on_ui_thread(lambda: self._receive_chat_error(error)),
        )
        self._update_chat_header()
        self._update_chat_known_models()
        self.query.buffer.on_text_changed += lambda _buffer: self._schedule_search()
        for item in (self.syllables_field, self.vowel_field, self.rhymes_field, self.sounds_field, self.meter_field):
            item.buffer.on_text_changed += lambda _buffer: self._schedule_search()

    def _progress_fragments(self):
        return [("bold", format_progress_line(self._stage_states, self._stage_details))]

    def _reset_progress(self) -> None:
        self._progress_visibility_token += 1
        if self._progress_clear_handle is not None:
            self._progress_clear_handle.cancel()
            self._progress_clear_handle = None
        self._stage_states = {stage.id: "pending" for stage in STAGES}
        self._stage_details = {}
        self.progress.text = self._progress_fragments()

    def _clear_completed_progress(self, token: int) -> None:
        if token != self._progress_visibility_token:
            return
        self._progress_clear_handle = None
        self.progress.text = []
        self.application.invalidate()

    def _schedule_completed_progress_clear(self) -> None:
        token = self._progress_visibility_token
        loop = self.application.loop
        if loop is not None:
            self._progress_clear_handle = loop.call_later(3, self._clear_completed_progress, token)

    def _hide_progress(self) -> None:
        self._progress_visibility_token += 1
        if self._progress_clear_handle is not None:
            self._progress_clear_handle.cancel()
            self._progress_clear_handle = None
        self.progress.text = []

    def _invoke_function_key(self, key: str) -> None:
        if key == "F1":
            self._show_help = not self._show_help
        elif key == "F2":
            self._show_controls = not self._show_controls
            self.application.layout.focus(self.sort_field if self._show_controls else self.query)
        elif key == "F3":
            self._show_preview = not self._show_preview
        elif key == "F4":
            self._toggle_chat()
        elif key == "F5":
            self._toggle_chat_settings()
        self.application.invalidate()

    def run(self) -> None:
        self.application.run()

    def close(self) -> None:
        self._chat_spinner_active = False
        if self._progress_clear_handle is not None:
            self._progress_clear_handle.cancel()
            self._progress_clear_handle = None
        self._controller.close()
        self._chat_controller.close()
        if self.application.future is not None:
            self.application.exit()

    def _clear_or_exit(self) -> None:
        if self.query.text:
            self.query.text = ""
        else:
            self.close()

    def _schedule_on_ui_thread(self, callback: Callable[[], None]) -> None:
        loop = self.application.loop
        if loop is not None:
            loop.call_soon_threadsafe(callback)

    def _read_controls(self) -> SearchControls:
        syllables = None
        if text := self.syllables_field.text.strip():
            try: syllables = int(text)
            except ValueError as error: raise ValidationError("Syllables must be a whole number.") from error
        return SearchControls(sort_mode=self.sort_field.current_value, category=self.category_field.current_value, syllables=syllables, primary_vowel=self.vowel_field.text.strip().upper() or None, rhymes_with=self.rhymes_field.text.strip() or None, sounds_like=self.sounds_field.text.strip() or None, meter=self.meter_field.text.strip() or None)

    def _schedule_search(self) -> None:
        query = self.query.text.strip()
        if not query:
            self._controller.clear(); self._rows = []; self._selected_index = 0; self.preview_control.fragments = []
            self._hide_progress(); self.status.text = ""; self.application.invalidate(); return
        try:
            controls = self._read_controls(); controls.validate()
        except ValidationError as error:
            self.status.text = f"Invalid filter: {error}"; self.application.invalidate(); return
        self._controls = controls; self._set_active_filters(); self._reset_progress()
        self.status.text = "Searching…"; self._controller.request(query, controls); self.application.invalidate()

    def _set_active_filters(self) -> None:
        active = [f"sort: {self._controls.sort_mode or 'relevance'}", f"category: {self._controls.category or 'all'}"]
        for item in fields(SearchControls):
            value = getattr(self._controls, item.name)
            if value is not None and item.name not in {"sort_mode", "category"}: active.append(f"{item.name.replace('_', ' ')}: {value}")
        self.active_filters.text = "  ".join(active)

    def _receive_progress(self, event: dict) -> None:
        self._stage_states[event["id"]] = event["state"]
        if detail := event.get("detail"):
            self._stage_details[event["id"]] = detail
        self.progress.text = self._progress_fragments()
        self.application.invalidate()

    def _receive_result(self, result: dict) -> None:
        self._rows = self._result_rows(result); self._selected_index = 0
        self.status.text = f"{len(self._rows)} result{'s' if len(self._rows) != 1 else ''}"; self._render_selection(); self._schedule_completed_progress_clear(); self.application.invalidate()

    def _receive_error(self, error: Exception) -> None:
        self.status.text = f"Search error: {error}"; self.application.invalidate()

    def _panes_for_width(self, columns: int):
        return self.stacked_panes if columns <= COMPACT_PANE_MAX_COLUMNS else self.side_by_side_panes

    def _panes_for_current_terminal_width(self):
        application = getattr(self, "application", None)
        if application is None:
            return self.side_by_side_panes
        return self._panes_for_width(application.output.get_size().columns)

    @staticmethod
    def _result_rows(result: dict) -> list[dict]:
        rows = []
        exact_match = result.get("exact_match")
        if exact_match and exact_match.get("senses"):
            sense = exact_match["senses"][0]
            rows.append({"headword": exact_match["headword"], "pos": sense["pos"], "definition": sense["definition"], "stress": sense.get("stress"), "label": sense["label"], "polarity": sense["polarity"], "synonyms": sense.get("synonyms") or [], "examples": sense.get("examples") or [], "relevance": 100})
        rows.extend(result.get("candidates") or [])
        return rows

    def _move_selection(self, step: int) -> None:
        if self._rows:
            self._selected_index = max(0, min(len(self._rows) - 1, self._selected_index + step)); self._render_selection()

    def _move_selection_from_mouse(self, step: int) -> None:
        self._move_selection(step)
        self.application.invalidate()

    def _select_result_from_mouse(self, index: int) -> None:
        if self._rows:
            self._selected_index = max(0, min(len(self._rows) - 1, index))
            self._render_selection()
            self.application.invalidate()

    def _render_selection(self) -> None:
        self.preview_control.reset_scroll()
        self.preview_control.fragments = candidate_preview_fragments(self._rows[self._selected_index], no_color=self.theme.no_color) if self._rows else []
        if self._show_chat:
            context = self._current_chat_context()
            if context is not None:
                self._activate_chat_session(context)
        self._update_chat_header()

    def _active_chat_provider(self) -> chat_module.ProviderSettings:
        return self._chat_settings.providers[self._chat_settings.active_provider]

    def _current_chat_context(self) -> chat_module.LexicalContext | None:
        if not self._rows:
            return None
        row = self._rows[self._selected_index]
        return chat_module.LexicalContext(self.query.text.strip() or "(no search query)", row["headword"], row["definition"], row["pos"])

    @property
    def _active_chat_session(self) -> ChatSession:
        assert self._active_chat_session_key is not None
        return self._chat_sessions[self._active_chat_session_key]

    def _activate_chat_session(self, context: chat_module.LexicalContext) -> ChatSessionKey:
        key = ChatSessionKey.from_context(context)
        previous_key = self._active_chat_session_key
        if previous_key is not None and previous_key != key:
            self._chat_sessions[previous_key].draft = self.chat_input.text
        if key not in self._chat_sessions:
            self._chat_sessions[key] = ChatSession(
                bootstrap=chat_module.lexical_bootstrap(context),
                history=[],
                draft=chat_module.default_writing_prompt(context),
            )
        self._active_chat_session_key = key
        self._chat_transcript_text = self._active_chat_session.transcript
        self._render_chat_transcript()
        self.chat_input.text = self._active_chat_session.draft
        self.chat_input.buffer.cursor_position = len(self.chat_input.text)
        return key

    def _sync_active_chat_transcript(self) -> None:
        if self._active_chat_session_key is None:
            return
        self._chat_transcript_text = self._active_chat_session.transcript
        self._render_chat_transcript()

    def _update_chat_header(self) -> None:
        provider = self._active_chat_provider()
        context = self._current_chat_context()
        header = [("class:muted", f"Provider: {provider.provider} · Model: {provider.model}")]
        if context is None:
            header.append(("class:muted", " · Select a result to add lexical context"))
        else:
            header.extend([
                ("class:muted", " · Context: "),
                ("class:result.headword", context.headword),
                ("class:result.pos", f" ({context.part_of_speech})"),
            ])
        self.chat_header.text = header

    def _update_chat_known_models(self) -> None:
        if self._chat_settings.active_provider != "gemini":
            self.chat_known_models.text = ""
            return
        models = self._chat_settings.gemini_models
        self.chat_known_models.text = "Cached Gemini models: " + (", ".join(models) if models else "none — run chat-config --provider gemini --test")

    def _toggle_chat(self) -> None:
        self._show_chat = not self._show_chat
        if self._show_chat:
            context = self._current_chat_context()
            if context is not None:
                self._activate_chat_session(context)
            elif context is None:
                self.status.text = "Select a result first; chat will then include its definition."
            self.application.layout.focus(self.chat_input)
        else:
            if self._active_chat_session_key is not None:
                self._active_chat_session.draft = self.chat_input.text
            self._show_chat_settings = False
            self.application.layout.focus(self.query)
        self._update_chat_header()
        self.application.invalidate()

    def _change_chat_provider(self) -> None:
        if self._chat_loading_settings:
            return
        self._save_chat_provider_form()
        self._chat_settings.active_provider = self.chat_provider_field.current_value
        self._load_chat_provider_form()

    def _load_chat_provider_form(self) -> None:
        provider = self._active_chat_provider()
        self._chat_form_provider = provider.provider
        self._chat_loading_settings = True
        try:
            self.chat_provider_field.current_value = provider.provider
            self.chat_model_field.text = provider.model
            self.chat_endpoint_field.text = provider.base_url
            self.chat_api_key_field.text = provider.api_key or ""
            for field in (self.chat_model_field, self.chat_endpoint_field, self.chat_api_key_field):
                field.buffer.cursor_position = len(field.text)
        finally:
            self._chat_loading_settings = False
        self._update_chat_header()
        self._update_chat_known_models()

    def _save_chat_provider_form(self) -> None:
        provider_name = self._chat_form_provider
        old = self._chat_settings.providers[provider_name]
        self._chat_settings.providers[provider_name] = chat_module.ProviderSettings(
            provider_name,
            self.chat_endpoint_field.text.strip(),
            self.chat_model_field.text.strip(),
            old.api_key_env,
            self.chat_api_key_field.text.strip() or None,
        )
        self._chat_settings.active_provider = provider_name

    def _toggle_chat_settings(self) -> None:
        if self._show_chat_settings:
            self._save_chat_provider_form()
            chat_module.save_settings(self._chat_settings)
            self._show_chat_settings = False
            self.status.text = "Chat provider settings saved locally."
            if self._chat_visible_before_settings:
                self.application.layout.focus(self.chat_input)
            else:
                self.application.layout.focus(self.query)
            self._show_chat = self._chat_visible_before_settings
        else:
            self._chat_visible_before_settings = self._show_chat
            self._show_chat_settings = True
            self._load_chat_provider_form()
            self.application.layout.focus(self.chat_endpoint_field)
        self.application.invalidate()

    def _focus_chat_setting(self, step: int) -> None:
        fields = (self.chat_endpoint_field, self.chat_model_field, self.chat_api_key_field, self.chat_provider_field)
        current = self.application.layout.current_window
        index = next((number for number, field in enumerate(fields) if field.window == current), 0)
        self.application.layout.focus(fields[(index + step) % len(fields)])

    def _send_chat(self) -> None:
        context = self._current_chat_context()
        message = self.chat_input.text.strip()
        if context is None:
            self.status.text = "Select a result before sending a chat message."
        elif not message:
            self.status.text = "Write a message before sending it."
        elif self._chat_controller.busy:
            self.status.text = "Chat is still responding; wait before sending another message."
        else:
            key = self._activate_chat_session(context)
            session = self._active_chat_session
            request_message = message if session.history else f"{session.bootstrap}\n\nUser request:\n{message}"
            request = (key, self._active_chat_provider(), list(session.history), request_message)
            if not self._chat_controller.send(request):
                self.status.text = "Write a message before sending it."
                self.application.invalidate()
                return
            session.history.append(("user", request_message))
            self._append_chat_turn("You", message, key)
            self._begin_chat_response(key)
            session.draft = ""
            self.chat_input.text = ""
            self.status.text = "Writing assistant is responding…"
        self.application.invalidate()

    def _execute_chat(self, request: tuple[ChatSessionKey, chat_module.ProviderSettings, list[tuple[str, str]], str]) -> tuple[ChatSessionKey, str]:
        key, provider, history, message = request
        try:
            answer = self._chat_client.stream(provider, history, message, lambda chunk: self._queue_chat_chunk(key, chunk))
        except Exception as error:
            raise ChatSessionRequestError(key, error) from error
        return key, answer

    def _receive_chat_answer(self, result: tuple[ChatSessionKey, str]) -> None:
        key, answer = result
        self._flush_chat_chunks()
        session = self._chat_sessions[key]
        session.history.append(("assistant", answer))
        if session.streamed_answer:
            session.transcript += "\n\n"
        else:
            session.transcript += f"{answer}\n\n"
        if key == self._active_chat_session_key:
            self._sync_active_chat_transcript()
        self._stop_chat_spinner()
        self.status.text = "Chat response received."
        self.application.invalidate()

    def _receive_chat_error(self, error: Exception) -> None:
        key = error.key if isinstance(error, ChatSessionRequestError) else self._active_chat_session_key
        if key is None:
            self.status.text = "Chat request failed."
            self._stop_chat_spinner()
            self.application.invalidate()
            return
        self._flush_chat_chunks()
        session = self._chat_sessions[key]
        if session.streamed_answer:
            session.transcript += f"\n\n**Chat error:** {error}\n\n"
        else:
            session.transcript += f"**Chat error:** {error}\n\n"
        if key == self._active_chat_session_key:
            self._sync_active_chat_transcript()
        self._stop_chat_spinner()
        self.status.text = "Chat request failed."
        self.application.invalidate()

    def _append_chat_turn(self, speaker: str, text: str, key: ChatSessionKey | None = None) -> None:
        key = self._active_chat_session_key if key is None else key
        if key is None:
            return
        session = self._chat_sessions[key]
        session.transcript += f"**{speaker}:**\n{text}\n\n"
        if key == self._active_chat_session_key:
            self._sync_active_chat_transcript()

    def _render_chat_transcript(self) -> None:
        self.chat_transcript_control.markdown = self._chat_transcript_text
        self.chat_transcript_control.set_cursor_line(1_000_000)

    def _begin_chat_response(self, key: ChatSessionKey) -> None:
        session = self._chat_sessions[key]
        session.streamed_answer = ""
        session.transcript += "**Assistant:**\n"
        if key == self._active_chat_session_key:
            self._sync_active_chat_transcript()
        self._chat_spinner_active = True
        self._chat_spinner_frame = 0
        self.chat_progress.text = "⠋ Writing assistant · streaming"
        if self.application.loop is not None:
            self.application.create_background_task(self._animate_chat_spinner())

    async def _animate_chat_spinner(self) -> None:
        frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        while self._chat_spinner_active:
            self.chat_progress.text = f"{frames[self._chat_spinner_frame % len(frames)]} Writing assistant · streaming"
            self._chat_spinner_frame += 1
            self.application.invalidate()
            await asyncio.sleep(0.12)

    def _stop_chat_spinner(self) -> None:
        self._chat_spinner_active = False
        self.chat_progress.text = ""

    def _queue_chat_chunk(self, key: ChatSessionKey, chunk: str) -> None:
        """Coalesce network chunks to at most one terminal redraw every 33 ms."""
        if not chunk:
            return
        with self._chat_chunk_lock:
            self._chat_pending_chunks.append((key, chunk))
            if self._chat_chunk_flush_scheduled:
                return
            self._chat_chunk_flush_scheduled = True
        self._schedule_on_ui_thread(self._schedule_chat_chunk_flush)

    def _schedule_chat_chunk_flush(self) -> None:
        loop = self.application.loop
        if loop is None:
            return
        loop.call_later(1 / 30, self._flush_chat_chunks)

    def _flush_chat_chunks(self) -> None:
        with self._chat_chunk_lock:
            chunks = self._chat_pending_chunks
            self._chat_pending_chunks = []
            self._chat_chunk_flush_scheduled = False
        if not chunks:
            return
        by_session: dict[ChatSessionKey, list[str]] = {}
        for key, chunk in chunks:
            by_session.setdefault(key, []).append(chunk)
        for key, session_chunks in by_session.items():
            session = self._chat_sessions[key]
            text = "".join(session_chunks)
            session.streamed_answer += text
            session.transcript += text
        if self._active_chat_session_key in by_session:
            self._sync_active_chat_transcript()
        self.application.invalidate()

    def _copy_selected(self) -> None:
        if self._rows:
            from revdict.cli import _run_copy_selection
            headword = self._rows[self._selected_index]["headword"]; _run_copy_selection(headword); self.status.text = f"Copied: {headword}"


def run(*, truecolor: bool | None = None) -> None:
    from revdict.cli import _get_search_result_with_progress
    execute = lambda query, on_progress=lambda _event: None, **kwargs: _get_search_result_with_progress(query, LIVE_TUI_TOP_N, on_progress, **kwargs)
    ui = NativeTui(execute) if truecolor is None else NativeTui(execute, truecolor=truecolor)
    ui.run()
