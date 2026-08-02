"""The native, daemon-backed terminal interface for revdict."""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from typing import Any

from prompt_toolkit.formatted_text import ANSI, to_formatted_text
from prompt_toolkit.application.current import get_app
from prompt_toolkit.layout import Window
from prompt_toolkit.layout.controls import UIContent, UIControl
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from prompt_toolkit.data_structures import Point
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.widgets import RadioList

from revdict.category import CATEGORIES
from revdict import chat as chat_module
from revdict.progress import STAGES
from revdict.sort import SORT_MODES


ARPABET_VOWELS = frozenset(
    {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
)
LIVE_TUI_TOP_N = 50
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


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


ACTIONS = (
    UiAction("F1", "Toggle generated help"), UiAction("F2", "Open or close filters"),
    UiAction("F3", "Toggle preview"), UiAction("F4", "Toggle writing chat"),
    UiAction("F5", "Open or save chat provider settings"),
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


def candidate_preview_fragments(candidate: dict):
    text = format_candidate_preview({**candidate, "stress": None})
    fragments = [("", text)]
    if candidate.get("stress"):
        fragments.extend([("", "\nStress: "), *to_formatted_text(ANSI(candidate["stress"].rstrip()))])
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

    def __init__(self, search_executor: Callable[..., dict]) -> None:
        from prompt_toolkit.application import Application
        from prompt_toolkit.filters import Condition
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, VSplit
        from prompt_toolkit.layout.dimension import Dimension
        from prompt_toolkit.layout.margins import ScrollbarMargin
        from prompt_toolkit.output.color_depth import ColorDepth
        from prompt_toolkit.styles import Style
        from prompt_toolkit.widgets import Frame, Label, TextArea

        self._search_executor, self._show_controls, self._show_help, self._show_preview = search_executor, False, False, True
        self._show_chat = False
        self._show_chat_settings = False
        self._chat_visible_before_settings = False
        self._rows: list[dict] = []
        self._selected_index = 0
        self._controls = SearchControls()
        self._stage_states = {stage.id: "pending" for stage in STAGES}
        self._stage_details: dict[str, str] = {}
        try:
            self._chat_settings = chat_module.load_settings()
            self._chat_settings_error = None
        except chat_module.ChatConfigurationError as error:
            self._chat_settings = chat_module.ChatSettings.defaults()
            self._chat_settings_error = str(error)
        self._chat_history: list[tuple[str, str]] = []
        self._chat_client = chat_module.ChatClient()
        self._chat_loading_settings = False
        self._chat_form_provider = self._chat_settings.active_provider
        self.query = TextArea(prompt="> ", multiline=False, height=1, style="class:query", focus_on_click=True)
        self.sort_field = TrackingRadioList(SORT_CHOICES, default=None, select_on_focus=True, on_change=self._schedule_search)
        self.category_field = TrackingRadioList(CATEGORY_CHOICES, default=None, select_on_focus=True, on_change=self._schedule_search)
        self.syllables_field = TextArea(prompt="Syllables: ", multiline=False, height=1)
        self.vowel_field = TextArea(prompt="Primary vowel: ", multiline=False, height=1)
        self.rhymes_field = TextArea(prompt="Rhymes with: ", multiline=False, height=1)
        self.sounds_field = TextArea(prompt="Sounds like: ", multiline=False, height=1)
        self.meter_field = TextArea(prompt="Meter: ", multiline=False, height=1)
        self.chat_header = Label(text="")
        self.chat_transcript = TextArea(text="", multiline=True, read_only=True, focusable=False, wrap_lines=True, scrollbar=True)
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
        self.results = Window(content=self.results_control, wrap_lines=False, right_margins=[ScrollbarMargin(display_arrows=True)], always_hide_cursor=True)
        self.preview_control = WordWrappedControl()
        self.preview = MouseScrollableWindow(content=self.preview_control, wrap_lines=False, right_margins=[ScrollbarMargin(display_arrows=True)], always_hide_cursor=True)
        self.progress = Label(text=self._progress_fragments(), dont_extend_height=True, wrap_lines=False)
        self.active_filters = Label(text="sort: relevance  category: all", style="class:muted")
        self.status = Label(text="F1 help · F2 filters · F3 preview · F4 chat · Ctrl-R sort · Enter copy", style="class:muted")
        controls = Frame(HSplit([self.sort_field, self.category_field, self.syllables_field, self.vowel_field, self.rhymes_field, self.sounds_field, self.meter_field], padding=0), title="Search controls — arrows choose Sort and Category")
        results_frame = Frame(self.results, title="Results", width=Dimension(weight=3))
        preview_frame = Frame(self.preview, title="Preview", width=Dimension(weight=2))
        panes = VSplit(
            [
                results_frame,
                ConditionalContainer(preview_frame, Condition(lambda: self._show_preview)),
            ],
            padding=1,
            height=Dimension(weight=1),
        )
        self.chat_settings_frame = Frame(
            HSplit([self.chat_provider_field, self.chat_model_field, self.chat_endpoint_field, self.chat_api_key_field, self.chat_known_models], padding=0),
            title="Chat provider settings — ↑/↓ provider · Tab fields · F5 saves locally",
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
        chat_panel = Frame(
            HSplit([
                chat_conversation,
                ConditionalContainer(self.chat_settings_frame, Condition(lambda: self._show_chat_settings)),
            ], padding=1),
            title="Writing assistant",
            height=Dimension(weight=1),
        )
        root = HSplit([
            Frame(self.query, title="revdict"), self.active_filters,
            ConditionalContainer(panes, Condition(lambda: not (self._show_chat or self._show_chat_settings))),
            ConditionalContainer(chat_panel, Condition(lambda: self._show_chat or self._show_chat_settings)),
            ConditionalContainer(controls, Condition(lambda: self._show_controls)),
            ConditionalContainer(Frame(Label(text=build_help_text()), title="Help"), Condition(lambda: self._show_help)),
            self.status,
            self.progress,
        ], padding=1)
        bindings = KeyBindings()
        @bindings.add("f1")
        def toggle_help(event): self._show_help = not self._show_help; event.app.invalidate()
        @bindings.add("f2")
        def toggle_controls(event):
            self._show_controls = not self._show_controls
            event.app.layout.focus(self.sort_field if self._show_controls else self.query)
            event.app.invalidate()
        @bindings.add("f3")
        def toggle_preview(event): self._show_preview = not self._show_preview; event.app.invalidate()
        @bindings.add("f4")
        def toggle_chat(event): self._toggle_chat()
        @bindings.add("f5")
        def toggle_chat_settings(event): self._toggle_chat_settings()
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
            layout=Layout(root, focused_element=self.query),
            key_bindings=bindings,
            style=Style.from_dict(
                {
                    "query": "bold",
                    "muted": "dim",
                    "result.headword": "bold",
                    "result.pos": "dim",
                    "result.selected": "reverse",
                }
            ),
            full_screen=True,
            mouse_support=True,
            color_depth=ColorDepth.DEPTH_8_BIT,
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
            lambda answer: self._schedule_on_ui_thread(lambda: self._receive_chat_answer(answer)),
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
        self._stage_states = {stage.id: "pending" for stage in STAGES}
        self._stage_details = {}
        self.progress.text = self._progress_fragments()

    def run(self) -> None:
        self.application.run()

    def close(self) -> None:
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
            self._reset_progress(); self.status.text = "F1 help · F2 filters · F3 preview · F4 chat · Ctrl-R sort · Enter copy"; self.application.invalidate(); return
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
        self.status.text = f"{len(self._rows)} result{'s' if len(self._rows) != 1 else ''}"; self._render_selection(); self.application.invalidate()

    def _receive_error(self, error: Exception) -> None:
        self.status.text = f"Search error: {error}"; self.application.invalidate()

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
        self.preview_control.fragments = candidate_preview_fragments(self._rows[self._selected_index]) if self._rows else []
        self._update_chat_header()

    def _active_chat_provider(self) -> chat_module.ProviderSettings:
        return self._chat_settings.providers[self._chat_settings.active_provider]

    def _current_chat_context(self) -> chat_module.LexicalContext | None:
        if not self._rows:
            return None
        row = self._rows[self._selected_index]
        return chat_module.LexicalContext(self.query.text.strip() or "(no search query)", row["headword"], row["definition"], row["pos"])

    def _update_chat_header(self) -> None:
        provider = self._active_chat_provider()
        context = self._current_chat_context()
        subject = f" · Context: {context.headword} ({context.part_of_speech})" if context else " · Select a result to add lexical context"
        self.chat_header.text = f"Provider: {provider.provider} · Model: {provider.model}{subject}"

    def _update_chat_known_models(self) -> None:
        models = self._chat_settings.gemini_models
        self.chat_known_models.text = "Cached Gemini models: " + (", ".join(models) if models else "none — run chat-config --provider gemini --test")

    def _toggle_chat(self) -> None:
        self._show_chat = not self._show_chat
        if self._show_chat:
            context = self._current_chat_context()
            if context is not None and not self._chat_history and not self.chat_input.text:
                self.chat_input.text = chat_module.default_writing_prompt(context)
                self.chat_input.buffer.cursor_position = len(self.chat_input.text)
            elif context is None:
                self.status.text = "Select a result first; chat will then include its definition."
            self.application.layout.focus(self.chat_input)
        else:
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
        elif self._chat_controller.busy:
            self.status.text = "Chat is still responding; wait before sending another message."
        elif self._chat_controller.send((self._active_chat_provider(), context, list(self._chat_history), message)):
            self._chat_history.append(("user", message))
            self._append_chat_turn("You", message)
            self.chat_input.text = ""
            self.status.text = "Writing assistant is responding…"
        else:
            self.status.text = "Write a message before sending it."
        self.application.invalidate()

    def _execute_chat(self, request: tuple[chat_module.ProviderSettings, chat_module.LexicalContext, list[tuple[str, str]], str]) -> str:
        provider, context, history, message = request
        return self._chat_client.complete(provider, context, history, message)

    def _receive_chat_answer(self, answer: str) -> None:
        self._chat_history.append(("assistant", answer))
        self._append_chat_turn("Assistant", answer)
        self.status.text = "Chat response received."
        self.application.invalidate()

    def _receive_chat_error(self, error: Exception) -> None:
        self._append_chat_turn("Chat error", str(error))
        self.status.text = "Chat request failed."
        self.application.invalidate()

    def _append_chat_turn(self, speaker: str, text: str) -> None:
        self.chat_transcript.text += f"{speaker}:\n{text}\n\n"
        self.chat_transcript.buffer.cursor_position = len(self.chat_transcript.text)

    def _copy_selected(self) -> None:
        if self._rows:
            from revdict.cli import _run_copy_selection
            headword = self._rows[self._selected_index]["headword"]; _run_copy_selection(headword); self.status.text = f"Copied: {headword}"


def run() -> None:
    from revdict.cli import _get_search_result_with_progress
    NativeTui(lambda query, on_progress=lambda _event: None, **kwargs: _get_search_result_with_progress(query, LIVE_TUI_TOP_N, on_progress, **kwargs)).run()
