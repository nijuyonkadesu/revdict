"""The native, daemon-backed terminal interface for revdict."""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from typing import Any

from prompt_toolkit.layout.controls import FormattedTextControl, UIContent, UIControl
from prompt_toolkit.mouse_events import MouseEvent
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.widgets import RadioList

from revdict.category import CATEGORIES
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
    UiAction("F3", "Toggle preview"), UiAction("Ctrl-R", "Cycle sort order"),
    UiAction("Ctrl-N / Ctrl-P", "Select next / previous result"), UiAction("Enter", "Copy selected headword"),
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
        fragments.append(("", "\nStress: "))
        fragments.extend(_stress_ansi_fragments(candidate["stress"].rstrip()))
    return fragments


def _stress_ansi_fragments(value: str):
    """Pass stressmark's SGR bytes through unchanged, independent of UI colour depth."""
    fragments = []
    position = 0
    for match in _ANSI_ESCAPE.finditer(value):
        if match.start() > position:
            fragments.append(("", value[position:match.start()]))
        fragments.append(("[ZeroWidthEscape]", match.group()))
        position = match.end()
    if position < len(value):
        fragments.append(("", value[position:]))
    return fragments


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
    def __init__(self, rows: Callable[[], list[dict]], selected: Callable[[], int]) -> None:
        self._rows, self._selected = rows, selected

    def create_content(self, width: int, height: int | None) -> UIContent:
        lines, _ = _wrap_result_fragments(self._rows(), self._selected(), width)
        return UIContent(get_line=lambda i: lines[i], line_count=len(lines), show_cursor=False)

    def mouse_handler(self, mouse_event: MouseEvent):
        return NotImplemented


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

    def __init__(self, execute: Callable[..., dict], on_result: Callable[[dict], None], on_error: Callable[[Exception], None], *, on_progress: Callable[[dict], None] | None = None, debounce_seconds: float = 0.1, callback_scheduler: Callable[[Callable[[], None]], None] | None = None) -> None:
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
        from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, VSplit, Window
        from prompt_toolkit.layout.dimension import Dimension
        from prompt_toolkit.layout.margins import ScrollbarMargin
        from prompt_toolkit.styles import Style
        from prompt_toolkit.widgets import Frame, Label, TextArea

        self._search_executor, self._show_controls, self._show_help, self._show_preview = search_executor, False, False, True
        self._rows: list[dict] = []
        self._selected_index = 0
        self._controls = SearchControls()
        self._stage_states = {stage.id: "pending" for stage in STAGES}
        self._stage_details: dict[str, str] = {}
        self.query = TextArea(prompt="> ", multiline=False, height=1, style="class:query", focus_on_click=True)
        self.sort_field = TrackingRadioList(SORT_CHOICES, default=None, select_on_focus=True, on_change=self._schedule_search)
        self.category_field = TrackingRadioList(CATEGORY_CHOICES, default=None, select_on_focus=True, on_change=self._schedule_search)
        self.syllables_field = TextArea(prompt="Syllables: ", multiline=False, height=1)
        self.vowel_field = TextArea(prompt="Primary vowel: ", multiline=False, height=1)
        self.rhymes_field = TextArea(prompt="Rhymes with: ", multiline=False, height=1)
        self.sounds_field = TextArea(prompt="Sounds like: ", multiline=False, height=1)
        self.meter_field = TextArea(prompt="Meter: ", multiline=False, height=1)
        self.results_control = ResultsControl(lambda: self._rows, lambda: self._selected_index)
        self.results = Window(content=self.results_control, wrap_lines=False, right_margins=[ScrollbarMargin(display_arrows=True)], always_hide_cursor=True)
        self.preview_control = FormattedTextControl("")
        self.preview = Window(content=self.preview_control, wrap_lines=True, right_margins=[ScrollbarMargin(display_arrows=True)], always_hide_cursor=True)
        self.progress = Label(text=self._progress_fragments(), dont_extend_height=True, wrap_lines=False)
        self.active_filters = Label(text="sort: relevance  category: all", style="class:muted")
        self.status = Label(text="F1 help · F2 filters · F3 preview · Ctrl-R sort · Enter copy", style="class:muted")
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
        root = HSplit([
            Frame(self.query, title="revdict"), self.active_filters,
            panes,
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
                    "result.pos": "underline",
                    "result.selected": "reverse",
                }
            ),
            full_screen=True,
            mouse_support=True,
        )
        self._controller = DebouncedSearchController(self._search_executor, self._receive_result, self._receive_error, on_progress=self._receive_progress, callback_scheduler=self._schedule_on_ui_thread)
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
            self._controller.clear(); self._rows = []; self._selected_index = 0; self.preview_control.text = ""
            self._reset_progress(); self.status.text = "F1 help · F2 filters · F3 preview · Ctrl-R sort · Enter copy"; self.application.invalidate(); return
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

    def _render_selection(self) -> None:
        self.preview_control.text = candidate_preview_fragments(self._rows[self._selected_index]) if self._rows else ""

    def _copy_selected(self) -> None:
        if self._rows:
            from revdict.cli import _run_copy_selection
            headword = self._rows[self._selected_index]["headword"]; _run_copy_selection(headword); self.status.text = f"Copied: {headword}"


def run() -> None:
    from revdict.cli import _get_search_result_with_progress
    NativeTui(lambda query, on_progress=lambda _event: None, **kwargs: _get_search_result_with_progress(query, LIVE_TUI_TOP_N, on_progress, **kwargs)).run()
