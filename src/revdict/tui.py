"""State and scheduling primitives for revdict's native terminal UI.

The prompt-toolkit rendering layer intentionally sits on top of these small,
terminal-independent objects so search correctness is covered without a TTY.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
from collections.abc import Callable
from typing import Any

from revdict.category import CATEGORIES
from revdict.sort import SORT_MODES


ARPABET_VOWELS = frozenset(
    {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
)
LIVE_TUI_TOP_N = 30


class ValidationError(ValueError):
    """A control value cannot be submitted to the search backend."""


def format_candidate_preview(candidate: dict) -> str:
    """Render the complete candidate payload shown in the native preview."""
    lines = [f"{candidate['headword']} ({candidate['pos']})", "", candidate["definition"]]
    synonyms = candidate.get("synonyms") or []
    if synonyms:
        lines.extend(["", "Synonyms: " + ", ".join(synonyms)])
    lines.append(f"Emotion: {candidate['label']} · {candidate['polarity']}")
    lines.append(f"Match confidence: {candidate['relevance']}%")
    if candidate.get("stress"):
        lines.append("Stress: " + candidate["stress"])
    examples = candidate.get("examples") or []
    if examples:
        lines.extend(["", 'Example: "' + examples[0] + '"'])
    return "\n".join(lines)


@dataclass
class SearchControls:
    """The non-text query constraints selected in the terminal UI."""

    sort_mode: str | None = None
    category: str | None = None
    syllables: int | None = None
    primary_vowel: str | None = None
    rhymes_with: str | None = None
    sounds_like: str | None = None
    meter: str | None = None

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
        return {
            "sort_mode": self.sort_mode,
            "category": self.category,
            "syllables": self.syllables,
            "primary_vowel": self.primary_vowel,
            "rhymes_with": self.rhymes_with,
            "sounds_like": self.sounds_like,
            "meter": self.meter,
        }


class DebouncedSearchController:
    """Run at most one backend search and coalesce obsolete UI input.

    A running daemon request is deliberately not cancelled: the daemon's
    protocol has no cancellation operation. Instead, changes made while it
    runs collapse into one latest request, and only the current generation is
    allowed to update the view.
    """

    def __init__(
        self,
        execute: Callable[..., dict],
        on_result: Callable[[dict], None],
        on_error: Callable[[Exception], None],
        *,
        debounce_seconds: float = 0.1,
        callback_scheduler: Callable[[Callable[[], None]], None] | None = None,
    ) -> None:
        self._execute = execute
        self._on_result = on_result
        self._on_error = on_error
        self._debounce_seconds = debounce_seconds
        self._schedule_callback = callback_scheduler or (lambda callback: callback())
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._pending: tuple[int, str, SearchControls] | None = None
        self._generation = 0
        self._running = False
        self._closed = False

    def request(self, query: str, controls: SearchControls) -> None:
        """Debounce a new UI state, replacing an older unstarted state."""
        with self._lock:
            if self._closed:
                return
            self._generation += 1
            generation = self._generation
            self._pending = (generation, query, controls)
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._debounce_seconds, self._launch_latest)
            self._timer.daemon = True
            self._timer.start()

    def close(self) -> None:
        """Ignore pending and in-flight responses after the UI closes."""
        with self._lock:
            self._closed = True
            self._clear_pending_locked()

    def clear(self) -> None:
        """Cancel an unstarted search and ignore any older in-flight response."""
        with self._lock:
            if not self._closed:
                self._clear_pending_locked()

    def _clear_pending_locked(self) -> None:
        self._generation += 1
        self._pending = None
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _launch_latest(self) -> None:
        with self._lock:
            self._timer = None
            if self._closed or self._running or self._pending is None:
                return
            generation, query, controls = self._pending
            self._pending = None
            self._running = True

        thread = threading.Thread(
            target=self._run,
            args=(generation, query, controls),
            name="revdict-search",
            daemon=True,
        )
        thread.start()

    def _run(self, generation: int, query: str, controls: SearchControls) -> None:
        try:
            result = self._execute(query, **controls.as_search_kwargs())
        except Exception as error:  # the CLI turns backend exceptions into user feedback too
            callback: Callable[[], None] = (
                lambda error=error: self._publish_error(generation, error)
            )
        else:
            callback = lambda: self._publish_result(generation, result)
        finally:
            with self._lock:
                self._running = False
                should_launch_pending = not self._closed and self._pending is not None

        self._schedule_callback(callback)
        if should_launch_pending:
            self._launch_latest()

    def _publish_result(self, generation: int, result: dict) -> None:
        with self._lock:
            current = not self._closed and generation == self._generation
        if current:
            self._on_result(result)

    def _publish_error(self, generation: int, error: Exception) -> None:
        with self._lock:
            current = not self._closed and generation == self._generation
        if current:
            self._on_error(error)


class NativeTui:
    """A prompt-toolkit interface over the existing daemon-backed search API."""

    def __init__(self, search_executor: Callable[..., dict]) -> None:
        from prompt_toolkit.application import Application
        from prompt_toolkit.completion import WordCompleter
        from prompt_toolkit.filters import Condition
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, VSplit
        from prompt_toolkit.styles import Style
        from prompt_toolkit.widgets import Frame, Label, TextArea

        self._search_executor = search_executor
        self._show_controls = False
        self._show_help = False
        self._show_preview = True
        self._rows: list[dict] = []
        self._selected_index = 0
        self._controls = SearchControls()

        self.query = TextArea(
            prompt="> ", multiline=False, height=1, style="class:query", focus_on_click=True
        )
        self.sort_field = TextArea(
            text="", prompt="Sort: ", multiline=False, height=1,
            completer=WordCompleter(list(SORT_MODES), ignore_case=True),
        )
        self.category_field = TextArea(
            text="", prompt="Category: ", multiline=False, height=1,
            completer=WordCompleter(list(CATEGORIES), ignore_case=True),
        )
        self.syllables_field = TextArea(prompt="Syllables: ", multiline=False, height=1)
        self.vowel_field = TextArea(
            prompt="Primary vowel: ", multiline=False, height=1,
            completer=WordCompleter(sorted(ARPABET_VOWELS), ignore_case=True),
        )
        self.rhymes_field = TextArea(prompt="Rhymes with: ", multiline=False, height=1)
        self.sounds_field = TextArea(prompt="Sounds like: ", multiline=False, height=1)
        self.meter_field = TextArea(prompt="Meter: ", multiline=False, height=1)
        self.results = TextArea(text="Type a meaning or word to search.", read_only=True, scrollbar=True)
        self.preview = TextArea(text="", read_only=True, scrollbar=True)
        self.active_filters = Label(text="sort: relevance  category: all")
        self.status = Label(text="F1 help · F2 filters · F3 preview · Ctrl-R sort · Enter copy")
        self.help = Label(
            text=(
                "Query syntax: blue*  *bird  bl????rd  //anagram  -abcd  +abcd  "
                "bl*:snow  :meaning  **word**  expand:nasa\n"
                "F2 opens filters; Tab moves controls; Ctrl-N/P selects a result; "
                "Esc clears then quits; Ctrl-C quits."
            )
        )

        controls = Frame(
            HSplit(
                [
                    self.sort_field,
                    self.category_field,
                    self.syllables_field,
                    self.vowel_field,
                    self.rhymes_field,
                    self.sounds_field,
                    self.meter_field,
                ],
                padding=0,
            ),
            title="Search controls — values apply as you type",
        )
        controls_container = ConditionalContainer(
            content=controls,
            filter=Condition(lambda: self._show_controls),
        )
        help_container = ConditionalContainer(
            content=Frame(self.help, title="Query syntax and keys"),
            filter=Condition(lambda: self._show_help),
        )
        preview_container = ConditionalContainer(
            content=Frame(self.preview, title="Preview"),
            filter=Condition(lambda: self._show_preview),
        )
        root = HSplit(
            [
                Frame(self.query, title="revdict"),
                self.active_filters,
                VSplit([Frame(self.results, title="Results"), preview_container], padding=1),
                controls_container,
                help_container,
                self.status,
            ],
            padding=1,
        )

        bindings = KeyBindings()

        @bindings.add("f1")
        def _toggle_help(event) -> None:
            self._show_help = not self._show_help
            event.app.invalidate()

        @bindings.add("f2")
        def _toggle_controls(event) -> None:
            self._show_controls = not self._show_controls
            if self._show_controls:
                event.app.layout.focus(self.sort_field)
            else:
                event.app.layout.focus(self.query)
            event.app.invalidate()

        @bindings.add("f3")
        def _toggle_preview(event) -> None:
            self._show_preview = not self._show_preview
            event.app.invalidate()

        @bindings.add("c-r")
        def _cycle_sort(event) -> None:
            current = self._controls.sort_mode or "relevance"
            next_index = (SORT_MODES.index(current) + 1) % len(SORT_MODES)
            self.sort_field.text = SORT_MODES[next_index]
            event.app.invalidate()

        @bindings.add("c-n")
        def _select_next(event) -> None:
            self._move_selection(1)
            event.app.invalidate()

        @bindings.add("c-p")
        def _select_previous(event) -> None:
            self._move_selection(-1)
            event.app.invalidate()

        @bindings.add("enter", filter=Condition(lambda: self.application.layout.current_window == self.query.window))
        def _copy_selection(event) -> None:
            self._copy_selected()
            event.app.invalidate()

        @bindings.add("escape")
        def _clear_or_exit(event) -> None:
            if self.query.text:
                self.query.text = ""
            else:
                self.close()

        @bindings.add("c-c")
        def _quit(event) -> None:
            self.close()

        self.application = Application(
            layout=Layout(root, focused_element=self.query),
            key_bindings=bindings,
            style=Style.from_dict(
                {
                    "query": "bold",
                    "frame.border": "fg:ansiblue",
                    "label": "fg:ansibrightblack",
                }
            ),
            full_screen=True,
        )
        self._controller = DebouncedSearchController(
            self._search_executor,
            self._receive_result,
            self._receive_error,
            callback_scheduler=self._schedule_on_ui_thread,
        )
        self.query.buffer.on_text_changed += lambda _buffer: self._schedule_search()
        for field in self._control_fields:
            field.buffer.on_text_changed += lambda _buffer: self._schedule_search()

    @property
    def _control_fields(self) -> tuple[Any, ...]:
        return (
            self.sort_field,
            self.category_field,
            self.syllables_field,
            self.vowel_field,
            self.rhymes_field,
            self.sounds_field,
            self.meter_field,
        )

    def run(self) -> None:
        self.application.run()

    def close(self) -> None:
        self._controller.close()
        if self.application.future is not None:
            self.application.exit()

    def _schedule_on_ui_thread(self, callback: Callable[[], None]) -> None:
        loop = self.application.loop
        if loop is not None:
            loop.call_soon_threadsafe(callback)

    def _read_controls(self) -> SearchControls:
        syllable_text = self.syllables_field.text.strip()
        if syllable_text:
            try:
                syllables = int(syllable_text)
            except ValueError as error:
                raise ValidationError("Syllables must be a whole number.") from error
        else:
            syllables = None
        return SearchControls(
            sort_mode=self.sort_field.text.strip().lower() or None,
            category=self.category_field.text.strip().lower() or None,
            syllables=syllables,
            primary_vowel=self.vowel_field.text.strip().upper() or None,
            rhymes_with=self.rhymes_field.text.strip() or None,
            sounds_like=self.sounds_field.text.strip() or None,
            meter=self.meter_field.text.strip() or None,
        )

    def _schedule_search(self) -> None:
        query = self.query.text.strip()
        if not query:
            self._controller.clear()
            self._rows = []
            self._selected_index = 0
            self.results.text = "Type a meaning or word to search."
            self.preview.text = ""
            self.status.text = "F1 help · F2 filters · F3 preview · Ctrl-R sort · Enter copy"
            self.application.invalidate()
            return
        try:
            controls = self._read_controls()
            controls.validate()
        except ValidationError as error:
            self.status.text = f"Invalid filter: {error}"
            self.application.invalidate()
            return
        self._controls = controls
        self._set_active_filters()
        self.status.text = "Searching…"
        self._controller.request(query, controls)
        self.application.invalidate()

    def _set_active_filters(self) -> None:
        active = [
            f"sort: {self._controls.sort_mode or 'relevance'}",
            f"category: {self._controls.category or 'all'}",
        ]
        if self._controls.syllables is not None:
            active.append(f"syllables: {self._controls.syllables}")
        if self._controls.primary_vowel:
            active.append(f"vowel: {self._controls.primary_vowel}")
        if self._controls.rhymes_with:
            active.append(f"rhymes: {self._controls.rhymes_with}")
        if self._controls.sounds_like:
            active.append(f"sounds: {self._controls.sounds_like}")
        if self._controls.meter:
            active.append(f"meter: {self._controls.meter}")
        self.active_filters.text = "  ".join(active)

    def _receive_result(self, result: dict) -> None:
        self._rows = self._result_rows(result)
        self._selected_index = 0
        self.status.text = f"{len(self._rows)} result{'s' if len(self._rows) != 1 else ''}"
        self._render_selection()
        self.application.invalidate()

    def _receive_error(self, error: Exception) -> None:
        self.status.text = f"Search error: {error}"
        self.application.invalidate()

    @staticmethod
    def _result_rows(result: dict) -> list[dict]:
        rows: list[dict] = []
        exact_match = result.get("exact_match")
        if exact_match and exact_match.get("senses"):
            sense = exact_match["senses"][0]
            rows.append(
                {
                    "headword": exact_match["headword"],
                    "pos": sense["pos"],
                    "definition": sense["definition"],
                    "stress": sense.get("stress"),
                    "label": sense["label"],
                    "polarity": sense["polarity"],
                    "synonyms": sense.get("synonyms") or [],
                    "examples": sense.get("examples") or [],
                    "relevance": 100,
                }
            )
        rows.extend(result.get("candidates") or [])
        return rows

    def _move_selection(self, step: int) -> None:
        if not self._rows:
            return
        self._selected_index = max(0, min(len(self._rows) - 1, self._selected_index + step))
        self._render_selection()

    def _render_selection(self) -> None:
        if not self._rows:
            self.results.text = "No results."
            self.preview.text = ""
            return
        display_lines = []
        for index, row in enumerate(self._rows):
            marker = "❯" if index == self._selected_index else " "
            display_lines.append(f"{marker} {row['headword']}  ({row['pos']})  {row['definition']}")
        self.results.text = "\n".join(display_lines)
        self.preview.text = format_candidate_preview(self._rows[self._selected_index])

    def _copy_selected(self) -> None:
        if not self._rows:
            return
        from revdict.cli import _run_copy_selection

        headword = self._rows[self._selected_index]["headword"]
        _run_copy_selection(headword)
        self.status.text = f"Copied: {headword}"


def run() -> None:
    """Launch the default interactive UI using the established CLI backend."""
    from revdict.cli import _get_search_result

    NativeTui(lambda query, **kwargs: _get_search_result(query, LIVE_TUI_TOP_N, **kwargs)).run()
