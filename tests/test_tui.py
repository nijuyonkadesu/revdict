import threading
import time

import pytest
from prompt_toolkit.application.current import create_app_session
from prompt_toolkit.data_structures import Point
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.keys import Keys
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.output import DummyOutput

from revdict import tui
from revdict.tui import (
    DebouncedSearchController,
    NativeTui,
    ResultsControl,
    SearchControls,
    ValidationError,
    _wrap_result_fragments,
    build_help_text,
    candidate_preview_fragments,
    format_progress_line,
    word_wrap_fragments,
    format_candidate_preview,
)


def test_search_controls_emit_cli_compatible_defaults():
    """Catches a UI default accidentally changing the CLI search semantics."""
    assert SearchControls().as_search_kwargs() == {
        "sort_mode": None,
        "category": None,
        "syllables": None,
        "primary_vowel": None,
        "rhymes_with": None,
        "sounds_like": None,
        "meter": None,
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("syllables", -1, "Syllables must be zero or greater."),
        ("primary_vowel", "ZZ", "Primary vowel must be an ARPAbet vowel."),
        ("meter", "/z", "Meter may contain only / and x."),
    ],
)
def test_search_controls_reject_invalid_filter_values(field, value, message):
    """Catches invalid controls reaching the daemon instead of surfacing locally."""
    controls = SearchControls()
    setattr(controls, field, value)

    with pytest.raises(ValidationError, match=message):
        controls.validate()


def test_debounced_controller_coalesces_changes_while_a_search_is_running():
    """Catches process/memory pressure from overlapping stale UI searches."""
    first_started = threading.Event()
    allow_first_to_finish = threading.Event()
    received_results = []
    calls = []
    active_count = 0
    peak_active_count = 0
    lock = threading.Lock()

    def execute(query, **_kwargs):
        nonlocal active_count, peak_active_count
        with lock:
            active_count += 1
            peak_active_count = max(peak_active_count, active_count)
            calls.append(query)
        if query == "first":
            first_started.set()
            assert allow_first_to_finish.wait(timeout=2)
        with lock:
            active_count -= 1
        return {"query": query}

    completed = threading.Event()

    def on_result(result):
        received_results.append(result)
        if result["query"] == "latest":
            completed.set()

    controller = DebouncedSearchController(
        execute,
        on_result,
        lambda _error: pytest.fail("the fake executor does not raise"),
        debounce_seconds=0,
    )
    try:
        controller.request("first", SearchControls())
        assert first_started.wait(timeout=2)
        controller.request("middle", SearchControls())
        controller.request("latest", SearchControls())
        allow_first_to_finish.set()

        assert completed.wait(timeout=2)
        assert calls == ["first", "latest"]
        assert peak_active_count == 1
        assert received_results == [{"query": "latest"}]
    finally:
        controller.close()


def test_debounced_controller_publishes_the_latest_backend_error():
    """Catches a failed daemon request disappearing instead of reaching the status line."""
    received_errors = []
    completed = threading.Event()

    def execute(_query, **_kwargs):
        raise RuntimeError("daemon is unavailable")

    def on_error(error):
        received_errors.append(str(error))
        completed.set()

    controller = DebouncedSearchController(
        execute,
        lambda _result: pytest.fail("the fake executor always raises"),
        on_error,
        debounce_seconds=0,
    )
    try:
        controller.request("happy", SearchControls())

        assert completed.wait(timeout=2)
        assert received_errors == ["daemon is unavailable"]
    finally:
        controller.close()


def test_debounced_controller_publishes_progress_before_the_search_finishes():
    """Progress must cross the worker/UI boundary live, not with the final result."""
    reported = threading.Event()
    release = threading.Event()

    def execute(_query, on_progress, **_kwargs):
        on_progress({"type": "stage", "id": "ready", "state": "active"})
        assert release.wait(timeout=2)
        return {"exact_match": None, "candidates": []}

    controller = DebouncedSearchController(
        execute,
        lambda _result: None,
        lambda error: pytest.fail(str(error)),
        on_progress=lambda _event: reported.set(),
        debounce_seconds=0,
    )
    try:
        controller.request("happy", SearchControls())
        assert reported.wait(timeout=2)
    finally:
        release.set()
        controller.close()


def test_debounced_controller_clear_cancels_an_unstarted_search():
    """Catches clearing the query still sending an empty daemon request."""
    calls = []

    controller = DebouncedSearchController(
        lambda query, **_kwargs: calls.append(query),
        lambda _result: pytest.fail("the cleared request must not publish a result"),
        lambda _error: pytest.fail("the fake executor does not raise"),
        debounce_seconds=0.1,
    )
    try:
        controller.request("happy", SearchControls())
        controller.clear()
        threading.Event().wait(timeout=0.2)

        assert calls == []
    finally:
        controller.close()


def test_default_debounce_waits_about_two_hundred_milliseconds():
    """Typing must not invoke the daemon before the intended 200ms pause."""
    completed = threading.Event()
    started_at = []
    requested_at = []

    def execute(_query, **_kwargs):
        started_at.append(time.monotonic())
        completed.set()
        return {"exact_match": None, "candidates": []}

    controller = DebouncedSearchController(execute, lambda _result: None, lambda error: pytest.fail(str(error)))
    try:
        requested_at.append(time.monotonic())
        controller.request("happy", SearchControls())

        assert not completed.wait(timeout=0.15)
        assert completed.wait(timeout=0.25)
        assert 0.18 <= started_at[0] - requested_at[0] < 0.28
    finally:
        controller.close()


def test_candidate_preview_includes_all_available_search_details():
    """Catches the native preview dropping data already available in fzf."""
    preview = format_candidate_preview(
        {
            "headword": "joyful",
            "pos": "adjective",
            "definition": "feeling great happiness",
            "examples": ["a joyful noise"],
            "synonyms": ["happy", "glad"],
            "stress": "JOYful",
            "label": "joy",
            "polarity": "positive",
            "relevance": 93,
        }
    )

    assert "joyful (adjective)" in preview
    assert "feeling great happiness" in preview
    assert "Synonyms: happy, glad" in preview
    assert "Emotion: joy · positive" in preview
    assert "Match confidence: 93%" in preview
    assert "Stress: JOYful" in preview
    assert 'Example: "a joyful noise"' in preview


def test_stress_preview_parses_ansi_instead_of_rendering_escape_characters():
    """Regression for the literal ^[[38;5… text shown by the old TextArea."""
    fragments = candidate_preview_fragments(
        {"headword": "console", "pos": "noun", "definition": "comfort", "examples": [], "synonyms": [], "stress": "\x1b[1;33mCON\x1b[0msole", "label": "joy", "polarity": "positive", "relevance": 90}
    )

    visible_text = "".join(text for _, text, *_ in fragments)
    assert any("ansiyellow" in style and "bold" in style for style, *_ in fragments)
    assert "\x1b" not in visible_text
    assert "CON" in visible_text


def test_preview_wrapper_moves_a_whole_word_to_the_next_line():
    lines = word_wrap_fragments([("", "a fine closely woven cotton fabric")], width=15)

    assert ["".join(text for _, text in line) for line in lines] == [
        "a fine closely",
        "woven cotton",
        "fabric",
    ]


def test_result_renderer_wraps_before_a_definition_word_and_marks_selection():
    """A narrow result pane must preserve words while retaining visible selection."""
    lines, _ = _wrap_result_fragments(
        [{"headword": "consolation", "pos": "noun", "definition": "comfort during disappointment"}],
        selected_index=0,
        width=28,
    )

    rendered_words = [text.strip() for line in lines for _, text in line if text.strip()]
    assert "disappointment" in rendered_words
    assert all("class:result.selected" in style for style, _ in lines[0])
    assert any("class:result.headword" in style for style, _ in lines[0])


def test_results_control_exposes_the_selected_result_as_its_cursor():
    """A selection below the viewport must make prompt-toolkit scroll it into view."""
    rows = [
        {"headword": "first", "pos": "noun", "definition": "one two three four five six"},
        {"headword": "second", "pos": "noun", "definition": "visible selected result"},
    ]

    content = ResultsControl(lambda: rows, lambda: 1, lambda _step: None, lambda _index: None).create_content(width=18, height=4)

    assert content.cursor_position.y > 0


def test_repeated_result_navigation_moves_the_rendered_results_viewport():
    """Ctrl-N must scroll the real prompt-toolkit Window, not only its preview."""
    rows = [
        {
            "headword": f"word{index}", "pos": "noun",
            "definition": "a deliberately long definition with enough words to wrap", "stress": None,
            "synonyms": [], "examples": [], "label": "joy", "polarity": "positive", "relevance": 80,
        }
        for index in range(50)
    ]
    with create_pipe_input() as input:
        with create_app_session(input=input, output=DummyOutput()):
            ui = NativeTui(lambda _query, **_kwargs: {"exact_match": None, "candidates": []})
            ui._rows = rows
            ui._render_selection()
            runner = threading.Thread(target=ui.run)
            runner.start()
            try:
                input.send_bytes(b"\x0e" * 20)  # Ctrl-N
                deadline = time.monotonic() + 1
                while (ui._selected_index != 20 or ui.results.vertical_scroll == 0) and time.monotonic() < deadline:
                    time.sleep(0.01)

                assert ui._selected_index == 20
                assert ui.results.vertical_scroll > 0
            finally:
                input.send_bytes(b"\x03")
                runner.join(timeout=2)
                ui.close()


def test_generated_help_lists_every_filter_and_the_preview_key():
    help_text = build_help_text()

    assert "F3" in help_text
    assert "F4" in help_text
    assert "F5" in help_text
    assert "F6" not in help_text
    assert "Sort" in help_text
    assert "Sounds like" in help_text
    assert "Idioms and slang" in help_text


def test_progress_line_reports_percent_phase_count_and_live_detail_in_one_line():
    states = {"ready": "completed", "validate": "active"}

    line = format_progress_line(states, {"validate": "Checking selected filters"})

    assert line == "Searching 10% · 2/10 · Validate query and filters — Checking selected filters"
    assert "\n" not in line


def test_escape_clears_a_nonempty_query_before_it_can_quit():
    ui = NativeTui(lambda _query, **_kwargs: {"exact_match": None, "candidates": []})
    ui.query.text = "tailor"
    try:
        ui._clear_or_exit()

        assert ui.query.text == ""
        assert any(
            binding.eager()
            for binding in ui.application.key_bindings.get_bindings_for_keys((Keys.Escape,))
        )
        assert ui.application.ttimeoutlen <= 0.02
    finally:
        ui.close()


def test_chat_panel_prefills_a_writing_prompt_from_the_selected_result():
    ui = NativeTui(lambda _query, **_kwargs: {"exact_match": None, "candidates": []})
    ui._rows = [{"headword": "tailor", "pos": "noun", "definition": "a person who makes and alters garments", "stress": None, "synonyms": [], "examples": [], "label": "neutral", "polarity": "neutral", "relevance": 100}]
    ui.query.text = "make clothing fit"
    ui._render_selection()
    try:
        ui._toggle_chat()

        assert ui._show_chat is True
        assert "tailor" in ui.chat_input.text
        assert "writing and in spoken conversation" in ui.chat_input.text
        assert ui.chat_input.buffer.cursor_position == len(ui.chat_input.text)
        assert ui.chat_input.window.wrap_lines()
    finally:
        ui.close()


def test_chat_settings_are_editable_and_close_back_to_the_main_tui(monkeypatch):
    monkeypatch.setattr(tui.chat_module, "save_settings", lambda _settings: None)
    ui = NativeTui(lambda _query, **_kwargs: {"exact_match": None, "candidates": []})
    try:
        ui._toggle_chat_settings()

        assert ui._show_chat_settings is True
        assert ui.application.layout.current_window == ui.chat_endpoint_field.window
        assert "F5 saves" in ui.chat_settings_frame.title
        assert ui.chat_endpoint_field.buffer.cursor_position == len(ui.chat_endpoint_field.text)

        ui._toggle_chat_settings()

        assert ui._show_chat_settings is False
        assert ui._show_chat is False
        assert ui.application.layout.current_window == ui.query.window
    finally:
        ui.close()


def test_chat_settings_only_show_cached_gemini_models_for_gemini():
    ui = NativeTui(lambda _query, **_kwargs: {"exact_match": None, "candidates": []})
    try:
        ui._chat_settings.active_provider = "ollama"
        ui._update_chat_known_models()
        assert ui.chat_known_models_container.filter() is False

        ui._chat_settings.active_provider = "gemini"
        ui._update_chat_known_models()
        assert ui.chat_known_models_container.filter() is True
    finally:
        ui.close()


def test_markdown_fragments_render_emphasis_without_painting_chat_colours():
    fragments = tui.markdown_fragments("**bold** and *italic* and `code`")

    assert "".join(text for _style, text, *_ in fragments) == "bold and italic and code"
    assert any(style == "bold" and text == "bold" for style, text, *_ in fragments)
    assert any(style == "italic" and text == "italic" for style, text, *_ in fragments)
    assert any(style == "bold" and text == "code" for style, text, *_ in fragments)
    assert not any("ansicolor" in style or "bg:" in style for style, _text, *_ in fragments)


def test_chat_renders_streamed_chunks_before_the_final_reply_arrives():
    ui = NativeTui(lambda _query, **_kwargs: {"exact_match": None, "candidates": []})
    try:
        ui._begin_chat_response()
        ui._queue_chat_chunk("**Natural**")
        ui._flush_chat_chunks()

        assert ui._chat_spinner_active is True
        assert ui._chat_streamed_answer == "**Natural**"
        assert ui.chat_transcript_control.markdown.endswith("**Natural**")

        ui._receive_chat_answer("**Natural**")

        assert ui._chat_spinner_active is False
        assert ui._chat_history[-1] == ("assistant", "**Natural**")
        assert ui._chat_transcript_text.count("**Natural**") == 1
    finally:
        ui.close()


def test_results_mouse_wheel_moves_selection_and_updates_the_preview():
    ui = NativeTui(lambda _query, **_kwargs: {"exact_match": None, "candidates": []})
    ui._rows = [
        {"headword": "first", "pos": "noun", "definition": "first definition", "stress": None, "synonyms": [], "examples": [], "label": "joy", "polarity": "positive", "relevance": 80},
        {"headword": "second", "pos": "noun", "definition": "second definition", "stress": None, "synonyms": [], "examples": [], "label": "joy", "polarity": "positive", "relevance": 80},
    ]
    ui._render_selection()
    try:
        handled = ui.results_control.mouse_handler(
            MouseEvent(position=Point(x=0, y=0), event_type=MouseEventType.SCROLL_DOWN, button=None, modifiers=frozenset())
        )

        assert handled is None
        assert ui._selected_index == 1
        assert "second" in "".join(text for _, text, *_ in ui.preview_control.fragments)
        assert ui.application.mouse_support()
    finally:
        ui.close()


def test_native_tui_close_is_safe_before_the_terminal_loop_starts():
    """Catches cleanup raising when startup fails before Application.run()."""
    ui = NativeTui(lambda _query, **_kwargs: {"exact_match": None, "candidates": []})

    ui.close()


def test_run_adapts_the_cli_backend_to_the_native_ui_search_contract(monkeypatch):
    """Catches UI requests omitting the live session's fixed result count."""
    received_calls = []

    class FakeTui:
        def __init__(self, execute):
            self.execute = execute

        def run(self):
            self.execute("happy", sort_mode="alpha")

    monkeypatch.setattr(tui, "NativeTui", FakeTui)

    def fake_search(query, top_n, on_progress, **kwargs):
        received_calls.append((query, top_n, kwargs))
        return {"exact_match": None, "candidates": []}

    monkeypatch.setattr("revdict.cli._get_search_result_with_progress", fake_search)

    tui.run()

    assert received_calls == [("happy", 50, {"sort_mode": "alpha"})]
