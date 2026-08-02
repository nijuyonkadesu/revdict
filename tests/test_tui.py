import threading
import time
from types import SimpleNamespace

import pytest
from prompt_toolkit.application.current import create_app_session
from prompt_toolkit.data_structures import Point
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.keys import Keys
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.output.color_depth import ColorDepth

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
        {"headword": "console", "pos": "noun", "definition": "comfort", "examples": [], "synonyms": [], "stress": "\x1b[1;33mCON\x1b[0msole", "label": "joy", "polarity": "positive", "relevance": 90},
        no_color=False,
    )

    visible_text = "".join(text for _, text, *_ in fragments)
    assert "".join(text for style, text, *_ in fragments if "class:stress" in style) == "CON"
    assert "\x1b" not in visible_text
    assert "CON" in visible_text


def test_stress_preview_preserves_reverse_video_highlighter_spans():
    fragments = candidate_preview_fragments(
        {"headword": "pearlstone", "pos": "noun", "definition": "a pearl-like stone", "examples": [], "synonyms": [], "stress": "pearl\x1b[7mstone\x1b[0m", "label": "neutral", "polarity": "neutral", "relevance": 90},
        no_color=False,
    )

    assert "".join(text for style, text, *_ in fragments if "reverse" in style) == "stone"


def test_stress_preview_renders_a_nuclear_syllable_with_reverse_video():
    fragments = candidate_preview_fragments(
        {"headword": "pearlstone", "pos": "noun", "definition": "a pearl-like stone", "examples": [], "synonyms": [], "stress": "\x1b[1;7mPEARL\x1b[0m\x1b[4mstone\x1b[0m", "label": "neutral", "polarity": "neutral", "relevance": 90},
        no_color=True,
    )

    assert "".join(text for style, text, *_ in fragments if "reverse" in style) == "PEARL"
    assert "".join(text for style, text, *_ in fragments if "underline" in style) == "stone"


def test_stress_preview_preserves_reverse_video_when_no_color_is_set():
    fragments = candidate_preview_fragments(
        {"headword": "nuclear", "pos": "noun", "definition": "atomic", "examples": [], "synonyms": [], "stress": "\x1b[7mnuclear\x1b[0m", "label": "neutral", "polarity": "neutral", "relevance": 90},
        no_color=True,
    )

    assert "".join(text for style, text, *_ in fragments if "reverse" in style) == "nuclear"


def test_terminal_theme_uses_terminal_ansi_roles_and_gates_truecolor():
    theme = tui.TerminalTheme.from_environment({"COLORTERM": "truecolor", "COLORFGBG": "15;0"})

    assert theme.color_depth is ColorDepth.DEPTH_24_BIT
    assert theme.colorfgbg == "15;0"
    assert theme.styles["result.headword"] == "ansigreen bold"
    assert theme.styles["result.pos"] == "dim"
    assert theme.styles["border"] == "bold"
    assert theme.styles["section.title"] == "bold"
    assert theme.styles["result.sentiment.positive"] == "ansigreen"
    assert theme.styles["result.sentiment.negative"] == "ansired"
    assert theme.styles["result.confidence"] == "ansimagenta"
    assert theme.styles["result.selected"] == "reverse dim"

    custom_accent = tui.TerminalTheme.from_environment({"REVDICT_ACCENT": "magenta"})
    assert custom_accent.styles["result.headword"] == "ansimagenta bold"

    disabled = tui.TerminalTheme.from_environment({"COLORTERM": "24bit"}, truecolor_requested=False)
    assert disabled.color_depth is ColorDepth.DEPTH_8_BIT


def test_no_color_removes_ui_colours_and_stress_falls_back_to_bold():
    theme = tui.TerminalTheme.from_environment({"NO_COLOR": "1", "COLORTERM": "truecolor"}, truecolor_requested=True)
    fragments = candidate_preview_fragments(
        {"headword": "console", "pos": "noun", "definition": "comfort", "examples": [], "synonyms": [], "stress": "\x1b[1;33mCON\x1b[0msole", "label": "joy", "polarity": "positive", "relevance": 90},
        no_color=True,
    )

    assert theme.color_depth is ColorDepth.DEPTH_1_BIT
    assert all("ansi" not in style and "#" not in style for style in theme.styles.values())
    assert "".join(text for style, text, *_ in fragments if style == "class:stress.no_color") == "CON"


def test_preview_wrapper_moves_a_whole_word_to_the_next_line():
    lines = word_wrap_fragments([("", "a fine closely woven cotton fabric")], width=15)

    assert ["".join(text for _, text in line) for line in lines] == [
        "a fine closely",
        "woven cotton",
        "fabric",
    ]


def test_proportional_scrollbar_uses_a_themeable_thumb_without_arrow_chrome():
    margin = tui.ProportionalScrollbarMargin()
    fragments = margin.create_margin(
        SimpleNamespace(content_height=100, window_height=10, displayed_lines=range(10), vertical_scroll=45),
        width=1,
        height=10,
    )

    glyphs = "".join(text for _style, text, *_ in fragments)
    assert glyphs.count("█") == 1
    assert glyphs.count("░") == 9
    assert "^" not in glyphs and "v" not in glyphs
    assert any(style == "class:scrollbar.thumb" and text == "█" for style, text, *_ in fragments)


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


def test_result_renderer_uses_fixed_columns_and_hanging_definition_indents():
    rows = [
        {"headword": "cat", "pos": "noun", "definition": "a small domesticated carnivorous mammal"},
        {"headword": "extraordinarilylongword", "pos": "adjective", "definition": "longer than usual"},
    ]
    lines, line_rows = _wrap_result_fragments(rows, selected_index=0, width=60)
    rendered = ["".join(text for _style, text in line) for line in lines]
    definition_column = tui.RESULT_MARKER_WIDTH + tui.RESULT_HEADWORD_COL_WIDTH + tui.RESULT_POS_COL_WIDTH

    assert rendered[0][definition_column:].startswith("a small")
    assert rendered[1].startswith(" " * definition_column)
    assert rendered[2][definition_column:].startswith("longer than usual")
    assert line_rows == [0, 0, 1]
    assert "…" in rendered[2 - 0] or "…" in "".join(rendered)


def test_selected_result_paints_every_wrapped_physical_line_to_the_panel_width():
    lines, _ = _wrap_result_fragments(
        [{"headword": "consolation", "pos": "noun", "definition": "comfort during disappointment"}],
        selected_index=0,
        width=28,
    )

    assert len(lines) > 1
    for line in lines:
        assert len("".join(text for _style, text in line)) == 28
        assert all("class:result.selected" in style for style, _text in line)


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

    assert line == "Searching 11% · 2/9 · Validating query & filters — Checking selected filters"
    assert "\n" not in line


def test_completed_search_progress_expires_without_clearing_a_new_search():
    ui = NativeTui(lambda _query, **_kwargs: {"exact_match": None, "candidates": []})
    try:
        ui.progress.text = [("bold", "Searching 100% · 9/9 · Done")]
        token = ui._progress_visibility_token
        ui._clear_completed_progress(token)
        assert ui.progress.text == []

        ui.progress.text = [("bold", "Searching 100% · 9/9 · Done")]
        ui._reset_progress()
        ui._clear_completed_progress(token)
        assert ui.progress.text != []
    finally:
        ui.close()


def test_bottom_function_buttons_invoke_the_same_actions_as_f_keys():
    ui = NativeTui(lambda _query, **_kwargs: {"exact_match": None, "candidates": []})
    try:
        labels = " ".join(button.text for button in ui.function_key_buttons)
        click = ui.function_key_buttons[0].control.text()[0][2]
        click(MouseEvent(position=Point(x=0, y=0), event_type=MouseEventType.MOUSE_UP, button=MouseButton.LEFT, modifiers=frozenset()))

        assert ui._show_help is True
        assert all(f"[{key}]" in labels for key in ("F1", "F2", "F3", "F4", "F5"))
        assert ui.theme.styles["button.key"] == "bold"
        assert ui.theme.styles["button.label"] == ""
    finally:
        ui.close()


def test_tui_uses_hairline_sections_instead_of_box_frames():
    ui = NativeTui(lambda _query, **_kwargs: {"exact_match": None, "candidates": []})
    try:
        assert all(
            isinstance(section, tui.HairlineSection)
            for section in (ui.query_section, ui.results_section, ui.preview_section, ui.controls_section)
        )
        assert ui.results_section.title == "Results"
        assert ui.preview_section.title == "Preview"
    finally:
        ui.close()


def test_idle_status_is_hidden_and_function_buttons_are_a_single_compact_row():
    """Catches the footer duplicating shortcuts or consuming a second row."""
    ui = NativeTui(lambda _query, **_kwargs: {"exact_match": None, "candidates": []})
    try:
        assert ui.status.text == ""
        assert ui.status_container.filter() is False
        assert len(ui.function_key_bar.children) == len(ui.function_key_buttons)
        assert [child.content for child in ui.function_key_bar.children] == [button.control for button in ui.function_key_buttons]
        assert all(button.left_symbol == button.right_symbol == "" for button in ui.function_key_buttons)

        ui.status.text = "Searching…"
        assert ui.status_container.filter() is True
    finally:
        ui.close()


def test_idle_footer_reserves_no_blank_status_or_progress_rows():
    """Catches invisible footer content leaving vertical holes above the buttons."""
    ui = NativeTui(lambda _query, **_kwargs: {"exact_match": None, "candidates": []})
    try:
        assert ui.root.padding == 0
        assert ui.status_container.filter() is False
        assert ui.progress_container.filter() is False

        ui.progress.text = [("bold", "Searching 50% · 5/9 · Retrieving candidates")]
        assert ui.progress_container.filter() is True
    finally:
        ui.close()


def test_results_and_preview_stack_on_narrow_terminals():
    """Catches mobile layouts compressing two reading panes side by side."""
    ui = NativeTui(lambda _query, **_kwargs: {"exact_match": None, "candidates": []})
    try:
        assert ui._panes_for_width(80) is ui.stacked_panes
        assert ui._panes_for_width(120) is ui.side_by_side_panes
    finally:
        ui.close()


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


def test_chat_header_reuses_the_semantic_headword_and_pos_styles():
    ui = NativeTui(lambda _query, **_kwargs: {"exact_match": None, "candidates": []})
    ui._rows = [{"headword": "tailor", "pos": "noun", "definition": "a person who makes and alters garments", "stress": None, "synonyms": [], "examples": [], "label": "neutral", "polarity": "neutral", "relevance": 100}]
    try:
        ui._render_selection()

        assert ("class:result.headword", "tailor") in ui.chat_header.text
        assert ("class:result.pos", " (noun)") in ui.chat_header.text
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
    context = tui.chat_module.LexicalContext("fabric", "percale", "a fine cotton fabric", "noun")
    try:
        key = ui._activate_chat_session(context)
        ui._begin_chat_response(key)
        ui._queue_chat_chunk(key, "**Natural**")
        ui._flush_chat_chunks()

        assert ui._chat_spinner_active is True
        assert ui._active_chat_session.streamed_answer == "**Natural**"
        assert ui.chat_transcript_control.markdown.endswith("**Natural**")

        ui._receive_chat_answer((key, "**Natural**"))

        assert ui._chat_spinner_active is False
        assert ui._active_chat_session.history[-1] == ("assistant", "**Natural**")
        assert ui._chat_transcript_text.count("**Natural**") == 1
    finally:
        ui.close()


def test_chat_sessions_restore_per_sense_and_survive_provider_changes():
    ui = NativeTui(lambda _query, **_kwargs: {"exact_match": None, "candidates": []})
    percale = tui.chat_module.LexicalContext("fabric", "percale", "a fine cotton fabric", "noun")
    reverence = tui.chat_module.LexicalContext("respect", "reverence", "deep respect", "noun")
    try:
        ui._activate_chat_session(percale)
        ui._append_chat_turn("You", "How formal is it?")
        ui._active_chat_session.history.append(("user", "How formal is it?"))

        ui._activate_chat_session(reverence)
        assert ui._chat_transcript_text == ""

        ui._activate_chat_session(percale)
        ui._chat_settings.active_provider = "gemini"
        assert "How formal is it?" in ui._chat_transcript_text
        assert ui._active_chat_session.bootstrap == tui.chat_module.lexical_bootstrap(percale)
    finally:
        ui.close()


def test_changing_the_highlighted_result_switches_the_visible_chat_session():
    ui = NativeTui(lambda _query, **_kwargs: {"exact_match": None, "candidates": []})
    ui._rows = [
        {"headword": "percale", "pos": "noun", "definition": "a fine cotton fabric", "stress": None, "synonyms": [], "examples": [], "label": "neutral", "polarity": "neutral", "relevance": 100},
        {"headword": "reverence", "pos": "noun", "definition": "deep respect", "stress": None, "synonyms": [], "examples": [], "label": "neutral", "polarity": "neutral", "relevance": 100},
    ]
    try:
        ui._toggle_chat()
        ui._append_chat_turn("You", "Tell me about percale.")
        ui._selected_index = 1
        ui._render_selection()

        assert ui._chat_transcript_text == ""

        ui._selected_index = 0
        ui._render_selection()
        assert "Tell me about percale." in ui._chat_transcript_text
    finally:
        ui.close()


def test_each_new_sense_gets_its_own_prefilled_chat_draft():
    ui = NativeTui(lambda _query, **_kwargs: {"exact_match": None, "candidates": []})
    ui._rows = [
        {"headword": "percale", "pos": "noun", "definition": "a fine cotton fabric", "stress": None, "synonyms": [], "examples": [], "label": "neutral", "polarity": "neutral", "relevance": 100},
        {"headword": "reverence", "pos": "noun", "definition": "deep respect", "stress": None, "synonyms": [], "examples": [], "label": "neutral", "polarity": "neutral", "relevance": 100},
    ]
    try:
        ui._toggle_chat()
        assert "percale" in ui.chat_input.text
        ui._toggle_chat()
        ui._selected_index = 1
        ui._render_selection()
        ui._toggle_chat()

        assert "reverence" in ui.chat_input.text
        assert "percale" not in ui.chat_input.text
        ui._toggle_chat()
        ui._selected_index = 0
        ui._render_selection()
        ui._toggle_chat()
        assert "percale" in ui.chat_input.text
    finally:
        ui.close()


def test_empty_chat_input_is_not_queued_to_a_provider(monkeypatch):
    ui = NativeTui(lambda _query, **_kwargs: {"exact_match": None, "candidates": []})
    ui._rows = [{"headword": "percale", "pos": "noun", "definition": "a fine cotton fabric", "stress": None, "synonyms": [], "examples": [], "label": "neutral", "polarity": "neutral", "relevance": 100}]
    queued = []
    monkeypatch.setattr(ui._chat_controller, "send", lambda request: queued.append(request) or True)
    try:
        ui._send_chat()

        assert queued == []
        assert ui.status.text == "Write a message before sending it."
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
