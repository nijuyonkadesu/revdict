import threading
import time
from types import SimpleNamespace

import pytest
from prompt_toolkit.buffer import CompletionState
from prompt_toolkit.completion import Completion
from prompt_toolkit.document import Document
from prompt_toolkit.application.current import create_app_session
from prompt_toolkit.data_structures import Point
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.keys import Keys
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.output.color_depth import ColorDepth
from prompt_toolkit.styles import Style

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
    markdown_fragments,
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


def test_debounced_controller_drops_stale_result_queued_on_ui_thread():
    """Queued callbacks must retain the generation that produced their result."""
    scheduled = []
    scheduled_lock = threading.Lock()
    latest_started = threading.Event()
    release_latest = threading.Event()
    received_results = []

    def schedule(callback):
        with scheduled_lock:
            scheduled.append(callback)

    def pop_scheduled_callback():
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            with scheduled_lock:
                if scheduled:
                    return scheduled.pop(0)
            time.sleep(0.01)
        pytest.fail("worker did not schedule its UI callback")

    def execute(query, **_kwargs):
        if query == "latest":
            latest_started.set()
            assert release_latest.wait(timeout=2)
        return {"query": query}

    controller = DebouncedSearchController(
        execute,
        received_results.append,
        lambda error: pytest.fail(str(error)),
        debounce_seconds=0,
        callback_scheduler=schedule,
    )
    try:
        controller.request("stale", SearchControls())
        stale_callback = pop_scheduled_callback()
        controller.request("latest", SearchControls())
        assert latest_started.wait(timeout=2)

        stale_callback()

        assert received_results == []
        release_latest.set()
        latest_callback = pop_scheduled_callback()
        latest_callback()
        assert received_results == [{"query": "latest"}]
    finally:
        release_latest.set()
        controller.close()


def test_debounced_controller_drops_stale_error_queued_on_ui_thread():
    """An old failure must not masquerade as an error for the latest query."""
    scheduled = []
    scheduled_lock = threading.Lock()
    latest_started = threading.Event()
    release_latest = threading.Event()
    received_errors = []

    def schedule(callback):
        with scheduled_lock:
            scheduled.append(callback)

    def pop_scheduled_callback():
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            with scheduled_lock:
                if scheduled:
                    return scheduled.pop(0)
            time.sleep(0.01)
        pytest.fail("worker did not schedule its UI callback")

    def execute(query, **_kwargs):
        if query == "stale":
            raise RuntimeError("stale failure")
        latest_started.set()
        assert release_latest.wait(timeout=2)
        return {"query": query}

    controller = DebouncedSearchController(
        execute,
        lambda _result: None,
        lambda error: received_errors.append(str(error)),
        debounce_seconds=0,
        callback_scheduler=schedule,
    )
    try:
        controller.request("stale", SearchControls())
        stale_callback = pop_scheduled_callback()
        controller.request("latest", SearchControls())
        assert latest_started.wait(timeout=2)

        stale_callback()

        assert received_errors == []
    finally:
        release_latest.set()
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


def test_stress_syllables_use_dotted_classes_to_keep_the_pinned_hue():
    fragments = tui._stress_fragments(
        "\x1b[1;7;33mPEARL\x1b[0m\x1b[1;33mPROM\x1b[0m"
        "\x1b[4;38;5;184mstone\x1b[0m\x1b[2;38;5;184mreduced\x1b[0m",
        no_color=False,
    )
    theme = tui.TerminalTheme.from_environment({"COLORTERM": "truecolor"})

    assert fragments == [
        ("class:stress.nuclear", "PEARL"),
        ("class:stress.prominent", "PROM"),
        ("class:stress.secondary", "stone"),
        ("class:stress.reduced", "reduced"),
    ]
    assert theme.styles["stress"] == "fg:#ffcc00"
    assert theme.styles["stress.nuclear"] == "reverse"
    assert theme.styles["stress.prominent"] == "bold"
    assert theme.styles["stress.secondary"] == "underline"
    assert theme.styles["stress.reduced"] == "dim"

    rendered = Style.from_dict(theme.styles).get_attrs_for_style_str("class:stress.nuclear")
    assert rendered.color == "ffcc00"
    assert rendered.reverse is True


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


def test_no_color_removes_ui_colours_and_stress_keeps_its_prominence_attribute():
    theme = tui.TerminalTheme.from_environment({"NO_COLOR": "1", "COLORTERM": "truecolor"}, truecolor_requested=True)
    fragments = candidate_preview_fragments(
        {"headword": "console", "pos": "noun", "definition": "comfort", "examples": [], "synonyms": [], "stress": "\x1b[1;33mCON\x1b[0msole", "label": "joy", "polarity": "positive", "relevance": 90},
        no_color=True,
    )

    assert theme.color_depth is ColorDepth.DEPTH_1_BIT
    assert all("ansi" not in style and "#" not in style for style in theme.styles.values())
    assert "".join(text for style, text, *_ in fragments if style == "class:stress.prominent") == "CON"


def test_preview_wrapper_moves_a_whole_word_to_the_next_line():
    lines = word_wrap_fragments([("", "a fine closely woven cotton fabric")], width=15)

    assert ["".join(text for _, text in line) for line in lines] == [
        "a fine closely",
        "woven cotton",
        "fabric",
    ]


def test_preview_wrapper_preserves_link_mouse_handlers_when_wrapping():
    handler = lambda _event: None

    lines = word_wrap_fragments(
        [("underline", "https://www.onelook.com/a-long-link", handler)],
        width=15,
    )

    assert all(
        len(fragment) == 3 and fragment[2] is handler
        for line in lines
        for fragment in line
    )
    assert all(sum(len(fragment[1]) for fragment in line) <= 15 for line in lines)


def test_onelook_preview_link_uses_osc8_for_the_selected_result():
    url = "https://www.onelook.com/thesaurus/?loc=revfp&s=periost"

    lines = word_wrap_fragments(tui.onelook_link_fragments("periost"), width=120)
    fragments = [fragment for line in lines for fragment in line]

    assert ("class:muted", "OneLook:") in fragments
    assert (
        "[ZeroWidthEscape]",
        f"\x1b]8;;{url}\x1b\\",
    ) in fragments
    assert any(text == url and "underline" in style for style, text, *_ in fragments)
    assert ("[ZeroWidthEscape]", "\x1b]8;;\x1b\\") in fragments
    assert "\u200b" not in "".join(text for _style, text, *_ in fragments)
    assert all(len(fragment) == 2 for fragment in fragments)


def test_wrapped_preview_preserves_osc8_sequences_without_counting_their_width():
    url = "https://www.onelook.com/thesaurus/?loc=revfp&s=periost"
    lines = word_wrap_fragments(tui.onelook_link_fragments("periost"), width=32)
    link_lines = [
        line for line in lines if any("class:osc8.link" in style for style, *_ in line)
    ]

    assert len(link_lines) > 1
    assert all(
        sum(text.startswith("\x1b]8;;") for _style, text, *_ in line) == 2
        for line in link_lines
    )
    assert "".join(
        text
        for line in link_lines
        for style, text, *_ in line
        if "class:osc8.link" in style
    ) == url
    assert all(line[-1][1] == " " for line in link_lines)
    assert all("\u200b" not in "".join(text for _style, text, *_ in line) for line in lines)
    assert all(
        sum(
            len(text)
            for style, text, *_ in line
            if "[ZeroWidthEscape]" not in style
        ) <= 32
        for line in lines
    )


def test_alt_enter_binding_copies_the_selected_link():
    ui = NativeTui(lambda _query, **_kwargs: {"exact_match": None, "candidates": []})
    actions = []
    ui._copy_selected_link = lambda: actions.append("copy-link")
    try:
        binding = next(
            binding
            for binding in ui.application.key_bindings.get_bindings_for_keys(
                (Keys.Escape, Keys.Enter)
            )
            if binding.filter()
        )
        binding.handler(SimpleNamespace(app=ui.application))

        assert actions == ["copy-link"]
    finally:
        ui.close()


def test_plain_enter_binding_keeps_accept_or_copy_behavior():
    ui = NativeTui(lambda _query, **_kwargs: {"exact_match": None, "candidates": []})
    actions = []
    ui._accept_or_copy = lambda: actions.append("accept-or-copy-word")
    try:
        binding = next(
            binding
            for binding in ui.application.key_bindings.get_bindings_for_keys(
                (Keys.Enter,)
            )
            if binding.filter()
        )
        binding.handler(SimpleNamespace(app=ui.application))

        assert actions == ["accept-or-copy-word"]
    finally:
        ui.close()


def test_copy_selected_link_copies_and_opens_the_current_results_exact_onelook_url(monkeypatch):
    copied = []
    opened = []
    monkeypatch.setattr("revdict.cli._is_remote_session", lambda: False)
    monkeypatch.setattr(
        "revdict.cli._run_copy_selection",
        lambda value: copied.append(value) or 0,
    )
    monkeypatch.setattr(
        tui.onelook,
        "open_url",
        lambda value: opened.append(value) or True,
        raising=False,
    )
    ui = NativeTui(lambda _query, **_kwargs: {"exact_match": None, "candidates": []})
    ui.query.text = "percale"
    ui._rows = [
        {
            "headword": "periost",
            "pos": "noun",
            "definition": "the connective tissue surrounding bone",
            "stress": None,
            "synonyms": [],
            "examples": [],
            "label": "neutral",
            "polarity": "neutral",
            "relevance": 90,
        }
    ]
    try:
        ui._copy_selected_link()

        assert copied == [
            "https://www.onelook.com/thesaurus/?loc=revfp&s=periost"
        ]
        assert opened == copied
        assert ui.status.text == "Copied and opened: periost"
    finally:
        ui.close()


def test_copy_selected_link_still_copies_when_browser_cannot_open(monkeypatch):
    copied = []
    monkeypatch.setattr("revdict.cli._is_remote_session", lambda: False)
    monkeypatch.setattr(
        "revdict.cli._run_copy_selection",
        lambda value: copied.append(value) or 0,
    )
    monkeypatch.setattr(tui.onelook, "open_url", lambda _value: False, raising=False)
    ui = NativeTui(lambda _query, **_kwargs: {"exact_match": None, "candidates": []})
    ui._rows = [
        {
            "headword": "periost",
            "pos": "noun",
            "definition": "the connective tissue surrounding bone",
            "stress": None,
            "synonyms": [],
            "examples": [],
            "label": "neutral",
            "polarity": "neutral",
            "relevance": 90,
        }
    ]
    try:
        ui._copy_selected_link()

        assert copied == [
            "https://www.onelook.com/thesaurus/?loc=revfp&s=periost"
        ]
        assert ui.status.text == "Copied link; browser did not open: periost"
    finally:
        ui.close()


def test_copy_selected_link_does_not_launch_a_remote_browser_over_ssh(monkeypatch):
    copied = []
    monkeypatch.setattr("revdict.cli._is_remote_session", lambda: True)
    monkeypatch.setattr(
        "revdict.cli._run_copy_selection",
        lambda value: copied.append(value) or 0,
    )
    monkeypatch.setattr(
        tui.onelook,
        "open_url",
        lambda _value: pytest.fail("remote session must not launch a browser"),
    )
    ui = NativeTui(lambda _query, **_kwargs: {"exact_match": None, "candidates": []})
    ui._rows = [
        {
            "headword": "epigeum",
            "pos": "noun",
            "definition": "a surface structure",
            "stress": None,
            "synonyms": [],
            "examples": [],
            "label": "neutral",
            "polarity": "neutral",
            "relevance": 90,
        }
    ]
    try:
        ui._copy_selected_link()

        assert copied == [
            "https://www.onelook.com/thesaurus/?loc=revfp&s=epigeum"
        ]
        assert ui.status.text == "Copied link over SSH: epigeum"
    finally:
        ui.close()


def test_selected_result_preview_url_targets_the_selected_headword():
    ui = NativeTui(lambda _query, **_kwargs: {"exact_match": None, "candidates": []})
    ui._controller.close()
    ui.query.text = "closely woven pill"
    ui._rows = [
        {
            "headword": "pillow",
            "pos": "noun",
            "definition": "a support for the head",
            "stress": None,
            "synonyms": [],
            "examples": [],
            "label": "neutral",
            "polarity": "neutral",
            "relevance": 90,
        }
    ]
    try:
        ui._render_selection()

        url = next(
            text
            for _style, text, *_ in ui.preview_control.fragments
            if text.startswith("https://")
        )
        assert url == "https://www.onelook.com/thesaurus/?loc=revfp&s=pillow"
    finally:
        ui.close()


def test_selected_result_preview_url_changes_with_the_selection():
    ui = NativeTui(lambda _query, **_kwargs: {"exact_match": None, "candidates": []})
    ui._controller.close()
    ui.query.text = "percale"
    ui._rows = [
        {
            "headword": headword,
            "pos": "adjective",
            "definition": "thecal",
            "stress": None,
            "synonyms": [],
            "examples": [],
            "label": "neutral",
            "polarity": "neutral",
            "relevance": 0,
        }
        for headword in ("percale", "periost")
    ]
    try:
        ui._selected_index = 0
        ui._render_selection()
        first_url = next(
            text
            for _style, text, *_ in ui.preview_control.fragments
            if text.startswith("https://")
        )

        ui._selected_index = 1
        ui._render_selection()
        second_url = next(
            text
            for _style, text, *_ in ui.preview_control.fragments
            if text.startswith("https://")
        )

        assert first_url == "https://www.onelook.com/thesaurus/?loc=revfp&s=percale"
        assert second_url == "https://www.onelook.com/thesaurus/?loc=revfp&s=periost"
        assert "%3Apercale" not in first_url
        assert "percale" not in second_url
    finally:
        ui.close()


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

    assert "## Query syntax" in help_text
    assert "## Filters" in help_text
    assert "## Keyboard" in help_text
    assert "**[F3]**" in help_text
    assert "F3" in help_text
    assert "F4" in help_text
    assert "F5" in help_text
    assert "F6" not in help_text
    assert "Sort" in help_text
    assert "Sounds like" in help_text
    assert "Idioms and slang" in help_text


def test_generated_help_markdown_emphasizes_sections_examples_and_keys():
    fragments = markdown_fragments(build_help_text(), width=100)

    assert any("underline" in style and "Query syntax" in text for style, text, *_ in fragments)
    assert any("bold" in style and "[F3]" in text for style, text, *_ in fragments)
    assert "\x1b" not in "".join(text for _style, text, *_ in fragments)


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


@pytest.mark.parametrize(
    ("first_key", "second_key", "expected_mode", "focus_name"),
    [
        ("F4", "F1", "help", "query"),
        ("F1", "F2", "controls", "sort"),
        ("F2", "F4", "chat", "chat"),
        ("F4", "F5", "settings", "settings"),
        ("F5", "F1", "help", "query"),
    ],
)
def test_function_keys_replace_the_previous_view(
    monkeypatch, first_key, second_key, expected_mode, focus_name
):
    monkeypatch.setattr(tui.chat_module, "save_settings", lambda _settings: None)
    ui = NativeTui(lambda _query, **_kwargs: {"exact_match": None, "candidates": []})
    try:
        ui._invoke_function_key(first_key)
        ui._invoke_function_key(second_key)

        expected_focus = {
            "query": ui.query.window,
            "sort": ui.sort_field.window,
            "chat": ui.chat_input.window,
            "settings": ui.chat_endpoint_field.window,
        }[focus_name]
        assert ui._navigation_mode() == expected_mode
        assert sum(
            (ui._show_help, ui._show_controls, ui._show_chat, ui._show_chat_settings)
        ) == 1
        assert ui.application.layout.current_window == expected_focus
    finally:
        ui.close()


@pytest.mark.parametrize("key", ["F1", "F2", "F4", "F5"])
def test_repeating_a_function_key_returns_to_results(monkeypatch, key):
    monkeypatch.setattr(tui.chat_module, "save_settings", lambda _settings: None)
    ui = NativeTui(lambda _query, **_kwargs: {"exact_match": None, "candidates": []})
    try:
        ui._invoke_function_key(key)
        ui._invoke_function_key(key)

        assert ui._navigation_mode() == "results"
        assert ui.application.layout.current_window == ui.query.window
    finally:
        ui.close()


def test_f5_returns_to_chat_when_setup_was_opened_from_chat(monkeypatch):
    monkeypatch.setattr(tui.chat_module, "save_settings", lambda _settings: None)
    ui = NativeTui(lambda _query, **_kwargs: {"exact_match": None, "candidates": []})
    try:
        ui._invoke_function_key("F4")
        ui._invoke_function_key("F5")
        assert ui._navigation_mode() == "settings"

        ui._invoke_function_key("F5")

        assert ui._navigation_mode() == "chat"
        assert ui.application.layout.current_window == ui.chat_input.window
    finally:
        ui.close()


def test_f3_closes_an_open_view_before_toggling_preview():
    ui = NativeTui(lambda _query, **_kwargs: {"exact_match": None, "candidates": []})
    try:
        ui._invoke_function_key("F4")
        ui._invoke_function_key("F3")

        assert ui._navigation_mode() == "results"
        assert ui._show_preview is False
        assert ui.application.layout.current_window == ui.query.window
    finally:
        ui.close()


def test_escape_closes_an_open_view_before_clearing_the_query():
    ui = NativeTui(lambda _query, **_kwargs: {"exact_match": None, "candidates": []})
    ui._controller.close()
    ui.query.text = "closely woven pill"
    try:
        ui._invoke_function_key("F4")

        ui._clear_or_exit()

        assert ui._navigation_mode() == "results"
        assert ui.query.text == "closely woven pill"
        assert ui.application.layout.current_window == ui.query.window
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


def test_new_input_hides_results_from_the_previous_query_immediately():
    """Only results for the latest settled input may remain on screen."""
    ui = NativeTui(lambda _query, **_kwargs: {"exact_match": None, "candidates": []})
    ui._controller.close()

    class RecordingController:
        def __init__(self):
            self.requests = []

        def request(self, query, controls):
            self.requests.append((query, controls))

        def clear(self):
            pass

        def close(self):
            pass

    controller = RecordingController()
    ui._controller = controller
    ui._rows = [
        {
            "headword": "old",
            "pos": "adjective",
            "definition": "belonging to the previous query",
            "stress": None,
            "synonyms": [],
            "examples": [],
            "label": "neutral",
            "polarity": "neutral",
            "relevance": 100,
        }
    ]
    ui._render_selection()
    try:
        ui.query.text = "latest query"

        assert len(controller.requests) == 1
        assert controller.requests[0][0] == "latest query"
        assert ui._rows == []
        assert ui._selected_index == 0
        assert ui.preview_control.fragments == []
        assert ui.status.text == "Searching…"
    finally:
        ui.close()


def test_invalid_filter_invalidates_an_active_search_without_starting_another():
    """A result for the old valid controls must not overwrite validation feedback."""
    ui = NativeTui(lambda _query, **_kwargs: {"exact_match": None, "candidates": []})
    ui._controller.close()

    class RecordingController:
        def __init__(self):
            self.requests = []
            self.clear_count = 0

        def request(self, query, controls):
            self.requests.append((query, controls))

        def clear(self):
            self.clear_count += 1

        def close(self):
            pass

    controller = RecordingController()
    ui._controller = controller
    try:
        ui.query.text = "happy"
        assert len(controller.requests) == 1

        ui.syllables_field.text = "not-a-number"

        assert len(controller.requests) == 1
        assert controller.clear_count == 1
        assert ui._rows == []
        assert "Invalid filter" in ui.status.text
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
        assert all(
            not binding.eager()
            for binding in ui.application.key_bindings.get_bindings_for_keys((Keys.Escape,))
        )
        assert ui.application.ttimeoutlen <= 0.02
    finally:
        ui.close()


def test_accept_completion_applies_first_when_none_highlighted():
    ui = NativeTui(lambda _query, **_kwargs: {"exact_match": None, "candidates": []})
    try:
        buf = ui.query.control.buffer
        buf.text = "hppy"
        buf.complete_state = CompletionState(
            original_document=buf.document,
            completions=[Completion("happy", -4), Completion("hippy", -4)],
            complete_index=None,
        )
        ui._accept_or_copy()
        assert buf.text == "happy"
    finally:
        ui.close()


def test_daemon_completer_uses_only_the_word_being_typed():
    calls = []
    completer = tui.DaemonCompleter(
        lambda prefix, limit: calls.append((prefix, limit)) or ["pillow"]
    )

    completions = list(
        completer.get_completions(Document("closely woven pill"), None)
    )

    assert calls == [("pill", 20)]
    assert [(item.text, item.start_position) for item in completions] == [
        ("pillow", -4)
    ]


def test_daemon_completer_never_drops_the_latest_rapid_call():
    calls = []
    completer = tui.DaemonCompleter(
        lambda prefix, _limit: calls.append(prefix) or [prefix + "ow"]
    )

    first = list(completer.get_completions(Document("pil"), None))
    latest = list(completer.get_completions(Document("pill"), None))

    assert calls == ["pil", "pill"]
    assert first[0].text == "pilow"
    assert latest[0].text == "pillow"


def test_daemon_completer_does_not_reopen_for_a_completed_word():
    completer = tui.DaemonCompleter(
        lambda _prefix, _limit: pytest.fail("no request expected after whitespace")
    )

    assert list(completer.get_completions(Document("closely woven "), None)) == []


def test_accept_completion_applies_highlighted_item():
    ui = NativeTui(lambda _query, **_kwargs: {"exact_match": None, "candidates": []})
    try:
        buf = ui.query.control.buffer
        buf.text = "hppy"
        buf.complete_state = CompletionState(
            original_document=buf.document,
            completions=[Completion("happy", -4), Completion("hippy", -4)],
            complete_index=1,
        )
        ui._accept_or_copy()
        assert buf.text == "hippy"
    finally:
        ui.close()


def test_accept_falls_back_to_copy_when_no_completions():
    ui = NativeTui(lambda _query, **_kwargs: {"exact_match": None, "candidates": []})
    copied = {}
    ui._copy_selected = lambda: copied.__setitem__("ran", True)
    try:
        ui._accept_or_copy()
        assert copied.get("ran") is True
    finally:
        ui.close()


def test_accept_falls_back_to_copy_when_empty_completions():
    ui = NativeTui(lambda _query, **_kwargs: {"exact_match": None, "candidates": []})
    ui._rows = [{"headword": "test", "pos": "n", "definition": "a test", "stress": None, "synonyms": [], "examples": [], "label": "neutral", "polarity": "neutral", "relevance": 100}]
    try:
        buf = ui.query.control.buffer
        buf.text = "hppy"
        buf.complete_state = CompletionState(
            original_document=buf.document,
            completions=[],
            complete_index=None,
        )
        ui._accept_or_copy()
        assert buf.text == "hppy"
    finally:
        ui.close()


def test_chat_panel_prefills_a_writing_prompt_from_the_selected_result():
    ui = NativeTui(lambda _query, **_kwargs: {"exact_match": None, "candidates": []})
    ui.query.text = "make clothing fit"
    ui._rows = [{"headword": "tailor", "pos": "noun", "definition": "a person who makes and alters garments", "stress": None, "synonyms": [], "examples": [], "label": "neutral", "polarity": "neutral", "relevance": 100}]
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


def test_chat_transcript_aligns_user_right_and_assistant_left():
    lines = tui.chat_transcript_lines(
        [
            tui.ChatTurn("user", "Can I use **percale** in formal writing?"),
            tui.ChatTurn("assistant", "Yes. It is a precise textile term."),
        ],
        width=80,
    )
    visible = ["".join(fragment[1] for fragment in line) for line in lines]

    assert visible[0].endswith("You")
    assert len(visible[0]) == 80
    assert visible[1].index("Can I use") >= 35
    assistant_label = visible.index("Assistant")
    assert visible[assistant_label + 1].startswith(" Yes.")
    assert any(
        "class:chat.user.bubble bold" in style and text == "percale"
        for line in lines
        for style, text, *_ in line
    )


def test_chat_theme_uses_existing_accent_and_has_no_color_fallback():
    themed = tui.TerminalTheme.from_environment({"REVDICT_ACCENT": "magenta"})
    plain = tui.TerminalTheme.from_environment({"NO_COLOR": "1"})

    assert themed.styles["chat.user.label"] == "ansimagenta bold"
    assert themed.styles["chat.user.bubble"] == "bg:ansimagenta ansiblack"
    assert themed.styles["chat.assistant.bubble"] == "reverse"
    assert plain.styles["chat.user.bubble"] == "reverse"
    assert "ansi" not in plain.styles["chat.user.bubble"]


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
        assert ui._active_chat_session.turns[-1].text == "**Natural**"
        assert ui._active_chat_session.turns[-1].pending is True

        ui._receive_chat_answer((key, "**Natural**"))

        assert ui._chat_spinner_active is False
        assert ui._active_chat_session.history[-1] == ("assistant", "**Natural**")
        assert ui._active_chat_session.turns[-1].text == "**Natural**"
        assert ui._active_chat_session.turns[-1].pending is False
    finally:
        ui.close()


def test_chat_undo_restores_user_message_and_rolls_back_provider_history():
    ui = NativeTui(lambda _query, **_kwargs: {"exact_match": None, "candidates": []})
    context = tui.chat_module.LexicalContext("fabric", "percale", "a fine cotton fabric", "noun")
    try:
        ui._activate_chat_session(context)
        session = ui._active_chat_session
        request = f"{session.bootstrap}\n\nUser request:\nHow formal is it?"
        session.history = [("user", request), ("assistant", "It is fairly formal.")]
        session.turns = [
            tui.ChatTurn("user", "How formal is it?", request_text=request),
            tui.ChatTurn("assistant", "It is fairly formal."),
        ]

        ui._undo_chat()

        assert session.history == []
        assert session.turns == []
        assert ui.chat_input.text == "How formal is it?"
        assert session.draft == "How formal is it?"
    finally:
        ui.close()


def test_chat_retry_replaces_response_without_duplicating_user_history(monkeypatch):
    ui = NativeTui(lambda _query, **_kwargs: {"exact_match": None, "candidates": []})
    context = tui.chat_module.LexicalContext("fabric", "percale", "a fine cotton fabric", "noun")
    requests = []
    monkeypatch.setattr(ui._chat_controller, "send", lambda request: requests.append(request) or True)
    try:
        key = ui._activate_chat_session(context)
        session = ui._active_chat_session
        request = f"{session.bootstrap}\n\nUser request:\nHow formal is it?"
        session.history = [("user", request), ("assistant", "Old answer")]
        session.turns = [
            tui.ChatTurn("user", "How formal is it?", request_text=request),
            tui.ChatTurn("assistant", "Old answer"),
        ]

        ui._retry_chat()

        assert requests[0][0] == key
        assert requests[0][2] == []
        assert requests[0][3] == request
        assert session.history == [("user", request)]
        assert [turn.role for turn in session.turns] == ["user", "assistant"]
        assert session.turns[-1].pending is True
        assert session.turns[-1].text == ""
    finally:
        ui.close()


def test_gemini_503_surfaces_popup_and_popup_retry_uses_failed_session(monkeypatch):
    ui = NativeTui(lambda _query, **_kwargs: {"exact_match": None, "candidates": []})
    context = tui.chat_module.LexicalContext("fabric", "percale", "a fine cotton fabric", "noun")
    requests = []
    monkeypatch.setattr(ui._chat_controller, "send", lambda request: requests.append(request) or True)
    try:
        key = ui._activate_chat_session(context)
        session = ui._active_chat_session
        request = f"{session.bootstrap}\n\nUser request:\nHow formal is it?"
        session.history = [("user", request)]
        session.turns = [
            tui.ChatTurn("user", "How formal is it?", request_text=request),
            tui.ChatTurn("assistant", "", pending=True),
        ]
        provider = tui.chat_module.ProviderSettings(
            "gemini", "https://example.test", "gemini-test", "GEMINI_KEY"
        )

        ui._receive_chat_error(
            tui.ChatSessionRequestError(
                key,
                provider,
                tui.chat_module.ChatRequestError("Provider returned HTTP 503."),
            )
        )

        assert ui._chat_error_message is not None
        assert "Gemini is temporarily unavailable (HTTP 503)" in ui.chat_error_text.text
        assert session.turns[-1].error == "Provider returned HTTP 503."
        assert ui.application.layout.current_window == ui.chat_error_retry_button.window

        ui._retry_from_error_popup()

        assert ui._chat_error_message is None
        assert requests[0][0] == key
        assert requests[0][1] is provider
        assert requests[0][3] == request
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
        assert ui._active_chat_turns() == []

        ui._activate_chat_session(percale)
        ui._chat_settings.active_provider = "gemini"
        assert ui._active_chat_turns()[0].text == "How formal is it?"
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

        assert ui._active_chat_turns() == []

        ui._selected_index = 0
        ui._render_selection()
        assert ui._active_chat_turns()[0].text == "Tell me about percale."
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
