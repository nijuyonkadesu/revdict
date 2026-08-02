import threading

import pytest

from revdict import tui
from revdict.tui import (
    DebouncedSearchController,
    NativeTui,
    SearchControls,
    ValidationError,
    _wrap_result_fragments,
    build_help_text,
    candidate_preview_fragments,
    format_progress_line,
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

    assert "\x1b" not in "".join(text for _, text, *_ in fragments)
    assert "CON" in "".join(text for _, text, *_ in fragments)


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


def test_generated_help_lists_every_filter_and_the_preview_key():
    help_text = build_help_text()

    assert "F3" in help_text
    assert "Sort" in help_text
    assert "Sounds like" in help_text
    assert "Idioms and slang" in help_text


def test_progress_line_reports_percent_phase_count_and_live_detail_in_one_line():
    states = {"ready": "completed", "validate": "active"}

    line = format_progress_line(states, {"validate": "Checking selected filters"})

    assert line == "Searching 10% · 2/10 · Validate query and filters — Checking selected filters"
    assert "\n" not in line


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
