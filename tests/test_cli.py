import json
import sys

from revdict import cli
from revdict.picker import PickerError


def test_main_returns_1_with_clean_error_on_non_numeric_n_value(capsys):
    """Regression guard: -n used to be parsed via manual list-slicing +
    int(), which raised an uncaught ValueError for a non-numeric value
    instead of a clean, handled error."""
    code = cli.main(["happy", "-n", "not-a-number"])

    captured = capsys.readouterr()
    assert code == 1
    assert "invalid int value" in captured.out
    assert "Traceback" not in captured.out


def test_main_returns_1_with_clean_error_on_missing_n_value(capsys):
    """Regression guard: -n with nothing after it used to raise an
    uncaught IndexError instead of a clean, handled error."""
    code = cli.main(["happy", "-n"])

    captured = capsys.readouterr()
    assert code == 1
    assert "expected one argument" in captured.out
    assert "Traceback" not in captured.out


def test_main_error_message_is_not_mangled_by_rich_markup(capsys):
    """Regression guard: argparse's usage text contains literal square
    brackets (e.g. "[query ...]"), which Rich would otherwise try to parse
    as its own markup tags and silently swallow."""
    code = cli.main(["happy", "-n", "abc"])

    captured = capsys.readouterr()
    assert code == 1
    assert "[query ...]" in captured.out


def test_main_help_flag_exits_cleanly_and_prints_usage(capsys):
    import pytest

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])

    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "usage: revdict" in captured.out


def test_main_prints_error_and_returns_1_when_index_missing(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_index_exists", lambda: False)

    code = cli.main(["happy"])

    captured = capsys.readouterr()
    assert code == 1
    assert "build-index" in captured.out


def test_main_routes_the_build_index_subcommand(monkeypatch):
    called = {}

    def fake_build_index(skip_confirm):
        called["skip_confirm"] = skip_confirm

    monkeypatch.setattr(cli, "_build_index", fake_build_index)

    code = cli.main(["build-index", "--yes"])

    assert code == 0
    assert called["skip_confirm"] is True


def test_build_index_warns_when_a_daemon_is_still_running_afterward(monkeypatch, capsys):
    import revdict.data.build_index as build_index_module

    monkeypatch.setattr(build_index_module, "build", lambda skip_confirm: None)
    monkeypatch.setattr(cli.daemon, "is_daemon_running", lambda: True)

    cli._build_index(skip_confirm=True)

    captured = capsys.readouterr()
    assert "daemon stop" in captured.out


def test_build_index_says_nothing_when_no_daemon_is_running(monkeypatch, capsys):
    import revdict.data.build_index as build_index_module

    monkeypatch.setattr(build_index_module, "build", lambda skip_confirm: None)
    monkeypatch.setattr(cli.daemon, "is_daemon_running", lambda: False)

    cli._build_index(skip_confirm=True)

    captured = capsys.readouterr()
    assert captured.out == ""


def test_main_routes_daemon_start_to_run_server(monkeypatch):
    called = {"ran": False}
    monkeypatch.setattr(cli, "_daemon_start", lambda: called.__setitem__("ran", True))

    code = cli.main(["daemon", "start"])

    assert code == 0
    assert called["ran"] is True


def test_main_routes_daemon_stop_and_reports_when_nothing_was_running(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_daemon_stop", lambda: False)

    code = cli.main(["daemon", "stop"])

    captured = capsys.readouterr()
    assert code == 0
    assert "not running" in captured.out.lower()


def test_main_routes_daemon_stop_and_reports_success(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_daemon_stop", lambda: True)

    code = cli.main(["daemon", "stop"])

    captured = capsys.readouterr()
    assert code == 0
    assert "stopped" in captured.out.lower()


def test_main_routes_daemon_status(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_daemon_status", lambda: "revdict daemon is running (pid 123).")

    code = cli.main(["daemon", "status"])

    captured = capsys.readouterr()
    assert code == 0
    assert "pid 123" in captured.out


def test_main_daemon_subcommand_with_unknown_or_missing_action_prints_usage(capsys):
    code = cli.main(["daemon"])

    captured = capsys.readouterr()
    assert code == 1
    assert "start" in captured.out and "stop" in captured.out and "status" in captured.out


def test_get_search_result_uses_daemon_when_it_answers(monkeypatch):
    monkeypatch.setattr(
        cli.daemon,
        "send_query",
        lambda query, top_n, sort_mode=None, category=None, syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None: {"exact_match": None, "candidates": []},
    )

    def fail_if_called():
        raise AssertionError("ensure_daemon_running should not be called if send_query answers")

    monkeypatch.setattr(cli.daemon, "ensure_daemon_running", fail_if_called)

    result = cli._get_search_result("happy", 10)

    assert result == {"exact_match": None, "candidates": []}


def test_get_search_result_starts_daemon_and_retries_when_first_attempt_fails(monkeypatch):
    attempts = {"count": 0}

    def fake_send_query(query, top_n, sort_mode=None, category=None, syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None):
        attempts["count"] += 1
        if attempts["count"] == 1:
            return None
        return {"exact_match": None, "candidates": [{"headword": "joyful"}]}

    monkeypatch.setattr(cli.daemon, "send_query", fake_send_query)
    monkeypatch.setattr(cli.daemon, "ensure_daemon_running", lambda: True)

    result = cli._get_search_result("happy", 10)

    assert attempts["count"] == 2
    assert result == {"exact_match": None, "candidates": [{"headword": "joyful"}]}


def test_get_search_result_falls_back_to_local_search_when_daemon_unavailable(monkeypatch):
    monkeypatch.setattr(cli.daemon, "send_query", lambda query, top_n, sort_mode=None, category=None, syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None: None)
    monkeypatch.setattr(cli.daemon, "ensure_daemon_running", lambda: False)

    fake_result = {"exact_match": None, "candidates": [{"headword": "fallback-used"}]}
    monkeypatch.setattr(
        cli, "_local_search_fallback", lambda query, top_n, sort_mode=None, category=None, syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None: fake_result
    )

    result = cli._get_search_result("happy", 10)

    assert result == fake_result


def test_run_query_warns_and_returns_0_on_blank_query(capsys):
    code = cli._run_query("   ", top_n=10, interactive=False)

    captured = capsys.readouterr()
    assert code == 0
    assert "word or phrase" in captured.out


def test_run_query_prints_exact_match_emotion_and_synonyms_when_present(monkeypatch, capsys):
    """Fix 1 + Fix 2: the exact-match table must show an emotion badge per
    sense (the headline feature that was previously silently dropped for the
    exact match) and synonyms when present, skipping the synonyms line
    cleanly when they're absent."""
    fake_result = {
        "exact_match": {
            "headword": "happy",
            "senses": [
                {
                    "pos": "adjective",
                    "definition": "feeling great pleasure",
                    "examples": [],
                    "source": "wordnet",
                    "synonyms": ["glad", "content"],
                    "label": "joy",
                    "polarity": "positive",
                },
                {
                    "pos": "adjective",
                    "definition": "willing to do something",
                    "examples": [],
                    "source": "wiktionary",
                    "synonyms": None,
                    "label": "neutral",
                    "polarity": "neutral",
                },
            ],
        },
        "candidates": [],
    }
    monkeypatch.setattr(cli, "_get_search_result", lambda query, top_n, sort_mode=None, category=None, syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None: fake_result)

    code = cli._run_query("happy", top_n=10, interactive=False)

    captured = capsys.readouterr()
    assert code == 0
    assert "joy · positive" in captured.out
    assert "neutral · neutral" in captured.out
    assert "glad" in captured.out and "content" in captured.out
    assert "Synonyms: \n" not in captured.out


def test_run_query_prints_stress_column_when_present(monkeypatch, capsys):
    fake_result = {
        "exact_match": {
            "headword": "happy",
            "senses": [
                {
                    "pos": "adjective",
                    "definition": "feeling great pleasure",
                    "examples": [],
                    "source": "wordnet",
                    "synonyms": None,
                    "label": "joy",
                    "polarity": "positive",
                    "stress": "HAPpy",
                }
            ],
        },
        "candidates": [],
    }
    monkeypatch.setattr(cli, "_get_search_result", lambda query, top_n, sort_mode=None, category=None, syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None: fake_result)

    code = cli._run_query("happy", top_n=10, interactive=False)

    captured = capsys.readouterr()
    assert code == 0
    assert "HAPpy" in captured.out


def test_run_query_prints_candidate_synonyms_column_when_present(monkeypatch, capsys):
    fake_result = {
        "exact_match": None,
        "candidates": [
            {
                "headword": "joyful",
                "pos": "adjective",
                "definition": "feeling great happiness",
                "examples": [],
                "label": "joy",
                "polarity": "positive",
                "relevance": 90,
                "stress": None,
                "synonyms": ["glad", "elated"],
            }
        ],
    }
    monkeypatch.setattr(cli, "_get_search_result", lambda query, top_n, sort_mode=None, category=None, syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None: fake_result)

    code = cli._run_query("happy", top_n=10, interactive=False)

    captured = capsys.readouterr()
    assert code == 0
    assert "glad" in captured.out and "elated" in captured.out


def test_run_query_prints_static_results_when_not_interactive(monkeypatch, capsys):
    fake_result = {
        "exact_match": None,
        "candidates": [
            {
                "headword": "joyful",
                "pos": "adjective",
                "definition": "feeling great happiness",
                "examples": [],
                "label": "joy",
                "polarity": "positive",
                "relevance": 90,
            }
        ],
    }
    monkeypatch.setattr(cli, "_get_search_result", lambda query, top_n, sort_mode=None, category=None, syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None: fake_result)

    code = cli._run_query("happy", top_n=10, interactive=False)

    captured = capsys.readouterr()
    assert code == 0
    assert "joyful" in captured.out


_FAKE_INTERACTIVE_RESULT = {
    "exact_match": None,
    "candidates": [
        {
            "headword": "joyful",
            "pos": "adjective",
            "definition": "feeling great happiness",
            "examples": [],
            "label": "joy",
            "polarity": "positive",
            "relevance": 90,
        }
    ],
}


def test_run_query_falls_back_to_static_results_when_fzf_missing(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_get_search_result", lambda query, top_n, sort_mode=None, category=None, syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None: _FAKE_INTERACTIVE_RESULT)
    monkeypatch.setattr(cli, "run_picker", lambda candidates, exact_match: None)
    monkeypatch.setattr(cli, "_fzf_missing", lambda: True)

    code = cli._run_query("happy", top_n=10, interactive=True)

    captured = capsys.readouterr()
    assert code == 0
    assert "joyful" in captured.out


def test_run_query_returns_quietly_when_user_cancels_the_picker(monkeypatch, capsys):
    """fzf present, user just pressed Esc/Ctrl-C (run_picker -> None) --
    this is a deliberate cancellation, not an error, so nothing should be
    printed and there should be no static-table fallback."""
    monkeypatch.setattr(cli, "_get_search_result", lambda query, top_n, sort_mode=None, category=None, syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None: _FAKE_INTERACTIVE_RESULT)
    monkeypatch.setattr(cli, "run_picker", lambda candidates, exact_match: None)
    monkeypatch.setattr(cli, "_fzf_missing", lambda: False)

    code = cli._run_query("happy", top_n=10, interactive=True)

    captured = capsys.readouterr()
    assert code == 0
    assert captured.out == ""


def test_run_query_falls_back_to_static_results_and_warns_on_picker_runtime_error(
    monkeypatch, capsys
):
    """Root requirement of Fix 3: a genuine fzf runtime failure (fzf present
    but erroring, e.g. no controlling terminal) must never produce zero
    output -- it must fall back to the static table and mention the error."""
    monkeypatch.setattr(cli, "_get_search_result", lambda query, top_n, sort_mode=None, category=None, syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None: _FAKE_INTERACTIVE_RESULT)

    def fake_run_picker(candidates, exact_match):
        raise PickerError(2, "inappropriate ioctl for device")

    monkeypatch.setattr(cli, "run_picker", fake_run_picker)
    monkeypatch.setattr(cli, "_fzf_missing", lambda: False)

    code = cli._run_query("happy", top_n=10, interactive=True)

    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.strip() != ""
    assert "joyful" in captured.out
    assert "ioctl" in captured.out or "fzf" in captured.out.lower()


def test_main_with_no_args_checks_isatty_before_going_interactive(monkeypatch):
    """The argv-parsing path already guards interactive with
    `sys.stdout.isatty()`; the no-arg path previously set interactive=True
    unconditionally. Both paths must apply the same guard."""
    monkeypatch.setattr(cli, "_index_exists", lambda: True)
    monkeypatch.setattr(cli.console, "input", lambda prompt: "happy")

    calls = {}

    def fake_run_query(query, top_n, interactive):
        calls["interactive"] = interactive
        return 0

    monkeypatch.setattr(cli, "_run_query", fake_run_query)

    class _NonTtyStdout:
        def isatty(self):
            return False

    monkeypatch.setattr(cli.sys, "stdout", _NonTtyStdout())

    code = cli.main([])

    assert code == 0
    assert calls["interactive"] is False


def test_query_only_prints_candidate_lines_into_the_given_preview_dir(monkeypatch, capsys, tmp_path):
    fake_result = {
        "exact_match": None,
        "candidates": [
            {
                "headword": "joyful",
                "pos": "adjective",
                "definition": "feeling great happiness",
                "examples": [],
                "label": "joy",
                "polarity": "positive",
                "relevance": 90,
            }
        ],
    }
    monkeypatch.setattr(cli, "_get_search_result", lambda query, top_n, sort_mode=None, category=None, syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None: fake_result)
    monkeypatch.setenv("REVDICT_LIVE_PREVIEW_DIR", str(tmp_path))

    code = cli.main(["--query-only", "happy"])

    captured = capsys.readouterr()
    assert code == 0
    assert "joyful" in captured.out
    assert (tmp_path / "0.txt").exists()


def test_query_only_with_blank_query_prints_nothing(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("REVDICT_LIVE_PREVIEW_DIR", str(tmp_path))

    code = cli.main(["--query-only", ""])

    captured = capsys.readouterr()
    assert code == 0
    assert captured.out == ""


def test_jsonl_query_prints_one_json_object_per_candidate(monkeypatch, capsys):
    fake_result = {
        "exact_match": None,
        "candidates": [
            {
                "headword": "joyful",
                "pos": "adjective",
                "definition": "feeling great happiness",
                "examples": [],
                "label": "joy",
                "polarity": "positive",
                "relevance": 90,
            }
        ],
    }
    monkeypatch.setattr(cli, "_get_search_result", lambda query, top_n, sort_mode=None, category=None, syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None: fake_result)

    code = cli.main(["--jsonl-query", "happy"])

    captured = capsys.readouterr()
    assert code == 0
    lines = captured.out.strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["headword"] == "joyful"
    assert row["is_exact"] is False
    assert row["relevance"] == 90


def test_jsonl_query_flattens_exact_match_first_sense_as_first_row(monkeypatch, capsys):
    fake_result = {
        "exact_match": {
            "headword": "happy",
            "senses": [
                {
                    "pos": "adjective",
                    "definition": "feeling or showing pleasure",
                    "stress": "\x1b[1mHAP\x1b[0mpy",
                    "label": "joy",
                    "polarity": "positive",
                    "synonyms": ["glad", "cheerful"],
                    "examples": ["a happy childhood"],
                }
            ],
        },
        "candidates": [],
    }
    monkeypatch.setattr(cli, "_get_search_result", lambda query, top_n, sort_mode=None, category=None, syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None: fake_result)

    code = cli.main(["--jsonl-query", "happy"])

    captured = capsys.readouterr()
    lines = captured.out.strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert code == 0
    assert row["headword"] == "happy"
    assert row["is_exact"] is True
    assert row["relevance"] == 100
    assert row["synonyms"] == ["glad", "cheerful"]
    assert row["stress"] == "\x1b[1mHAP\x1b[0mpy"


def test_jsonl_query_with_blank_query_prints_nothing(monkeypatch, capsys):
    code = cli.main(["--jsonl-query", ""])

    captured = capsys.readouterr()
    assert code == 0
    assert captured.out == ""


def test_jsonl_query_candidate_without_synonyms_or_stress_defaults_cleanly(monkeypatch, capsys):
    fake_result = {
        "exact_match": None,
        "candidates": [
            {
                "headword": "joyful",
                "pos": "adjective",
                "definition": "feeling great happiness",
                "examples": [],
                "label": "joy",
                "polarity": "positive",
                "relevance": 90,
            }
        ],
    }
    monkeypatch.setattr(cli, "_get_search_result", lambda query, top_n, sort_mode=None, category=None, syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None: fake_result)

    code = cli.main(["--jsonl-query", "happy"])

    row = json.loads(capsys.readouterr().out.strip())
    assert code == 0
    assert row["synonyms"] == []
    assert row["stress"] is None


def test_tui_query_prints_one_json_object_per_candidate(monkeypatch, capsys):
    fake_result = {
        "exact_match": None,
        "candidates": [
            {
                "headword": "joyful", "pos": "adjective",
                "definition": "feeling great happiness",
                "examples": [], "label": "joy", "polarity": "positive",
                "relevance": 90,
            }
        ],
    }
    captured_kwargs = {}

    def fake_get_search_result(query, top_n, sort_mode=None, category=None, syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None):
        captured_kwargs.update(
            query=query, top_n=top_n, sort_mode=sort_mode, category=category,
            syllables=syllables, primary_vowel=primary_vowel, rhymes_with=rhymes_with,
            sounds_like=sounds_like, meter=meter,
        )
        return fake_result

    monkeypatch.setattr(cli, "_get_search_result", fake_get_search_result)

    payload = json.dumps({"query": "happy", "sort": "most_formal", "category": "noun"})
    code = cli.main(["--tui-query", payload])

    captured = capsys.readouterr()
    assert code == 0
    lines = captured.out.strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["headword"] == "joyful"
    assert row["is_exact"] is False
    assert captured_kwargs["query"] == "happy"
    assert captured_kwargs["sort_mode"] == "most_formal"
    assert captured_kwargs["category"] == "noun"


def test_tui_query_defaults_top_n_when_omitted(monkeypatch, capsys):
    captured_kwargs = {}

    def fake_get_search_result(query, top_n, **kwargs):
        captured_kwargs["top_n"] = top_n
        return {"exact_match": None, "candidates": []}

    monkeypatch.setattr(cli, "_get_search_result", fake_get_search_result)

    code = cli.main(["--tui-query", json.dumps({"query": "happy"})])

    assert code == 0
    assert captured_kwargs["top_n"] == cli.LIVE_SESSION_TOP_N


def test_tui_query_flattens_exact_match_first_sense_as_first_row(monkeypatch, capsys):
    fake_result = {
        "exact_match": {
            "headword": "happy",
            "senses": [
                {
                    "pos": "adjective", "definition": "feeling or showing pleasure",
                    "stress": "\x1b[1mHAP\x1b[0mpy", "label": "joy", "polarity": "positive",
                    "synonyms": ["glad", "cheerful"], "examples": ["a happy childhood"],
                }
            ],
        },
        "candidates": [],
    }
    monkeypatch.setattr(
        cli, "_get_search_result",
        lambda query, top_n, **kwargs: fake_result,
    )

    code = cli.main(["--tui-query", json.dumps({"query": "happy"})])

    captured = capsys.readouterr()
    lines = captured.out.strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert code == 0
    assert row["is_exact"] is True
    assert row["relevance"] == 100
    assert row["synonyms"] == ["glad", "cheerful"]


def test_tui_query_with_blank_query_prints_nothing(monkeypatch, capsys):
    code = cli.main(["--tui-query", json.dumps({"query": ""})])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out == ""


def test_tui_query_with_empty_payload_prints_nothing(capsys):
    code = cli.main(["--tui-query", ""])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out == ""


def test_tui_query_with_invalid_json_prints_a_clean_error(capsys):
    code = cli.main(["--tui-query", "{not valid json"])
    captured = capsys.readouterr()
    assert code == 1
    assert "revdict: error:" in captured.out


def test_tui_query_propagates_search_value_error_as_a_clean_message(monkeypatch, capsys):
    def fake_get_search_result(query, top_n, **kwargs):
        raise ValueError("Unknown sort mode: 'bogus'")

    monkeypatch.setattr(cli, "_get_search_result", fake_get_search_result)

    code = cli.main(["--tui-query", json.dumps({"query": "happy", "sort": "bogus"})])

    captured = capsys.readouterr()
    assert code == 1
    assert "revdict: error: Unknown sort mode" in captured.out


def test_jsonl_query_and_tui_query_produce_identical_rows_for_the_same_result(monkeypatch, capsys):
    """Both flags must share the exact same row-building logic (DRY) --
    this test locks in that they can never silently drift apart."""
    fake_result = {
        "exact_match": None,
        "candidates": [
            {
                "headword": "joyful", "pos": "adjective",
                "definition": "feeling great happiness",
                "examples": [], "label": "joy", "polarity": "positive",
                "relevance": 90,
            }
        ],
    }
    monkeypatch.setattr(
        cli, "_get_search_result",
        lambda query, top_n, **kwargs: fake_result,
    )

    code_jsonl = cli.main(["--jsonl-query", "happy"])
    jsonl_row = json.loads(capsys.readouterr().out.strip())

    code_tui = cli.main(["--tui-query", json.dumps({"query": "happy"})])
    tui_row = json.loads(capsys.readouterr().out.strip())

    assert code_jsonl == 0
    assert code_tui == 0
    assert jsonl_row == tui_row


def test_main_with_no_args_and_a_tty_launches_the_live_session(monkeypatch):
    # Patches only `isatty` on the real (capsys-managed) sys.stdout object in
    # place, rather than replacing sys.stdout wholesale -- a wholesale
    # replacement breaks Rich's Console and capsys (see the sibling test
    # below). Also explicitly mocks _fzf_missing: previously this test only
    # passed because fzf happened to be on PATH on the dev machine, so the
    # real fzf-missing branch (which crashes on the old-style fake stdout)
    # was never exercised. Without fzf installed, this test would have
    # failed with the same AttributeError the sibling test's fix addressed.
    monkeypatch.setattr(cli, "_index_exists", lambda: True)
    monkeypatch.setattr(cli, "_fzf_missing", lambda: False)

    called = {"ran": False}
    monkeypatch.setattr(cli.picker, "run_live_session", lambda: called.__setitem__("ran", True))

    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)

    code = cli.main([])

    assert code == 0
    assert called["ran"] is True


def test_main_with_no_args_and_a_tty_but_missing_fzf_prints_a_clear_message(monkeypatch, capsys):
    # Deviation from the task brief's literal test code: the brief's
    # _TtyStdout fake only implements isatty(), then does
    # `monkeypatch.setattr(cli.sys, "stdout", _TtyStdout())`. Since `cli.sys`
    # is the actual process-wide `sys` module, that replaces the real
    # sys.stdout everywhere -- including inside Rich's Console, which
    # resolves `self.file` to `sys.stdout` dynamically on every print() call.
    # Because the fake has no `write()`, the console.print() call this branch
    # requires crashes with AttributeError instead of returning code 1. It
    # also disconnects capsys's own capture object, so even a working
    # write() on the fake wouldn't reach `captured.out`. The minimal fix that
    # preserves the brief's assertions and capsys capture is to patch just
    # `isatty` on the real (capsys-managed) sys.stdout object in place,
    # rather than replacing sys.stdout wholesale.
    monkeypatch.setattr(cli, "_index_exists", lambda: True)
    monkeypatch.setattr(cli, "_fzf_missing", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)

    code = cli.main([])

    captured = capsys.readouterr()
    assert code == 1
    assert "fzf" in captured.out.lower()


def test_is_remote_session_true_when_tmux_is_set(monkeypatch):
    monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,12345,0")
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_CLIENT", raising=False)

    assert cli._is_remote_session() is True


def test_is_remote_session_true_when_ssh_tty_is_set(monkeypatch):
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setenv("SSH_TTY", "/dev/pts/3")
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_CLIENT", raising=False)

    assert cli._is_remote_session() is True


def test_is_remote_session_true_when_ssh_connection_is_set(monkeypatch):
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.setenv("SSH_CONNECTION", "10.0.0.1 22 10.0.0.2 22")
    monkeypatch.delenv("SSH_CLIENT", raising=False)

    assert cli._is_remote_session() is True


def test_is_remote_session_true_when_ssh_client_is_set(monkeypatch):
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.setenv("SSH_CLIENT", "10.0.0.1 22 22")

    assert cli._is_remote_session() is True


def test_is_remote_session_false_when_nothing_is_set(monkeypatch):
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_CLIENT", raising=False)

    assert cli._is_remote_session() is False


def test_build_osc52_sequence_base64_encodes_and_wraps_correctly():
    result = cli._build_osc52_sequence("joy")

    assert result == "\x1b]52;c;am95\x07"


def test_copy_via_system_clipboard_uses_the_first_available_tool(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cli.shutil, "which", lambda name: "/usr/bin/xclip" if name == "xclip" else None
    )

    def fake_run(command, input, check, timeout):
        calls.append((command, input))

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    cli._copy_via_system_clipboard("joy")

    assert len(calls) == 1
    assert calls[0][0] == ["xclip", "-selection", "clipboard"]
    assert calls[0][1] == b"joy"


def test_copy_via_system_clipboard_prefers_wl_copy_when_multiple_tools_exist(monkeypatch):
    calls = []
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/usr/bin/{name}")

    def fake_run(command, input, check, timeout):
        calls.append(command)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    cli._copy_via_system_clipboard("joy")

    assert calls == [["wl-copy"]]


def test_copy_via_system_clipboard_does_nothing_when_no_tool_is_available(monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    calls = []
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: calls.append(a))

    cli._copy_via_system_clipboard("joy")

    assert calls == []


def test_run_copy_selection_uses_osc52_for_a_remote_session(monkeypatch):
    monkeypatch.setattr(cli, "_is_remote_session", lambda: True)
    osc52_calls = []
    clipboard_calls = []
    monkeypatch.setattr(cli, "_copy_via_osc52", lambda text: osc52_calls.append(text))
    monkeypatch.setattr(cli, "_copy_via_system_clipboard", lambda text: clipboard_calls.append(text))

    code = cli._run_copy_selection("joy")

    assert code == 0
    assert osc52_calls == ["joy"]
    assert clipboard_calls == []


def test_run_copy_selection_uses_system_clipboard_for_a_local_session(monkeypatch):
    monkeypatch.setattr(cli, "_is_remote_session", lambda: False)
    osc52_calls = []
    clipboard_calls = []
    monkeypatch.setattr(cli, "_copy_via_osc52", lambda text: osc52_calls.append(text))
    monkeypatch.setattr(cli, "_copy_via_system_clipboard", lambda text: clipboard_calls.append(text))

    code = cli._run_copy_selection("joy")

    assert code == 0
    assert clipboard_calls == ["joy"]
    assert osc52_calls == []


def test_run_copy_selection_does_nothing_for_blank_input(monkeypatch):
    osc52_calls = []
    clipboard_calls = []
    monkeypatch.setattr(cli, "_copy_via_osc52", lambda text: osc52_calls.append(text))
    monkeypatch.setattr(cli, "_copy_via_system_clipboard", lambda text: clipboard_calls.append(text))

    code = cli._run_copy_selection("")

    assert code == 0
    assert osc52_calls == []
    assert clipboard_calls == []


def test_main_dispatches_copy_selection(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "_run_copy_selection", lambda headword: calls.append(headword) or 0)

    code = cli.main(["--copy-selection", "joy"])

    assert code == 0
    assert calls == ["joy"]


def test_main_dispatches_copy_selection_with_empty_string_when_no_argument_given(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "_run_copy_selection", lambda headword: calls.append(headword) or 0)

    code = cli.main(["--copy-selection"])

    assert code == 0
    assert calls == [""]


def test_leading_dash_query_requires_the_argparse_separator(monkeypatch, capsys):
    """A leading '-' in a one-shot query (the disallow-letters pattern
    syntax, e.g. '-abcd') collides with argparse's own flag parsing --
    'revdict -- -abcd' is the documented workaround (POSIX '--' end-of-
    options marker), not a bug in the query parser itself. The live fzf
    session is unaffected: --query-only/--jsonl-query read argv[1] directly
    and never go through this argparse path.

    Deviation from the task brief's literal test code: mocks
    _index_exists/_get_search_result so this doesn't silently depend on a
    real index being built on whichever machine runs the suite -- every
    other test in this file whose cli.main(...) call reaches past the
    index-exists check does the same (see e.g.
    test_main_prints_error_and_returns_1_when_index_missing above). Without
    this, the test would return 1 instead of 0 on a fresh checkout/CI with
    no index, for a reason unrelated to the '--' behavior it's meant to
    lock in."""
    from revdict import cli

    monkeypatch.setattr(cli, "_index_exists", lambda: True)
    monkeypatch.setattr(
        cli, "_get_search_result", lambda query, top_n, sort_mode=None, category=None, syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None: {"exact_match": None, "candidates": []}
    )

    code = cli.main(["--no-interactive", "--", "-abcd"])

    assert code == 0


def test_query_parser_accepts_all_seven_sort_modes():
    from revdict import cli

    parser = cli._query_parser()

    for mode in ("relevance", "alpha", "alpha_desc", "shortest", "longest", "most_common", "least_common"):
        args = parser.parse_args(["happy", "--sort", mode])
        assert args.sort == mode


def test_query_parser_rejects_an_invalid_sort_mode():
    import pytest

    from revdict import cli

    parser = cli._query_parser()

    # Deviation from the task brief's literal test code: parse_args() on
    # this file's parser goes through _QuietArgumentParser.error(), which
    # raises cli._ArgumentError rather than calling SystemExit directly (see
    # _ArgumentError's use elsewhere in this file, e.g.
    # test_main_error_message_is_not_mangled_by_rich_markup, which triggers
    # the same override via cli.main()). SystemExit is only raised for
    # argparse's built-in exit() paths like --help, not error() paths like
    # an invalid --sort choice.
    with pytest.raises(cli._ArgumentError):
        parser.parse_args(["happy", "--sort", "nonsense"])


def test_query_parser_sort_defaults_to_none():
    from revdict import cli

    parser = cli._query_parser()

    args = parser.parse_args(["happy"])
    assert args.sort is None


def test_main_passes_sort_flag_through_to_get_search_result(monkeypatch):
    from revdict import cli

    monkeypatch.setattr(cli, "_index_exists", lambda: True)
    calls = {}

    def fake_get_search_result(query, top_n, sort_mode=None, category=None, syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None):
        calls["sort_mode"] = sort_mode
        return {"exact_match": None, "candidates": []}

    monkeypatch.setattr(cli, "_get_search_result", fake_get_search_result)

    code = cli.main(["happy", "--sort", "alpha", "--no-interactive"])

    assert code == 0
    assert calls["sort_mode"] == "alpha"


def test_main_without_sort_flag_passes_none(monkeypatch):
    from revdict import cli

    monkeypatch.setattr(cli, "_index_exists", lambda: True)
    calls = {}

    def fake_get_search_result(query, top_n, sort_mode=None, category=None, syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None):
        calls["sort_mode"] = sort_mode
        return {"exact_match": None, "candidates": []}

    monkeypatch.setattr(cli, "_get_search_result", fake_get_search_result)

    code = cli.main(["happy", "--no-interactive"])

    assert code == 0
    assert calls["sort_mode"] is None


def test_query_parser_accepts_all_seven_categories():
    from revdict import cli

    parser = cli._query_parser()

    for value in ("all", "noun", "adjective", "verb", "adverb", "idiom_slang", "old"):
        args = parser.parse_args(["happy", "--category", value])
        assert args.category == value


def test_query_parser_rejects_an_invalid_category():
    import pytest

    from revdict import cli

    parser = cli._query_parser()

    with pytest.raises(cli._ArgumentError):
        parser.parse_args(["happy", "--category", "nonsense"])


def test_query_parser_category_defaults_to_none():
    from revdict import cli

    parser = cli._query_parser()

    args = parser.parse_args(["happy"])
    assert args.category is None


def test_main_passes_category_flag_through_to_get_search_result(monkeypatch):
    from revdict import cli

    monkeypatch.setattr(cli, "_index_exists", lambda: True)
    calls = {}

    def fake_get_search_result(query, top_n, sort_mode=None, category=None, syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None):
        calls["category"] = category
        return {"exact_match": None, "candidates": []}

    monkeypatch.setattr(cli, "_get_search_result", fake_get_search_result)

    code = cli.main(["happy", "--category", "noun", "--no-interactive"])

    assert code == 0
    assert calls["category"] == "noun"


def test_main_without_category_flag_passes_none(monkeypatch):
    from revdict import cli

    monkeypatch.setattr(cli, "_index_exists", lambda: True)
    calls = {}

    def fake_get_search_result(query, top_n, sort_mode=None, category=None, syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None):
        calls["category"] = category
        return {"exact_match": None, "candidates": []}

    monkeypatch.setattr(cli, "_get_search_result", fake_get_search_result)

    code = cli.main(["happy", "--no-interactive"])

    assert code == 0
    assert calls["category"] is None


def test_query_parser_accepts_all_five_phonetic_flags():
    from revdict import cli

    parser = cli._query_parser()

    args = parser.parse_args([
        "happy", "--syllables", "2", "--primary-vowel", "AE",
        "--rhymes-with", "cat", "--sounds-like", "bat", "--meter", "/x",
    ])
    assert args.syllables == 2
    assert args.primary_vowel == "AE"
    assert args.rhymes_with == "cat"
    assert args.sounds_like == "bat"
    assert args.meter == "/x"


def test_query_parser_phonetic_flags_default_to_none():
    from revdict import cli

    parser = cli._query_parser()

    args = parser.parse_args(["happy"])
    assert args.syllables is None
    assert args.primary_vowel is None
    assert args.rhymes_with is None
    assert args.sounds_like is None
    assert args.meter is None


def test_query_parser_rejects_a_non_integer_syllables_value():
    import pytest

    from revdict import cli

    parser = cli._query_parser()

    with pytest.raises(cli._ArgumentError):
        parser.parse_args(["happy", "--syllables", "two"])


def test_query_parser_rejects_an_invalid_primary_vowel():
    """--primary-vowel is a closed ARPAbet vowel set -- a typo or a stray
    stress digit (e.g. "AE1" instead of "AE") must fail loudly via
    argparse's choices=, not silently pass through and match nothing."""
    import pytest

    from revdict import cli

    parser = cli._query_parser()

    with pytest.raises(cli._ArgumentError):
        parser.parse_args(["happy", "--primary-vowel", "AE1"])


def test_query_parser_primary_vowel_is_case_insensitive():
    from revdict import cli

    parser = cli._query_parser()

    args = parser.parse_args(["happy", "--primary-vowel", "ae"])
    assert args.primary_vowel == "AE"


def test_query_parser_rejects_an_invalid_meter_pattern():
    """A --meter value with anything other than '/' and 'x' must fail
    loudly, not silently match nothing."""
    import pytest

    from revdict import cli

    parser = cli._query_parser()

    with pytest.raises(cli._ArgumentError):
        parser.parse_args(["happy", "--meter", "/-x"])


def test_main_passes_all_five_phonetic_flags_through_to_get_search_result(monkeypatch):
    from revdict import cli

    monkeypatch.setattr(cli, "_index_exists", lambda: True)
    calls = {}

    def fake_get_search_result(query, top_n, sort_mode=None, category=None, syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None):
        calls.update(
            syllables=syllables, primary_vowel=primary_vowel, rhymes_with=rhymes_with,
            sounds_like=sounds_like, meter=meter,
        )
        return {"exact_match": None, "candidates": []}

    monkeypatch.setattr(cli, "_get_search_result", fake_get_search_result)

    code = cli.main([
        "happy", "--syllables", "2", "--primary-vowel", "AE",
        "--rhymes-with", "cat", "--sounds-like", "bat", "--meter", "/x",
        "--no-interactive",
    ])

    assert code == 0
    assert calls == {"syllables": 2, "primary_vowel": "AE", "rhymes_with": "cat", "sounds_like": "bat", "meter": "/x"}


def test_main_without_phonetic_flags_passes_all_none(monkeypatch):
    from revdict import cli

    monkeypatch.setattr(cli, "_index_exists", lambda: True)
    calls = {}

    def fake_get_search_result(query, top_n, sort_mode=None, category=None, syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None):
        calls.update(
            syllables=syllables, primary_vowel=primary_vowel, rhymes_with=rhymes_with,
            sounds_like=sounds_like, meter=meter,
        )
        return {"exact_match": None, "candidates": []}

    monkeypatch.setattr(cli, "_get_search_result", fake_get_search_result)

    code = cli.main(["happy", "--no-interactive"])

    assert code == 0
    assert calls == {"syllables": None, "primary_vowel": None, "rhymes_with": None, "sounds_like": None, "meter": None}


def test_main_prints_a_clean_error_message_when_a_phonetic_target_cannot_be_resolved(monkeypatch, capsys):
    """Regression guard: search()/resolve_phonetic_target deliberately
    raises ValueError when --rhymes-with/--sounds-like can't resolve their
    target (e.g. stressmark missing/outdated) -- this must surface as a
    clean `revdict: error: ...` message and exit code 1, not an unhandled
    traceback."""
    from revdict import cli

    monkeypatch.setattr(cli, "_index_exists", lambda: True)

    def fake_get_search_result(query, top_n, sort_mode=None, category=None, syllables=None, primary_vowel=None, rhymes_with=None, sounds_like=None, meter=None):
        raise ValueError(
            "--rhymes-with requires the stressmark library (>= 0.2.0) to be installed and importable."
        )

    monkeypatch.setattr(cli, "_get_search_result", fake_get_search_result)

    code = cli.main(["cat", "--rhymes-with", "hat", "--no-interactive"])

    captured = capsys.readouterr()
    assert code == 1
    assert "Traceback" not in captured.out
    assert "Traceback" not in captured.err
    assert "stressmark" in captured.out or "stressmark" in captured.err
