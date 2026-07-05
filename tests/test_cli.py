import sys

from revdict import cli
from revdict.picker import PickerError


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
    monkeypatch.setattr(cli.daemon, "daemon_status", lambda: "revdict daemon is running (pid 1).")

    cli._build_index(skip_confirm=True)

    captured = capsys.readouterr()
    assert "daemon stop" in captured.out


def test_build_index_says_nothing_when_no_daemon_is_running(monkeypatch, capsys):
    import revdict.data.build_index as build_index_module

    monkeypatch.setattr(build_index_module, "build", lambda skip_confirm: None)
    monkeypatch.setattr(cli.daemon, "daemon_status", lambda: "revdict daemon is not running.")

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
        cli.daemon, "send_query", lambda query, top_n: {"exact_match": None, "candidates": []}
    )

    def fail_if_called():
        raise AssertionError("ensure_daemon_running should not be called if send_query answers")

    monkeypatch.setattr(cli.daemon, "ensure_daemon_running", fail_if_called)

    result = cli._get_search_result("happy", 10)

    assert result == {"exact_match": None, "candidates": []}


def test_get_search_result_starts_daemon_and_retries_when_first_attempt_fails(monkeypatch):
    attempts = {"count": 0}

    def fake_send_query(query, top_n):
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
    monkeypatch.setattr(cli.daemon, "send_query", lambda query, top_n: None)
    monkeypatch.setattr(cli.daemon, "ensure_daemon_running", lambda: False)

    fake_result = {"exact_match": None, "candidates": [{"headword": "fallback-used"}]}
    monkeypatch.setattr(cli, "_local_search_fallback", lambda query, top_n: fake_result)

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
    monkeypatch.setattr(cli, "_get_search_result", lambda query, top_n: fake_result)

    code = cli._run_query("happy", top_n=10, interactive=False)

    captured = capsys.readouterr()
    assert code == 0
    assert "joy · positive" in captured.out
    assert "neutral · neutral" in captured.out
    assert "glad" in captured.out and "content" in captured.out
    assert "Synonyms: \n" not in captured.out


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
    monkeypatch.setattr(cli, "_get_search_result", lambda query, top_n: fake_result)

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
    monkeypatch.setattr(cli, "_get_search_result", lambda query, top_n: _FAKE_INTERACTIVE_RESULT)
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
    monkeypatch.setattr(cli, "_get_search_result", lambda query, top_n: _FAKE_INTERACTIVE_RESULT)
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
    monkeypatch.setattr(cli, "_get_search_result", lambda query, top_n: _FAKE_INTERACTIVE_RESULT)

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
