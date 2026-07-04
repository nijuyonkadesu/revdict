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

    def fake_build(skip_confirm):
        called["skip_confirm"] = skip_confirm

    monkeypatch.setattr(cli, "build", fake_build)

    code = cli.main(["build-index", "--yes"])

    assert code == 0
    assert called["skip_confirm"] is True


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
    monkeypatch.setattr(cli.search_mod, "search", lambda query, top_n: fake_result)

    code = cli._run_query("happy", top_n=10, interactive=False)

    captured = capsys.readouterr()
    assert code == 0
    assert "joy · positive" in captured.out
    assert "neutral · neutral" in captured.out
    assert "glad" in captured.out and "content" in captured.out
    # No dangling "Synonyms:" label with nothing after it for the sense with none.
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
    monkeypatch.setattr(cli.search_mod, "search", lambda query, top_n: fake_result)

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
    monkeypatch.setattr(cli.search_mod, "search", lambda query, top_n: _FAKE_INTERACTIVE_RESULT)
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
    monkeypatch.setattr(cli.search_mod, "search", lambda query, top_n: _FAKE_INTERACTIVE_RESULT)
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
    monkeypatch.setattr(cli.search_mod, "search", lambda query, top_n: _FAKE_INTERACTIVE_RESULT)

    def fake_run_picker(candidates, exact_match):
        raise PickerError(2, "inappropriate ioctl for device")

    monkeypatch.setattr(cli, "run_picker", fake_run_picker)
    monkeypatch.setattr(cli, "_fzf_missing", lambda: False)

    code = cli._run_query("happy", top_n=10, interactive=True)

    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.strip() != ""
    assert "joyful" in captured.out  # static fallback table was printed
    assert "ioctl" in captured.out or "fzf" in captured.out.lower()  # error mentioned


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
