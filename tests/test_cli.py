from revdict import cli


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
