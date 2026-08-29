from urllib.parse import parse_qs, urlparse

import pytest

from revdict import onelook


def test_build_result_url_targets_only_the_selected_headword():
    parsed = urlparse(onelook.build_result_url("periost"))

    assert parsed.scheme == "https"
    assert parsed.netloc == "www.onelook.com"
    assert parsed.path == "/thesaurus/"
    assert parse_qs(parsed.query) == {"loc": ["revfp"], "s": ["periost"]}


def test_build_result_urls_differ_for_each_selected_result():
    first = onelook.build_result_url("percale")
    second = onelook.build_result_url("periost")

    assert first != second
    assert parse_qs(urlparse(first).query)["s"] == ["percale"]
    assert parse_qs(urlparse(second).query)["s"] == ["periost"]


def test_build_result_url_rejects_an_empty_headword():
    with pytest.raises(ValueError, match="non-empty headword"):
        onelook.build_result_url("  ")


def test_open_url_requests_a_new_browser_tab(monkeypatch):
    calls = []
    monkeypatch.setattr(
        onelook.webbrowser,
        "open",
        lambda url, new: calls.append((url, new)) or True,
    )

    assert onelook.open_url("https://www.onelook.com/thesaurus/?s=periost") is True
    assert calls == [("https://www.onelook.com/thesaurus/?s=periost", 2)]


@pytest.mark.parametrize("error", [OSError("no browser"), onelook.webbrowser.Error("failed")])
def test_open_url_reports_browser_launch_failure(monkeypatch, error):
    def fail(_url, new):
        assert new == 2
        raise error

    monkeypatch.setattr(onelook.webbrowser, "open", fail)

    assert onelook.open_url("https://www.onelook.com/thesaurus/?s=periost") is False
