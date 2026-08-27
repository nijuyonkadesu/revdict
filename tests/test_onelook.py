from urllib.parse import parse_qs, urlparse

import pytest

from revdict import onelook


def test_build_search_url_preserves_complete_revdict_query():
    query = "??lon:synthetic fabric -ei //letters"

    parsed = urlparse(onelook.build_search_url(query))

    assert parsed.scheme == "https"
    assert parsed.netloc == "www.onelook.com"
    assert parsed.path == "/thesaurus/"
    assert parse_qs(parsed.query) == {"loc": ["revfp"], "s": [query]}


def test_build_search_url_rejects_an_empty_query():
    with pytest.raises(ValueError, match="non-empty query"):
        onelook.build_search_url("  ")
