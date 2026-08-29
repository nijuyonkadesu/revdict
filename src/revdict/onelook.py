"""Links to selected words in the OneLook reverse dictionary."""

import webbrowser
from urllib.parse import urlencode


ONELOOK_SEARCH_URL = "https://www.onelook.com/thesaurus/"


def build_result_url(headword: str) -> str:
    """Build a OneLook URL for the selected result headword."""
    headword = headword.strip()
    if not headword:
        raise ValueError("A OneLook result URL requires a non-empty headword.")
    return f"{ONELOOK_SEARCH_URL}?{urlencode({'loc': 'revfp', 's': headword})}"


def open_url(url: str) -> bool:
    """Open a URL in the default browser without disrupting the TUI on failure."""
    try:
        return webbrowser.open(url, new=2)
    except (OSError, webbrowser.Error):
        return False
