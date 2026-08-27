"""Links to equivalent searches in the OneLook reverse dictionary."""

import webbrowser
from urllib.parse import urlencode


ONELOOK_SEARCH_URL = "https://www.onelook.com/thesaurus/"


def build_search_url(query: str) -> str:
    """Build a OneLook reverse-dictionary URL preserving the exact query."""
    query = query.strip()
    if not query:
        raise ValueError("A OneLook search URL requires a non-empty query.")
    return f"{ONELOOK_SEARCH_URL}?{urlencode({'loc': 'revfp', 's': query})}"


def open_url(url: str) -> bool:
    """Open a OneLook URL in the user's preferred browser."""
    return webbrowser.open(url, new=2)
