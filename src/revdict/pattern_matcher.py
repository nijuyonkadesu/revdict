import re
from typing import Callable

# `y` is treated as a consonant (not a vowel) for the `#`/`@` wildcards --
# TODO.md's own symbol legend doesn't specify either way, so this is a
# documented interpretation call: English orthography favors the consonant
# reading far more often (yellow, yes, yard) than the reverse.
_VOWELS = set("aeiou")
_CONSONANTS = set("abcdefghijklmnopqrstuvwxyz") - _VOWELS

_WILDCARD_TRANSLATION = {
    "*": ".*",
    "?": ".",
    "#": f"[{''.join(sorted(_CONSONANTS))}]",
    "@": f"[{''.join(sorted(_VOWELS))}]",
}


def _translate_wildcard_clause(clause: str) -> str:
    # Full anchoring (^...$) is correct uniformly for every wildcard shape
    # here, not just fixed-length ones: 'blue*' -> '^blue.*$' still reads as
    # "starts with blue" because '.*$' already permits anything through to
    # the end of the string.
    parts = [_WILDCARD_TRANSLATION.get(char, re.escape(char)) for char in clause]
    return "^" + "".join(parts) + "$"


def compile_pattern(clause: str) -> Callable[[str], bool]:
    pattern = re.compile(_translate_wildcard_clause(clause), re.IGNORECASE)
    return lambda word: pattern.match(word) is not None
