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


def _compile_wildcard(clause: str) -> Callable[[str], bool]:
    pattern = re.compile(_translate_wildcard_clause(clause), re.IGNORECASE)
    return lambda word: pattern.match(word) is not None


def _compile_disallow(clause: str) -> Callable[[str], bool]:
    excluded = set(clause[1:].lower())
    return lambda word: excluded.isdisjoint(word.lower())


def _compile_restrict(clause: str) -> Callable[[str], bool]:
    allowed = set(clause[1:].lower())
    return lambda word: set(word.lower()) <= allowed


def _compile_anagram(clause: str) -> Callable[[str], bool]:
    letters = sorted(clause.strip("/").lower())
    return lambda word: sorted(word.lower()) == letters


def compile_pattern(clause: str) -> Callable[[str], bool]:
    if "//" in clause:
        return _compile_anagram(clause)
    if clause.startswith("-"):
        return _compile_disallow(clause)
    if clause.startswith("+"):
        return _compile_restrict(clause)
    return _compile_wildcard(clause)


def compile_clauses(clauses: list[str]) -> Callable[[str], bool]:
    predicates = [compile_pattern(clause) for clause in clauses]
    return lambda word: all(predicate(word) for predicate in predicates)
