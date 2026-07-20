from revdict import category as category_module
from revdict import phonetics
from revdict.pattern_matcher import compile_clauses
from revdict.query_syntax import ParsedQuery

# Real acronym expansion skips small function words rather than taking
# every token's initial literally -- "national aeronautics and space
# administration" only reduces to n-a-s-a once "and" is dropped (verified
# by hand: naively including it gives n-a-a-s-a, which never matches).
_EXPAND_SKIP_WORDS = {"and", "of", "the", "for", "a", "an", "&"}


def matching_headwords(parsed: ParsedQuery, word_index: dict[str, list[int]]) -> list[str]:
    if parsed.mode == "structural":
        predicate = compile_clauses(parsed.pattern_clauses)
        return [word for word in word_index if predicate(word)]

    if parsed.mode == "expand":
        target = parsed.expand_target
        matches = []
        for word in word_index:
            tokens = [token for token in word.split() if token.lower() not in _EXPAND_SKIP_WORDS]
            if len(tokens) < 2:
                continue
            initials = "".join(token[0] for token in tokens if token).lower()
            if initials == target:
                matches.append(word)
        return matches

    if parsed.mode == "phrase_contains":
        target = parsed.phrase_word
        matches = []
        for word in word_index:
            tokens = [token.lower() for token in word.split()]
            if len(tokens) < 2:
                continue
            if target in tokens:
                matches.append(word)
        return matches

    raise ValueError(f"matching_headwords does not support mode {parsed.mode!r}")


def _score_and_sort(headwords: list[str], literary_frequency: dict[str, float]) -> list[tuple[str, float]]:
    scored = [(word, literary_frequency.get(word, 0.0)) for word in headwords]
    return sorted(scored, key=lambda pair: (-pair[1], pair[0]))


def run_structural(
    parsed: ParsedQuery,
    state: dict,
    top_n: int,
    category: str | None = None,
    syllables: int | None = None,
    primary_vowel: str | None = None,
    rhyme_key: str | None = None,
    sounds_like_phonemes: list[str] | None = None,
    meter: str | None = None,
) -> dict:
    # Callers are responsible for validating `category` against
    # category.CATEGORIES before calling this function; search() does that
    # eagerly before dispatch, so this function doesn't duplicate it.

    # Deferred import: search.py imports structural_search for dispatch
    # (Task 6), so importing search.py at module load time here would be
    # circular. Matches the lazy-import pattern already used elsewhere in
    # this codebase (cli.py's _local_search_fallback, daemon.py's
    # run_server) to defer a heavy/cyclic import until it's actually needed.
    from revdict.search import build_candidate, relative_relevance

    word_index = state["word_index"]
    metadata = state["metadata"]
    literary_frequency = state["literary_frequency"]

    headwords = matching_headwords(parsed, word_index)
    if category and category != "all":
        headwords = [
            word
            for word in headwords
            if category_module.matches_category(metadata[word_index[word][0]], category)
        ]
    # syllables is checked with `is not None` rather than folded into the
    # same any([...]) truthiness check as the other 4 -- 0 is a real,
    # meaningful filter value for syllable count (no real word has 0
    # syllables, so syllables=0 should exclude everything), but Python's
    # any() treats 0 as falsy, which would otherwise make this guard
    # silently skip filtering whenever syllables was exactly 0.
    if syllables is not None or any([primary_vowel, rhyme_key, sounds_like_phonemes, meter]):
        headwords = [
            word
            for word in headwords
            if phonetics.matches_syllable_count(metadata[word_index[word][0]], syllables)
            and phonetics.matches_primary_vowel(metadata[word_index[word][0]], primary_vowel)
            and phonetics.matches_rhyme(metadata[word_index[word][0]], rhyme_key)
            and phonetics.matches_sounds_like(metadata[word_index[word][0]], sounds_like_phonemes)
            and phonetics.matches_meter(metadata[word_index[word][0]], meter)
        ]
    ranked = _score_and_sort(headwords, literary_frequency)[:top_n]
    relevances = relative_relevance([score for _, score in ranked])

    candidates = [
        build_candidate(metadata[word_index[headword][0]], relevance, state)
        for (headword, _), relevance in zip(ranked, relevances)
    ]

    return {"exact_match": None, "candidates": candidates}


def matching_row_indices(parsed: ParsedQuery, word_index: dict[str, list[int]]) -> list[int]:
    predicate = compile_clauses(parsed.pattern_clauses)
    indices = []
    for word, rows in word_index.items():
        if predicate(word):
            indices.extend(rows)
    return indices
