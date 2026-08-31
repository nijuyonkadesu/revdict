import bisect
import heapq

import numpy as np

from revdict.pattern_matcher import compile_clauses
from revdict.progress import ProgressReporter
from revdict.query_syntax import ParsedQuery
from revdict.structural_index import (
    EXPAND_SKIP_WORDS,
    CompactStructuralIndex,
    ascii_letter_mask,
)

# Real acronym expansion skips small function words rather than taking
# every token's initial literally -- "national aeronautics and space
# administration" only reduces to n-a-s-a once "and" is dropped (verified
# by hand: naively including it gives n-a-a-s-a, which never matches).
_EXPAND_SKIP_WORDS = EXPAND_SKIP_WORDS
_PREFIX_WILDCARDS = frozenset("*?#@")


def _literal_prefix(clauses: list[str]) -> str | None:
    """Return the lowercase literal in a provably pure ``prefix*`` query."""
    if len(clauses) != 1:
        return None
    clause = clauses[0]
    if (
        len(clause) <= 1
        or not clause.endswith("*")
        or clause.count("*") != 1
        or any(character in _PREFIX_WILDCARDS for character in clause[:-1])
        or clause.startswith(("+", "-"))
        or "//" in clause
    ):
        return None
    return clause[:-1].lower()


def _prefix_bounds(words: list[str], prefix: str) -> tuple[int, int]:
    start = bisect.bisect_left(words, prefix)
    successor = None
    for index in range(len(prefix) - 1, -1, -1):
        codepoint = ord(prefix[index])
        if codepoint < 0x10FFFF:
            successor = prefix[:index] + chr(codepoint + 1)
            break
    end = len(words) if successor is None else bisect.bisect_left(words, successor)
    return start, end


def _prefix_matches(words: list[str], prefix: str) -> list[str]:
    start, end = _prefix_bounds(words, prefix)
    return words[start:end]


def _frequency_rank(pair: tuple[str, float]) -> tuple[float, str]:
    return -pair[1], pair[0]


def _intersect_candidates(candidates: list[np.ndarray]) -> np.ndarray | None:
    usable = [np.asarray(values, dtype="uint32") for values in candidates if values is not None]
    if not usable:
        return None
    usable.sort(key=len)
    result = usable[0]
    for values in usable[1:]:
        if not len(result):
            break
        result = np.intersect1d(result, values, assume_unique=True)
    return np.asarray(result, dtype="uint32")


def _prefix_word_ids(words: list[str], prefix: str) -> np.ndarray:
    start, end = _prefix_bounds(words, prefix)
    return np.arange(start, end, dtype="uint32")


def _wildcard_clause_plan(
    clause: str,
    planner: CompactStructuralIndex,
    words: list[str],
) -> tuple[np.ndarray | None, bool]:
    lowered = clause.lower()
    wildcard_positions = [
        index for index, character in enumerate(lowered) if character in _PREFIX_WILDCARDS
    ]
    if not wildcard_positions:
        candidates = [
            _prefix_word_ids(words, lowered),
            planner.ids_with_length(len(lowered)),
        ]
        return _intersect_candidates(candidates), True

    first_wildcard = wildcard_positions[0]
    last_wildcard = wildcard_positions[-1]
    literal_prefix = lowered[:first_wildcard]
    literal_suffix = lowered[last_wildcard + 1 :]
    candidates: list[np.ndarray] = []
    if literal_prefix:
        candidates.append(_prefix_word_ids(words, literal_prefix))
    if literal_suffix:
        candidates.append(planner.suffix_ids(literal_suffix))
    if "*" not in lowered:
        candidates.append(planner.ids_with_length(len(lowered)))

    literal_letters = "".join(
        character for character in lowered if character not in _PREFIX_WILDCARDS
    )
    required_mask = ascii_letter_mask(literal_letters)
    if required_mask:
        candidates.append(planner.ids_with_letters(required_mask))

    pure_prefix = _literal_prefix([lowered]) is not None
    pure_suffix = (
        lowered.startswith("*")
        and lowered.count("*") == 1
        and len(lowered) > 1
        and not any(character in _PREFIX_WILDCARDS for character in lowered[1:])
    )
    exact_contains_letter = (
        len(lowered) == 3
        and lowered[0] == lowered[2] == "*"
        and "a" <= lowered[1] <= "z"
    )
    return _intersect_candidates(candidates), (
        lowered == "*" or pure_prefix or pure_suffix or exact_contains_letter
    )


def _clause_plan(
    clause: str,
    planner: CompactStructuralIndex,
    words: list[str],
) -> tuple[np.ndarray | None, bool]:
    if "//" in clause:
        signature = clause.strip("/").lower()
        return planner.anagram_ids(signature), False
    if clause.startswith("-"):
        excluded = set(clause[1:].lower())
        mask = ascii_letter_mask("".join(excluded))
        candidates = planner.ids_without_letters(mask) if mask else None
        return candidates, all("a" <= character <= "z" for character in excluded)
    if clause.startswith("+"):
        allowed = set(clause[1:].lower())
        if all("a" <= character <= "z" for character in allowed):
            return planner.ids_restricted_to_letters(
                ascii_letter_mask("".join(allowed))
            ), True
        return None, False
    return _wildcard_clause_plan(clause, planner, words)


def planned_word_ids(
    parsed: ParsedQuery,
    planner: CompactStructuralIndex,
    words: list[str],
) -> np.ndarray:
    """Return exact matching word IDs using indexes plus predicate confirmation."""
    if parsed.mode == "expand":
        return planner.acronym_ids(parsed.expand_target or "")
    if parsed.mode == "phrase_contains":
        return planner.phrase_ids(parsed.phrase_word or "")
    if parsed.mode not in ("structural", "combined"):
        raise ValueError(f"planned_word_ids does not support mode {parsed.mode!r}")

    clause_plans = [
        _clause_plan(clause, planner, words)
        for clause in parsed.pattern_clauses
    ]
    candidates = _intersect_candidates(
        [values for values, _exact in clause_plans if values is not None]
    )
    if candidates is None:
        candidates = np.arange(len(words), dtype="uint32")
    if all(exact for _values, exact in clause_plans):
        return candidates

    predicate = compile_clauses(parsed.pattern_clauses)
    return np.fromiter(
        (word_id for word_id in candidates if predicate(words[int(word_id)])),
        dtype="uint32",
    )


def matching_headwords(
    parsed: ParsedQuery,
    word_index: dict[str, list[int]] | list[str],
    *,
    sorted_headwords: bool = False,
) -> list[str]:
    if parsed.mode == "structural":
        prefix = _literal_prefix(parsed.pattern_clauses)
        if prefix is not None and sorted_headwords:
            return _prefix_matches(word_index, prefix)
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


def _score_and_sort(
    headwords: list[str],
    literary_frequency: dict[str, float],
    limit: int | None = None,
    word_id_start: int | None = None,
) -> list[tuple[str, float]]:
    compact_words = getattr(literary_frequency, "words", None)
    if (
        word_id_start is not None
        and compact_words is not None
        and word_id_start + len(headwords) <= len(compact_words)
    ):
        scored = (
            (
                word,
                compact_words.frequency_by_id(word_id_start + offset, 0.0),
            )
            for offset, word in enumerate(headwords)
        )
    else:
        scored = (
            (word, literary_frequency.get(word, 0.0))
            for word in headwords
        )
    if limit is None:
        return sorted(scored, key=_frequency_rank)
    return heapq.nsmallest(max(limit, 0), scored, key=_frequency_rank)


def _run_compact_structural(
    parsed: ParsedQuery,
    state: dict,
    top_n: int,
    category: str | None,
    syllables: int | None,
    primary_vowel: str | None,
    rhyme_key: str | None,
    sounds_like_phonemes: list[str] | None,
    meter: str | None,
    progress: ProgressReporter,
) -> dict:
    from revdict.search import build_candidate, preload_emotions, relative_relevance

    word_index = state["word_index"]
    metadata = state["metadata"]
    facets = state["facets"]
    planner = state["structural_index"]
    words = state["headwords"]

    progress.active("scope")
    word_ids = planned_word_ids(parsed, planner, words)
    progress.completed("scope")

    progress.active("retrieve")
    has_filters = bool(category and category != "all") or syllables is not None or any(
        [primary_vowel, rhyme_key, sounds_like_phonemes, meter]
    )
    if has_filters:
        candidate_rows = word_index.rows_for_ids(word_ids)
        matched_rows = facets.matching_rows(
            candidate_rows,
            category=category,
            syllables=syllables,
            primary_vowel=primary_vowel,
            rhyme_key=rhyme_key,
            sounds_like_phonemes=sounds_like_phonemes,
            meter=meter,
        )
        matched_rows = np.asarray(matched_rows, dtype="uint32")
        if len(matched_rows):
            matched_word_ids = np.asarray(
                word_index.row_headwords[matched_rows], dtype="uint32"
            )
            word_ids, first_positions = np.unique(
                matched_word_ids, return_index=True
            )
            selected_rows = matched_rows[first_positions]
        else:
            word_ids = np.empty(0, dtype="uint32")
            selected_rows = np.empty(0, dtype="uint32")
    else:
        selected_rows = word_index.first_rows_for_ids(word_ids)

    frequencies = np.asarray(word_index.frequencies[word_ids], dtype="float64")
    frequencies = np.nan_to_num(frequencies, copy=True, nan=0.0)
    ranked_positions = heapq.nsmallest(
        max(top_n, 0),
        range(len(word_ids)),
        key=lambda position: (-float(frequencies[position]), int(word_ids[position])),
    )
    ranked_scores = [float(frequencies[position]) for position in ranked_positions]
    progress.completed("retrieve")
    progress.active("filter")
    progress.completed("filter")
    relevances = relative_relevance(ranked_scores)

    progress.active("enrich")
    ranked_rows = [int(selected_rows[position]) for position in ranked_positions]
    selected_records = [metadata[row] for row in ranked_rows]
    preload_emotions(selected_records, state)
    candidates = [
        build_candidate(record, relevance, state)
        for record, relevance in zip(selected_records, relevances)
    ]
    progress.completed("enrich")
    return {"exact_match": None, "candidates": candidates}


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
    progress: ProgressReporter | None = None,
) -> dict:
    # Callers are responsible for validating `category` against
    # category.CATEGORIES before calling this function; search() does that
    # eagerly before dispatch, so this function doesn't duplicate it.

    progress = progress or ProgressReporter()
    if (
        state.get("structural_index") is not None
        and state.get("facets") is not None
        and state.get("headwords") is not None
        and hasattr(state.get("word_index"), "rows_for_ids")
    ):
        return _run_compact_structural(
            parsed,
            state,
            top_n,
            category,
            syllables,
            primary_vowel,
            rhyme_key,
            sounds_like_phonemes,
            meter,
            progress,
        )

    # Deferred import: search.py imports structural_search for dispatch
    # (Task 6), so importing search.py at module load time here would be
    # circular. Matches the lazy-import pattern already used elsewhere in
    # this codebase (cli.py's _local_search_fallback, daemon.py's
    # run_server) to defer a heavy/cyclic import until it's actually needed.
    from revdict.search import build_candidate, preload_emotions, relative_relevance

    word_index = state["word_index"]
    metadata = state["metadata"]
    literary_frequency = state["literary_frequency"]

    progress.active("scope")
    sorted_words = state.get("headwords")
    prefix = _literal_prefix(parsed.pattern_clauses)
    prefix_word_id_start = None
    if prefix is not None and sorted_words is not None:
        prefix_word_id_start, _ = _prefix_bounds(sorted_words, prefix)
    headwords = matching_headwords(
        parsed,
        sorted_words if sorted_words is not None else word_index,
        sorted_headwords=sorted_words is not None,
    )
    progress.completed("scope")
    progress.active("retrieve")
    facets = state.get("facets")
    has_filters = bool(category and category != "all") or syllables is not None or any(
        [primary_vowel, rhyme_key, sounds_like_phonemes, meter]
    )
    selected_rows = None
    if facets is not None and has_filters:
        candidate_rows = []
        for word in headwords:
            candidate_rows.extend(word_index[word])
        matched_rows = facets.matching_rows(
            candidate_rows,
            category=category,
            syllables=syllables,
            primary_vowel=primary_vowel,
            rhyme_key=rhyme_key,
            sounds_like_phonemes=sounds_like_phonemes,
            meter=meter,
        )
        accepted = set(matched_rows.tolist())
        selected_rows = {}
        for word in headwords:
            matching_rows = (row for row in word_index[word] if row in accepted)
            row = next(matching_rows, None)
            if row is not None:
                selected_rows[word] = row
        headwords = list(selected_rows)
    elif category and category != "all":
        from revdict import category as category_module
        headwords = [word for word in headwords if category_module.matches_category(metadata[word_index[word][0]], category)]
    # syllables is checked with `is not None` rather than folded into the
    # same any([...]) truthiness check as the other 4 -- 0 is a real,
    # meaningful filter value for syllable count (no real word has 0
    # syllables, so syllables=0 should exclude everything), but Python's
    # any() treats 0 as falsy, which would otherwise make this guard
    # silently skip filtering whenever syllables was exactly 0.
    if facets is None and (syllables is not None or any([primary_vowel, rhyme_key, sounds_like_phonemes, meter])):
        from revdict import phonetics
        headwords = [
            word
            for word in headwords
            if phonetics.matches_syllable_count(metadata[word_index[word][0]], syllables)
            and phonetics.matches_primary_vowel(metadata[word_index[word][0]], primary_vowel)
            and phonetics.matches_rhyme(metadata[word_index[word][0]], rhyme_key)
            and phonetics.matches_sounds_like(metadata[word_index[word][0]], sounds_like_phonemes)
            and phonetics.matches_meter(metadata[word_index[word][0]], meter)
        ]
    ranked = _score_and_sort(
        headwords,
        literary_frequency,
        top_n,
        word_id_start=(
            prefix_word_id_start
            if not has_filters and getattr(literary_frequency, "words", None) is word_index
            else None
        ),
    )
    if selected_rows is None:
        selected_rows = {word: word_index[word][0] for word, _ in ranked}
    progress.completed("retrieve")
    progress.active("filter")
    progress.completed("filter")
    relevances = relative_relevance([score for _, score in ranked])

    progress.active("enrich")
    selected_records = [metadata[selected_rows[headword]] for headword, _ in ranked]
    preload_emotions(selected_records, state)
    candidates = [
        build_candidate(metadata[selected_rows[headword]], relevance, state)
        for (headword, _), relevance in zip(ranked, relevances)
    ]
    progress.completed("enrich")

    return {"exact_match": None, "candidates": candidates}


def matching_row_indices(
    parsed: ParsedQuery,
    word_index: dict[str, list[int]],
    sorted_headwords: list[str] | None = None,
    structural_index: CompactStructuralIndex | None = None,
) -> list[int] | np.ndarray:
    if structural_index is not None and sorted_headwords is not None:
        word_ids = planned_word_ids(parsed, structural_index, sorted_headwords)
        rows_for_ids = getattr(word_index, "rows_for_ids", None)
        if rows_for_ids is not None:
            return rows_for_ids(word_ids)
    prefix = _literal_prefix(parsed.pattern_clauses)
    if prefix is not None and sorted_headwords is not None:
        start, end = _prefix_bounds(sorted_headwords, prefix)
        rows_for_range = getattr(word_index, "rows_for_id_range", None)
        if rows_for_range is not None and len(word_index) == len(sorted_headwords):
            return rows_for_range(start, end).tolist()
        matching_words = sorted_headwords[start:end]
    else:
        predicate = compile_clauses(parsed.pattern_clauses)
        matching_words = (word for word in word_index if predicate(word))
    indices = []
    for word in matching_words:
        indices.extend(word_index[word])
    return indices
