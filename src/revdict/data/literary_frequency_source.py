import gzip
import math
from pathlib import Path

from revdict.data.download import (
    download_cached_file,
    validate_gzip_header,
    validate_nonempty_file,
)

RAW_NGRAM_FICTION_URL = (
    "https://storage.googleapis.com/books/ngrams/books/20200217/eng-fiction/1-00000-of-00001.gz"
)
RAW_NGRAM_FICTION_TOTALCOUNTS_URL = (
    "https://storage.googleapis.com/books/ngrams/books/20200217/eng-fiction/totalcounts-1"
)
NGRAM_RELEASE = "20200217"

# Restricting to 2010-2019 favors late-corpus usage over the 19th-century
# diction that dominates this corpus's older years (it stretches back to the
# 1500s) -- this makes the signal "words used in 2010s fiction"
# rather than "words used in fiction, including Shakespeare-era archaisms".
YEAR_RANGE_START = 2010
YEAR_RANGE_END = 2019

_POS_TAGS = {
    "NOUN", "PROPN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "NUM", "CONJ", "PRT", "X"
}


def download_raw_ngram_fiction(dest_path: str, refresh: bool = False) -> dict:
    return download_cached_file(
        RAW_NGRAM_FICTION_URL,
        Path(dest_path),
        validator=_validate_ngram_dump,
        refresh=refresh,
        validation_id="google-ngram-1gram-v1",
    )


def download_raw_ngram_fiction_totalcounts(dest_path: str, refresh: bool = False) -> dict:
    return download_cached_file(
        RAW_NGRAM_FICTION_TOTALCOUNTS_URL,
        Path(dest_path),
        validator=_validate_totalcounts,
        refresh=refresh,
        validation_id="google-ngram-totalcounts-2010-2019-v1",
    )


def _sum_recent_years(year_count_fields: list[str]) -> int:
    total = 0
    for field in year_count_fields:
        year_str, match_count_str = field.split(",", 2)[:2]
        year = int(year_str)
        if YEAR_RANGE_START <= year <= YEAR_RANGE_END:
            total += int(match_count_str)
    return total


def _corpus_total_for_recent_years(totalcounts_text: str) -> int:
    fields = [field for field in totalcounts_text.strip().split("\t") if field]
    return _sum_recent_years(fields)


def _validate_totalcounts(path: Path) -> None:
    validate_nonempty_file(path)
    with Path(path).open(encoding="utf-8") as f:
        total = _corpus_total_for_recent_years(f.read())
    if total <= 0:
        raise ValueError(f"Ngram totalcounts has no tokens in the configured year range: {path}")


def _validate_ngram_dump(path: Path) -> None:
    validate_gzip_header(path)
    parsed = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            token, separator, fields_text = line.partition("\t")
            if not separator or not token or not fields_text.strip():
                raise ValueError(f"Invalid Ngram row on line {line_number}")
            fields = fields_text.rstrip("\n").split("\t")
            try:
                _sum_recent_years(fields)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid Ngram counts on line {line_number}: {error}") from error
            parsed += 1
    if parsed == 0:
        raise ValueError(f"Ngram dump contains no rows: {path}")


def _strip_pos_suffix(token: str) -> str:
    underscore = token.rfind("_")
    if underscore == -1:
        return token
    if token[underscore + 1 :] in _POS_TAGS:
        return token[:underscore]
    return token


def _has_pos_suffix(token: str) -> bool:
    underscore = token.rfind("_")
    return underscore != -1 and token[underscore + 1 :] in _POS_TAGS


def compute_literary_frequencies(
    headwords: set[str], raw_gz_path: str, totalcounts_path: str
) -> dict[str, float]:
    """Builds a headword -> zipf-scale literary-frequency score (log10 of
    matches per billion words, floored at 0.0) from the Google Books Ngram
    "English Fiction" corpus, restricted to YEAR_RANGE_START-YEAR_RANGE_END.

    Only computes scores for the given `headwords` (revdict's own corpus
    vocabulary) -- the raw file has tens of millions of entries, most of
    which are irrelevant, so filtering during the single streaming pass
    keeps this bounded instead of materializing the whole file.
    """
    with open(totalcounts_path, encoding="utf-8") as f:
        corpus_total = _corpus_total_for_recent_years(f.read())
    if corpus_total <= 0:
        raise ValueError("Ngram corpus total must be positive for the configured year range")

    target = {word.lower() for word in headwords}
    # The export contains both bare tokens and POS-tagged projections of the
    # same underlying occurrences. For example, the cached 2019 fiction data
    # has 13,791,481 bare `run` matches in 2010-2019 and 13,790,767 matches
    # across run_NOUN/run_VERB/run_ADJ. Adding both nearly doubles its score.
    # Prefer bare-token counts; tagged rows are only a fallback for a token
    # absent from the bare export.
    bare_counts: dict[str, int] = {}
    tagged_counts: dict[str, int] = {}
    with gzip.open(raw_gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            tab = line.find("\t")
            if tab == -1:
                continue
            token = line[:tab]
            tagged = _has_pos_suffix(token)
            word = _strip_pos_suffix(token).lower()
            if word not in target:
                continue
            fields = line[tab + 1 :].rstrip("\n").split("\t")
            matches = _sum_recent_years(fields)
            if matches:
                destination = tagged_counts if tagged else bare_counts
                destination[word] = destination.get(word, 0) + matches

    frequencies = {}
    for word in bare_counts.keys() | tagged_counts.keys():
        count = bare_counts.get(word, tagged_counts.get(word, 0))
        per_billion = (count / corpus_total) * 1_000_000_000
        frequencies[word] = math.log10(per_billion) if per_billion > 0 else 0.0
    return frequencies
