import gzip
import math
import urllib.request
from pathlib import Path

RAW_NGRAM_FICTION_URL = (
    "http://storage.googleapis.com/books/ngrams/books/20200217/eng-fiction/1-00000-of-00001.gz"
)
RAW_NGRAM_FICTION_TOTALCOUNTS_URL = (
    "http://storage.googleapis.com/books/ngrams/books/20200217/eng-fiction/totalcounts-1"
)

# Restricting to 2010-2019 favors contemporary usage over the 19th-century
# diction that dominates this corpus's older years (it stretches back to the
# 1500s) -- this is what makes the signal "words used in modern fiction"
# rather than "words used in fiction, including Shakespeare-era archaisms".
YEAR_RANGE_START = 2010
YEAR_RANGE_END = 2019

_POS_TAGS = {"NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "NUM", "CONJ", "PRT", "X"}


def download_raw_ngram_fiction(dest_path: str) -> None:
    dest = Path(dest_path)
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(RAW_NGRAM_FICTION_URL, tmp)
    tmp.rename(dest)


def download_raw_ngram_fiction_totalcounts(dest_path: str) -> None:
    dest = Path(dest_path)
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(RAW_NGRAM_FICTION_TOTALCOUNTS_URL, tmp)
    tmp.rename(dest)


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


def _strip_pos_suffix(token: str) -> str:
    underscore = token.rfind("_")
    if underscore == -1:
        return token
    if token[underscore + 1 :] in _POS_TAGS:
        return token[:underscore]
    return token


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

    target = {word.lower() for word in headwords}
    counts: dict[str, int] = {}
    with gzip.open(raw_gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            tab = line.find("\t")
            if tab == -1:
                continue
            token = line[:tab]
            word = _strip_pos_suffix(token).lower()
            if word not in target:
                continue
            fields = line[tab + 1 :].rstrip("\n").split("\t")
            matches = _sum_recent_years(fields)
            if matches:
                counts[word] = counts.get(word, 0) + matches

    frequencies = {}
    for word, count in counts.items():
        per_billion = (count / corpus_total) * 1_000_000_000
        frequencies[word] = math.log10(per_billion) if per_billion > 0 else 0.0
    return frequencies
