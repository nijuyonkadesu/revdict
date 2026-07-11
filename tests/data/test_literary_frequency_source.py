import gzip
import tempfile
from pathlib import Path

from revdict.data.literary_frequency_source import (
    _corpus_total_for_recent_years,
    _strip_pos_suffix,
    _sum_recent_years,
    compute_literary_frequencies,
)


def test_strip_pos_suffix_removes_a_known_pos_tag():
    assert _strip_pos_suffix("shleep_VERB") == "shleep"
    assert _strip_pos_suffix("firie_NOUN") == "firie"


def test_strip_pos_suffix_leaves_a_bare_word_unchanged():
    assert _strip_pos_suffix("Hesperides") == "Hesperides"


def test_strip_pos_suffix_leaves_a_non_pos_underscore_unchanged():
    # A word containing an underscore that isn't a recognized POS tag
    # shouldn't be mistaken for a POS-suffixed entry.
    assert _strip_pos_suffix("some_thing") == "some_thing"


def test_sum_recent_years_only_counts_the_configured_year_window():
    fields = ["2009,100,5", "2010,20,2", "2015,30,3", "2019,50,4", "2020,999,9"]
    assert _sum_recent_years(fields) == 20 + 30 + 50


def test_corpus_total_for_recent_years_sums_tab_separated_totalcounts_format():
    text = "\t2009,1000,10,1\t2010,200,5,1\t2019,300,6,1\t2020,9999,50,3\t"
    assert _corpus_total_for_recent_years(text) == 200 + 300


def test_compute_literary_frequencies_only_returns_scores_for_requested_headwords():
    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / "fiction.jsonl.gz"
        totalcounts_path = Path(tmp) / "totalcounts-1"
        with gzip.open(raw_path, "wt", encoding="utf-8") as f:
            f.write("murmur\t2010,1000,10\t2019,2000,15\t2020,50,1\n")
            f.write("irrelevant\t2010,999999,999\n")
        totalcounts_path.write_text("\t2010,1000000000,1,1\t2019,1000000000,1,1\t2020,1,1,1\t")

        result = compute_literary_frequencies({"murmur"}, str(raw_path), str(totalcounts_path))

        assert set(result.keys()) == {"murmur"}


def test_compute_literary_frequencies_merges_pos_suffixed_variants_of_the_same_word():
    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / "fiction.jsonl.gz"
        totalcounts_path = Path(tmp) / "totalcounts-1"
        with gzip.open(raw_path, "wt", encoding="utf-8") as f:
            f.write("run_NOUN\t2010,100,5\n")
            f.write("run_VERB\t2010,300,10\n")
            f.write("run\t2010,50,3\n")
        totalcounts_path.write_text("\t2010,1000000000,1,1\t")

        result = compute_literary_frequencies({"run"}, str(raw_path), str(totalcounts_path))

        # (100 + 300 + 50) / 1_000_000_000 * 1_000_000_000 = 450 matches per
        # billion words -> log10(450)
        import math

        assert result["run"] == math.log10(450)


def test_compute_literary_frequencies_is_case_insensitive_and_lowercases_output_keys():
    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / "fiction.jsonl.gz"
        totalcounts_path = Path(tmp) / "totalcounts-1"
        with gzip.open(raw_path, "wt", encoding="utf-8") as f:
            f.write("Hesperides\t2010,10,2\n")
        totalcounts_path.write_text("\t2010,1000000000,1,1\t")

        result = compute_literary_frequencies(
            {"hesperides"}, str(raw_path), str(totalcounts_path)
        )

        assert "hesperides" in result


def test_compute_literary_frequencies_returns_nothing_for_a_word_with_zero_recent_matches():
    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / "fiction.jsonl.gz"
        totalcounts_path = Path(tmp) / "totalcounts-1"
        with gzip.open(raw_path, "wt", encoding="utf-8") as f:
            # All matches fall outside the 2010-2019 window.
            f.write("archaic\t1800,500,10\t2025,300,5\n")
        totalcounts_path.write_text("\t2010,1000000000,1,1\t")

        result = compute_literary_frequencies({"archaic"}, str(raw_path), str(totalcounts_path))

        assert "archaic" not in result
