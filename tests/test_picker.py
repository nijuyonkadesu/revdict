# tests/test_picker.py
from revdict.picker import format_candidate_line, parse_selection


def test_format_candidate_line_has_five_tab_fields_and_marks_exact_match():
    line = format_candidate_line(
        "happy", "adjective", "feeling pleasure", "Joy", "positive", 92, index=3, is_exact=True
    )
    fields = line.split("\t")
    assert len(fields) == 5
    assert fields[-1] == "3"
    assert fields[0].startswith("★")


def test_format_candidate_line_truncates_long_definitions():
    long_definition = "x" * 200
    line = format_candidate_line(
        "word", "noun", long_definition, "neutral", "neutral", 50, index=0
    )
    gloss_field = line.split("\t")[1]
    assert len(gloss_field) < 100


def test_parse_selection_extracts_trailing_index_or_none_for_empty_input():
    line = format_candidate_line("joyful", "adjective", "x", "Joy", "positive", 80, index=5)
    assert parse_selection(line + "\n") == 5
    assert parse_selection("") is None
    assert parse_selection("   ") is None
