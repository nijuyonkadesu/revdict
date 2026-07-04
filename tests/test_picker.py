# tests/test_picker.py
from revdict.picker import (
    _render_exact_preview,
    format_candidate_line,
    parse_selection,
)

_EXACT_MATCH_FIXTURE = {
    "headword": "happy",
    "senses": [
        {
            "pos": "adjective",
            "definition": "feeling great pleasure",
            "examples": ["a happy child"],
            "source": "wordnet",
            "synonyms": ["glad", "content"],
            "label": "joy",
            "polarity": "positive",
        },
        {
            "pos": "adjective",
            "definition": "willing to do something",
            "examples": [],
            "source": "wiktionary",
            "synonyms": None,
            "label": "neutral",
            "polarity": "neutral",
        },
    ],
}


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


def test_render_exact_preview_shows_real_per_sense_emotion_badge_and_synonyms():
    """Fix 1 + Fix 2: the exact-match preview pane must show each sense's real
    emotion tag (previously nothing was shown at all for the exact match) and
    synonyms when present, without a dangling "Synonyms:" label when absent."""
    preview = _render_exact_preview(_EXACT_MATCH_FIXTURE)

    assert "joy · positive" in preview
    assert "neutral · neutral" in preview
    assert "glad, content" in preview
    # The second sense has no synonyms -- must not print an empty label.
    assert "Synonyms: \n" not in preview
    assert "Synonyms:\n" not in preview
    assert preview.count("Synonyms:") == 1
