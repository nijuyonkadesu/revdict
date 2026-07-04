# tests/data/test_build_index.py
from revdict.data.build_index import estimate_full_duration, group_by_definition


def test_estimate_full_duration_extrapolates_linearly_from_a_sample():
    assert estimate_full_duration(100, 10.0, 1000) == 100.0


def test_estimate_full_duration_handles_an_empty_sample():
    assert estimate_full_duration(0, 0.0, 1000) == 0.0


def test_group_by_definition_groups_identical_texts_and_preserves_first_seen_order():
    records = [{"definition": "a"}, {"definition": "b"}, {"definition": "a"}]

    unique_texts, index_groups = group_by_definition(records)

    assert unique_texts == ["a", "b"]
    assert index_groups == [[0, 2], [1]]
