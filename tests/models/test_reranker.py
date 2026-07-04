from revdict.models.reranker import build_pairs


def test_build_pairs_pairs_the_query_with_each_definition_in_order():
    pairs = build_pairs("joy", ["feeling happy", "a legal document"])
    assert pairs == [("joy", "feeling happy"), ("joy", "a legal document")]
