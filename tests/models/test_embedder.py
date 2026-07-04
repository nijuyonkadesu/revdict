from revdict.models.embedder import format_query_text


def test_format_query_text_prepends_the_bge_retrieval_instruction():
    result = format_query_text("feeling of intense annoyance")
    assert result == (
        "Represent this sentence for searching relevant passages: "
        "feeling of intense annoyance"
    )
