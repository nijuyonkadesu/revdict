from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MODEL_REVISION = "c5ee24cb16019beea0893ab7796b1df96625c6b8"


def build_pairs(query: str, definitions: list[str]) -> list[tuple[str, str]]:
    return [(query, definition) for definition in definitions]


class Reranker:
    def __init__(self, model_name: str = MODEL_NAME, revision: str = MODEL_REVISION):
        self._model = CrossEncoder(model_name, revision=revision)

    def score(self, query: str, definitions: list[str]) -> list[float]:
        pairs = build_pairs(query, definitions)
        scores = self._model.predict(pairs)
        return [float(score) for score in scores]
