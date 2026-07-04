import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def format_query_text(query: str) -> str:
    return QUERY_INSTRUCTION + query


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME):
        self._model = SentenceTransformer(model_name)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return vectors.astype("float32")

    def encode_query(self, query: str) -> np.ndarray:
        vector = self._model.encode(
            [format_query_text(query)],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        return vector.astype("float32")
