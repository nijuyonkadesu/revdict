import torch
from sentence_transformers import CrossEncoder
from sentence_transformers.util import batch_to_device

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MODEL_REVISION = "c5ee24cb16019beea0893ab7796b1df96625c6b8"
RERANK_BATCH_SIZE = 32


def build_pairs(query: str, definitions: list[str]) -> list[tuple[str, str]]:
    return [(query, definition) for definition in definitions]


class Reranker:
    def __init__(self, model_name: str = MODEL_NAME, revision: str = MODEL_REVISION):
        self._model = CrossEncoder(model_name, revision=revision)

    def score(self, query: str, definitions: list[str]) -> list[float]:
        pairs = build_pairs(query, definitions)
        if not pairs:
            return []

        # CrossEncoder.predict sorts text pairs by character count and then
        # tokenizes each batch separately. Character count is a poor proxy for
        # WordPiece length in dictionary definitions, leaving enough padding
        # in a 600-candidate rerank to dominate the search latency. Tokenize
        # the unchanged pairs once, group them by their exact token lengths,
        # and feed those same token IDs through the same float32 model.
        transformer = self._model[0]
        processor = transformer.processor
        encoded = processor(
            [pair[0] for pair in pairs],
            [pair[1] for pair in pairs],
            padding=False,
            truncation="longest_first",
        )
        sorted_indices = sorted(
            range(len(pairs)),
            key=lambda index: len(encoded["input_ids"][index]),
            reverse=True,
        )

        device = str(self._model.device)
        self._model.to(device)
        self._model.eval()
        activation = self._model.activation_fn
        scores = [0.0] * len(pairs)

        with torch.inference_mode():
            for start in range(0, len(sorted_indices), RERANK_BATCH_SIZE):
                batch_indices = sorted_indices[start : start + RERANK_BATCH_SIZE]
                samples = [
                    {
                        name: values[index]
                        for name, values in encoded.items()
                    }
                    for index in batch_indices
                ]
                features = processor.pad(samples, padding=True, return_tensors="pt")
                features["modality"] = "text"
                features = batch_to_device(features, device)
                batch_scores = self._model.forward(features)["scores"]
                if activation is not None:
                    batch_scores = activation(batch_scores)
                if self._model.num_labels == 1 and batch_scores.ndim > 1:
                    batch_scores = batch_scores.squeeze(-1)

                values = batch_scores.detach().cpu().float().tolist()
                for index, value in zip(batch_indices, values):
                    scores[index] = float(value)

        return scores
