from typing import Dict, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class DenseRetriever:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - dependency/runtime guard
            raise RuntimeError(
                "sentence-transformers is required for dense retrieval. "
                "Install requirements and ensure internet access for model download."
            ) from exc
        self._model = SentenceTransformer(model_name)
        self._chunks: List[Dict] = []
        self._embeddings: np.ndarray | None = None

    def fit(self, chunks: List[Dict]) -> None:
        self._chunks = chunks
        texts = [c["text"] for c in chunks]
        self._embeddings = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def search(self, question: str, top_k: int = 3) -> List[Dict]:
        if self._embeddings is None or not self._chunks:
            return []
        q = self._model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
        sim = cosine_similarity(q, self._embeddings)[0]
        ranked = np.argsort(-sim)[:top_k]
        out: List[Dict] = []
        for idx in ranked.tolist():
            item = dict(self._chunks[idx])
            item["score"] = round(float(sim[idx]), 4)
            out.append(item)
        return out
