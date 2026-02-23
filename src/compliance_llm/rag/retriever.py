from typing import List, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFRetriever:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)
        self._chunks: List[Dict] = []
        self._x = None

    def fit(self, chunks: List[Dict]) -> None:
        self._chunks = chunks
        self._x = self.vectorizer.fit_transform([c["text"] for c in chunks])

    def search(self, question: str, top_k: int = 3) -> List[Dict]:
        if self._x is None or not self._chunks:
            return []
        qv = self.vectorizer.transform([question])
        sim = cosine_similarity(qv, self._x)[0]
        ranked = sorted(enumerate(sim.tolist()), key=lambda t: t[1], reverse=True)[:top_k]
        out = []
        for idx, score in ranked:
            item = dict(self._chunks[idx])
            item["score"] = round(float(score), 4)
            out.append(item)
        return out


def answer_from_context(question: str, contexts: List[Dict]) -> str:
    if not contexts:
        return "No evidence found."
    best = contexts[0]["text"]
    q_terms = set(question.lower().split())
    sentences = [s.strip() for s in best.split(".") if s.strip()]
    if not sentences:
        return best[:240]
    scored = []
    for s in sentences:
        overlap = len(q_terms.intersection(set(s.lower().split())))
        scored.append((overlap, s))
    scored.sort(reverse=True)
    return scored[0][1]
