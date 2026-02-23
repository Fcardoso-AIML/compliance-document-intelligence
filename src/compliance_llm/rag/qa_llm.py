from typing import Dict, List


class GroundedQAModel:
    def __init__(self, model_name: str = "deepset/roberta-base-squad2") -> None:
        try:
            from transformers import pipeline
        except Exception as exc:  # pragma: no cover - dependency/runtime guard
            raise RuntimeError(
                "transformers is required for grounded QA. "
                "Install requirements and ensure internet access for model download."
            ) from exc
        self._qa = pipeline("question-answering", model=model_name)

    def answer(self, question: str, contexts: List[Dict]) -> str:
        if not contexts:
            return "No evidence found."
        best_answer = ""
        best_score = -1.0
        for ctx in contexts:
            pred = self._qa(question=question, context=ctx["text"])
            score = float(pred.get("score", 0.0))
            if score > best_score:
                best_score = score
                best_answer = str(pred.get("answer", "")).strip()
        return best_answer or "No answer found in retrieved evidence."
