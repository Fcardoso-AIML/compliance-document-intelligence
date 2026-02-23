from typing import Dict, List


class ZeroShotComplianceClassifier:
    def __init__(self, model_name: str = "facebook/bart-large-mnli") -> None:
        try:
            from transformers import pipeline
        except Exception as exc:  # pragma: no cover - dependency/runtime guard
            raise RuntimeError(
                "transformers is required for advanced classification. "
                "Install requirements and ensure internet access for model download."
            ) from exc
        self._pipeline = pipeline("zero-shot-classification", model=model_name)

    def predict(
        self,
        text: str,
        candidate_labels: List[str],
        threshold: float = 0.35,
    ) -> Dict[str, float]:
        out = self._pipeline(text, candidate_labels, multi_label=True)
        probs = {
            label.upper(): float(score)
            for label, score in zip(out["labels"], out["scores"])
            if float(score) >= threshold
        }
        return probs
