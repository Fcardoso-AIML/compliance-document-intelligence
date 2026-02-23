from typing import Dict, List


class TransformerNER:
    def __init__(self, model_name: str = "dslim/bert-base-NER") -> None:
        try:
            from transformers import pipeline
        except Exception as exc:  # pragma: no cover - dependency/runtime guard
            raise RuntimeError(
                "transformers is required for advanced NER. "
                "Install requirements and ensure internet access for model download."
            ) from exc
        self._pipeline = pipeline(
            "token-classification",
            model=model_name,
            aggregation_strategy="simple",
        )

    def extract(self, text: str) -> List[Dict[str, str]]:
        ents = self._pipeline(text)
        out = []
        for ent in ents:
            out.append({"text": ent["word"], "label": ent["entity_group"]})
        return out
