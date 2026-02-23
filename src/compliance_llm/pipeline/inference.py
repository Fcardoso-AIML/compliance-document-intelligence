from pathlib import Path
from typing import Dict

import joblib

from compliance_llm.models.classifier import ComplianceClassifier
from compliance_llm.models.ner import extract_entities
from compliance_llm.models.risk import risk_score
from compliance_llm.rag.retriever import answer_from_context


class InferenceEngine:
    def __init__(self, artifacts_dir: Path) -> None:
        self.classifier = ComplianceClassifier.load(str(artifacts_dir / "classifier.joblib"))
        self.retriever = joblib.load(artifacts_dir / "retriever.joblib")

    def classify(self, doc_id: str, text: str) -> Dict:
        probs = self.classifier.predict(text)
        return {
            "doc_id": doc_id,
            "labels": sorted(probs.keys()),
            "probabilities": probs,
            "risk_score": risk_score(probs),
        }

    def ner(self, text: str):
        return extract_entities(text)

    def qa(self, question: str, top_k: int = 3) -> Dict:
        evidence = self.retriever.search(question, top_k=top_k)
        answer = answer_from_context(question, evidence)
        return {"answer": answer, "evidence": evidence}
