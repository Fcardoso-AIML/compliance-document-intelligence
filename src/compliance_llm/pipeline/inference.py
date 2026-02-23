from pathlib import Path
from typing import Dict, List

import joblib

from compliance_llm.data.chunking import to_chunk_records
from compliance_llm.data.ingestion import load_corpus
from compliance_llm.models.advanced_classifier import ZeroShotComplianceClassifier
from compliance_llm.models.advanced_ner import TransformerNER
from compliance_llm.models.classifier import ComplianceClassifier
from compliance_llm.models.ner import extract_entities
from compliance_llm.models.risk import risk_score
from compliance_llm.rag.dense_retriever import DenseRetriever
from compliance_llm.rag.qa_llm import GroundedQAModel
from compliance_llm.rag.retriever import answer_from_context


class InferenceEngine:
    def __init__(
        self,
        artifacts_dir: Path,
        mode: str = "baseline",
        data_dir: Path | None = None,
        labels: List[str] | None = None,
        zero_shot_model: str = "facebook/bart-large-mnli",
        ner_model: str = "dslim/bert-base-NER",
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        qa_model: str = "deepset/roberta-base-squad2",
    ) -> None:
        self.mode = mode
        self.labels = labels or ["AML", "PRIVACY", "ESG", "REPORTING", "AI_ACT", "RISK"]
        if mode == "advanced":
            if data_dir is None:
                raise ValueError("data_dir is required in advanced mode.")
            self.classifier = ZeroShotComplianceClassifier(model_name=zero_shot_model)
            self.ner_model = TransformerNER(model_name=ner_model)
            self.retriever = DenseRetriever(model_name=embed_model)
            self.qa_model = GroundedQAModel(model_name=qa_model)
            corpus = load_corpus(data_dir)
            chunks = []
            for doc in corpus:
                chunks.extend(to_chunk_records(doc))
            self.retriever.fit(chunks)
        else:
            self.classifier = ComplianceClassifier.load(str(artifacts_dir / "classifier.joblib"))
            self.retriever = joblib.load(artifacts_dir / "retriever.joblib")

    def classify(self, doc_id: str, text: str) -> Dict:
        if self.mode == "advanced":
            probs = self.classifier.predict(text, candidate_labels=self.labels)
        else:
            probs = self.classifier.predict(text)
        return {
            "doc_id": doc_id,
            "labels": sorted(probs.keys()),
            "probabilities": probs,
            "risk_score": risk_score(probs),
        }

    def ner(self, text: str):
        if self.mode == "advanced":
            return self.ner_model.extract(text)
        return extract_entities(text)

    def qa(self, question: str, top_k: int = 3) -> Dict:
        evidence = self.retriever.search(question, top_k=top_k)
        if self.mode == "advanced":
            answer = self.qa_model.answer(question, evidence)
        else:
            answer = answer_from_context(question, evidence)
        return {"answer": answer, "evidence": evidence}
