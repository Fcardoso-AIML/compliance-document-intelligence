from pathlib import Path
from typing import Dict

from compliance_llm.data.ingestion import load_corpus
from compliance_llm.data.chunking import to_chunk_records
from compliance_llm.models.classifier import ComplianceClassifier
from compliance_llm.rag.retriever import TFIDFRetriever


def train_all(data_dir: Path, artifacts_dir: Path) -> Dict[str, str]:
    corpus = load_corpus(data_dir)
    texts = [d["text"] for d in corpus]
    labels = [d.get("labels", []) for d in corpus]

    clf = ComplianceClassifier()
    clf.fit(texts, labels)

    clf_path = artifacts_dir / "classifier.joblib"
    clf.save(str(clf_path))

    chunks = []
    for doc in corpus:
        chunks.extend(to_chunk_records(doc))

    retr = TFIDFRetriever()
    retr.fit(chunks)

    import joblib
    ret_path = artifacts_dir / "retriever.joblib"
    joblib.dump(retr, ret_path)

    return {"classifier": str(clf_path), "retriever": str(ret_path)}
