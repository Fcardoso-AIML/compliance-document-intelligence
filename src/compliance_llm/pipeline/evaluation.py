from pathlib import Path
from typing import Dict

from compliance_llm.data.ingestion import load_corpus
from compliance_llm.pipeline.inference import InferenceEngine


def evaluate(data_dir: Path, artifacts_dir: Path) -> Dict:
    data = load_corpus(data_dir)
    engine = InferenceEngine(artifacts_dir)

    exact_hits = 0
    for row in data:
        pred = set(engine.classify(row["doc_id"], row["text"])["labels"])
        gold = set(row.get("labels", []))
        if pred == gold:
            exact_hits += 1

    acc = exact_hits / max(1, len(data))
    return {"docs": len(data), "exact_match_accuracy": round(acc, 4)}
