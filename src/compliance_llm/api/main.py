from functools import lru_cache

from fastapi import FastAPI

from compliance_llm.config import get_settings
from compliance_llm.governance.audit import log_event
from compliance_llm.pipeline.inference import InferenceEngine
from compliance_llm.schemas import (
    ClassificationResponse,
    DocumentRequest,
    NEREntity,
    NERResponse,
    QARequest,
    QAResponse,
)

app = FastAPI(title="Compliance Document Intelligence API", version="0.1.0")


@lru_cache
def get_engine() -> InferenceEngine:
    s = get_settings()
    return InferenceEngine(s.artifacts_dir)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/classify", response_model=ClassificationResponse)
def classify(req: DocumentRequest) -> ClassificationResponse:
    s = get_settings()
    out = get_engine().classify(req.doc_id, req.text)
    log_event(s.logs_dir / "audit.log", "classify", {"doc_id": req.doc_id, "labels": out["labels"]})
    return ClassificationResponse(**out)


@app.post("/ner", response_model=NERResponse)
def ner(req: DocumentRequest) -> NERResponse:
    s = get_settings()
    entities = [NEREntity(**e) for e in get_engine().ner(req.text)]
    log_event(s.logs_dir / "audit.log", "ner", {"doc_id": req.doc_id, "entities": len(entities)})
    return NERResponse(entities=entities)


@app.post("/qa", response_model=QAResponse)
def qa(req: QARequest) -> QAResponse:
    s = get_settings()
    out = get_engine().qa(req.question, top_k=req.top_k)
    log_event(s.logs_dir / "audit.log", "qa", {"question": req.question, "top_k": req.top_k})
    return QAResponse(**out)
