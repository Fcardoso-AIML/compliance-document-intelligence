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
def get_engine(mode: str) -> InferenceEngine:
    s = get_settings()
    labels = [x.strip() for x in s.class_labels.split(",") if x.strip()]
    return InferenceEngine(
        artifacts_dir=s.artifacts_dir,
        mode=mode,
        data_dir=s.data_dir,
        labels=labels,
        zero_shot_model=s.zero_shot_model,
        ner_model=s.ner_model,
        embed_model=s.embed_model,
        qa_model=s.qa_model,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/classify", response_model=ClassificationResponse)
def classify(req: DocumentRequest, mode: str = "baseline") -> ClassificationResponse:
    s = get_settings()
    out = get_engine(mode).classify(req.doc_id, req.text)
    log_event(s.logs_dir / "audit.log", "classify", {"doc_id": req.doc_id, "labels": out["labels"], "mode": mode})
    return ClassificationResponse(**out)


@app.post("/ner", response_model=NERResponse)
def ner(req: DocumentRequest, mode: str = "baseline") -> NERResponse:
    s = get_settings()
    entities = [NEREntity(**e) for e in get_engine(mode).ner(req.text)]
    log_event(s.logs_dir / "audit.log", "ner", {"doc_id": req.doc_id, "entities": len(entities), "mode": mode})
    return NERResponse(entities=entities)


@app.post("/qa", response_model=QAResponse)
def qa(req: QARequest, mode: str = "baseline") -> QAResponse:
    s = get_settings()
    out = get_engine(mode).qa(req.question, top_k=req.top_k)
    log_event(s.logs_dir / "audit.log", "qa", {"question": req.question, "top_k": req.top_k, "mode": mode})
    return QAResponse(**out)
