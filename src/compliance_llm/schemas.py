from typing import Dict, List
from pydantic import BaseModel, Field


class DocumentRequest(BaseModel):
    doc_id: str = Field(..., description="Document identifier")
    text: str = Field(..., min_length=20, description="Document content")


class ClassificationResponse(BaseModel):
    doc_id: str
    labels: List[str]
    probabilities: Dict[str, float]
    risk_score: float


class QARequest(BaseModel):
    question: str
    top_k: int = 3


class QAResponse(BaseModel):
    answer: str
    evidence: List[Dict[str, str]]


class NEREntity(BaseModel):
    text: str
    label: str


class NERResponse(BaseModel):
    entities: List[NEREntity]
