# AI-Powered Compliance Document Intelligence

Production-oriented baseline for regulated NLP systems: multi-label document classification, rule-assisted NER, risk scoring, and retrieval-augmented QA with governance logging.

## Why this project
- Domain fit: AML, KYC, GDPR, ESG, EU AI Act
- Architecture fit: classification + RAG QA + auditability
- Hiring signal: regulated AI engineering with testability and deployment

## Architecture
- `Classifier`: TF-IDF + One-vs-Rest Logistic Regression (deterministic, fast baseline)
- `NER`: compliance-focused rule patterns (upgrade path to transformer NER)
- `RAG`: TF-IDF retriever over chunked documents + extractive answer synthesis
- `Risk`: weighted risk scoring by predicted labels
- `Governance`: structured audit logs + model card generation

## Project structure
- `src/compliance_llm/`: core package
- `scripts/`: train/evaluate/demo entrypoints
- `tests/`: unit and API smoke tests
- `notebooks/testing.ipynb`: end-to-end validation notebook
- `notebooks/parts/`: notebook split into reusable python modules

## Quickstart
```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
pip install -e .
python scripts/train.py
pytest
python scripts/evaluate.py
python scripts/run_demo.py
uvicorn compliance_llm.api.main:app --reload
```

## API
- `GET /health`
- `POST /classify` with `{doc_id, text}`
- `POST /ner` with `{doc_id, text}`
- `POST /qa` with `{question, top_k}`

## SOTA Upgrade Path
- Swap TF-IDF classifier for `microsoft/deberta-v3-base` multi-label fine-tuning
- Replace retriever with sentence-transformers embeddings + FAISS
- Plug LLM QA (`Llama 3.1`, `Mistral`, or API model) with citation prompting
- Add drift monitoring, calibration, and weak-supervision label QA
- Integrate MLflow/DVC for experiment and dataset lineage

## Docker
```bash
docker build -t compliance-llm .
docker run -p 8000:8000 compliance-llm
```
