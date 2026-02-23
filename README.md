# AI-Powered Compliance Document Intelligence

Production-oriented regulated NLP system with two modes:
- `baseline`: TF-IDF + logistic regression + rule NER + sparse retrieval
- `advanced`: zero-shot transformer classification + transformer NER + dense retrieval + grounded QA

## Why this project
- Domain fit: AML, KYC, GDPR, ESG, EU AI Act
- Architecture fit: classification + RAG QA + auditability
- Hiring signal: regulated AI engineering with testability and deployment

## Architecture
- `Classifier`
  - Baseline: TF-IDF + One-vs-Rest Logistic Regression
  - Advanced: `facebook/bart-large-mnli` zero-shot multi-label classification
- `NER`
  - Baseline: compliance-focused rules
  - Advanced: `dslim/bert-base-NER`
- `RAG`
  - Baseline: sparse TF-IDF retrieval + extractive heuristic answer
  - Advanced: sentence-transformer dense retrieval + grounded QA model (`deepset/roberta-base-squad2`)
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
python scripts/run_sota_benchmark.py
uvicorn compliance_llm.api.main:app --reload
```

Advanced evaluation:
```bash
python scripts/evaluate.py --mode advanced
```

## API
- `GET /health`
- `POST /classify?mode=baseline|advanced` with `{doc_id, text}`
- `POST /ner?mode=baseline|advanced` with `{doc_id, text}`
- `POST /qa?mode=baseline|advanced` with `{question, top_k}`

## Current SOTA-like Stack
- Zero-shot transformer classification
- Transformer NER
- Dense retrieval with sentence embeddings
- Grounded extractive QA on retrieved evidence

## Next Upgrade Path
- Fine-tuned DeBERTa-v3 multi-label classifier on larger curated corpora
- Hybrid retrieval (BM25 + dense + reranker)
- Generative answer model with strict citation enforcement
- MLflow + DVC + drift monitoring in CI/CD

## Docker
```bash
docker build -t compliance-llm .
docker run -p 8000:8000 compliance-llm
```
