# Personal Build Notes (Decision Log)

## What we built
End-to-end compliance NLP platform with four capabilities:
1. Multi-label classification of regulatory documents
2. Compliance NER extraction
3. Retrieval-augmented QA
4. Risk scoring + governance/audit logging

## Why these choices
- We used a strong baseline first (TF-IDF + Logistic Regression) to guarantee reproducible training with small data and no GPU dependency.
- We added NER and QA in parallel to compare extraction-centric vs question-centric workflows, matching your target narrative.
- We included governance components (audit logs + model card) because regulated AI systems need traceability beyond raw model metrics.
- We split notebook logic into maintainable script parts so experimentation can graduate into production code without rewrite.

## Full sequence executed
1. Scaffolded package layout (`src`, `scripts`, `tests`, `notebooks`, `data`, `reports`).
2. Implemented ingestion/chunking and sample compliance dataset.
3. Built classifier, risk scoring, NER, retriever, QA synthesis.
4. Added FastAPI serving and governance logging.
5. Added tests and evaluation script.
6. Created notebook + split notebook modules.
7. Added Docker and compose config.
8. Trained/evaluated/demoed and documented findings.
9. Prepared GitHub and Hugging Face push commands.

## Findings interpretation (baseline run)
- Baseline reaches useful signal on seeded corpus but is not yet benchmark-grade due tiny dataset.
- Risk score can prioritize analyst review queues effectively when paired with confidence thresholds.
- NER gives explicit obligation/risk tokens; QA gives analyst-friendly natural-language answers.
- Combined NER+QA improves trust: entities justify what answer was based on.

## How to evolve to SOTA
1. Replace classifier with DeBERTa-v3 multi-label fine-tuning on larger labeled corpora.
2. Move RAG to dense retrieval (e5/bge embeddings + FAISS/hybrid BM25).
3. Use LLM answer generation with strict citation grounding and answerability checks.
4. Add active learning and human-in-the-loop labeling for hard classes.
5. Add MLflow, DVC, and CI/CD model promotion gates.

## Implemented advanced mode (2026-02-23)
1. Added zero-shot classifier (`facebook/bart-large-mnli`) for fast multi-label coverage.
2. Added transformer NER (`dslim/bert-base-NER`) for richer entities.
3. Added dense retrieval (`sentence-transformers/all-MiniLM-L6-v2`) for semantic search.
4. Added grounded QA (`deepset/roberta-base-squad2`) over retrieved evidence.
5. Added side-by-side benchmark script: `scripts/run_sota_benchmark.py`.

## Files to run for comparison
1. `python scripts/train.py`
2. `python scripts/evaluate.py --mode baseline`
3. `python scripts/evaluate.py --mode advanced`
4. `python scripts/run_sota_benchmark.py`
