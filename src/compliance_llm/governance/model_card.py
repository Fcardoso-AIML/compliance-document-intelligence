from pathlib import Path


def write_model_card(path: Path) -> None:
    text = """# Model Card - Compliance LLM (v0.1.0)

## Intended Use
Assist compliance analysts with triage: document classification, risk scoring, and evidence-backed QA.

## Limitations
- Uses a light TF-IDF retriever baseline unless upgraded to embedding model.
- NER is rule-based in this baseline.
- Not legal advice; human review required for decisions.

## Monitoring
- Track label drift, risk score distribution, and unanswered-question rate.

## Governance
- All inference actions should emit structured audit events.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
