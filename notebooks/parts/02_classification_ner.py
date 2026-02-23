"""Notebook part 02: classification and NER checks."""
from compliance_llm.config import get_settings
from compliance_llm.pipeline.inference import InferenceEngine

s = get_settings()
engine = InferenceEngine(s.artifacts_dir)
text = "Under GDPR institutions must report data breaches and maintain KYC controls."
print(engine.classify("nb-doc", text))
print(engine.ner(text))
