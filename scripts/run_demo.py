import json

from compliance_llm.config import get_settings
from compliance_llm.pipeline.inference import InferenceEngine


if __name__ == "__main__":
    s = get_settings()
    engine = InferenceEngine(s.artifacts_dir)

    text = "Under GDPR, institutions must report personal data breaches within 72 hours and maintain KYC controls to reduce AML risk."
    out = engine.classify("demo", text)
    ner = engine.ner(text)
    qa = engine.qa("What are reporting obligations?", top_k=2)

    print(json.dumps({"classification": out, "ner": ner, "qa": qa}, indent=2))
