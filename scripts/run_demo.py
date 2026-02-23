import json

from compliance_llm.config import get_settings
from compliance_llm.pipeline.inference import InferenceEngine


if __name__ == "__main__":
    s = get_settings()
    baseline = InferenceEngine(s.artifacts_dir, mode="baseline", data_dir=s.data_dir)

    text = "Under GDPR, institutions must report personal data breaches within 72 hours and maintain KYC controls to reduce AML risk."
    b_out = baseline.classify("demo", text)
    b_ner = baseline.ner(text)
    b_qa = baseline.qa("What are reporting obligations?", top_k=2)

    adv_payload = {}
    try:
        advanced = InferenceEngine(s.artifacts_dir, mode="advanced", data_dir=s.data_dir)
        adv_payload = {
            "classification": advanced.classify("demo", text),
            "ner": advanced.ner(text),
            "qa": advanced.qa("What are reporting obligations?", top_k=2),
        }
    except Exception as exc:
        adv_payload = {"error": str(exc)}

    print(
        json.dumps(
            {
                "baseline": {"classification": b_out, "ner": b_ner, "qa": b_qa},
                "advanced": adv_payload,
            },
            indent=2,
        )
    )
