import json
from pathlib import Path

from compliance_llm.config import get_settings
from compliance_llm.pipeline.evaluation import evaluate
from compliance_llm.pipeline.inference import InferenceEngine


def main() -> None:
    s = get_settings()
    baseline_metrics = evaluate(s.data_dir, s.artifacts_dir, mode="baseline")

    advanced_metrics = None
    advanced_error = None
    try:
        advanced_metrics = evaluate(s.data_dir, s.artifacts_dir, mode="advanced")
    except Exception as exc:
        advanced_error = str(exc)

    sample_text = (
        "Financial institutions shall implement KYC controls, monitor suspicious "
        "transactions, and report personal-data breaches under GDPR timelines."
    )
    sample_question = "What are the key reporting obligations?"

    baseline = InferenceEngine(s.artifacts_dir, mode="baseline", data_dir=s.data_dir)
    baseline_out = {
        "classification": baseline.classify("bench-doc", sample_text),
        "ner": baseline.ner(sample_text),
        "qa": baseline.qa(sample_question, top_k=3),
    }

    advanced_out = None
    if advanced_error is None:
        adv = InferenceEngine(s.artifacts_dir, mode="advanced", data_dir=s.data_dir)
        advanced_out = {
            "classification": adv.classify("bench-doc", sample_text),
            "ner": adv.ner(sample_text),
            "qa": adv.qa(sample_question, top_k=3),
        }

    report = {
        "baseline_metrics": baseline_metrics,
        "advanced_metrics": advanced_metrics,
        "advanced_error": advanced_error,
        "baseline_output": baseline_out,
        "advanced_output": advanced_out,
    }

    out_path = Path("reports/sota_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved comparison report to {out_path}")


if __name__ == "__main__":
    main()
