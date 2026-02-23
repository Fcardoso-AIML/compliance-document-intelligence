import json

from compliance_llm.config import get_settings
from compliance_llm.pipeline.evaluation import evaluate


if __name__ == "__main__":
    s = get_settings()
    metrics = evaluate(s.data_dir, s.artifacts_dir)
    print(json.dumps(metrics, indent=2))
