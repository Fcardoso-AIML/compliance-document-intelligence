import json
import argparse

from compliance_llm.config import get_settings
from compliance_llm.pipeline.evaluation import evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "advanced"], default="baseline")
    args = parser.parse_args()
    s = get_settings()
    metrics = evaluate(s.data_dir, s.artifacts_dir, mode=args.mode)
    print(json.dumps(metrics, indent=2))
