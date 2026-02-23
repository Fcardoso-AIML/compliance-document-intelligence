"""Notebook part 03: QA and evaluation."""
from compliance_llm.config import get_settings
from compliance_llm.pipeline.evaluation import evaluate
from compliance_llm.pipeline.inference import InferenceEngine

s = get_settings()
engine = InferenceEngine(s.artifacts_dir)
print(engine.qa("What are AML obligations?", top_k=3))
print(evaluate(s.data_dir, s.artifacts_dir))
