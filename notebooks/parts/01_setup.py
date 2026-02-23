"""Notebook part 01: setup and training."""
from compliance_llm.config import get_settings
from compliance_llm.pipeline.training import train_all

s = get_settings()
paths = train_all(s.data_dir, s.artifacts_dir)
print(paths)
