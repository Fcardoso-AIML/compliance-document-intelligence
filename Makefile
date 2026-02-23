.PHONY: venv install train test eval demo api

venv:
	python -m venv .venv

install:
	. .venv/Scripts/Activate.ps1 && pip install -r requirements.txt && pip install -e .

train:
	python scripts/train.py

test:
	pytest

eval:
	python scripts/evaluate.py

demo:
	python scripts/run_demo.py

api:
	uvicorn compliance_llm.api.main:app --host 0.0.0.0 --port 8000
