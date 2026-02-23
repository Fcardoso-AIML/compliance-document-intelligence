from compliance_llm.config import get_settings
from compliance_llm.pipeline.training import train_all
from compliance_llm.governance.model_card import write_model_card


if __name__ == "__main__":
    s = get_settings()
    paths = train_all(s.data_dir, s.artifacts_dir)
    write_model_card(s.reports_dir / "MODEL_CARD.md")
    print(paths)
