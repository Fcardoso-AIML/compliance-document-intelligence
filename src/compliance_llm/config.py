from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    project_name: str = "Compliance LLM"
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    reports_dir: Path = Path("reports")
    logs_dir: Path = Path("logs")
    random_state: int = 42

    model_config = SettingsConfigDict(env_prefix="COMPLIANCE_", env_file=".env")


def get_settings() -> Settings:
    s = Settings()
    s.artifacts_dir.mkdir(parents=True, exist_ok=True)
    s.reports_dir.mkdir(parents=True, exist_ok=True)
    s.logs_dir.mkdir(parents=True, exist_ok=True)
    return s
