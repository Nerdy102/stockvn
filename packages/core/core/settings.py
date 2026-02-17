from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings loaded from environment variables (and optional .env)."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    APP_ENV: str = Field(default="local")
    DEV_MODE: bool = Field(default=True)
    LOG_LEVEL: str = Field(default="INFO")

    DATA_PROVIDER: str = Field(default="csv")
    DEMO_DATA_DIR: str = Field(default="data_demo")

    DATABASE_URL: str = Field(default="sqlite:///./vn_invest.db")

    MARKET_RULES_PATH: str = Field(default="configs/market_rules_vn.yaml")
    FEES_TAXES_PATH: str = Field(default="configs/fees_taxes.yaml")
    EXECUTION_MODEL_PATH: str = Field(default="configs/execution_model.yaml")
    BROKER_NAME: str = Field(default="demo_broker")

    API_BASE_URL: str = Field(default="http://localhost:8000")

    SSI_CONSUMER_ID: str = Field(default="")
    SSI_CONSUMER_SECRET: str = Field(default="")
    SSI_PRIVATE_KEY_PATH: str = Field(default="")

    INGEST_BATCH_SIZE: int = Field(default=500)
    INGEST_ERROR_KILL_SWITCH_THRESHOLD: int = Field(default=100)
    INGEST_SCHEMA_DRIFT_KILL_SWITCH_THRESHOLD: int = Field(default=200)

    API_DEFAULT_LIMIT: int = Field(default=200)
    API_MAX_LIMIT: int = Field(default=2000)
    API_DEFAULT_DAYS: int = Field(default=365)


def get_settings() -> Settings:
    return Settings()
