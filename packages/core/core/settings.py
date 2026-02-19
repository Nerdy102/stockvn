from __future__ import annotations

import logging

from pydantic import Field
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Settings loaded from environment variables (and optional .env)."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    APP_ENV: str = Field(default="local")
    DEV_MODE: bool = Field(default=True)
    LOG_LEVEL: str = Field(default="INFO")

    DATA_PROVIDER: str = Field(default="csv")
    DEMO_DATA_DIR: str = Field(default="data_demo")
    CRYPTO_DEFAULT_EXCHANGE: str = Field(default="binance_public")

    DATABASE_URL: str = Field(default="sqlite:///./vn_invest.db")
    REDIS_URL: str = Field(default="redis://localhost:6379/0")
    REALTIME_ENABLED: bool = Field(default=False)

    SSI_FCDATA_BASE_URL: str = Field(default="https://fc-data.ssi.com.vn/api/v2")
    SSI_STREAM_URL: str = Field(default="wss://iboard-stream.ssi.com.vn")

    MARKET_RULES_PATH: str = Field(default="configs/market_rules_vn.yaml")
    FEES_TAXES_PATH: str = Field(default="configs/fees_taxes.yaml")
    EXECUTION_MODEL_PATH: str = Field(default="configs/execution_model.yaml")
    BROKER_NAME: str = Field(default="demo_broker")

    API_BASE_URL: str = Field(default="http://localhost:8000")

    TRADING_ENV: str = Field(default="dev")
    ENABLE_LIVE_TRADING: bool = Field(default=False)
    LIVE_BROKER: str = Field(default="paper")
    LIVE_SANDBOX: bool = Field(default=True)
    KILL_SWITCH: bool = Field(default=False)

    ALERT_EMAIL_ENABLED: bool = Field(default=False)
    ALERT_DIGEST_RECIPIENT: str = Field(default="")

    SSI_CONSUMER_ID: str = Field(default="")
    SSI_CONSUMER_SECRET: str = Field(default="")
    SSI_PRIVATE_KEY_PATH: str = Field(default="")

    INGEST_BATCH_SIZE: int = Field(default=500)
    INGEST_ERROR_KILL_SWITCH_THRESHOLD: int = Field(default=100)
    INGEST_SCHEMA_DRIFT_KILL_SWITCH_THRESHOLD: int = Field(default=200)

    API_DEFAULT_LIMIT: int = Field(default=200)
    API_MAX_LIMIT: int = Field(default=2000)
    API_DEFAULT_DAYS: int = Field(default=365)

    PARQUET_LAKE_ROOT: str = Field(default="artifacts/parquet_lake")
    DUCKDB_PATH: str = Field(default="artifacts/duckdb/cache.duckdb")
    ENABLE_DUCKDB_FAST_PATH: bool = Field(default=True)

    @model_validator(mode="after")
    def validate_runtime_requirements(self) -> Settings:
        if self.TRADING_ENV not in {"dev", "paper", "live"}:
            raise RuntimeError("TRADING_ENV phải thuộc dev|paper|live")

        required_urls = {
            "DATABASE_URL": self.DATABASE_URL,
            "REDIS_URL": self.REDIS_URL,
            "SSI_FCDATA_BASE_URL": self.SSI_FCDATA_BASE_URL,
            "SSI_STREAM_URL": self.SSI_STREAM_URL,
        }
        missing_urls = [name for name, value in required_urls.items() if not str(value).strip()]
        if missing_urls:
            missing_msg = ", ".join(missing_urls)
            raise RuntimeError(f"Missing required environment variables: {missing_msg}")

        missing_ssi_credentials = [
            key
            for key, value in {
                "SSI_CONSUMER_ID": self.SSI_CONSUMER_ID,
                "SSI_CONSUMER_SECRET": self.SSI_CONSUMER_SECRET,
                "SSI_PRIVATE_KEY_PATH": self.SSI_PRIVATE_KEY_PATH,
            }.items()
            if not str(value).strip()
        ]

        if missing_ssi_credentials and not self.DEV_MODE:
            missing_msg = ", ".join(missing_ssi_credentials)
            raise RuntimeError(
                "SSI credentials are required when DEV_MODE=false. " f"Missing: {missing_msg}"
            )

        if missing_ssi_credentials and self.DEV_MODE:
            log.warning(
                "DEV_MODE=true and SSI credentials are missing; SSI live provider will be disabled. "
                "Missing: %s",
                ", ".join(missing_ssi_credentials),
            )

        return self


def get_settings() -> Settings:
    return Settings()
