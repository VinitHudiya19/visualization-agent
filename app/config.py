"""
Centralized configuration for Visualization Agent.
All secrets come from .env — nothing hardcoded.
"""
from __future__ import annotations

import logging
import sys
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("viz-agent")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(handler)
    return logger


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # Storage
    STORAGE_TYPE: Literal["local"] = "local"
    CHART_OUTPUT_PATH: str = "./charts"
    MAX_DATA_ROWS: int = 1000

    # Server
    PORT: int = 8003

    # CORS — comma-separated origins, "*" means allow all
    CORS_ORIGINS: str = "*"

    @property
    def cors_origins_list(self) -> list[str]:
        if self.CORS_ORIGINS.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


settings = Settings()
logger = setup_logging()
