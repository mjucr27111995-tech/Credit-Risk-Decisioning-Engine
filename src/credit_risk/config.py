"""Application configuration.

Every value comes from the environment (prefix ``CRM_``) or an ``.env`` file,
is bound to a typed model, and is validated at startup. Missing or malformed
configuration crashes immediately with ``CRM-INT-001`` rather than silently
defaulting to something unsafe.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic import ValidationError as PydanticValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from credit_risk.errors import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class AppEnv(StrEnum):
    """Deployment environment. Drives log rendering and debug affordances."""

    DEV = "dev"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(StrEnum):
    """Supported log levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class Settings(BaseSettings):
    """Typed, validated application settings.

    Attributes:
        artifact_path: Location of the serialised model bundle. Relative paths
            resolve against the project root so the app runs from any cwd.
        app_env: Deployment environment.
        log_level: Minimum level emitted by the logger.
        log_json: Force JSON log rendering. Always on outside ``dev``.
        base_score: Lower bound of the credit score range. Valid 0-1000.
        scale_length: Width of the credit score range. Valid 1-1000.
        stp_enabled: Surface the Straight-Through-Processing recommendation.
    """

    model_config = SettingsConfigDict(
        env_prefix="CRM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    artifact_path: Path = Field(default=Path("artifacts/model_data.joblib"))
    app_env: AppEnv = Field(default=AppEnv.DEV)
    log_level: LogLevel = Field(default=LogLevel.INFO)
    log_json: bool = Field(default=False)
    base_score: int = Field(default=300, ge=0, le=1000)
    scale_length: int = Field(default=600, ge=1, le=1000)
    stp_enabled: bool = Field(default=True)

    @field_validator("artifact_path")
    @classmethod
    def _resolve_artifact_path(cls, value: Path) -> Path:
        """Anchor relative artifact paths to the project root."""
        return value if value.is_absolute() else PROJECT_ROOT / value

    @property
    def max_score(self) -> int:
        """Upper bound of the credit score range."""
        return self.base_score + self.scale_length

    @property
    def render_json_logs(self) -> bool:
        """JSON logs are mandatory outside development."""
        return self.log_json or self.app_env is not AppEnv.DEV


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and validate settings once per process.

    Raises:
        ConfigurationError: ``CRM-INT-001`` when the environment is invalid.
    """
    try:
        return Settings()
    except PydanticValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(part) for part in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise ConfigurationError(f"Invalid configuration: {details}") from exc
