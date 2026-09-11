"""Unit tests for configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from credit_risk.config import PROJECT_ROOT, AppEnv, LogLevel, Settings, get_settings
from credit_risk.errors import ConfigurationError


class TestSettingsDefaults:
    def should_loadSettings_withNoOverrides_returnDocumentedDefaults(self):
        # Act
        settings = Settings(_env_file=None)

        # Assert
        assert settings.app_env is AppEnv.DEV
        assert settings.log_level is LogLevel.INFO
        assert settings.base_score == 300
        assert settings.scale_length == 600
        assert settings.max_score == 900

    def should_resolvePath_withRelativeArtifactPath_anchorToProjectRoot(self):
        settings = Settings(_env_file=None, artifact_path=Path("artifacts/model_data.joblib"))
        assert settings.artifact_path.is_absolute()
        assert settings.artifact_path == PROJECT_ROOT / "artifacts" / "model_data.joblib"

    def should_resolvePath_withAbsoluteArtifactPath_preserveIt(self, tmp_path: Path):
        target = tmp_path / "custom.joblib"
        assert Settings(_env_file=None, artifact_path=target).artifact_path == target


class TestSettingsFromEnvironment:
    def should_loadSettings_withEnvironmentOverrides_applyThem(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        # Arrange
        monkeypatch.setenv("CRM_APP_ENV", "production")
        monkeypatch.setenv("CRM_LOG_LEVEL", "warning")
        monkeypatch.setenv("CRM_BASE_SCORE", "100")
        monkeypatch.setenv("CRM_SCALE_LENGTH", "900")
        monkeypatch.setenv("CRM_ARTIFACT_PATH", str(tmp_path / "m.joblib"))

        # Act
        settings = Settings(_env_file=None)

        # Assert
        assert settings.app_env is AppEnv.PRODUCTION
        assert settings.log_level is LogLevel.WARNING
        assert settings.max_score == 1000

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("CRM_BASE_SCORE", "-1"),
            ("CRM_BASE_SCORE", "1001"),
            ("CRM_SCALE_LENGTH", "0"),
            ("CRM_SCALE_LENGTH", "1001"),
            ("CRM_APP_ENV", "sandbox"),
            ("CRM_LOG_LEVEL", "trace"),
            ("CRM_BASE_SCORE", "not-a-number"),
        ],
    )
    def should_loadSettings_withInvalidEnvironment_throwConfigurationError(
        self, monkeypatch: pytest.MonkeyPatch, key: str, value: str
    ):
        # Arrange
        monkeypatch.setenv(key, value)
        get_settings.cache_clear()

        # Act / Assert - fail fast, never silently default
        with pytest.raises(ConfigurationError) as exc_info:
            get_settings()
        assert exc_info.value.code == "CRM-INT-001"
        get_settings.cache_clear()


class TestLogRendering:
    def should_renderJsonLogs_withDevEnvironment_returnFalse(self):
        settings = Settings(_env_file=None, app_env=AppEnv.DEV, log_json=False)
        assert settings.render_json_logs is False

    def should_renderJsonLogs_withDevAndExplicitFlag_returnTrue(self):
        assert Settings(_env_file=None, app_env=AppEnv.DEV, log_json=True).render_json_logs is True

    @pytest.mark.parametrize("env", [AppEnv.STAGING, AppEnv.PRODUCTION])
    def should_renderJsonLogs_withNonDevEnvironment_forceJson(self, env: AppEnv):
        # Assert - structured logs are mandatory outside development
        assert Settings(_env_file=None, app_env=env, log_json=False).render_json_logs is True


class TestSettingsCaching:
    def should_getSettings_withRepeatedCalls_returnSameInstance(self):
        # Arrange
        get_settings.cache_clear()

        # Act / Assert
        assert get_settings() is get_settings()
        get_settings.cache_clear()
