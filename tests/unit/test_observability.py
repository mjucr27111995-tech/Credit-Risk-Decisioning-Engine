"""Unit tests for structured logging and the Business Process Event contract."""

from __future__ import annotations

import json
from typing import Any

import pytest
import structlog

from credit_risk.config import AppEnv, LogLevel, Settings
from credit_risk.observability import (
    SERVICE_NAME,
    EventName,
    EventStatus,
    bind_correlation_id,
    configure_logging,
    get_logger,
    log_event,
    new_correlation_id,
)


@pytest.fixture
def captured() -> list[dict[str, Any]]:
    """Capture emitted events without touching stdout."""
    logs = structlog.testing.LogCapture()
    structlog.configure(processors=[logs])
    yield logs.entries
    structlog.reset_defaults()


class TestLogEvent:
    def should_logEvent_withSuccess_emitThreePartMessage(self, captured: list[dict[str, Any]]):
        # Act
        log_event(
            get_logger(),
            event_name=EventName.RISK_ASSESSMENT,
            status=EventStatus.SUCCESS,
            domain_id="RA-0001",
            status_message="Assessed as Good",
        )

        # Assert
        entry = captured[0]
        assert entry["event"] == f"{SERVICE_NAME} | Event: RISK_ASSESSMENT SUCCESS | RA-0001"
        assert entry["log_level"] == "info"

    def should_logEvent_withSuccess_includeFullEventContract(self, captured: list[dict[str, Any]]):
        # Act
        log_event(
            get_logger(),
            event_name=EventName.SCORECARD_MAPPING,
            status=EventStatus.SUCCESS,
            domain_id="RA-0002",
            status_message="Mapped PD to Excellent",
            creditScore=880,
        )

        # Assert
        payload = captured[0]["__bpe__"]
        assert payload == {
            "domainId": "RA-0002",
            "domainType": "LOAN_APPLICATION",
            "eventName": "SCORECARD_MAPPING",
            "status": "SUCCESS",
            "statusMessage": "Mapped PD to Excellent",
        }
        assert captured[0]["creditScore"] == 880

    def should_logEvent_withFailure_emitAtErrorLevel(self, captured: list[dict[str, Any]]):
        # Act
        log_event(
            get_logger(),
            event_name=EventName.MODEL_ARTIFACT_LOAD,
            status=EventStatus.FAILURE,
            domain_id="model.joblib",
            domain_type="MODEL_ARTIFACT",
            status_message="Artifact missing",
            errorCode="CRM-NOT-001",
        )

        # Assert
        entry = captured[0]
        assert entry["log_level"] == "error"
        assert entry["__bpe__"]["status"] == "FAILURE"
        assert entry["errorCode"] == "CRM-NOT-001"

    def should_logEvent_withCustomDomainType_overrideDefault(self, captured: list[dict[str, Any]]):
        log_event(
            get_logger(),
            event_name=EventName.MODEL_ARTIFACT_LOAD,
            status=EventStatus.SUCCESS,
            domain_id="lr-abc",
            domain_type="MODEL_ARTIFACT",
            status_message="Loaded",
        )
        assert captured[0]["__bpe__"]["domainType"] == "MODEL_ARTIFACT"

    @pytest.mark.parametrize("status", list(EventStatus))
    def should_logEvent_withAnyStatus_useOnlyPermittedValues(
        self, captured: list[dict[str, Any]], status: EventStatus
    ):
        # Assert - the contract admits SUCCESS and FAILURE only
        log_event(
            get_logger(),
            event_name=EventName.FEATURE_VECTOR_BUILD,
            status=status,
            domain_id="RA-0003",
            status_message="checked",
        )
        assert captured[0]["__bpe__"]["status"] in {"SUCCESS", "FAILURE"}


class TestCorrelationId:
    def should_generateCorrelationId_withRepeatedCalls_returnUniqueValues(self):
        assert new_correlation_id() != new_correlation_id()

    def should_bindCorrelationId_withValue_attachToSubsequentEvents(
        self, captured: list[dict[str, Any]]
    ):
        # Arrange
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                *structlog.get_config()["processors"],
            ]
        )
        correlation_id = new_correlation_id()

        # Act
        bind_correlation_id(correlation_id)
        try:
            log_event(
                get_logger(),
                event_name=EventName.RISK_ASSESSMENT,
                status=EventStatus.SUCCESS,
                domain_id="RA-0004",
                status_message="done",
            )

            # Assert
            assert captured[0]["correlationID"] == correlation_id
        finally:
            structlog.contextvars.clear_contextvars()


class TestConfigureLogging:
    def should_configureLogging_withProductionSettings_renderValidJson(
        self, capsys: pytest.CaptureFixture[str]
    ):
        # Arrange
        structlog.reset_defaults()
        configure_logging(
            Settings(_env_file=None, app_env=AppEnv.PRODUCTION, log_level=LogLevel.INFO)
        )

        # Act
        log_event(
            get_logger(),
            event_name=EventName.RISK_ASSESSMENT,
            status=EventStatus.SUCCESS,
            domain_id="RA-0005",
            status_message="Assessed as Poor",
        )

        # Assert - the BPE contract requires `message` text and nested `event`
        payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert payload["message"].startswith(f"{SERVICE_NAME} | Event:")
        assert payload["event"]["eventName"] == "RISK_ASSESSMENT"
        assert payload["event"]["status"] == "SUCCESS"
        assert payload["service"] == SERVICE_NAME
        assert "timestamp" in payload
        structlog.reset_defaults()

    def should_configureLogging_withRepeatedCalls_remainIdempotent(self):
        # Arrange - Streamlit reruns the script on every interaction
        structlog.reset_defaults()
        settings = Settings(_env_file=None)

        # Act
        configure_logging(settings)
        first = structlog.get_config()["processors"]
        configure_logging(settings)

        # Assert
        assert structlog.get_config()["processors"] is first
        structlog.reset_defaults()
