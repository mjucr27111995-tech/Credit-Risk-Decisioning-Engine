"""Typed error hierarchy for the credit risk scorecard.

Error code format: ``CRM-[TYPE]-[NUM]`` where TYPE is one of
VAL (validation), NOT (not found), CON (conflict), EXT (external),
INT (internal).
"""

from __future__ import annotations

from datetime import UTC, datetime
from http import HTTPStatus


class CreditRiskError(Exception):
    """Base class for every error raised by this application."""

    def __init__(self, code: str, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.timestamp = datetime.now(UTC).isoformat()

    def to_dict(self) -> dict[str, str | int]:
        """Render the error in the standard response shape."""
        return {
            "timestamp": self.timestamp,
            "status": self.status_code,
            "error": self.code,
            "message": self.message,
        }

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


class ValidationError(CreditRiskError):
    """A loan application violates a domain invariant. ``CRM-VAL-001``."""

    def __init__(self, message: str, code: str = "CRM-VAL-001") -> None:
        super().__init__(code, message, HTTPStatus.BAD_REQUEST)


class ArtifactNotFoundError(CreditRiskError):
    """The serialised model bundle is absent from disk. ``CRM-NOT-001``."""

    def __init__(self, message: str) -> None:
        super().__init__("CRM-NOT-001", message, HTTPStatus.NOT_FOUND)


class ArtifactLoadError(CreditRiskError):
    """The model bundle exists but could not be deserialised. ``CRM-EXT-001``."""

    def __init__(self, message: str) -> None:
        super().__init__("CRM-EXT-001", message, HTTPStatus.SERVICE_UNAVAILABLE)


class ArtifactSchemaError(CreditRiskError):
    """The model bundle does not match the expected feature contract. ``CRM-CON-001``."""

    def __init__(self, message: str) -> None:
        super().__init__("CRM-CON-001", message, HTTPStatus.CONFLICT)


class ConfigurationError(CreditRiskError):
    """Startup configuration is missing or invalid. ``CRM-INT-001``."""

    def __init__(self, message: str) -> None:
        super().__init__("CRM-INT-001", message, HTTPStatus.INTERNAL_SERVER_ERROR)


class ScoringError(CreditRiskError):
    """Inference failed for an otherwise valid application. ``CRM-INT-002``."""

    def __init__(self, message: str) -> None:
        super().__init__("CRM-INT-002", message, HTTPStatus.INTERNAL_SERVER_ERROR)
