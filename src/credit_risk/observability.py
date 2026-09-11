"""Structured logging and Business Process Events (BPE).

L1 events mark major milestones (a full risk assessment) and are never
sampled. L2 events cover sub-operations (artifact load, feature build,
scorecard mapping) for debugging and performance analysis.

No PII is ever logged: applicant identity is not captured by this system, and
raw monetary inputs are excluded from event payloads.
"""

from __future__ import annotations

import logging
import sys
import uuid
from enum import StrEnum
from typing import Final

import structlog

from credit_risk.config import LogLevel, Settings

SERVICE_NAME: Final[str] = "credit-risk-scorecard"

_LEVEL_MAP: Final[dict[LogLevel, int]] = {
    LogLevel.DEBUG: logging.DEBUG,
    LogLevel.INFO: logging.INFO,
    LogLevel.WARNING: logging.WARNING,
    LogLevel.ERROR: logging.ERROR,
}


# structlog reserves the ``event`` key for the log message, but the BPE contract
# requires ``message`` for the text and ``event`` for the nested payload. The
# payload therefore travels under this internal key and is renamed on the way
# out by :func:`_rename_bpe_fields`.
_BPE_KEY: Final[str] = "__bpe__"


class EventStatus(StrEnum):
    """The only permitted BPE status values."""

    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


class EventName(StrEnum):
    """Business process event names.

    ``RISK_ASSESSMENT`` is the L1 milestone; the rest are L2 sub-operations.
    """

    RISK_ASSESSMENT = "RISK_ASSESSMENT"
    MODEL_ARTIFACT_LOAD = "MODEL_ARTIFACT_LOAD"
    FEATURE_VECTOR_BUILD = "FEATURE_VECTOR_BUILD"
    SCORECARD_MAPPING = "SCORECARD_MAPPING"


def _rename_bpe_fields(
    _logger: object, _method: str, event_dict: structlog.typing.EventDict
) -> structlog.typing.EventDict:
    """Reshape structlog's keys into the Business Process Event contract.

    Moves the log text from ``event`` to ``message`` and promotes the BPE
    payload into ``event``.
    """
    if "event" in event_dict:
        event_dict["message"] = event_dict.pop("event")
    if _BPE_KEY in event_dict:
        event_dict["event"] = event_dict.pop(_BPE_KEY)
    return event_dict


def configure_logging(settings: Settings) -> None:
    """Install the structlog pipeline. Idempotent, safe on Streamlit reruns."""
    if structlog.is_configured():
        return

    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer()
        if settings.render_json_logs
        else structlog.dev.ConsoleRenderer(colors=False, event_key="message")
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=_LEVEL_MAP[settings.log_level],
        force=True,
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            _rename_bpe_fields,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(_LEVEL_MAP[settings.log_level]),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger() -> structlog.stdlib.BoundLogger:
    """Return a logger bound to this service."""
    logger: structlog.stdlib.BoundLogger = structlog.get_logger().bind(service=SERVICE_NAME)
    return logger


def new_correlation_id() -> str:
    """Mint a correlation id. Generated once per session, never regenerated."""
    return str(uuid.uuid4())


def bind_correlation_id(correlation_id: str) -> None:
    """Bind a correlation id to the ambient logging context."""
    structlog.contextvars.bind_contextvars(correlationID=correlation_id)


def log_event(
    logger: structlog.stdlib.BoundLogger,
    *,
    event_name: EventName,
    status: EventStatus,
    domain_id: str,
    domain_type: str = "LOAN_APPLICATION",
    status_message: str,
    **context: str | int | float | bool | None,
) -> None:
    """Emit a business process event in the standard 3-part contract.

    Args:
        logger: Bound logger to write through.
        event_name: Business event being reported.
        status: ``SUCCESS`` or ``FAILURE`` only.
        domain_id: Business identifier, e.g. the assessment id.
        domain_type: Domain entity type.
        status_message: Human-readable summary.
        **context: Extra non-PII fields to attach.
    """
    message = f"{SERVICE_NAME} | Event: {event_name} {status} | {domain_id}"
    payload = {
        "domainId": domain_id,
        "domainType": domain_type,
        "eventName": str(event_name),
        "status": str(status),
        "statusMessage": status_message,
    }
    log = logger.info if status is EventStatus.SUCCESS else logger.error
    log(message, **{_BPE_KEY: payload}, **context)
