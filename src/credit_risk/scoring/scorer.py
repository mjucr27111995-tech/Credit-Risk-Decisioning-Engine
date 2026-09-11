"""The scoring service: loan application -> risk assessment."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from credit_risk.config import Settings, get_settings
from credit_risk.domain.models import (
    FeatureContribution,
    LoanApplication,
    RiskAssessment,
)
from credit_risk.domain.scorecard import Scorecard
from credit_risk.errors import CreditRiskError, ScoringError
from credit_risk.observability import EventName, EventStatus, get_logger, log_event
from credit_risk.scoring.artifact import ModelArtifact, load_artifact
from credit_risk.scoring.features import FeatureVectorBuilder
from credit_risk.scoring.labels import humanise


def _sigmoid(log_odds: float) -> float:
    """Numerically stable logistic function.

    The naive ``1 / (1 + exp(-x))`` overflows for large negative ``x``; this
    formulation is stable across the whole real line.
    """
    if log_odds >= 0:
        return 1.0 / (1.0 + math.exp(-log_odds))
    exponentiated = math.exp(log_odds)
    return exponentiated / (1.0 + exponentiated)


class CreditRiskScorer:
    """Scores loan applications and explains each decision.

    Dependencies are injected via the constructor; there is no global state, so
    the same instance is safe to reuse across requests and Streamlit reruns.

    Args:
        artifact: Validated model bundle.
        feature_builder: Converts domain objects into the model matrix.
        scorecard: Maps probabilities onto scores and rating bands.
    """

    def __init__(
        self,
        artifact: ModelArtifact,
        feature_builder: FeatureVectorBuilder,
        scorecard: Scorecard,
    ) -> None:
        self._artifact = artifact
        self._feature_builder = feature_builder
        self._scorecard = scorecard
        self._logger = get_logger()

    @property
    def scorecard(self) -> Scorecard:
        """The scorecard calibration in use."""
        return self._scorecard

    @property
    def model_version(self) -> str:
        """Version of the loaded model artifact."""
        return self._artifact.version

    def score(self, application: LoanApplication) -> RiskAssessment:
        """Assess a single loan application.

        Args:
            application: A validated loan application.

        Returns:
            An immutable :class:`RiskAssessment` including per-feature
            contributions.

        Raises:
            ScoringError: ``CRM-INT-002`` if inference fails.
        """
        pending = self._new_assessment()
        domain_id = pending.assessment_id

        try:
            result = self._assess(application, pending)
        except CreditRiskError as exc:
            self._log_failure(domain_id, exc.message, exc.code)
            raise
        except Exception as exc:  # noqa: BLE001 - boundary: never leak raw errors
            error = ScoringError(f"Unexpected failure while scoring application: {exc}")
            self._log_failure(domain_id, error.message, error.code)
            raise error from exc

        log_event(
            self._logger,
            event_name=EventName.RISK_ASSESSMENT,
            status=EventStatus.SUCCESS,
            domain_id=domain_id,
            status_message=f"Assessed as {result.rating}",
            probabilityOfDefault=round(result.probability_of_default, 6),
            creditScore=result.credit_score,
            rating=str(result.rating),
            modelVersion=result.model_version,
        )
        return result

    def _new_assessment(self) -> RiskAssessment:
        """Mint an assessment shell so its id can label events from the start."""
        top = self._scorecard.max_score
        return RiskAssessment(
            probability_of_default=0.0,
            credit_score=top,
            rating=self._scorecard.rating_for(top),
            recommended_action=self._scorecard.action_for(top),
        )

    def _assess(self, application: LoanApplication, pending: RiskAssessment) -> RiskAssessment:
        """Run the inference pipeline and fill in the assessment."""
        domain_id = pending.assessment_id

        features = self._feature_builder.build(application)
        log_event(
            self._logger,
            event_name=EventName.FEATURE_VECTOR_BUILD,
            status=EventStatus.SUCCESS,
            domain_id=domain_id,
            status_message=f"Built {features.shape[1]} scaled features",
        )

        probability, contributions = self._predict(features)
        credit_score = self._scorecard.to_credit_score(probability)
        band = self._scorecard.band_for(credit_score)
        log_event(
            self._logger,
            event_name=EventName.SCORECARD_MAPPING,
            status=EventStatus.SUCCESS,
            domain_id=domain_id,
            status_message=f"Mapped PD to {band.rating}",
            creditScore=credit_score,
        )

        return pending.model_copy(
            update={
                "probability_of_default": probability,
                "credit_score": credit_score,
                "rating": band.rating,
                "recommended_action": band.action,
                "contributions": contributions,
                "model_version": self._artifact.version,
            }
        )

    def _log_failure(self, domain_id: str, message: str, code: str) -> None:
        """Emit the L1 failure event for a rejected assessment."""
        log_event(
            self._logger,
            event_name=EventName.RISK_ASSESSMENT,
            status=EventStatus.FAILURE,
            domain_id=domain_id,
            status_message=message,
            errorCode=code,
        )

    def _predict(self, features: pd.DataFrame) -> tuple[float, tuple[FeatureContribution, ...]]:
        """Compute the default probability and the exact feature contributions.

        For a logistic regression the log-odds are a plain sum of
        ``coefficient * value`` terms, so each term *is* that feature's
        contribution - an exact explanation, not an approximation.
        """
        values = np.asarray(features.to_numpy(dtype=float)).ravel()
        coefficients = self._artifact.coefficients
        if values.size != coefficients.size:
            raise ScoringError(
                f"Feature/coefficient mismatch: {values.size} features "
                f"vs {coefficients.size} coefficients"
            )

        terms = coefficients * values
        log_odds = float(terms.sum()) + self._artifact.intercept
        probability = _sigmoid(log_odds)

        contributions = tuple(
            FeatureContribution(
                feature=name,
                display_name=humanise(name),
                raw_value=float(value),
                coefficient=float(coefficient),
                contribution=float(term),
            )
            for name, value, coefficient, term in zip(
                self._artifact.features, values, coefficients, terms, strict=True
            )
        )
        return probability, contributions


def build_scorer(settings: Settings | None = None) -> CreditRiskScorer:
    """Compose a fully wired scorer from configuration.

    Args:
        settings: Optional override; loaded from the environment when omitted.

    Returns:
        A ready-to-use :class:`CreditRiskScorer`.

    Raises:
        CreditRiskError: If configuration or the model artifact is invalid.
    """
    resolved = settings or get_settings()
    artifact = load_artifact(resolved.artifact_path)
    return CreditRiskScorer(
        artifact=artifact,
        feature_builder=FeatureVectorBuilder(artifact),
        scorecard=Scorecard(
            base_score=resolved.base_score,
            scale_length=resolved.scale_length,
        ),
    )
