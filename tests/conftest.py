"""Shared fixtures.

Unit tests never touch disk or the real model. Integration tests explicitly
opt in via the ``integration`` marker.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from credit_risk.config import Settings
from credit_risk.domain.models import LoanApplication, LoanPurpose, LoanType, ResidenceType
from credit_risk.domain.scorecard import Scorecard
from credit_risk.scoring.artifact import EXPECTED_FEATURES, ModelArtifact
from credit_risk.scoring.features import PLACEHOLDER_COLUMNS, FeatureVectorBuilder
from credit_risk.scoring.scorer import CreditRiskScorer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "model_data.joblib"


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip ``CRM_*`` variables so the suite is independent of the shell.

    Without this, a developer who has exported e.g. ``CRM_LOG_LEVEL`` sees
    phantom failures in tests that assert on documented defaults. Individual
    tests still set the variables they need via ``monkeypatch.setenv``.
    """
    for key in list(os.environ):
        if key.startswith("CRM_"):
            monkeypatch.delenv(key, raising=False)


class FakeEstimator:
    """Minimal linear estimator stand-in with known coefficients."""

    def __init__(self, coefficients: np.ndarray, intercept: float) -> None:
        self.coef_ = coefficients.reshape(1, -1)
        self.intercept_ = np.array([intercept])


class IdentityScaler:
    """Scaler stub that passes values through untouched."""

    def __init__(self, expected_columns: tuple[str, ...]) -> None:
        self.expected_columns = expected_columns

    def transform(self, frame: object) -> np.ndarray:
        """Return the frame's values unchanged."""
        return np.asarray(frame.to_numpy(dtype=float))  # type: ignore[attr-defined]


@pytest.fixture
def scaled_columns() -> tuple[str, ...]:
    """The 18 columns the real scaler expects."""
    return EXPECTED_FEATURES + PLACEHOLDER_COLUMNS


@pytest.fixture
def fake_artifact(scaled_columns: tuple[str, ...]) -> ModelArtifact:
    """A deterministic artifact: coefficient i == i + 1, intercept -5."""
    coefficients = np.arange(1, len(EXPECTED_FEATURES) + 1, dtype=float)
    return ModelArtifact(
        model=FakeEstimator(coefficients, -5.0),
        features=EXPECTED_FEATURES,
        scaler=IdentityScaler(scaled_columns),
        cols_to_scale=scaled_columns,
        version="test-artifact",
    )


@pytest.fixture
def scorer(fake_artifact: ModelArtifact) -> CreditRiskScorer:
    """A scorer wired to the deterministic fake artifact."""
    return CreditRiskScorer(
        artifact=fake_artifact,
        feature_builder=FeatureVectorBuilder(fake_artifact),
        scorecard=Scorecard(),
    )


@pytest.fixture
def scorecard() -> Scorecard:
    """The default 300-900 scorecard."""
    return Scorecard()


@pytest.fixture
def valid_application() -> LoanApplication:
    """The reference application used by the legacy prototype's defaults."""
    return LoanApplication(
        age=28,
        income=1_200_000,
        loan_amount=2_560_000,
        loan_tenure_months=36,
        avg_dpd_per_delinquency=20.0,
        delinquency_ratio=30.0,
        credit_utilization_ratio=30.0,
        number_of_open_accounts=2,
        residence_type=ResidenceType.OWNED,
        loan_purpose=LoanPurpose.EDUCATION,
        loan_type=LoanType.UNSECURED,
    )


@pytest.fixture
def low_risk_application() -> LoanApplication:
    """A clean applicant: no delinquency, low utilisation, secured home loan."""
    return LoanApplication(
        age=45,
        income=3_000_000,
        loan_amount=900_000,
        loan_tenure_months=24,
        avg_dpd_per_delinquency=0.0,
        delinquency_ratio=0.0,
        credit_utilization_ratio=10.0,
        number_of_open_accounts=1,
        residence_type=ResidenceType.OWNED,
        loan_purpose=LoanPurpose.HOME,
        loan_type=LoanType.SECURED,
    )


@pytest.fixture
def default_settings(tmp_path: Path) -> Settings:
    """Settings pointing at a temporary, non-existent artifact path."""
    return Settings(artifact_path=tmp_path / "missing.joblib")
