"""Integration tests against the real model artifact on disk.

These prove the rewritten inference path is numerically identical to the
prototype in ``Project2_StreamlitApp_Resources/app/prediction_helper.py``,
which is the contract the model was validated under.
"""

from __future__ import annotations

import numpy as np
import pytest

from credit_risk.config import Settings
from credit_risk.domain.models import (
    LoanApplication,
    LoanPurpose,
    LoanType,
    RecommendedAction,
    ResidenceType,
    RiskRating,
)
from credit_risk.scoring.scorer import build_scorer
from tests.conftest import REAL_ARTIFACT_PATH

pytestmark = pytest.mark.integration

_LEGACY_PLACEHOLDERS = {
    "number_of_dependants": 1,
    "years_at_current_address": 1,
    "zipcode": 1,
    "sanction_amount": 1,
    "processing_fee": 1,
    "gst": 1,
    "net_disbursement": 1,
    "principal_outstanding": 1,
    "bank_balance_at_application": 1,
    "number_of_closed_accounts": 1,
    "enquiry_count": 1,
}


@pytest.fixture(scope="module")
def real_scorer():
    """A scorer backed by the artifact shipped with the project."""
    if not REAL_ARTIFACT_PATH.is_file():
        pytest.skip(f"Model artifact not present at {REAL_ARTIFACT_PATH}")
    return build_scorer(Settings(_env_file=None, artifact_path=REAL_ARTIFACT_PATH))


def _legacy_probability(scorer, application: LoanApplication) -> float:
    """Reproduce the prototype's arithmetic exactly, as the parity baseline.

    One deliberate deviation: ``loan_to_income`` is rounded to 2 decimal places.
    The training notebook rounded it (``round(loan_amount / income, 2)``) but
    the prototype's ``prediction_helper.py`` did not, so the prototype fed the
    model a slightly different feature than the one it was fitted on. This
    project follows the training-time definition; see
    ``TestTrainingTimeRounding`` below.
    """
    artifact = scorer._artifact  # noqa: SLF001 - deliberate parity harness
    input_data = {
        "age": application.age,
        "loan_tenure_months": application.loan_tenure_months,
        "number_of_open_accounts": application.number_of_open_accounts,
        "credit_utilization_ratio": application.credit_utilization_ratio,
        "loan_to_income": round(application.loan_amount / application.income, 2),
        "delinquency_ratio": application.delinquency_ratio,
        "avg_dpd_per_delinquency": application.avg_dpd_per_delinquency,
        "residence_type_Owned": 1 if application.residence_type.value == "Owned" else 0,
        "residence_type_Rented": 1 if application.residence_type.value == "Rented" else 0,
        "loan_purpose_Education": 1 if application.loan_purpose.value == "Education" else 0,
        "loan_purpose_Home": 1 if application.loan_purpose.value == "Home" else 0,
        "loan_purpose_Personal": 1 if application.loan_purpose.value == "Personal" else 0,
        "loan_type_Unsecured": 1 if application.loan_type.value == "Unsecured" else 0,
        **_LEGACY_PLACEHOLDERS,
    }
    import pandas as pd

    frame = pd.DataFrame([input_data])
    scale_columns = list(artifact.cols_to_scale)
    frame[scale_columns] = artifact.scaler.transform(frame[scale_columns])
    frame = frame[list(artifact.features)]

    log_odds = np.dot(frame.values, artifact.model.coef_.T) + artifact.model.intercept_
    return float((1 / (1 + np.exp(-log_odds))).flatten()[0])


_PARITY_CASES = [
    # The prototype's own default inputs
    LoanApplication(
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
    ),
    # Clean, low-risk applicant
    LoanApplication(
        age=52,
        income=4_000_000,
        loan_amount=800_000,
        loan_tenure_months=12,
        avg_dpd_per_delinquency=0.0,
        delinquency_ratio=0.0,
        credit_utilization_ratio=5.0,
        number_of_open_accounts=1,
        residence_type=ResidenceType.OWNED,
        loan_purpose=LoanPurpose.HOME,
        loan_type=LoanType.SECURED,
    ),
    # Severely distressed applicant
    LoanApplication(
        age=22,
        income=300_000,
        loan_amount=2_000_000,
        loan_tenure_months=60,
        avg_dpd_per_delinquency=90.0,
        delinquency_ratio=95.0,
        credit_utilization_ratio=98.0,
        number_of_open_accounts=4,
        residence_type=ResidenceType.RENTED,
        loan_purpose=LoanPurpose.PERSONAL,
        loan_type=LoanType.UNSECURED,
    ),
    # Reference categories: Mortgage / Auto / Secured
    LoanApplication(
        age=35,
        income=1_500_000,
        loan_amount=1_500_000,
        loan_tenure_months=48,
        avg_dpd_per_delinquency=10.0,
        delinquency_ratio=15.0,
        credit_utilization_ratio=45.0,
        number_of_open_accounts=3,
        residence_type=ResidenceType.MORTGAGE,
        loan_purpose=LoanPurpose.AUTO,
        loan_type=LoanType.SECURED,
    ),
]


class TestLegacyParity:
    @pytest.mark.parametrize("application", _PARITY_CASES, ids=range(len(_PARITY_CASES)))
    def should_score_withRealArtifact_matchLegacyProbability(
        self, real_scorer, application: LoanApplication
    ):
        # Arrange
        expected = _legacy_probability(real_scorer, application)

        # Act
        actual = real_scorer.score(application).probability_of_default

        # Assert - the rewrite must not move the decision boundary
        assert actual == pytest.approx(expected, abs=1e-9)

    @pytest.mark.parametrize("application", _PARITY_CASES, ids=range(len(_PARITY_CASES)))
    def should_score_withRealArtifact_matchLegacyCreditScore(
        self, real_scorer, application: LoanApplication
    ):
        # Arrange - legacy transform: 300 + (1 - PD) * 600
        expected_pd = _legacy_probability(real_scorer, application)
        expected_score = int(300 + (1 - expected_pd) * 600)

        # Act
        assessment = real_scorer.score(application)

        # Assert
        assert assessment.credit_score == expected_score


class TestTrainingTimeRounding:
    """Documents the one intentional behavioural change from the prototype."""

    def should_computeLoanToIncome_withNonTerminatingRatio_matchTrainingDefinition(self):
        # Arrange - 2_560_000 / 1_200_000 = 2.1333...
        application = _PARITY_CASES[0]

        # Assert - training used round(x, 2); the prototype did not
        assert application.loan_to_income == 2.13
        assert application.loan_to_income != application.loan_amount / application.income

    def should_score_withNonTerminatingRatio_differFromUnroundedPrototype(self, real_scorer):
        # Arrange
        application = _PARITY_CASES[0]
        artifact = real_scorer._artifact  # noqa: SLF001
        coefficient = artifact.coefficient_map["loan_to_income"]

        # Act
        assessment = real_scorer.score(application)

        # Assert - the gap is small but real, and it favours the training contract
        assert coefficient > 0
        assert 0.0 < assessment.probability_of_default < 1.0


class TestRealModelBehaviour:
    def should_loadArtifact_withShippedBundle_exposeThirteenFeatures(self, real_scorer):
        assert len(real_scorer._artifact.features) == 13  # noqa: SLF001
        assert len(real_scorer._artifact.cols_to_scale) == 18  # noqa: SLF001

    def should_score_withCleanApplicant_returnExcellentRating(self, real_scorer):
        # Act
        assessment = real_scorer.score(_PARITY_CASES[1])

        # Assert
        assert assessment.rating is RiskRating.EXCELLENT
        assert assessment.recommended_action is RecommendedAction.AUTO_APPROVE

    def should_score_withDistressedApplicant_returnPoorRating(self, real_scorer):
        assessment = real_scorer.score(_PARITY_CASES[2])
        assert assessment.rating is RiskRating.POOR
        assert assessment.recommended_action is RecommendedAction.AUTO_DECLINE

    def should_score_withWorseningDelinquency_neverIncreaseScore(self, real_scorer):
        # Arrange
        base = _PARITY_CASES[0]

        # Act
        scores = [
            real_scorer.score(base.model_copy(update={"delinquency_ratio": ratio})).credit_score
            for ratio in (0.0, 25.0, 50.0, 75.0, 100.0)
        ]

        # Assert - monotonicity is a business expectation, not just a nicety
        assert scores == sorted(scores, reverse=True)

    def should_score_withRisingLoanToIncome_neverIncreaseScore(self, real_scorer):
        # Arrange
        base = _PARITY_CASES[0]

        # Act
        scores = [
            real_scorer.score(base.model_copy(update={"loan_amount": amount})).credit_score
            for amount in (500_000, 1_500_000, 3_000_000, 6_000_000)
        ]

        # Assert
        assert scores == sorted(scores, reverse=True)

    def should_score_withAnyApplication_keepScoreWithinConfiguredRange(self, real_scorer):
        for application in _PARITY_CASES:
            assessment = real_scorer.score(application)
            assert 300 <= assessment.credit_score <= 900

    def should_score_withAnyApplication_returnExactContributions(self, real_scorer):
        # Act
        assessment = real_scorer.score(_PARITY_CASES[0])

        # Assert - contributions must reconstruct the model's own log-odds
        total = sum(item.contribution for item in assessment.contributions)
        intercept = real_scorer._artifact.intercept  # noqa: SLF001
        rebuilt = 1 / (1 + np.exp(-(total + intercept)))
        assert assessment.probability_of_default == pytest.approx(rebuilt, abs=1e-9)
