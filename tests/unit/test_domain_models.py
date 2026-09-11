"""Unit tests for the LoanApplication aggregate and its value objects."""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from credit_risk.domain.models import (
    FeatureContribution,
    LoanApplication,
    LoanPurpose,
    LoanType,
    RecommendedAction,
    ResidenceType,
    RiskAssessment,
    RiskRating,
)

_BASE_INPUT = {
    "age": 30,
    "income": 1_000_000,
    "loan_amount": 500_000,
    "loan_tenure_months": 24,
    "avg_dpd_per_delinquency": 0.0,
    "delinquency_ratio": 0.0,
    "credit_utilization_ratio": 20.0,
    "number_of_open_accounts": 2,
    "residence_type": ResidenceType.OWNED,
    "loan_purpose": LoanPurpose.HOME,
    "loan_type": LoanType.SECURED,
}


class TestLoanApplicationValidation:
    def should_create_withValidInput_returnApplication(self):
        # Act
        application = LoanApplication(**_BASE_INPUT)

        # Assert
        assert application.age == 30
        assert application.residence_type is ResidenceType.OWNED

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("age", 17),
            ("age", 101),
            ("income", 0),
            ("income", -1),
            ("loan_amount", 0),
            ("loan_tenure_months", 0),
            ("loan_tenure_months", 361),
            ("avg_dpd_per_delinquency", -1.0),
            ("avg_dpd_per_delinquency", 366.0),
            ("delinquency_ratio", -0.1),
            ("delinquency_ratio", 100.1),
            ("credit_utilization_ratio", -1.0),
            ("credit_utilization_ratio", 101.0),
            ("number_of_open_accounts", 0),
            ("number_of_open_accounts", 21),
        ],
    )
    def should_create_withOutOfBoundsField_throwValidationError(self, field: str, value: float):
        # Arrange
        payload = {**_BASE_INPUT, field: value}

        # Act / Assert
        with pytest.raises(PydanticValidationError):
            LoanApplication(**payload)

    def should_create_withUnknownCategory_throwValidationError(self):
        with pytest.raises(PydanticValidationError):
            LoanApplication(**{**_BASE_INPUT, "residence_type": "Houseboat"})

    def should_create_withExtraField_throwValidationError(self):
        with pytest.raises(PydanticValidationError):
            LoanApplication(**{**_BASE_INPUT, "credit_bureau_name": "CIBIL"})

    def should_mutate_withFrozenAggregate_throwValidationError(self):
        # Arrange
        application = LoanApplication(**_BASE_INPUT)

        # Act / Assert
        with pytest.raises(PydanticValidationError):
            application.age = 40


class TestLoanToIncome:
    def should_computeLoanToIncome_withValidAmounts_returnRoundedRatio(self):
        application = LoanApplication(
            **{**_BASE_INPUT, "loan_amount": 2_560_000, "income": 1_200_000}
        )
        assert application.loan_to_income == pytest.approx(2.13)

    def should_computeLoanToIncome_withMinimalIncome_avoidDivisionByZero(self):
        # Arrange - income is constrained positive, so this must stay finite
        application = LoanApplication(**{**_BASE_INPUT, "income": 1, "loan_amount": 1})

        # Assert
        assert application.loan_to_income == pytest.approx(1.0)


class TestRiskAssessment:
    def should_createAssessment_withDefaults_generateUniqueIdentifier(self):
        # Act
        first = RiskAssessment(
            probability_of_default=0.1,
            credit_score=840,
            rating=RiskRating.EXCELLENT,
            recommended_action=RecommendedAction.AUTO_APPROVE,
        )
        second = RiskAssessment(
            probability_of_default=0.1,
            credit_score=840,
            rating=RiskRating.EXCELLENT,
            recommended_action=RecommendedAction.AUTO_APPROVE,
        )

        # Assert
        assert first.assessment_id.startswith("RA-")
        assert first.assessment_id != second.assessment_id

    def should_reportProbabilityPercent_withProbability_returnScaledValue(self):
        assessment = RiskAssessment(
            probability_of_default=0.1234,
            credit_score=800,
            rating=RiskRating.EXCELLENT,
            recommended_action=RecommendedAction.AUTO_APPROVE,
        )
        assert assessment.probability_percent == pytest.approx(12.34)

    def should_createAssessment_withInvalidProbability_throwValidationError(self):
        with pytest.raises(PydanticValidationError):
            RiskAssessment(
                probability_of_default=1.5,
                credit_score=800,
                rating=RiskRating.EXCELLENT,
                recommended_action=RecommendedAction.AUTO_APPROVE,
            )

    def should_rankTopDrivers_withMixedSigns_orderByAbsoluteInfluence(self):
        # Arrange
        contributions = (
            _contribution("a", 0.5),
            _contribution("b", -4.0),
            _contribution("c", 2.0),
        )
        assessment = RiskAssessment(
            probability_of_default=0.5,
            credit_score=600,
            rating=RiskRating.AVERAGE,
            recommended_action=RecommendedAction.REFER_TO_UNDERWRITER,
            contributions=contributions,
        )

        # Act
        drivers = assessment.top_drivers(2)

        # Assert
        assert [item.feature for item in drivers] == ["b", "c"]


class TestFeatureContribution:
    def should_flagDirection_withPositiveContribution_returnIncreasesRisk(self):
        assert _contribution("x", 1.5).increases_risk is True

    def should_flagDirection_withNegativeContribution_returnReducesRisk(self):
        assert _contribution("x", -1.5).increases_risk is False

    def should_flagDirection_withZeroContribution_returnReducesRisk(self):
        assert _contribution("x", 0.0).increases_risk is False


def _contribution(feature: str, value: float) -> FeatureContribution:
    """Build a contribution with the given signed effect."""
    return FeatureContribution(
        feature=feature,
        display_name=feature.upper(),
        raw_value=1.0,
        coefficient=value,
        contribution=value,
    )
