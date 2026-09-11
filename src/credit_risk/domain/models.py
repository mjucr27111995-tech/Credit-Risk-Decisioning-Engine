"""Domain entities and value objects.

``LoanApplication`` is the aggregate root and the single consistency boundary:
all input invariants are enforced here, at the boundary, so no downstream layer
re-validates. ``RiskAssessment`` and its value objects are immutable.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, computed_field


class ResidenceType(StrEnum):
    """Applicant residence status. Mirrors the trained model's categories."""

    OWNED = "Owned"
    RENTED = "Rented"
    MORTGAGE = "Mortgage"


class LoanPurpose(StrEnum):
    """Purpose of the requested loan."""

    EDUCATION = "Education"
    HOME = "Home"
    AUTO = "Auto"
    PERSONAL = "Personal"


class LoanType(StrEnum):
    """Whether the loan is backed by collateral."""

    SECURED = "Secured"
    UNSECURED = "Unsecured"


class RiskRating(StrEnum):
    """Scorecard rating band mandated by the SOW."""

    POOR = "Poor"
    AVERAGE = "Average"
    GOOD = "Good"
    EXCELLENT = "Excellent"


class RecommendedAction(StrEnum):
    """Operational recommendation derived from the rating.

    ``AUTO_APPROVE`` / ``AUTO_DECLINE`` are the Straight-Through-Processing
    candidates described in SOW Phase 2; the middle bands stay with a human.
    """

    AUTO_APPROVE = "Auto-approve (STP eligible)"
    APPROVE_WITH_REVIEW = "Approve with manual review"
    REFER_TO_UNDERWRITER = "Refer to underwriter"
    AUTO_DECLINE = "Decline (STP eligible)"


class LoanApplication(BaseModel):
    """Aggregate root: a validated loan application ready for scoring.

    Bounds mirror the training distribution. Values outside them are rejected
    rather than silently extrapolated, because a linear model gives no warning
    when asked to predict far outside the data it saw.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", use_enum_values=False)

    age: int = Field(ge=18, le=100, description="Applicant age in years.")
    income: int = Field(gt=0, le=1_000_000_000, description="Annual income in INR.")
    loan_amount: int = Field(gt=0, le=1_000_000_000, description="Requested principal in INR.")
    loan_tenure_months: int = Field(ge=1, le=360, description="Repayment tenure in months.")
    avg_dpd_per_delinquency: float = Field(
        ge=0.0, le=365.0, description="Mean days past due across delinquent months."
    )
    delinquency_ratio: float = Field(
        ge=0.0, le=100.0, description="Delinquent months as a percentage of total loan months."
    )
    credit_utilization_ratio: float = Field(
        ge=0.0, le=100.0, description="Bureau credit utilisation percentage."
    )
    number_of_open_accounts: int = Field(
        ge=1, le=20, description="Open credit accounts reported by the bureau."
    )
    residence_type: ResidenceType
    loan_purpose: LoanPurpose
    loan_type: LoanType

    @computed_field  # type: ignore[prop-decorator]
    @property
    def loan_to_income(self) -> float:
        """Loan-to-income ratio, the strongest single predictor in the model.

        ``income`` is constrained to be positive, so this cannot divide by zero.
        """
        return round(self.loan_amount / self.income, 2)


class FeatureContribution(BaseModel):
    """One feature's additive push on the log-odds of default.

    Explainability is a contractual SOW deliverable: because the model is a
    logistic regression, ``coefficient * scaled_value`` is the exact, not
    approximate, contribution to the decision.
    """

    model_config = ConfigDict(frozen=True)

    feature: str
    display_name: str
    raw_value: float
    coefficient: float
    contribution: float

    @property
    def increases_risk(self) -> bool:
        """True when this feature pushed the application toward default."""
        return self.contribution > 0


class RiskAssessment(BaseModel):
    """Immutable result of scoring a single loan application."""

    model_config = ConfigDict(frozen=True)

    assessment_id: str = Field(default_factory=lambda: f"RA-{uuid.uuid4().hex[:12].upper()}")
    assessed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    probability_of_default: float = Field(ge=0.0, le=1.0)
    credit_score: int
    rating: RiskRating
    recommended_action: RecommendedAction
    contributions: tuple[FeatureContribution, ...] = ()
    model_version: str = "unknown"

    @property
    def probability_percent(self) -> float:
        """Probability of default as a percentage."""
        return self.probability_of_default * 100.0

    def top_drivers(self, limit: int = 5) -> tuple[FeatureContribution, ...]:
        """Return the features with the largest absolute influence."""
        ranked = sorted(self.contributions, key=lambda c: abs(c.contribution), reverse=True)
        return tuple(ranked[:limit])
