"""Translate a :class:`LoanApplication` into the model's input matrix.

Inherited technical debt, isolated here on purpose
-------------------------------------------------
The persisted ``MinMaxScaler`` was fitted on 18 columns during training, but
the final model consumes only 13. The 11 unused columns must still be present
for ``scaler.transform`` to accept the frame, so they are supplied as explicit
named placeholders and dropped immediately afterwards. They never reach the
model and cannot influence the prediction.

Removing this requires refitting the scaler on the model's feature set only.
Until then, everything about the workaround lives in this one module.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

from credit_risk.domain.models import LoanApplication, LoanPurpose, ResidenceType
from credit_risk.errors import ScoringError
from credit_risk.scoring.artifact import ModelArtifact

# Columns the fitted scaler expects but the model never consumes.
PLACEHOLDER_COLUMNS: Final[tuple[str, ...]] = (
    "number_of_dependants",
    "years_at_current_address",
    "zipcode",
    "sanction_amount",
    "processing_fee",
    "gst",
    "net_disbursement",
    "principal_outstanding",
    "bank_balance_at_application",
    "number_of_closed_accounts",
    "enquiry_count",
)

# Matches the constant used by the original prototype, which guarantees
# numeric parity with the model as it was validated during training.
PLACEHOLDER_VALUE: Final[float] = 1.0


class FeatureVectorBuilder:
    """Builds scaled, correctly ordered model input from domain objects.

    Args:
        artifact: The validated model bundle whose column contract to honour.
    """

    def __init__(self, artifact: ModelArtifact) -> None:
        self._artifact = artifact

    def build(self, application: LoanApplication) -> pd.DataFrame:
        """Return a single-row frame of scaled features in model order.

        Args:
            application: A validated loan application.

        Returns:
            One-row DataFrame whose columns are exactly ``artifact.features``.

        Raises:
            ScoringError: ``CRM-INT-002`` if scaling fails or a contracted
                column is missing.
        """
        frame = pd.DataFrame([self._to_row(application)])
        missing = [column for column in self._artifact.cols_to_scale if column not in frame.columns]
        if missing:
            raise ScoringError(
                f"Feature frame is missing columns required by the scaler: {missing}"
            )

        scale_columns = list(self._artifact.cols_to_scale)
        try:
            frame[scale_columns] = self._artifact.scaler.transform(frame[scale_columns])
        except (ValueError, TypeError, AttributeError) as exc:
            raise ScoringError(f"Failed to scale features: {exc}") from exc

        return frame[list(self._artifact.features)]

    @staticmethod
    def _to_row(application: LoanApplication) -> dict[str, float]:
        """Flatten an application into raw (unscaled) model + placeholder columns.

        One-hot encoding follows the training-time ``drop_first=True`` scheme:
        ``Mortgage``, ``Auto`` and ``Secured`` are the reference levels and are
        represented by all-zero indicators.
        """
        row: dict[str, float] = {
            "age": float(application.age),
            "loan_tenure_months": float(application.loan_tenure_months),
            "number_of_open_accounts": float(application.number_of_open_accounts),
            "credit_utilization_ratio": float(application.credit_utilization_ratio),
            "loan_to_income": float(application.loan_to_income),
            "delinquency_ratio": float(application.delinquency_ratio),
            "avg_dpd_per_delinquency": float(application.avg_dpd_per_delinquency),
            "residence_type_Owned": float(application.residence_type is ResidenceType.OWNED),
            "residence_type_Rented": float(application.residence_type is ResidenceType.RENTED),
            "loan_purpose_Education": float(application.loan_purpose is LoanPurpose.EDUCATION),
            "loan_purpose_Home": float(application.loan_purpose is LoanPurpose.HOME),
            "loan_purpose_Personal": float(application.loan_purpose is LoanPurpose.PERSONAL),
            "loan_type_Unsecured": float(application.loan_type.value == "Unsecured"),
        }
        row.update(dict.fromkeys(PLACEHOLDER_COLUMNS, PLACEHOLDER_VALUE))
        return row
