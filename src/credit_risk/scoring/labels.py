"""Display labels for model features.

Kept separate from the artifact loader so the presentation vocabulary can grow
without touching validation logic.
"""

from __future__ import annotations

from typing import Final

_HUMAN_LABELS: Final[dict[str, str]] = {
    "age": "Age",
    "loan_tenure_months": "Loan tenure (months)",
    "number_of_open_accounts": "Open loan accounts",
    "credit_utilization_ratio": "Credit utilisation ratio",
    "loan_to_income": "Loan-to-income ratio",
    "delinquency_ratio": "Delinquency ratio",
    "avg_dpd_per_delinquency": "Average DPD per delinquency",
    "residence_type_Owned": "Residence: owned",
    "residence_type_Rented": "Residence: rented",
    "loan_purpose_Education": "Purpose: education",
    "loan_purpose_Home": "Purpose: home",
    "loan_purpose_Personal": "Purpose: personal",
    "loan_type_Unsecured": "Unsecured loan",
}


def humanise(feature: str) -> str:
    """Return a display label for a model feature name.

    Args:
        feature: Raw model feature name.

    Returns:
        A curated label, or a sentence-cased fallback for unknown features.
    """
    return _HUMAN_LABELS.get(feature, feature.replace("_", " ").capitalize())
