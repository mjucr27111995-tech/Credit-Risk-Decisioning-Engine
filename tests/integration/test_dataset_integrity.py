"""Structural checks on the source dataset backing the model.

These exist because an earlier draft of the model card asserted target leakage
via ``delinquency_ratio`` and ``avg_dpd_per_delinquency``. The claim was wrong.
These tests encode the evidence that refuted it, so the conclusion is
machine-checked rather than a matter of prose that can silently rot.

The finding: the bureau table describes the applicant's history with *other*
lenders (CIBIL-style), which is available at application time. It does not
describe repayment of the loan being scored.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.integration

# The datasets live outside this project, alongside the course resources.
_DATASET_DIR = Path(__file__).resolve().parents[2].parent / "Project2_DataCollection_Resources"


@pytest.fixture(scope="module")
def merged() -> pd.DataFrame:
    """The three source tables joined on ``cust_id``, as the notebook does."""
    required = ["customers.csv", "loans.csv", "bureau_data.csv"]
    if not all((_DATASET_DIR / name).is_file() for name in required):
        pytest.skip(f"Source datasets not present in {_DATASET_DIR}")

    customers = pd.read_csv(_DATASET_DIR / "customers.csv")
    loans = pd.read_csv(_DATASET_DIR / "loans.csv")
    bureau = pd.read_csv(_DATASET_DIR / "bureau_data.csv")
    frame = customers.merge(loans, on="cust_id").merge(bureau, on="cust_id")
    frame["default"] = frame["default"].astype(int)
    frame["delinquency_ratio"] = (
        frame["delinquent_months"] * 100 / frame["total_loan_months"]
    ).round(1)
    return frame


class TestBureauDescribesExternalHistory:
    """Evidence that bureau fields are not derived from the scored loan."""

    def should_compareLoanCounts_withBureauAccounts_showMultipleExternalAccounts(
        self, merged: pd.DataFrame
    ):
        # Arrange - each customer holds exactly one loan with this lender
        loans_with_us = merged.groupby("cust_id")["loan_id"].nunique().max()

        # Act
        bureau_accounts = (
            merged["number_of_open_accounts"] + merged["number_of_closed_accounts"]
        ).mean()

        # Assert - ~3.5 accounts reported vs 1 held here => external lenders
        assert loans_with_us == 1
        assert bureau_accounts > 2.0

    def should_compareTenures_withBureauMonths_exceedCurrentLoanTenure(self, merged: pd.DataFrame):
        # Act
        bureau_months = merged["total_loan_months"].mean()
        current_tenure = merged["loan_tenure_months"].mean()

        # Assert - bureau history spans far longer than this loan
        assert bureau_months > 2 * current_tenure

    def should_compareTenures_withBureauMonths_rarelyMatchExactly(self, merged: pd.DataFrame):
        # Act - if bureau described this loan, these would coincide often
        exact_matches = (merged["total_loan_months"] == merged["loan_tenure_months"]).mean()

        # Assert
        assert exact_matches < 0.05


class TestDelinquencyIsPredictiveNotDeterministic:
    """A leaked target would be near-deterministic. These features are not."""

    def should_measureDefaultRate_withSevereDelinquency_stayWellBelowCertainty(
        self, merged: pd.DataFrame
    ):
        # Act
        severe = merged[merged["delinquency_ratio"] > 75]
        default_rate = severe["default"].mean()

        # Assert - ~58%, not ~100%: informative, not the answer
        assert 0.3 < default_rate < 0.8

    def should_measureDefaultRate_withNoDelinquency_remainNonZero(self, merged: pd.DataFrame):
        # Act
        clean = merged[merged["delinquent_months"] == 0]

        # Assert - clean applicants still default, so this is not the label
        assert clean["default"].mean() > 0.01

    def should_measureDefaultRate_acrossDelinquencyBuckets_increaseMonotonically(
        self, merged: pd.DataFrame
    ):
        # Arrange
        buckets = pd.cut(merged["delinquency_ratio"], [-0.1, 0, 10, 25, 50, 75, 100])

        # Act
        rates = merged.groupby(buckets, observed=True)["default"].mean().tolist()

        # Assert - a clean risk gradient, which is what a good predictor looks like
        assert rates == sorted(rates)

    def should_checkInvariant_withDelinquentMonths_neverExceedTotalMonths(
        self, merged: pd.DataFrame
    ):
        # Assert - internal consistency of the bureau table
        assert (merged["delinquent_months"] <= merged["total_loan_months"]).all()
