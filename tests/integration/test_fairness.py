"""Fairness regression tests for the shipped scorecard.

These encode the conclusions of ``docs/FAIRNESS_AUDIT.md`` as executable
assertions, so a future change to features or the artifact cannot silently
introduce disparate impact or degrade calibration without a test failing.

Read-only: nothing here modifies the model, the features, or the scoring path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

pytestmark = pytest.mark.integration

APPROVAL_CUTOFF = 650
# Four-fifths rule: a US EEOC convention used here as an industry benchmark.
# Not an Indian legal standard - see the jurisdiction caveat in the audit doc.
MIN_IMPACT_RATIO = 0.80
PROXY_AUC_CEILING = 0.75


def _impact_ratios(df: pd.DataFrame, attr: str) -> pd.Series:
    """Approval rate per group, divided by the best group's approval rate."""
    approved = df["score"] >= APPROVAL_CUTOFF
    rates = approved.groupby(df[attr], observed=True).mean()
    return rates / rates.max()


class TestDisparateImpact:
    """Every group must clear the four-fifths threshold."""

    @pytest.mark.parametrize("attr", ["gender", "marital_status", "employment_status", "city"])
    def should_measureApprovalRates_withProtectedAttribute_clearFourFifthsRule(
        self, scored_population: pd.DataFrame, attr: str
    ) -> None:
        # Act
        ratios = _impact_ratios(scored_population, attr)

        # Assert
        assert ratios.min() >= MIN_IMPACT_RATIO, (
            f"{attr} worst impact ratio {ratios.min():.3f} < {MIN_IMPACT_RATIO}; "
            f"per-group ratios:\n{ratios.sort_values().to_string()}"
        )

    def should_measureApprovalRates_withAgeBand_clearFourFifthsRule(
        self, scored_population: pd.DataFrame
    ) -> None:
        # Arrange - age is a model feature, so it is the likeliest to disparately impact
        df = scored_population.copy()
        df["age_band"] = pd.cut(df["age"], [17, 25, 35, 45, 55, 100])

        # Act
        ratios = _impact_ratios(df, "age_band")

        # Assert
        assert ratios.min() >= MIN_IMPACT_RATIO

    def should_comparePredictiveAccuracy_acrossGenders_stayEquivalent(
        self, scored_population: pd.DataFrame
    ) -> None:
        # Act - equal AUC means the score is equally trustworthy for both groups
        aucs = scored_population.groupby("gender", observed=True).apply(
            lambda g: roc_auc_score(g["default"], g["pd"]), include_groups=False
        )

        # Assert
        assert abs(aucs.max() - aucs.min()) < 0.02


class TestAgeDisparityIsRiskJustified:
    """Age differentiation must track measured risk, not amplify it."""

    def should_compareApprovalGap_withRiskGap_notAmplifyDisparity(
        self, scored_population: pd.DataFrame
    ) -> None:
        # Arrange
        df = scored_population.copy()
        df["age_band"] = pd.cut(df["age"], [17, 25, 35, 45, 55, 100])
        grouped = df.groupby("age_band", observed=True)
        stats = grouped.agg(
            approval=("score", lambda s: (s >= APPROVAL_CUTOFF).mean()),
            actual_default=("default", "mean"),
        )

        # Act - relative risk spread vs relative approval spread
        risk_spread = stats["actual_default"].max() / stats["actual_default"].min()
        approval_shortfall = 1 - (stats["approval"].min() / stats["approval"].max())

        # Assert - 2.7x risk spread should not be met by a larger approval penalty
        assert risk_spread > 2.0
        assert approval_shortfall < 0.20


class TestProxyDiscrimination:
    """Excluded attributes must not be reconstructible from the features."""

    @pytest.mark.parametrize("attr", ["gender", "marital_status", "employment_status"])
    def should_reconstructProtectedAttribute_fromModelFeatures_findWeakSignal(
        self, scored_population: pd.DataFrame, attr: str
    ) -> None:
        # fairness_audit becomes importable via the scored_population fixture,
        # which puts scripts/ on sys.path only when the datasets are present.
        from fairness_audit import proxy_strength  # noqa: PLC0415

        # Act
        auc = proxy_strength(scored_population, attr)

        # Assert
        assert auc < PROXY_AUC_CEILING, f"{attr} reconstructible from features at {auc:.3f}"

    def should_reconstructGender_fromModelFeatures_findNoSignalAtAll(
        self, scored_population: pd.DataFrame
    ) -> None:
        from fairness_audit import proxy_strength  # noqa: PLC0415

        # Act
        auc = proxy_strength(scored_population, "gender")

        # Assert - gender is genuinely absent, not merely excluded by name
        assert auc < 0.55


class TestConditionalParity:
    """A given score must mean the same thing for every group."""

    @pytest.mark.parametrize("attr", ["gender", "marital_status", "employment_status"])
    def should_compareDefaultRates_withinScoreBand_stayCloseAcrossGroups(
        self, scored_population: pd.DataFrame, attr: str
    ) -> None:
        # Arrange
        df = scored_population.copy()
        df["band"] = pd.cut(df["score"], [299, 499, 649, 749, 900])

        # Act
        table = df.pivot_table(
            index="band", columns=attr, values="default", aggfunc="mean", observed=True
        )
        largest_gap = (table.iloc[:, 0] - table.iloc[:, 1]).abs().max()

        # Assert
        assert largest_gap < 0.05, f"{attr} default-rate gap {largest_gap:.4f} within a band"


class TestCalibrationDefect:
    """Pin the known miscalibration so a future fix is a visible, deliberate change.

    The model overpredicts PD by ~1.78x because it was fitted on a class-balanced
    sample. This is documented in ``docs/FAIRNESS_AUDIT.md`` and deliberately NOT
    corrected, to preserve numeric parity with the prototype. If someone applies
    the prior correction, these tests fail loudly and the docs must be updated.
    """

    def should_comparePredictedRate_withActualRate_remainOverpredictedAsDocumented(
        self, scored_population: pd.DataFrame
    ) -> None:
        # Act
        ratio = scored_population["pd"].mean() / scored_population["default"].mean()

        # Assert - documented as 1.78x
        assert 1.7 < ratio < 1.9, (
            f"Calibration ratio moved to {ratio:.2f}x. If this was intentional, "
            "update docs/FAIRNESS_AUDIT.md and the parity tests."
        )

    def should_measureRankOrdering_despiteMiscalibration_remainStrong(
        self, scored_population: pd.DataFrame
    ) -> None:
        # Act - miscalibration shifts probabilities but must not harm ranking
        auc = roc_auc_score(scored_population["default"], scored_population["pd"])

        # Assert
        assert auc > 0.97

    def should_measureDefaultRate_acrossScoreBands_decreaseMonotonically(
        self, scored_population: pd.DataFrame
    ) -> None:
        # Arrange
        df = scored_population.copy()
        df["band"] = pd.cut(df["score"], [299, 499, 649, 749, 900])

        # Act
        rates = df.groupby("band", observed=True)["default"].mean().tolist()

        # Assert - higher score must always mean lower risk
        assert rates == sorted(rates, reverse=True)


class TestBandSaturation:
    """Pin the band-distribution finding that makes a naive fix unsafe.

    The scorecard is a linear map of a heavily skewed PD distribution, so the
    top band is already saturated. A prior-correction intercept shift would push
    ~90% of applicants into `Excellent`. Documented in ``docs/FAIRNESS_AUDIT.md``.
    """

    def should_measureScoreDistribution_withRealPopulation_showTopBandSaturated(
        self, scored_population: pd.DataFrame
    ) -> None:
        # Act
        top_band_share = (scored_population["score"] >= 750).mean()

        # Assert - documented as 81.4%
        assert top_band_share > 0.75, (
            f"Top-band share moved to {top_band_share:.1%}. If band cutoffs or "
            "calibration changed, update docs/FAIRNESS_AUDIT.md."
        )

    def should_measureMedianPd_withRealPopulation_remainNearZero(
        self, scored_population: pd.DataFrame
    ) -> None:
        # Act - the skew that drives the saturation above
        median_pd = scored_population["pd"].median()

        # Assert
        assert median_pd < 0.01

    def should_identifyPromotedApplicants_underPriorCorrection_beHigherRisk(
        self, scored_population: pd.DataFrame
    ) -> None:
        # Arrange - apply the King & Zeng shift in-test, without changing the app
        df = scored_population
        tau = df["default"].mean()
        shift = np.log((1 - tau) / tau)
        clipped = df["pd"].clip(1e-12, 1 - 1e-12)
        corrected = 1 / (1 + np.exp(-(np.log(clipped / (1 - clipped)) - shift)))
        corrected_score = (300 + (1 - corrected) * 600).round()

        # Act
        promoted = df[(df["score"] < APPROVAL_CUTOFF) & (corrected_score >= APPROVAL_CUTOFF)]

        # Assert - they default well above portfolio average, so the naive fix
        # would approve genuinely riskier applicants
        assert promoted["default"].mean() > 2 * tau
