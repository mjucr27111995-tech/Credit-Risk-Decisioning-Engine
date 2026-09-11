"""Unit tests for the Altair chart builders.

The builders are pure functions, so they are verified by inspecting the chart
specification rather than by rendering a browser.
"""

from __future__ import annotations

import pytest

from credit_risk.domain.models import (
    FeatureContribution,
    RecommendedAction,
    RiskAssessment,
    RiskRating,
)
from credit_risk.domain.scorecard import Scorecard
from credit_risk.ui.charts import build_contribution_chart, build_score_band_chart


def _rows(spec: dict[str, object], layer_index: int | None = None) -> list[dict[str, object]]:
    """Resolve a chart's inline rows.

    Altair 6 hoists data into a top-level ``datasets`` map and leaves a named
    reference behind, so tests must dereference it rather than read
    ``data.values`` directly.
    """
    node = spec if layer_index is None else spec["layer"][layer_index]  # type: ignore[index]
    data = node["data"]  # type: ignore[index]
    if "values" in data:
        return list(data["values"])
    datasets = spec["datasets"]  # type: ignore[index]
    return list(datasets[data["name"]])


@pytest.fixture
def contributions() -> tuple[FeatureContribution, ...]:
    """Twelve contributions with mixed signs and magnitudes."""
    return tuple(
        FeatureContribution(
            feature=f"feature_{index}",
            display_name=f"Feature {index}",
            raw_value=0.5,
            coefficient=float(index) * (-1 if index % 2 else 1),
            contribution=float(index) * (-1 if index % 2 else 1),
        )
        for index in range(1, 13)
    )


@pytest.fixture
def assessment(contributions: tuple[FeatureContribution, ...]) -> RiskAssessment:
    """A Good-rated assessment carrying the contributions."""
    return RiskAssessment(
        probability_of_default=0.2,
        credit_score=780,
        rating=RiskRating.EXCELLENT,
        recommended_action=RecommendedAction.AUTO_APPROVE,
        contributions=contributions,
        model_version="test",
    )


class TestScoreBandChart:
    def should_buildScoreBandChart_withAssessment_returnLayeredSpec(
        self, assessment: RiskAssessment, scorecard: Scorecard
    ):
        # Act
        spec = build_score_band_chart(assessment, scorecard).to_dict()

        # Assert - bands + marker rule + score label
        assert len(spec["layer"]) == 3

    def should_buildScoreBandChart_withAssessment_includeEveryBand(
        self, assessment: RiskAssessment, scorecard: Scorecard
    ):
        # Act
        spec = build_score_band_chart(assessment, scorecard).to_dict()

        # Assert
        ratings = {row["rating"] for row in _rows(spec, 0)}
        assert ratings == {str(band.rating) for band in scorecard.bands}

    def should_buildScoreBandChart_withAssessment_markScorePosition(
        self, assessment: RiskAssessment, scorecard: Scorecard
    ):
        spec = build_score_band_chart(assessment, scorecard).to_dict()
        assert _rows(spec, 1)[0]["score"] == 780

    def should_buildScoreBandChart_withCustomCalibration_spanFullRange(
        self, assessment: RiskAssessment
    ):
        # Arrange
        card = Scorecard(base_score=0, scale_length=1000)

        # Act
        spec = build_score_band_chart(assessment, card).to_dict()

        # Assert
        assert spec["layer"][0]["encoding"]["x"]["scale"]["domain"] == [0, 1001]


class TestContributionChart:
    def should_buildContributionChart_withManyFeatures_respectLimit(
        self, contributions: tuple[FeatureContribution, ...]
    ):
        # Act
        spec = build_contribution_chart(contributions, limit=5).to_dict()

        # Assert
        assert len(_rows(spec)) == 5

    def should_buildContributionChart_withManyFeatures_orderByAbsoluteInfluence(
        self, contributions: tuple[FeatureContribution, ...]
    ):
        # Act
        rows = _rows(build_contribution_chart(contributions, limit=4).to_dict())

        # Assert
        magnitudes = [abs(row["contribution"]) for row in rows]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def should_buildContributionChart_withMixedSigns_labelDirection(
        self, contributions: tuple[FeatureContribution, ...]
    ):
        # Act
        rows = _rows(build_contribution_chart(contributions).to_dict())

        # Assert
        for row in rows:
            expected = "Increases risk" if row["contribution"] > 0 else "Reduces risk"
            assert row["direction"] == expected

    def should_buildContributionChart_withSingleFeature_keepMinimumHeight(self):
        # Arrange
        single = (
            FeatureContribution(
                feature="age",
                display_name="Age",
                raw_value=1.0,
                coefficient=0.5,
                contribution=0.5,
            ),
        )

        # Act
        spec = build_contribution_chart(single).to_dict()

        # Assert - the minimum floor applies below ~6 features
        assert spec["height"] == 160
        assert len(_rows(spec)) == 1
