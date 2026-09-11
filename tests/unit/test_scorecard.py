"""Unit tests for the scorecard transform and rating bands."""

from __future__ import annotations

import pytest

from credit_risk.domain.models import RecommendedAction, RiskRating
from credit_risk.domain.scorecard import Scorecard


class TestToCreditScore:
    def should_convertProbability_withZeroRisk_returnMaximumScore(self, scorecard: Scorecard):
        # Arrange / Act
        score = scorecard.to_credit_score(0.0)

        # Assert
        assert score == 900

    def should_convertProbability_withCertainDefault_returnMinimumScore(self, scorecard: Scorecard):
        assert scorecard.to_credit_score(1.0) == 300

    def should_convertProbability_withHalfRisk_returnMidpointScore(self, scorecard: Scorecard):
        assert scorecard.to_credit_score(0.5) == 600

    def should_convertProbability_withKnownValue_matchLegacyTransform(self, scorecard: Scorecard):
        # Arrange - transform used by the approved training notebook
        probability = 0.6654998816431852

        # Act
        score = scorecard.to_credit_score(probability)

        # Assert
        assert score == int(300 + (1 - probability) * 600)

    @pytest.mark.parametrize("probability", [-0.01, 1.01, float("nan")])
    def should_convertProbability_withOutOfRangeValue_throwValueError(
        self, scorecard: Scorecard, probability: float
    ):
        with pytest.raises(ValueError, match="probability_of_default"):
            scorecard.to_credit_score(probability)

    def should_convertProbability_withDescendingRisk_returnMonotonicScores(
        self, scorecard: Scorecard
    ):
        # Arrange
        probabilities = [0.9, 0.7, 0.5, 0.3, 0.1]

        # Act
        scores = [scorecard.to_credit_score(value) for value in probabilities]

        # Assert
        assert scores == sorted(scores)


class TestRatingBands:
    @pytest.mark.parametrize(
        ("score", "expected"),
        [
            (300, RiskRating.POOR),
            (499, RiskRating.POOR),
            (500, RiskRating.AVERAGE),
            (649, RiskRating.AVERAGE),
            (650, RiskRating.GOOD),
            (749, RiskRating.GOOD),
            (750, RiskRating.EXCELLENT),
            (900, RiskRating.EXCELLENT),
        ],
    )
    def should_rateScore_withBandEdge_returnExpectedRating(
        self, scorecard: Scorecard, score: int, expected: RiskRating
    ):
        assert scorecard.rating_for(score) is expected

    def should_buildBands_withDefaultCalibration_returnContiguousCoverage(
        self, scorecard: Scorecard
    ):
        # Act
        bands = scorecard.bands

        # Assert - no gaps, no overlaps, full range covered
        assert bands[0].lower == 300
        assert bands[-1].upper == 900
        for lower, upper in zip(bands, bands[1:], strict=False):
            assert upper.lower == lower.upper + 1

    def should_rateScore_withScoreBelowRange_clampToPoor(self, scorecard: Scorecard):
        assert scorecard.rating_for(-50) is RiskRating.POOR

    def should_rateScore_withScoreAboveRange_clampToExcellent(self, scorecard: Scorecard):
        assert scorecard.rating_for(5_000) is RiskRating.EXCELLENT

    @pytest.mark.parametrize(
        ("score", "expected"),
        [
            (350, RecommendedAction.AUTO_DECLINE),
            (600, RecommendedAction.REFER_TO_UNDERWRITER),
            (700, RecommendedAction.APPROVE_WITH_REVIEW),
            (880, RecommendedAction.AUTO_APPROVE),
        ],
    )
    def should_recommendAction_withScore_returnBandAction(
        self, scorecard: Scorecard, score: int, expected: RecommendedAction
    ):
        assert scorecard.action_for(score) is expected

    def should_labelBand_withGoodBand_returnRangeText(self, scorecard: Scorecard):
        good = scorecard.band_for(700)
        assert good.range_label == "650-749"
        assert good.label == "Good (650-749)"


class TestCalibration:
    def should_construct_withCustomRange_rescaleBands(self):
        # Arrange / Act
        card = Scorecard(base_score=0, scale_length=1000)

        # Assert
        assert card.min_score == 0
        assert card.max_score == 1000
        assert card.bands[1].lower == 333
        assert card.rating_for(1000) is RiskRating.EXCELLENT

    def should_construct_withTooSmallScale_throwValueError(self):
        with pytest.raises(ValueError, match="scale_length"):
            Scorecard(base_score=300, scale_length=2)
