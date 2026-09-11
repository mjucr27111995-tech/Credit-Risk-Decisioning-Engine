"""Unit tests for the scoring service."""

from __future__ import annotations

import math

import numpy as np
import pytest

from credit_risk.domain.models import LoanApplication
from credit_risk.domain.scorecard import Scorecard
from credit_risk.errors import ScoringError
from credit_risk.scoring.artifact import EXPECTED_FEATURES, ModelArtifact
from credit_risk.scoring.features import FeatureVectorBuilder
from credit_risk.scoring.scorer import CreditRiskScorer, _sigmoid


class TestSigmoid:
    def should_computeSigmoid_withZero_returnHalf(self):
        assert _sigmoid(0.0) == pytest.approx(0.5)

    @pytest.mark.parametrize("log_odds", [-800.0, -50.0, -1.0, 0.0, 1.0, 50.0, 800.0])
    def should_computeSigmoid_withExtremeInput_stayWithinUnitInterval(self, log_odds: float):
        # Act - the naive formulation overflows here; this must not
        result = _sigmoid(log_odds)

        # Assert
        assert 0.0 <= result <= 1.0
        assert math.isfinite(result)

    def should_computeSigmoid_withModerateInput_matchReferenceFormula(self):
        assert _sigmoid(2.5) == pytest.approx(1 / (1 + math.exp(-2.5)))

    def should_computeSigmoid_withNegatedInput_returnComplement(self):
        assert _sigmoid(-3.0) == pytest.approx(1 - _sigmoid(3.0))


class TestScore:
    def should_score_withValidApplication_returnConsistentAssessment(
        self, scorer: CreditRiskScorer, valid_application: LoanApplication
    ):
        # Act
        assessment = scorer.score(valid_application)

        # Assert
        assert 0.0 <= assessment.probability_of_default <= 1.0
        assert scorer.scorecard.min_score <= assessment.credit_score <= scorer.scorecard.max_score
        assert assessment.rating is scorer.scorecard.rating_for(assessment.credit_score)
        assert assessment.recommended_action is scorer.scorecard.action_for(assessment.credit_score)
        assert assessment.model_version == "test-artifact"

    def should_score_withValidApplication_returnOneContributionPerFeature(
        self, scorer: CreditRiskScorer, valid_application: LoanApplication
    ):
        assessment = scorer.score(valid_application)
        assert len(assessment.contributions) == len(EXPECTED_FEATURES)
        assert tuple(item.feature for item in assessment.contributions) == EXPECTED_FEATURES

    def should_score_withValidApplication_returnContributionsThatReconstructLogOdds(
        self, scorer: CreditRiskScorer, valid_application: LoanApplication, fake_artifact
    ):
        # Act
        assessment = scorer.score(valid_application)

        # Assert - explainability must be exact, not indicative
        total = sum(item.contribution for item in assessment.contributions)
        expected = _sigmoid(total + fake_artifact.intercept)
        assert assessment.probability_of_default == pytest.approx(expected)

    def should_score_withValidApplication_computeContributionAsCoefficientTimesValue(
        self, scorer: CreditRiskScorer, valid_application: LoanApplication
    ):
        assessment = scorer.score(valid_application)
        for item in assessment.contributions:
            assert item.contribution == pytest.approx(item.coefficient * item.raw_value)

    def should_score_withRepeatedCall_returnDeterministicResult(
        self, scorer: CreditRiskScorer, valid_application: LoanApplication
    ):
        first = scorer.score(valid_application)
        second = scorer.score(valid_application)
        assert first.probability_of_default == second.probability_of_default
        assert first.credit_score == second.credit_score

    def should_score_withDistinctApplications_returnDistinctAssessmentIds(
        self,
        scorer: CreditRiskScorer,
        valid_application: LoanApplication,
        low_risk_application: LoanApplication,
    ):
        first = scorer.score(valid_application)
        second = scorer.score(low_risk_application)
        assert first.assessment_id != second.assessment_id


class TestScoreErrorHandling:
    def should_score_withMismatchedCoefficients_throwScoringError(
        self, fake_artifact: ModelArtifact, valid_application: LoanApplication
    ):
        # Arrange - estimator declares fewer coefficients than features
        class ShortEstimator:
            coef_ = np.arange(1, 4, dtype=float).reshape(1, -1)
            intercept_ = np.array([0.0])

        broken = ModelArtifact(
            model=ShortEstimator(),
            features=fake_artifact.features,
            scaler=fake_artifact.scaler,
            cols_to_scale=fake_artifact.cols_to_scale,
            version="broken",
        )
        broken_scorer = CreditRiskScorer(
            artifact=broken,
            feature_builder=FeatureVectorBuilder(broken),
            scorecard=Scorecard(),
        )

        # Act / Assert
        with pytest.raises(ScoringError) as exc_info:
            broken_scorer.score(valid_application)
        assert exc_info.value.code == "CRM-INT-002"

    def should_score_withUnexpectedFailure_wrapAsScoringError(
        self, fake_artifact: ModelArtifact, valid_application: LoanApplication
    ):
        # Arrange
        class ExplodingBuilder(FeatureVectorBuilder):
            def build(self, application: LoanApplication):
                raise RuntimeError("unexpected boom")

        broken_scorer = CreditRiskScorer(
            artifact=fake_artifact,
            feature_builder=ExplodingBuilder(fake_artifact),
            scorecard=Scorecard(),
        )

        # Act / Assert - raw exceptions must never escape the boundary
        with pytest.raises(ScoringError, match="Unexpected failure"):
            broken_scorer.score(valid_application)
