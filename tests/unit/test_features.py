"""Unit tests for the feature vector contract."""

from __future__ import annotations

import pandas as pd
import pytest

from credit_risk.domain.models import LoanApplication, LoanPurpose, LoanType, ResidenceType
from credit_risk.errors import ScoringError
from credit_risk.scoring.artifact import EXPECTED_FEATURES, ModelArtifact
from credit_risk.scoring.features import (
    PLACEHOLDER_COLUMNS,
    PLACEHOLDER_VALUE,
    FeatureVectorBuilder,
)


class TestFeatureVectorShape:
    def should_build_withValidApplication_returnSingleRowInModelOrder(
        self, fake_artifact: ModelArtifact, valid_application: LoanApplication
    ):
        # Act
        frame = FeatureVectorBuilder(fake_artifact).build(valid_application)

        # Assert
        assert frame.shape == (1, len(EXPECTED_FEATURES))
        assert list(frame.columns) == list(EXPECTED_FEATURES)

    def should_build_withValidApplication_excludePlaceholderColumns(
        self, fake_artifact: ModelArtifact, valid_application: LoanApplication
    ):
        # Act
        frame = FeatureVectorBuilder(fake_artifact).build(valid_application)

        # Assert - placeholders exist only to satisfy the scaler
        assert not set(PLACEHOLDER_COLUMNS) & set(frame.columns)

    def should_build_withValidApplication_returnNumericValuesOnly(
        self, fake_artifact: ModelArtifact, valid_application: LoanApplication
    ):
        frame = FeatureVectorBuilder(fake_artifact).build(valid_application)
        assert frame.notna().all().all()
        assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes)


class TestCategoricalEncoding:
    @pytest.mark.parametrize(
        ("residence", "owned", "rented"),
        [
            (ResidenceType.OWNED, 1.0, 0.0),
            (ResidenceType.RENTED, 0.0, 1.0),
            (ResidenceType.MORTGAGE, 0.0, 0.0),
        ],
    )
    def should_encodeResidence_withEachCategory_returnDropFirstIndicators(
        self,
        fake_artifact: ModelArtifact,
        valid_application: LoanApplication,
        residence: ResidenceType,
        owned: float,
        rented: float,
    ):
        # Arrange
        application = valid_application.model_copy(update={"residence_type": residence})

        # Act
        frame = FeatureVectorBuilder(fake_artifact).build(application)

        # Assert
        assert frame["residence_type_Owned"].iloc[0] == owned
        assert frame["residence_type_Rented"].iloc[0] == rented

    @pytest.mark.parametrize(
        ("purpose", "education", "home", "personal"),
        [
            (LoanPurpose.EDUCATION, 1.0, 0.0, 0.0),
            (LoanPurpose.HOME, 0.0, 1.0, 0.0),
            (LoanPurpose.PERSONAL, 0.0, 0.0, 1.0),
            (LoanPurpose.AUTO, 0.0, 0.0, 0.0),
        ],
    )
    def should_encodePurpose_withEachCategory_returnDropFirstIndicators(
        self,
        fake_artifact: ModelArtifact,
        valid_application: LoanApplication,
        purpose: LoanPurpose,
        education: float,
        home: float,
        personal: float,
    ):
        application = valid_application.model_copy(update={"loan_purpose": purpose})
        frame = FeatureVectorBuilder(fake_artifact).build(application)

        assert frame["loan_purpose_Education"].iloc[0] == education
        assert frame["loan_purpose_Home"].iloc[0] == home
        assert frame["loan_purpose_Personal"].iloc[0] == personal

    @pytest.mark.parametrize(
        ("loan_type", "expected"),
        [(LoanType.UNSECURED, 1.0), (LoanType.SECURED, 0.0)],
    )
    def should_encodeLoanType_withEachCategory_returnIndicator(
        self,
        fake_artifact: ModelArtifact,
        valid_application: LoanApplication,
        loan_type: LoanType,
        expected: float,
    ):
        application = valid_application.model_copy(update={"loan_type": loan_type})
        frame = FeatureVectorBuilder(fake_artifact).build(application)
        assert frame["loan_type_Unsecured"].iloc[0] == expected


class TestPlaceholderContract:
    def should_buildRow_withAnyApplication_setPlaceholdersToLegacyConstant(
        self, valid_application: LoanApplication
    ):
        # Act - private helper covered deliberately: it is the debt boundary
        row = FeatureVectorBuilder._to_row(valid_application)

        # Assert
        for column in PLACEHOLDER_COLUMNS:
            assert row[column] == PLACEHOLDER_VALUE

    def should_buildRow_withAnyApplication_coverEveryScaledColumn(
        self, valid_application: LoanApplication, scaled_columns: tuple[str, ...]
    ):
        row = FeatureVectorBuilder._to_row(valid_application)
        assert set(scaled_columns) <= row.keys()


class TestFeatureBuilderErrors:
    def should_build_withScalerMissingColumn_throwScoringError(
        self, fake_artifact: ModelArtifact, valid_application: LoanApplication
    ):
        # Arrange - artifact demands a column the builder never produces
        broken = ModelArtifact(
            model=fake_artifact.model,
            features=fake_artifact.features,
            scaler=fake_artifact.scaler,
            cols_to_scale=(*fake_artifact.cols_to_scale, "unknown_column"),
            version="broken",
        )

        # Act / Assert
        with pytest.raises(ScoringError) as exc_info:
            FeatureVectorBuilder(broken).build(valid_application)
        assert exc_info.value.code == "CRM-INT-002"

    def should_build_withFailingScaler_throwScoringError(
        self, fake_artifact: ModelArtifact, valid_application: LoanApplication
    ):
        # Arrange
        class ExplodingScaler:
            def transform(self, frame: object) -> object:
                raise ValueError("scaler misconfigured")

        broken = ModelArtifact(
            model=fake_artifact.model,
            features=fake_artifact.features,
            scaler=ExplodingScaler(),
            cols_to_scale=fake_artifact.cols_to_scale,
            version="broken",
        )

        # Act / Assert
        with pytest.raises(ScoringError, match="Failed to scale features"):
            FeatureVectorBuilder(broken).build(valid_application)
