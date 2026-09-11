"""Unit tests for artifact loading, validation and the error hierarchy."""

from __future__ import annotations

from http import HTTPStatus
from pathlib import Path

import joblib
import numpy as np
import pytest

from credit_risk.errors import (
    ArtifactLoadError,
    ArtifactNotFoundError,
    ArtifactSchemaError,
    ConfigurationError,
    CreditRiskError,
    ScoringError,
    ValidationError,
)
from credit_risk.scoring.artifact import EXPECTED_FEATURES, load_artifact
from credit_risk.scoring.labels import humanise
from tests.conftest import FakeEstimator, IdentityScaler


def _write_bundle(path: Path, bundle: object) -> Path:
    """Persist a bundle to disk for loader tests."""
    joblib.dump(bundle, path)
    return path


def _valid_bundle(scaled_columns: tuple[str, ...]) -> dict[str, object]:
    """A bundle that satisfies the full contract."""
    return {
        "model": FakeEstimator(np.ones(len(EXPECTED_FEATURES)), -1.0),
        "features": list(EXPECTED_FEATURES),
        "scaler": IdentityScaler(scaled_columns),
        "cols_to_scale": list(scaled_columns),
    }


class TestLoadArtifact:
    def should_load_withValidBundle_returnArtifact(
        self, tmp_path: Path, scaled_columns: tuple[str, ...]
    ):
        # Arrange
        path = _write_bundle(tmp_path / "model.joblib", _valid_bundle(scaled_columns))

        # Act
        artifact = load_artifact(path)

        # Assert
        assert artifact.features == EXPECTED_FEATURES
        assert artifact.intercept == pytest.approx(-1.0)
        assert artifact.version.startswith("lr-")

    def should_load_withValidBundle_deriveStableVersion(
        self, tmp_path: Path, scaled_columns: tuple[str, ...]
    ):
        path = _write_bundle(tmp_path / "model.joblib", _valid_bundle(scaled_columns))
        assert load_artifact(path).version == load_artifact(path).version

    def should_load_withMissingFile_throwArtifactNotFoundError(self, tmp_path: Path):
        with pytest.raises(ArtifactNotFoundError) as exc_info:
            load_artifact(tmp_path / "absent.joblib")
        assert exc_info.value.code == "CRM-NOT-001"
        assert exc_info.value.status_code == HTTPStatus.NOT_FOUND

    def should_load_withCorruptFile_throwArtifactLoadError(self, tmp_path: Path):
        # Arrange
        path = tmp_path / "corrupt.joblib"
        path.write_bytes(b"this is not a joblib payload")

        # Act / Assert
        with pytest.raises(ArtifactLoadError) as exc_info:
            load_artifact(path)
        assert exc_info.value.code == "CRM-EXT-001"

    def should_load_withNonDictPayload_throwArtifactSchemaError(self, tmp_path: Path):
        path = _write_bundle(tmp_path / "list.joblib", ["not", "a", "bundle"])
        with pytest.raises(ArtifactSchemaError, match="must deserialise to a dict"):
            load_artifact(path)

    def should_load_withMissingKeys_throwArtifactSchemaError(
        self, tmp_path: Path, scaled_columns: tuple[str, ...]
    ):
        # Arrange
        bundle = _valid_bundle(scaled_columns)
        del bundle["scaler"]
        path = _write_bundle(tmp_path / "partial.joblib", bundle)

        # Act / Assert
        with pytest.raises(ArtifactSchemaError) as exc_info:
            load_artifact(path)
        assert exc_info.value.code == "CRM-CON-001"
        assert "scaler" in exc_info.value.message

    def should_load_withUnexpectedFeatures_throwArtifactSchemaError(
        self, tmp_path: Path, scaled_columns: tuple[str, ...]
    ):
        # Arrange
        bundle = _valid_bundle(scaled_columns)
        bundle["features"] = ["age", "income"]
        path = _write_bundle(tmp_path / "drifted.joblib", bundle)

        # Act / Assert
        with pytest.raises(ArtifactSchemaError, match="feature contract"):
            load_artifact(path)

    def should_load_withNonLinearEstimator_throwArtifactSchemaError(
        self, tmp_path: Path, scaled_columns: tuple[str, ...]
    ):
        # Arrange
        bundle = _valid_bundle(scaled_columns)
        bundle["model"] = {"not": "an estimator"}
        path = _write_bundle(tmp_path / "nonlinear.joblib", bundle)

        # Act / Assert
        with pytest.raises(ArtifactSchemaError, match="linear estimator"):
            load_artifact(path)

    def should_load_withCoefficientCountMismatch_throwArtifactSchemaError(
        self, tmp_path: Path, scaled_columns: tuple[str, ...]
    ):
        # Arrange
        bundle = _valid_bundle(scaled_columns)
        bundle["model"] = FakeEstimator(np.ones(5), 0.0)
        path = _write_bundle(tmp_path / "mismatch.joblib", bundle)

        # Act / Assert
        with pytest.raises(ArtifactSchemaError, match="coefficients"):
            load_artifact(path)

    def should_load_withNonTransformingScaler_throwArtifactSchemaError(
        self, tmp_path: Path, scaled_columns: tuple[str, ...]
    ):
        # Arrange - an object without a callable `transform`
        bundle = _valid_bundle(scaled_columns)
        bundle["scaler"] = {"not": "a scaler"}
        path = _write_bundle(tmp_path / "noscaler.joblib", bundle)

        # Act / Assert
        with pytest.raises(ArtifactSchemaError, match="exposing 'transform'"):
            load_artifact(path)

    @pytest.mark.parametrize("key", ["features", "cols_to_scale"])
    def should_load_withNonSequenceColumnNames_throwArtifactSchemaError(
        self, tmp_path: Path, scaled_columns: tuple[str, ...], key: str
    ):
        # Arrange - a scalar where a sequence of column names is required
        bundle = _valid_bundle(scaled_columns)
        bundle[key] = 42
        path = _write_bundle(tmp_path / f"bad_{key}.joblib", bundle)

        # Act / Assert
        with pytest.raises(ArtifactSchemaError, match="non-sequence"):
            load_artifact(path)


class TestArtifactAccessors:
    def should_mapCoefficients_withValidArtifact_pairEveryFeature(
        self, tmp_path: Path, scaled_columns: tuple[str, ...]
    ):
        # Arrange
        bundle = _valid_bundle(scaled_columns)
        bundle["model"] = FakeEstimator(np.arange(1, len(EXPECTED_FEATURES) + 1, dtype=float), 0.0)
        artifact = load_artifact(_write_bundle(tmp_path / "m.joblib", bundle))

        # Act
        mapping = artifact.coefficient_map

        # Assert
        assert set(mapping) == set(EXPECTED_FEATURES)
        assert mapping["age"] == pytest.approx(1.0)


class TestHumanise:
    def should_humanise_withKnownFeature_returnCuratedLabel(self):
        assert humanise("loan_to_income") == "Loan-to-income ratio"

    def should_humanise_withUnknownFeature_returnFallbackLabel(self):
        assert humanise("some_new_field") == "Some new field"


class TestErrorHierarchy:
    @pytest.mark.parametrize(
        ("error", "code", "status"),
        [
            (ValidationError("bad"), "CRM-VAL-001", HTTPStatus.BAD_REQUEST),
            (ArtifactNotFoundError("gone"), "CRM-NOT-001", HTTPStatus.NOT_FOUND),
            (ArtifactLoadError("broken"), "CRM-EXT-001", HTTPStatus.SERVICE_UNAVAILABLE),
            (ArtifactSchemaError("drift"), "CRM-CON-001", HTTPStatus.CONFLICT),
            (ConfigurationError("cfg"), "CRM-INT-001", HTTPStatus.INTERNAL_SERVER_ERROR),
            (ScoringError("boom"), "CRM-INT-002", HTTPStatus.INTERNAL_SERVER_ERROR),
        ],
    )
    def should_raiseError_withEachType_exposeCodeAndStatus(
        self, error: CreditRiskError, code: str, status: HTTPStatus
    ):
        assert error.code == code
        assert error.status_code == status
        assert isinstance(error, CreditRiskError)

    def should_serialiseError_withAnyError_returnStandardResponseShape(self):
        # Act
        payload = ValidationError("Age must be at least 18").to_dict()

        # Assert
        assert set(payload) == {"timestamp", "status", "error", "message"}
        assert payload["status"] == HTTPStatus.BAD_REQUEST
        assert payload["error"] == "CRM-VAL-001"

    def should_stringifyError_withAnyError_prefixWithCode(self):
        assert str(ScoringError("failed")) == "[CRM-INT-002] failed"
