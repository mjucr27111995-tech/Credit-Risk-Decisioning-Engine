"""Model artifact loading and schema validation.

The bundle produced by the training notebook is a dict of
``{model, features, scaler, cols_to_scale}``. Deserialisation is a trust
boundary, so the file is validated for existence, structure and feature
contract before any prediction is attempted.

Security note: joblib uses pickle, which executes arbitrary code on load. The
artifact is therefore treated as trusted first-party code shipped with the
application - never load a bundle supplied by an end user.
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, cast

import joblib
import numpy as np
import pandas as pd

from credit_risk.errors import (
    ArtifactLoadError,
    ArtifactNotFoundError,
    ArtifactSchemaError,
    CreditRiskError,
)
from credit_risk.observability import (
    SERVICE_NAME,
    EventName,
    EventStatus,
    get_logger,
    log_event,
)


class LinearEstimator(Protocol):
    """The slice of a fitted linear classifier this application relies on."""

    coef_: np.ndarray
    intercept_: np.ndarray


class Scaler(Protocol):
    """The slice of a fitted scikit-learn scaler this application relies on."""

    def transform(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803 - sklearn API
        """Scale a frame of raw feature values."""
        ...


_REQUIRED_KEYS: Final[frozenset[str]] = frozenset({"model", "features", "scaler", "cols_to_scale"})

# The exact feature contract this application was written against. A mismatch
# means the artifact and the code have diverged: fail loudly (CRM-CON-001)
# rather than silently scoring against the wrong columns.
EXPECTED_FEATURES: Final[tuple[str, ...]] = (
    "age",
    "loan_tenure_months",
    "number_of_open_accounts",
    "credit_utilization_ratio",
    "loan_to_income",
    "delinquency_ratio",
    "avg_dpd_per_delinquency",
    "residence_type_Owned",
    "residence_type_Rented",
    "loan_purpose_Education",
    "loan_purpose_Home",
    "loan_purpose_Personal",
    "loan_type_Unsecured",
)


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    """A validated, ready-to-use model bundle.

    Attributes:
        model: Fitted logistic regression.
        features: Model input columns, in order.
        scaler: Fitted MinMaxScaler.
        cols_to_scale: Columns the scaler expects, in order.
        version: Short content hash of the artifact file, used as model version.
    """

    model: LinearEstimator
    features: tuple[str, ...]
    scaler: Scaler
    cols_to_scale: tuple[str, ...]
    version: str

    @property
    def coefficients(self) -> np.ndarray:
        """Model coefficients as a 1-D array."""
        return np.asarray(self.model.coef_).ravel()

    @property
    def intercept(self) -> float:
        """Model intercept."""
        return float(np.asarray(self.model.intercept_).ravel()[0])

    @property
    def coefficient_map(self) -> dict[str, float]:
        """Feature name -> coefficient."""
        return dict(zip(self.features, (float(c) for c in self.coefficients), strict=True))


def _content_version(path: Path) -> str:
    """Derive a stable version string from the artifact's contents."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return f"lr-{digest}"


def _validate_structure(raw: object, path: Path) -> dict[str, object]:
    """Ensure the deserialised object is a bundle with the required keys."""
    if not isinstance(raw, dict):
        raise ArtifactSchemaError(
            f"Model artifact at '{path}' must deserialise to a dict, got {type(raw).__name__}"
        )
    missing = _REQUIRED_KEYS - raw.keys()
    if missing:
        raise ArtifactSchemaError(
            f"Model artifact at '{path}' is missing required keys: {sorted(missing)}"
        )
    return raw


def _validate_feature_contract(features: tuple[str, ...], path: Path) -> None:
    """Ensure the artifact's features match the contract this code implements."""
    if features != EXPECTED_FEATURES:
        raise ArtifactSchemaError(
            f"Model artifact at '{path}' does not match the expected feature contract. "
            f"Expected {list(EXPECTED_FEATURES)}, got {list(features)}"
        )


def _validate_model(model: object, feature_count: int, path: Path) -> LinearEstimator:
    """Ensure the estimator is linear and dimensionally consistent.

    Returns:
        The same object, narrowed to :class:`LinearEstimator`.
    """
    if not hasattr(model, "coef_") or not hasattr(model, "intercept_"):
        raise ArtifactSchemaError(
            f"Model artifact at '{path}' must contain a linear estimator exposing "
            f"'coef_' and 'intercept_'; got {type(model).__name__}"
        )
    estimator = cast(LinearEstimator, model)
    coef_count = int(np.asarray(estimator.coef_).ravel().size)
    if coef_count != feature_count:
        raise ArtifactSchemaError(
            f"Model artifact at '{path}' has {coef_count} coefficients "
            f"but declares {feature_count} features"
        )
    return estimator


def _validate_scaler(scaler: object, path: Path) -> Scaler:
    """Ensure the scaler exposes the ``transform`` method the builder calls.

    Returns:
        The same object, narrowed to :class:`Scaler`.
    """
    if not callable(getattr(scaler, "transform", None)):
        raise ArtifactSchemaError(
            f"Model artifact at '{path}' must contain a scaler exposing 'transform'; "
            f"got {type(scaler).__name__}"
        )
    return cast(Scaler, scaler)


def _log_load_warnings(path: Path, caught: list[warnings.WarningMessage]) -> None:
    """Report deserialisation warnings once each, loudest first.

    Version-skew warnings matter: a bundle pickled by a different scikit-learn
    release can silently mis-predict, so they are surfaced explicitly rather
    than suppressed. Duplicates are collapsed to keep the log readable.
    """
    if not caught:
        return
    logger = get_logger()
    seen: set[str] = set()
    for warning in caught:
        detail = str(warning.message).strip()
        first_line = detail.splitlines()[0] if detail else ""
        if first_line in seen:
            continue
        seen.add(first_line)
        is_version_skew = "version" in first_line.lower()
        log = logger.warning if is_version_skew else logger.debug
        log(
            f"{SERVICE_NAME} | Artifact deserialisation warning",
            artifact=path.name,
            category=warning.category.__name__,
            detail=first_line,
            versionSkew=is_version_skew,
        )


def _fail_load(path: Path, error: CreditRiskError) -> None:
    """Emit the failure event for an artifact that could not be loaded."""
    log_event(
        get_logger(),
        event_name=EventName.MODEL_ARTIFACT_LOAD,
        status=EventStatus.FAILURE,
        domain_id=path.name,
        domain_type="MODEL_ARTIFACT",
        status_message=error.message,
        errorCode=error.code,
    )


def _deserialise(path: Path) -> object:
    """Read the bundle from disk, converting any failure into a typed error.

    Raises:
        ArtifactNotFoundError: ``CRM-NOT-001`` if the file is absent.
        ArtifactLoadError: ``CRM-EXT-001`` if it cannot be deserialised.
    """
    if not path.is_file():
        missing = ArtifactNotFoundError(f"Model artifact not found at '{path}'")
        _fail_load(path, missing)
        raise missing

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            raw: object = joblib.load(path)
    except Exception as exc:  # noqa: BLE001 - pickle can raise almost anything
        # Deserialisation is a trust boundary: a truncated or hostile payload
        # surfaces as IndexError, UnpicklingError, ImportError, OSError and
        # more. Catch broadly, re-raise as one typed error, keep the cause.
        unreadable = ArtifactLoadError(f"Failed to deserialise model artifact at '{path}': {exc}")
        _fail_load(path, unreadable)
        raise unreadable from exc

    _log_load_warnings(path, caught)
    return raw


def _names(bundle: dict[str, object], key: str, path: Path) -> tuple[str, ...]:
    """Read a sequence of column names from the bundle.

    Raises:
        ArtifactSchemaError: ``CRM-CON-001`` if the value is not a sequence.
    """
    value = bundle[key]
    if not isinstance(value, Sequence | np.ndarray | pd.Index):
        raise ArtifactSchemaError(
            f"Model artifact at '{path}' has a non-sequence '{key}': {type(value).__name__}"
        )
    return tuple(str(name) for name in value)


def load_artifact(path: Path) -> ModelArtifact:
    """Load and validate the model bundle from disk.

    Args:
        path: Absolute path to the ``.joblib`` bundle.

    Returns:
        A validated :class:`ModelArtifact`.

    Raises:
        ArtifactNotFoundError: ``CRM-NOT-001`` if the file is absent.
        ArtifactLoadError: ``CRM-EXT-001`` if it cannot be deserialised.
        ArtifactSchemaError: ``CRM-CON-001`` if it violates the contract.
    """
    bundle = _validate_structure(_deserialise(path), path)
    features = _names(bundle, "features", path)
    cols_to_scale = _names(bundle, "cols_to_scale", path)
    _validate_feature_contract(features, path)

    artifact = ModelArtifact(
        model=_validate_model(bundle["model"], len(features), path),
        features=features,
        scaler=_validate_scaler(bundle["scaler"], path),
        cols_to_scale=cols_to_scale,
        version=_content_version(path),
    )
    log_event(
        get_logger(),
        event_name=EventName.MODEL_ARTIFACT_LOAD,
        status=EventStatus.SUCCESS,
        domain_id=artifact.version,
        domain_type="MODEL_ARTIFACT",
        status_message=f"Loaded {len(features)} features from '{path.name}'",
        featureCount=len(features),
        scaledColumnCount=len(cols_to_scale),
    )
    return artifact
