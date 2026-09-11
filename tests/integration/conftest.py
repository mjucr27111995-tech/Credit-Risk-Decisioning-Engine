"""Shared fixtures for integration tests that need the source datasets."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _PROJECT_ROOT / "scripts"
_DATA = _PROJECT_ROOT.parent / "Project2_DataCollection_Resources"
_REQUIRED = ("customers.csv", "loans.csv", "bureau_data.csv")


@pytest.fixture(scope="session")
def scored_population() -> pd.DataFrame:
    """The full dataset scored by the shipped artifact.

    Reuses ``scripts/fairness_audit.py`` so the audit document and these tests
    can never disagree about how the population was scored.
    """
    if not all((_DATA / name).is_file() for name in _REQUIRED):
        pytest.skip(f"Source datasets not present in {_DATA}")
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))

    from fairness_audit import load_scored  # noqa: PLC0415 - optional, data-dependent

    return load_scored()
