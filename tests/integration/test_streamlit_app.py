"""End-to-end tests driving the real Streamlit app via ``AppTest``.

These exercise the composition root, the form, validation and rendering in one
pass - the closest thing to a browser without launching one.
"""

from __future__ import annotations

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from credit_risk.config import get_settings
from tests.conftest import PROJECT_ROOT, REAL_ARTIFACT_PATH

pytestmark = pytest.mark.integration

_APP_PATH = PROJECT_ROOT / "app.py"
_TIMEOUT_SECONDS = 60


@pytest.fixture
def app() -> AppTest:
    """A freshly initialised app instance."""
    if not REAL_ARTIFACT_PATH.is_file():
        pytest.skip(f"Model artifact not present at {REAL_ARTIFACT_PATH}")
    instance = AppTest.from_file(str(_APP_PATH), default_timeout=_TIMEOUT_SECONDS)
    return instance.run()


class TestInitialRender:
    def should_renderApp_withFirstLoad_raiseNoException(self, app: AppTest):
        assert not app.exception

    def should_renderApp_withFirstLoad_showEveryInputControl(self, app: AppTest):
        # Assert - 8 numeric inputs and 3 category selectors
        assert len(app.number_input) == 8
        assert len(app.selectbox) == 3

    def should_renderApp_withFirstLoad_showSubmitButton(self, app: AppTest):
        assert len(app.button) == 1
        assert "Assess" in app.button[0].label

    def should_renderApp_withFirstLoad_showRatingBandReference(self, app: AppTest):
        # Assert - the empty state lists the bands so officers know the scale
        assert len(app.dataframe) == 1
        assert list(app.dataframe[0].value["Rating"]) == [
            "Excellent",
            "Good",
            "Average",
            "Poor",
        ]

    def should_renderApp_withFirstLoad_populateSidebarModelCard(self, app: AppTest):
        markdown = " ".join(block.value for block in app.sidebar.markdown)
        assert "Model card" in markdown
        assert "Logistic regression" in markdown


class TestAssessmentFlow:
    def should_submitForm_withPrototypeDefaults_renderAssessment(self, app: AppTest):
        # Act
        app.button[0].click().run()

        # Assert
        assert not app.exception
        rendered = " ".join(block.value for block in app.markdown)
        assert "Probability of default" in rendered
        assert "Indicative credit score" in rendered
        assert "Recommendation" in rendered

    def should_submitForm_withPrototypeDefaults_renderBothCharts(self, app: AppTest):
        # Act
        app.button[0].click().run()

        # Assert - score band chart + contribution chart
        assert len(app.get("vega_lite_chart")) == 2

    def should_submitForm_withCleanApplicant_renderExcellentRating(self, app: AppTest):
        # Arrange - a low-risk profile
        for widget in app.number_input:
            if widget.label.startswith("Delinquency"):
                widget.set_value(0.0)
            elif widget.label.startswith("Credit utilisation"):
                widget.set_value(5.0)
            elif widget.label.startswith("Average DPD"):
                widget.set_value(0.0)
            elif widget.label.startswith("Loan amount"):
                widget.set_value(500_000)
            elif widget.label.startswith("Annual income"):
                widget.set_value(4_000_000)

        # Act
        app.button[0].click().run()

        # Assert
        assert not app.exception
        rendered = " ".join(block.value for block in app.markdown)
        assert "Excellent" in rendered

    def should_submitForm_withDistressedApplicant_renderPoorRating(self, app: AppTest):
        # Arrange
        for widget in app.number_input:
            if widget.label.startswith("Delinquency"):
                widget.set_value(95.0)
            elif widget.label.startswith("Credit utilisation"):
                widget.set_value(98.0)
            elif widget.label.startswith("Average DPD"):
                widget.set_value(90.0)
            elif widget.label.startswith("Loan amount"):
                widget.set_value(3_000_000)
            elif widget.label.startswith("Annual income"):
                widget.set_value(400_000)

        # Act
        app.button[0].click().run()

        # Assert
        assert not app.exception
        rendered = " ".join(block.value for block in app.markdown)
        assert "Poor" in rendered

    def should_submitForm_withSecuredHomeLoan_renderWithoutError(self, app: AppTest):
        # Arrange - exercise the model's reference categories
        app.selectbox(key="residence_type").set_value("Mortgage")
        app.selectbox(key="loan_purpose").set_value("Auto")
        app.selectbox(key="loan_type").set_value("Secured")

        # Act
        app.button[0].click().run()

        # Assert
        assert not app.exception

    def should_submitForm_withRepeatedRuns_keepCorrelationIdStable(self, app: AppTest):
        # Act
        app.button[0].click().run()
        first = app.session_state["crm_correlation_id"]
        app.button[0].click().run()

        # Assert - one correlation id per session, never regenerated
        assert app.session_state["crm_correlation_id"] == first


class TestFailureModes:
    def should_renderApp_withMissingArtifact_showTypedErrorCode(self, monkeypatch):
        # Arrange - settings and the scorer are both cached, so clear each
        monkeypatch.setenv("CRM_ARTIFACT_PATH", "/nonexistent/model.joblib")
        get_settings.cache_clear()
        st.cache_resource.clear()

        # Act
        instance = AppTest.from_file(str(_APP_PATH), default_timeout=_TIMEOUT_SECONDS).run()

        # Assert - fails visibly and safely, with no stack trace leaked
        try:
            assert not instance.exception
            assert any("CRM-NOT-001" in block.value for block in instance.error)
        finally:
            get_settings.cache_clear()
            st.cache_resource.clear()
