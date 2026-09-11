"""Streamlit application shell: composition root and control flow.

All rendering lives in :mod:`credit_risk.ui.panels`; this module wires
dependencies together and decides what to show.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import streamlit as st

from credit_risk.config import AppEnv, Settings, get_settings
from credit_risk.errors import CreditRiskError
from credit_risk.observability import (
    bind_correlation_id,
    configure_logging,
    new_correlation_id,
)
from credit_risk.scoring.scorer import CreditRiskScorer, build_scorer
from credit_risk.ui.inputs import FormResult, render_application_form
from credit_risk.ui.panels import (
    render_empty_state,
    render_footer,
    render_masthead,
    render_results,
    render_sidebar,
    render_startup_failure,
)
from credit_risk.ui.theme import apply_theme

_PAGE_TITLE: Final[str] = "Lauki Finance | Credit Assessment"
_CORRELATION_KEY: Final[str] = "crm_correlation_id"


@st.cache_resource(show_spinner="Loading credit risk model...")
def _load_scorer(artifact_path: str, base_score: int, scale_length: int) -> CreditRiskScorer:
    """Build the scorer once per configuration and share it across reruns.

    The configuration is passed by value so it forms part of the cache key:
    changing ``CRM_ARTIFACT_PATH`` or the calibration rebuilds the scorer
    instead of silently serving a stale one.
    """
    return build_scorer(
        Settings(
            artifact_path=Path(artifact_path),
            base_score=base_score,
            scale_length=scale_length,
        )
    )


def _session_correlation_id() -> str:
    """Return this session's correlation id, minting it on first use."""
    if _CORRELATION_KEY not in st.session_state:
        st.session_state[_CORRELATION_KEY] = new_correlation_id()
    correlation_id: str = st.session_state[_CORRELATION_KEY]
    return correlation_id


def _render_result_column(result: FormResult, scorer: CreditRiskScorer, settings: Settings) -> None:
    """Render whichever of the three result states applies."""
    if result.errors:
        st.error("Please correct the following before continuing:")
        for message in result.errors:
            st.markdown(f"- {message}")
        return

    if result.application is None:
        render_empty_state(scorer.scorecard)
        return

    try:
        assessment = scorer.score(result.application)
    except CreditRiskError as error:
        st.error(f"**{error.code}** &mdash; {error.message}")
        if settings.app_env is AppEnv.DEV:
            st.exception(error)
        return

    render_results(assessment, scorer.scorecard, settings)


def render_app() -> None:
    """Application entry point: configure, wire, and render the page."""
    settings = get_settings()
    configure_logging(settings)

    st.set_page_config(
        page_title=_PAGE_TITLE,
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_theme()

    correlation_id = _session_correlation_id()
    bind_correlation_id(correlation_id)

    try:
        scorer = _load_scorer(
            artifact_path=str(settings.artifact_path),
            base_score=settings.base_score,
            scale_length=settings.scale_length,
        )
    except CreditRiskError as error:
        render_startup_failure(error)
        return

    render_masthead(scorer, settings)
    render_sidebar(settings, scorer, correlation_id)

    form_column, result_column = st.columns([1.05, 1], gap="large")
    with form_column:
        result = render_application_form()
    with result_column:
        _render_result_column(result, scorer, settings)

    render_footer(settings)
