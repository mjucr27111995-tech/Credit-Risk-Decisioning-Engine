"""Shared form vocabulary: section headings and per-section value carriers.

Typed carriers rather than dicts, so every raw widget value has a declared
type before it reaches domain validation.
"""

from __future__ import annotations

from typing import NamedTuple

import streamlit as st


class ProfileInputs(NamedTuple):
    """Raw values collected by the applicant-profile section."""

    age: int
    income: int
    residence_type: str


class RequestInputs(NamedTuple):
    """Raw values collected by the loan-request section."""

    loan_amount: int
    loan_tenure_months: int
    loan_purpose: str
    loan_type: str


class BureauInputs(NamedTuple):
    """Raw values collected by the bureau-history section."""

    delinquency_ratio: float
    avg_dpd_per_delinquency: float
    credit_utilization_ratio: float
    number_of_open_accounts: int


def section_heading(index: int, title: str, hint: str) -> None:
    """Render a numbered section heading.

    Args:
        index: One-based section number shown in the rule.
        title: Section title.
        hint: Right-aligned contextual hint.
    """
    st.markdown(
        f"""<div class="crm-section">
              <span class="crm-section-index">{index:02d}</span>
              <span class="crm-section-title">{title}</span>
              <span class="crm-section-hint">{hint}</span>
            </div>""",
        unsafe_allow_html=True,
    )
