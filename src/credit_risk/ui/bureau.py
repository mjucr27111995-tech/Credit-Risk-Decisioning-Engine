"""Bureau-history input section.

Split out of :mod:`credit_risk.ui.inputs` to keep both modules small; the
section owns only widget rendering and returns raw values for validation.
"""

from __future__ import annotations

import streamlit as st

from credit_risk.ui.sections import BureauInputs, section_heading

_DEFAULT_DELINQUENCY_RATIO: float = 30.0
_DEFAULT_AVG_DPD: float = 20.0
_DEFAULT_UTILISATION: float = 30.0
_DEFAULT_OPEN_ACCOUNTS: int = 2


def render_bureau_section() -> BureauInputs:
    """Render section 3 and return the raw widget values it collected."""
    section_heading(3, "Bureau record", "As reported by the credit bureau")
    columns = st.columns([1, 1, 1, 1])
    with columns[0]:
        delinquency_ratio = st.number_input(
            "Delinquency ratio (%)",
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            value=_DEFAULT_DELINQUENCY_RATIO,
            help="Delinquent months as a percentage of total loan months.",
        )
    with columns[1]:
        avg_dpd_per_delinquency = st.number_input(
            "Average DPD",
            min_value=0.0,
            max_value=365.0,
            step=1.0,
            value=_DEFAULT_AVG_DPD,
            help="Mean days past due across delinquent months.",
        )
    with columns[2]:
        credit_utilization_ratio = st.number_input(
            "Credit utilisation (%)",
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            value=_DEFAULT_UTILISATION,
            help="Share of available revolving credit in use.",
        )
    with columns[3]:
        number_of_open_accounts = st.number_input(
            "Open accounts",
            min_value=1,
            max_value=20,
            step=1,
            value=_DEFAULT_OPEN_ACCOUNTS,
            help="Active credit accounts on the bureau report.",
        )
    return BureauInputs(
        delinquency_ratio=float(delinquency_ratio),
        avg_dpd_per_delinquency=float(avg_dpd_per_delinquency),
        credit_utilization_ratio=float(credit_utilization_ratio),
        number_of_open_accounts=int(number_of_open_accounts),
    )
