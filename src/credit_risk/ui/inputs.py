"""Input form. Collects raw widget values and validates them into the aggregate.

The form is the boundary: nothing leaves this module until pydantic has
accepted it as a :class:`LoanApplication`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final, NamedTuple

import streamlit as st
from pydantic import ValidationError as PydanticValidationError

from credit_risk.domain.models import LoanApplication, LoanPurpose, LoanType, ResidenceType
from credit_risk.ui.bureau import render_bureau_section
from credit_risk.ui.sections import (
    BureauInputs,
    ProfileInputs,
    RequestInputs,
    section_heading,
)

# Defaults mirror the reference prototype so results are directly comparable.
_DEFAULTS: Final[dict[str, int]] = {
    "age": 28,
    "income": 1_200_000,
    "loan_amount": 2_560_000,
    "loan_tenure_months": 36,
}

_LTI_WARN_THRESHOLD: Final[float] = 3.0
# Above this, the ratio is displayed in compact form so it cannot overflow the
# callout. Reachable with a very low income against a large facility.
_LTI_COMPACT_THRESHOLD: Final[float] = 1_000.0

# Category defaults, stated explicitly so the form never depends on the
# declaration order of the domain enums.
_DEFAULT_RESIDENCE: Final[ResidenceType] = ResidenceType.OWNED
_DEFAULT_PURPOSE: Final[LoanPurpose] = LoanPurpose.EDUCATION
_DEFAULT_LOAN_TYPE: Final[LoanType] = LoanType.UNSECURED


def _index_of(enum_type: type[StrEnum], member: StrEnum) -> int:
    """Return the position of ``member`` within its enum's declaration order."""
    return list(enum_type).index(member)


class FormResult(NamedTuple):
    """Outcome of rendering the form.

    Attributes:
        application: The validated aggregate, or ``None`` when invalid.
        errors: Human-readable validation messages.
        submitted: Whether the officer pressed Assess on this rerun.
    """

    application: LoanApplication | None
    errors: tuple[str, ...]
    submitted: bool


def format_ratio(ratio: float) -> str:
    """Format the loan-to-income ratio so it never overflows its callout.

    Args:
        ratio: Loan amount divided by annual income.

    Returns:
        Two decimal places normally; scientific notation once the value grows
        beyond four digits, which happens with an implausibly low income.
    """
    if ratio >= _LTI_COMPACT_THRESHOLD:
        return f"{ratio:.1e}"
    return f"{ratio:.2f}"


def _render_ratio_callout(loan_amount: int, income: int) -> None:
    """Show the live loan-to-income ratio, flagged when it turns risky."""
    ratio = loan_amount / income if income > 0 else 0.0
    is_risky = ratio > _LTI_WARN_THRESHOLD
    css_class = "crm-ratio crm-ratio-warn" if is_risky else "crm-ratio"
    note = "Exceeds 3.0 threshold" if is_risky else "Requested amount over annual income"
    st.markdown(
        f"""<div class="{css_class}">
              <div class="crm-ratio-label">Loan to income</div>
              <div class="crm-ratio-value">{format_ratio(ratio)}</div>
              <div class="crm-ratio-note">{note}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def _humanise_errors(exc: PydanticValidationError) -> tuple[str, ...]:
    """Turn pydantic errors into messages a loan officer can act on."""
    messages: list[str] = []
    for error in exc.errors():
        field = ".".join(str(part) for part in error["loc"]) or "input"
        messages.append(f"{field.replace('_', ' ').capitalize()}: {error['msg']}")
    return tuple(messages)


def _render_profile_section() -> ProfileInputs:
    """Render section 1 and return the raw widget values it collected."""
    section_heading(1, "Applicant particulars", "Demographics and affordability")
    columns = st.columns([1, 1, 1])
    with columns[0]:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            step=1,
            value=_DEFAULTS["age"],
            help="Applicant age in years (18-100).",
        )
    with columns[1]:
        income = st.number_input(
            "Annual income (INR)",
            min_value=1,
            step=50_000,
            value=_DEFAULTS["income"],
            help="Gross annual income. Must be greater than zero.",
        )
    with columns[2]:
        residence_type = st.selectbox(
            "Residence type",
            options=[member.value for member in ResidenceType],
            index=_index_of(ResidenceType, _DEFAULT_RESIDENCE),
            key="residence_type",
            help="'Mortgage' is the model's reference category.",
        )
    return ProfileInputs(age=int(age), income=int(income), residence_type=str(residence_type))


def _render_request_row_two(loan_amount: int, income: int) -> str:
    """Render the loan-type selector and the live loan-to-income callout."""
    columns = st.columns([1, 1, 1])
    with columns[0]:
        loan_type = st.selectbox(
            "Loan type",
            options=[member.value for member in LoanType],
            index=_index_of(LoanType, _DEFAULT_LOAN_TYPE),
            key="loan_type",
            help="Unsecured lending carries a higher modelled risk.",
        )
    with columns[1]:
        _render_ratio_callout(loan_amount, income)
    return str(loan_type)


def _render_request_section(income: int) -> RequestInputs:
    """Render section 2 and return the raw widget values it collected.

    Args:
        income: Already-collected income, used for the live ratio callout.
    """
    section_heading(2, "Facility requested", "Product, quantum and tenure")
    columns = st.columns([1, 1, 1])
    with columns[0]:
        loan_amount = st.number_input(
            "Loan amount (INR)",
            min_value=1,
            step=50_000,
            value=_DEFAULTS["loan_amount"],
            help="Requested principal.",
        )
    with columns[1]:
        loan_tenure_months = st.number_input(
            "Tenure (months)",
            min_value=1,
            max_value=360,
            step=1,
            value=_DEFAULTS["loan_tenure_months"],
            help="Repayment period, 1-360 months.",
        )
    with columns[2]:
        loan_purpose = st.selectbox(
            "Loan purpose",
            options=[member.value for member in LoanPurpose],
            index=_index_of(LoanPurpose, _DEFAULT_PURPOSE),
            key="loan_purpose",
            help="'Auto' is the model's reference category.",
        )

    loan_type = _render_request_row_two(int(loan_amount), income)
    return RequestInputs(
        loan_amount=int(loan_amount),
        loan_tenure_months=int(loan_tenure_months),
        loan_purpose=str(loan_purpose),
        loan_type=loan_type,
    )


def _to_application(
    profile: ProfileInputs, request: RequestInputs, bureau: BureauInputs
) -> LoanApplication:
    """Assemble the validated aggregate from the three sections' raw values.

    Raises:
        PydanticValidationError: If any domain invariant is violated.
    """
    return LoanApplication(
        age=profile.age,
        income=profile.income,
        loan_amount=request.loan_amount,
        loan_tenure_months=request.loan_tenure_months,
        avg_dpd_per_delinquency=bureau.avg_dpd_per_delinquency,
        delinquency_ratio=bureau.delinquency_ratio,
        credit_utilization_ratio=bureau.credit_utilization_ratio,
        number_of_open_accounts=bureau.number_of_open_accounts,
        residence_type=ResidenceType(profile.residence_type),
        loan_purpose=LoanPurpose(request.loan_purpose),
        loan_type=LoanType(request.loan_type),
    )


def render_application_form() -> FormResult:
    """Render the three-part application form and validate it on submit.

    Returns:
        A :class:`FormResult` carrying the validated application or the errors.
    """
    with st.form("loan_application", border=False):
        profile = _render_profile_section()
        st.write("")
        request = _render_request_section(profile.income)
        st.write("")
        bureau = render_bureau_section()
        st.write("")
        submitted = st.form_submit_button("Assess credit risk", type="primary")

    if not submitted:
        return FormResult(application=None, errors=(), submitted=False)

    try:
        application = _to_application(profile, request, bureau)
    except PydanticValidationError as exc:
        return FormResult(application=None, errors=_humanise_errors(exc), submitted=True)

    return FormResult(application=application, errors=(), submitted=True)
