"""Presentational panels for the Streamlit page.

Pure rendering: these helpers take domain objects and settings and draw them.
They hold no business logic and make no decisions, which keeps ``app.py`` down
to composition and control flow.
"""

from __future__ import annotations

import html
from typing import Final

import pandas as pd
import streamlit as st

from credit_risk import __version__
from credit_risk.config import Settings
from credit_risk.domain.models import RecommendedAction, RiskAssessment
from credit_risk.domain.scorecard import Scorecard
from credit_risk.errors import CreditRiskError
from credit_risk.scoring.scorer import CreditRiskScorer
from credit_risk.ui.charts import build_contribution_chart, build_score_band_chart
from credit_risk.ui.palette import band_colour

_STP_ACTIONS: Final[frozenset[RecommendedAction]] = frozenset(
    {RecommendedAction.AUTO_APPROVE, RecommendedAction.AUTO_DECLINE}
)


def render_masthead(scorer: CreditRiskScorer, settings: Settings) -> None:
    """Render the document masthead with model provenance."""
    st.markdown(
        f"""<div class="crm-masthead">
             <div>
               <div class="crm-masthead-institution">Lauki Finance &middot;
                 Credit Risk</div>
               <h1>Credit Assessment</h1>
               <p class="crm-masthead-sub">Probability of default, indicative
                  credit score and rating band for a single loan application,
                  with the contribution of every factor disclosed.</p>
             </div>
             <div class="crm-meta">
               <div class="crm-meta-item">
                 <span class="crm-meta-key">Model</span>
                 <span class="crm-meta-val">Logistic regression</span>
               </div>
               <div class="crm-meta-item">
                 <span class="crm-meta-key">Version</span>
                 <span class="crm-meta-val">{html.escape(scorer.model_version)}</span>
               </div>
               <div class="crm-meta-item">
                 <span class="crm-meta-key">Discrimination</span>
                 <span class="crm-meta-val">AUC 0.98 &middot; Gini 0.96</span>
               </div>
               <div class="crm-meta-item">
                 <span class="crm-meta-key">Environment</span>
                 <span class="crm-meta-val">{html.escape(settings.app_env)}</span>
               </div>
             </div>
           </div>""",
        unsafe_allow_html=True,
    )


def _render_verdict(assessment: RiskAssessment, scorecard: Scorecard) -> None:
    """Render the headline verdict card."""
    band = scorecard.band_for(assessment.credit_score)
    st.markdown(
        f"""<div class="crm-verdict" style="--crm-band:{band_colour(band.rating)};">
              <div class="crm-verdict-head">
                <span class="crm-verdict-label">Indicative credit score</span>
                <span class="crm-verdict-rating">{html.escape(str(assessment.rating))}</span>
              </div>
              <div class="crm-verdict-score">{assessment.credit_score}
                <span>of {scorecard.max_score}</span></div>
              <div class="crm-verdict-range">{html.escape(str(assessment.rating))} band spans
                {html.escape(band.range_label)}</div>
              <div class="crm-verdict-action">
                <span class="crm-verdict-action-key">Recommendation</span>
                <span class="crm-verdict-action-val">
                  {html.escape(str(assessment.recommended_action))}</span>
              </div>
            </div>""",
        unsafe_allow_html=True,
    )


def _rail_cell(key: str, value: str, note: str, *, numeric: bool = True) -> str:
    """Build the markup for one metric in the rail beneath the verdict."""
    value_class = "crm-rail-val" if numeric else "crm-rail-val crm-rail-val-text"
    return (
        f'<div class="crm-rail-cell">'
        f'<div class="crm-rail-key">{html.escape(key)}</div>'
        f'<div class="{value_class}">{html.escape(value)}</div>'
        f'<div class="crm-rail-note">{html.escape(note)}</div>'
        f"</div>"
    )


def _render_metrics(assessment: RiskAssessment, settings: Settings) -> None:
    """Render the three headline metrics as a single joined rail."""
    cells = [
        _rail_cell(
            "Probability of default",
            f"{assessment.probability_percent:.2f}%",
            "Model output for this application.",
        ),
        _rail_cell(
            "Repayment confidence",
            f"{100 - assessment.probability_percent:.2f}%",
            "Complement of the default probability.",
        ),
    ]
    if settings.stp_enabled:
        is_stp = assessment.recommended_action in _STP_ACTIONS
        cells.append(
            _rail_cell(
                "Processing route",
                "Straight-through" if is_stp else "Manual review",
                "Referral path under SOW Phase 2.",
                numeric=False,
            )
        )
    else:
        cells.append(
            _rail_cell(
                "Model version",
                assessment.model_version,
                "Artifact content hash.",
                numeric=False,
            )
        )
    st.markdown(f'<div class="crm-rail">{"".join(cells)}</div>', unsafe_allow_html=True)


def contribution_table(assessment: RiskAssessment) -> pd.DataFrame:
    """Build the calculation-detail table, strongest influence first."""
    ranked = sorted(
        assessment.contributions,
        key=lambda entry: abs(entry.contribution),
        reverse=True,
    )
    return pd.DataFrame(
        [
            {
                "Feature": item.display_name,
                "Scaled value": round(item.raw_value, 4),
                "Coefficient": round(item.coefficient, 4),
                "Contribution": round(item.contribution, 4),
                "Direction": "Increases risk" if item.increases_risk else "Reduces risk",
            }
            for item in ranked
        ]
    )


def _render_calculation_detail(assessment: RiskAssessment, scorecard: Scorecard) -> None:
    """Render the transparency panel: provenance plus the full contribution table."""
    with st.expander("Calculation detail"):
        total = sum(item.contribution for item in assessment.contributions)
        st.markdown(
            f"""
            - **Assessment id** &nbsp;`{assessment.assessment_id}`
            - **Assessed at** &nbsp;`{assessment.assessed_at.isoformat(timespec="seconds")}`
            - **Model version** &nbsp;`{assessment.model_version}`
            - **Sum of contributions** &nbsp;`{total:.4f}`
            - **Score transform** &nbsp;`{scorecard.min_score} + (1 - PD) x
              {scorecard.max_score - scorecard.min_score}`
            """
        )
        st.dataframe(contribution_table(assessment), width="stretch", hide_index=True)


def _subhead(title: str, note: str = "") -> None:
    """Render a small-caps subheading above a chart."""
    markup = f'<div class="crm-subhead">{html.escape(title)}</div>'
    if note:
        markup += f'<p class="crm-subhead-note">{note}</p>'
    st.markdown(markup, unsafe_allow_html=True)


def render_results(assessment: RiskAssessment, scorecard: Scorecard, settings: Settings) -> None:
    """Render the verdict, metrics, charts and transparency panel."""
    _render_verdict(assessment, scorecard)
    _render_metrics(assessment, settings)

    _subhead("Position within rating bands")
    st.altair_chart(build_score_band_chart(assessment, scorecard), width="stretch")

    _subhead(
        "Factor attribution",
        "Each bar is exactly <code>coefficient &times; scaled value</code>, so these are "
        "precise contributions to the log-odds of default rather than approximations.",
    )
    st.altair_chart(build_contribution_chart(assessment.contributions), width="stretch")

    _render_calculation_detail(assessment, scorecard)


def render_empty_state(scorecard: Scorecard) -> None:
    """Render the placeholder shown before the first assessment."""
    st.markdown(
        """<div class="crm-empty">
             <div class="crm-empty-mark">&mdash;&nbsp;&mdash;&nbsp;&mdash;</div>
             <strong>Awaiting assessment</strong>
             <span>Complete the application particulars and select
               <em>Assess credit risk</em>.</span>
           </div>""",
        unsafe_allow_html=True,
    )
    _subhead("Rating bands and referral policy")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Rating": str(band.rating),
                    "Score range": band.range_label,
                    "Recommended action": str(band.action),
                }
                for band in reversed(scorecard.bands)
            ]
        ),
        width="stretch",
        hide_index=True,
    )


def render_sidebar(settings: Settings, scorer: CreditRiskScorer, correlation_id: str) -> None:
    """Render model provenance and known limitations."""
    with st.sidebar:
        st.markdown('<div class="crm-side-head">Model card</div>', unsafe_allow_html=True)
        st.markdown(
            f"""<dl class="crm-dl">
                  <dt>Algorithm</dt><dd>Logistic regression</dd>
                  <dt>Features</dt><dd>13</dd>
                  <dt>Test AUC</dt><dd>0.98</dd>
                  <dt>Gini</dt><dd>0.96</dd>
                  <dt>Score range</dt>
                    <dd>{scorecard_range(settings)}</dd>
                  <dt>Artifact</dt><dd><code>{html.escape(scorer.model_version)}</code></dd>
                  <dt>Application</dt><dd><code>v{__version__}</code></dd>
                </dl>""",
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown('<div class="crm-side-head">Basis of preparation</div>', unsafe_allow_html=True)
        st.markdown(
            """<p class="crm-side-note">Fitted on a synthetic teaching dataset with an
                 unusually clean signal; the reported AUC and Gini will not transfer to a
                 live portfolio. All thirteen features are observable at application time.
                 Refer to <code>docs/MODEL_CARD.md</code> and
                 <code>docs/FAIRNESS_AUDIT.md</code>.</p>""",
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(
            f"""<div class="crm-side-head">Correlation id</div>
                <p class="crm-side-note"><code>{html.escape(correlation_id)}</code></p>""",
            unsafe_allow_html=True,
        )


def scorecard_range(settings: Settings) -> str:
    """Format the configured score range for display."""
    return f"{settings.base_score}&ndash;{settings.base_score + settings.scale_length}"


def render_footer(settings: Settings) -> None:
    """Render the page footer."""
    st.markdown(
        f"""<div class="crm-foot">
              Decision support only. Every assessment remains subject to Lauki Finance
              credit policy and, where indicated, underwriter review.
              Model artifact <code>{html.escape(settings.artifact_path.name)}</code>.
            </div>""",
        unsafe_allow_html=True,
    )


def render_startup_failure(error: CreditRiskError) -> None:
    """Fail visibly and safely when the model cannot be loaded."""
    st.error(f"**{error.code}** &mdash; {error.message}")
    st.info(
        "Confirm that `CRM_ARTIFACT_PATH` points at a valid model bundle. "
        "The default is `artifacts/model_data.joblib`."
    )
