"""Altair chart builders.

Pure functions: they take domain objects and return charts, so they can be
unit-tested without a browser or a Streamlit runtime.
"""

from __future__ import annotations

from typing import Final, cast

import altair as alt
import pandas as pd

from credit_risk.domain.models import FeatureContribution, RiskAssessment
from credit_risk.domain.scorecard import Scorecard
from credit_risk.ui import palette

_MARKER_COLOUR: Final[str] = palette.INK
_AXIS_LABEL_COLOUR: Final[str] = palette.MUTED
_CHART_FONT: Final[str] = palette.SANS


def _band_frame(scorecard: Scorecard) -> pd.DataFrame:
    """Tabulate the rating bands for plotting."""
    return pd.DataFrame(
        [
            {
                "rating": str(band.rating),
                "start": band.lower,
                "end": band.upper + 1,
                "colour": palette.band_colour(band.rating),
                "range": band.range_label,
            }
            for band in scorecard.bands
        ]
    )


def _marker_layers(
    assessment: RiskAssessment, base_scale: alt.Scale
) -> tuple[alt.Chart, alt.Chart]:
    """Build the score position rule and its numeric label."""
    marker = pd.DataFrame([{"score": assessment.credit_score, "rating": str(assessment.rating)}])
    rule = (
        alt.Chart(marker)
        .mark_rule(color=_MARKER_COLOUR, size=2)
        .encode(
            x=alt.X("score:Q", scale=base_scale, title=None),
            tooltip=[
                alt.Tooltip("score:Q", title="Credit score"),
                alt.Tooltip("rating:N", title="Rating"),
            ],
        )
    )
    label = (
        alt.Chart(marker)
        .mark_text(
            dy=-19,
            fontSize=12,
            fontWeight=600,
            color=_MARKER_COLOUR,
            font=_CHART_FONT,
        )
        .encode(x=alt.X("score:Q", scale=base_scale, title=None), text="score:Q")
    )
    return rule, label


def build_score_band_chart(assessment: RiskAssessment, scorecard: Scorecard) -> alt.LayerChart:
    """Render the score against the four rating bands with a position marker.

    Args:
        assessment: The assessment to plot.
        scorecard: Supplies band edges.

    Returns:
        A layered Altair chart sized to its container.
    """
    base_scale = alt.Scale(domain=[scorecard.min_score, scorecard.max_score + 1], nice=False)

    band_layer = (
        alt.Chart(_band_frame(scorecard))
        .mark_bar(height=13)
        .encode(
            x=alt.X(
                "start:Q",
                scale=base_scale,
                title=None,
                axis=alt.Axis(values=_ticks(scorecard)),
            ),
            x2="end:Q",
            color=alt.Color("colour:N", scale=None, legend=None),
            tooltip=[
                alt.Tooltip("rating:N", title="Rating"),
                alt.Tooltip("range:N", title="Score range"),
            ],
        )
    )
    marker_layer, label_layer = _marker_layers(assessment, base_scale)

    chart = (
        (band_layer + marker_layer + label_layer)
        .properties(height=76, width="container", title="")
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor=_AXIS_LABEL_COLOUR,
            labelFontSize=10,
            labelFont=_CHART_FONT,
            labelPadding=6,
            domain=False,
            ticks=False,
            grid=False,
        )
    )
    return cast(alt.LayerChart, chart)


def _ticks(scorecard: Scorecard) -> list[int]:
    """Axis ticks at every band edge plus the range maximum."""
    edges = [band.lower for band in scorecard.bands]
    edges.append(scorecard.max_score)
    return edges


def _contribution_frame(contributions: tuple[FeatureContribution, ...], limit: int) -> pd.DataFrame:
    """Tabulate the strongest contributions for plotting."""
    ranked = sorted(contributions, key=lambda item: abs(item.contribution), reverse=True)[:limit]
    return pd.DataFrame(
        [
            {
                "feature": item.display_name,
                "contribution": round(item.contribution, 4),
                "direction": "Increases risk" if item.increases_risk else "Reduces risk",
                "coefficient": round(item.coefficient, 4),
                "scaled_value": round(item.raw_value, 4),
            }
            for item in ranked
        ]
    )


def build_contribution_chart(
    contributions: tuple[FeatureContribution, ...], limit: int = 8
) -> alt.Chart:
    """Plot the features that pushed the decision, largest influence first.

    Args:
        contributions: Per-feature log-odds contributions.
        limit: Maximum number of bars to draw.

    Returns:
        A diverging bar chart: red increases risk, green reduces it.
    """
    frame = _contribution_frame(contributions, limit)

    chart = (
        alt.Chart(frame)
        .mark_bar(height=12)
        .encode(
            x=alt.X(
                "contribution:Q",
                title="Contribution to log-odds of default",
                axis=alt.Axis(
                    grid=True,
                    gridColor=palette.HAIRLINE,
                    gridDash=[2, 3],
                    titleColor=palette.FAINT,
                    titleFontSize=10,
                    titleFontWeight=400,
                    titlePadding=12,
                ),
            ),
            y=alt.Y(
                "feature:N",
                sort=None,
                title=None,
                # Feature labels are long; do not let Vega truncate them.
                axis=alt.Axis(labelLimit=230, labelPadding=8),
            ),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(
                    domain=["Increases risk", "Reduces risk"],
                    range=[palette.RISK_COLOUR, palette.SAFE_COLOUR],
                ),
                legend=alt.Legend(
                    title=None,
                    orient="top",
                    direction="horizontal",
                    labelColor=palette.MUTED,
                    labelFontSize=10,
                    labelFont=_CHART_FONT,
                    symbolType="square",
                    symbolSize=70,
                    offset=4,
                ),
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("contribution:Q", title="Contribution", format=".4f"),
                alt.Tooltip("coefficient:Q", title="Model coefficient", format=".4f"),
                alt.Tooltip("scaled_value:Q", title="Scaled value", format=".4f"),
            ],
        )
        .properties(height=max(160, 27 * len(frame)), width="container")
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor=_AXIS_LABEL_COLOUR,
            labelFont=_CHART_FONT,
            labelFontSize=10,
            domain=False,
            ticks=False,
        )
    )
    return cast(alt.Chart, chart)
