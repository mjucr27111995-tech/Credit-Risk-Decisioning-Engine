"""Design tokens for the interface.

Presentation concerns live here, not in the domain. ``ScorecardBand`` describes
score ranges and actions; how a rating is *coloured* is a UI decision, so the
mapping belongs on this side of the dependency boundary.

Palette intent: an institutional lender's document surface. Near-monochrome
neutrals, one deep navy accent, and desaturated jewel tones for risk. Saturated
"traffic light" colours are deliberately avoided — they read as consumer app,
not regulated credit decisioning.
"""

from __future__ import annotations

from typing import Final

from credit_risk.domain.models import RiskRating

# --- Neutrals -----------------------------------------------------------------
# Canvas is warmed off the blue-grey axis so the page reads as paper stock
# rather than a dashboard chrome.
INK: Final[str] = "#0B1220"
INK_SECONDARY: Final[str] = "#334155"
MUTED: Final[str] = "#64748B"
# Darkened from #94A3B8, which failed WCAG AA at 2.56:1 for the small caption
# text it carries. Now 4.58:1 on SURFACE. Enforced by tests/unit/test_theme.py.
FAINT: Final[str] = "#697787"
HAIRLINE: Final[str] = "#E7E3DC"
HAIRLINE_STRONG: Final[str] = "#D3CCC1"
CANVAS: Final[str] = "#FAF8F5"
SURFACE: Final[str] = "#FFFFFF"

# --- Accent -------------------------------------------------------------------
ACCENT: Final[str] = "#1B365D"
ACCENT_SOFT: Final[str] = "#EFF2F7"

# --- Brass ---------------------------------------------------------------------
# The single "premium" note: engraved-letterhead brass. Deliberately outside the
# risk palette below, so it can never be mistaken for a decision signal.
#
# BRASS carries text and clears WCAG AA (>= 4.5:1) on all three backgrounds it
# appears on: SURFACE, CANVAS and BRASS_SOFT.
# BRASS_RULE is decorative only - rules, borders, the empty-state mark. It must
# never carry text: at 2.67:1 on white it would fail AA.
BRASS: Final[str] = "#7D6234"
BRASS_RULE: Final[str] = "#B99A62"
BRASS_SOFT: Final[str] = "#F6F1E7"

# --- Risk semantics (desaturated, archival rather than alarming) --------------
WINE: Final[str] = "#8C1D3F"
AMBER: Final[str] = "#A15C10"
STEEL: Final[str] = "#1D5A87"
TEAL: Final[str] = "#12655A"

_BAND_COLOURS: Final[dict[RiskRating, str]] = {
    RiskRating.POOR: WINE,
    RiskRating.AVERAGE: AMBER,
    RiskRating.GOOD: STEEL,
    RiskRating.EXCELLENT: TEAL,
}

# Chart series colours: muted so that the data, not the hue, carries the message.
RISK_COLOUR: Final[str] = "#9E3B4E"
SAFE_COLOUR: Final[str] = "#2C6E63"

# --- Typography ---------------------------------------------------------------
# Local stacks only: no webfont fetch, so the app renders identically offline.
SERIF: Final[str] = "ui-serif, 'Iowan Old Style', Georgia, 'Times New Roman', serif"
SANS: Final[str] = (
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', Inter, Roboto, system-ui, sans-serif"
)
MONO: Final[str] = "ui-monospace, 'SF Mono', SFMono-Regular, Menlo, Consolas, monospace"


def band_colour(rating: RiskRating) -> str:
    """Return the presentation colour for a rating band.

    Args:
        rating: The risk rating to colour.

    Returns:
        A hex colour string.
    """
    return _BAND_COLOURS[rating]
