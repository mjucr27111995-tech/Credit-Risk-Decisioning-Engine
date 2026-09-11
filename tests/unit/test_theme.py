"""Tests for presentation tokens and stylesheet assembly."""

from __future__ import annotations

import re

import pytest

from credit_risk.domain.models import RiskRating
from credit_risk.domain.scorecard import ScorecardBand
from credit_risk.ui import palette, theme

_HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")

_AA_SMALL = 4.5
_AA_LARGE = 3.0


def _relative_luminance(hex_colour: str) -> float:
    """WCAG 2.1 relative luminance for an sRGB hex colour."""
    raw = hex_colour.lstrip("#")
    channels = [int(raw[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast_ratio(foreground: str, background: str) -> float:
    """WCAG 2.1 contrast ratio between two hex colours."""
    lighter = max(_relative_luminance(foreground), _relative_luminance(background))
    darker = min(_relative_luminance(foreground), _relative_luminance(background))
    return (lighter + 0.05) / (darker + 0.05)


class TestBandColour:
    """Every rating must resolve to exactly one valid colour."""

    @pytest.mark.parametrize("rating", list(RiskRating))
    def should_resolveColour_withEveryRating_returnValidHex(self, rating: RiskRating):
        # Act
        colour = palette.band_colour(rating)

        # Assert
        assert _HEX.match(colour), colour

    def should_resolveColours_acrossRatings_beMutuallyDistinct(self):
        # Act
        colours = {palette.band_colour(rating) for rating in RiskRating}

        # Assert - bands must never be confusable
        assert len(colours) == len(list(RiskRating))


class TestDomainStaysPresentationFree:
    """The domain must not carry UI concerns across the dependency boundary."""

    def should_inspectScorecardBand_forPresentationFields_findNone(self):
        # Assert - colour lives in the ui layer, not the domain
        assert "colour" not in ScorecardBand._fields
        assert "color" not in ScorecardBand._fields


class TestColourContrast:
    """Text colours must clear WCAG AA. A premium look must stay legible."""

    @pytest.mark.parametrize("background", [palette.SURFACE, palette.CANVAS, palette.BRASS_SOFT])
    def should_measureContrast_withBrassText_clearAaSmall(self, background: str):
        # Act - BRASS is used for the institution line, section numerals and labels
        ratio = contrast_ratio(palette.BRASS, background)

        # Assert
        assert ratio >= _AA_SMALL, f"BRASS on {background} is {ratio:.2f}:1"

    @pytest.mark.parametrize(
        "token",
        ["INK", "INK_SECONDARY", "MUTED", "FAINT", "BRASS"],
    )
    def should_measureContrast_withEveryTextToken_clearAaSmallOnSurface(self, token: str):
        # Act - every token that carries text, including small captions
        ratio = contrast_ratio(getattr(palette, token), palette.SURFACE)

        # Assert
        assert ratio >= _AA_SMALL, f"{token} is {ratio:.2f}:1 on white"

    @pytest.mark.parametrize("rating", list(RiskRating))
    def should_measureContrast_withBandColours_clearAaLarge(self, rating: RiskRating):
        # Act - band colours appear as the rating label and chart fills
        ratio = contrast_ratio(palette.band_colour(rating), palette.SURFACE)

        # Assert
        assert ratio >= _AA_LARGE, f"{rating} band is {ratio:.2f}:1"

    def should_measureContrast_withButtonLabel_clearAaSmall(self):
        # Act
        ratio = contrast_ratio("#FFFFFF", palette.ACCENT)

        # Assert
        assert ratio >= _AA_SMALL

    def should_useBrassRule_forDecorationOnly_notForText(self):
        # BRASS_RULE is intentionally too light for text. This test documents
        # that constraint so nobody promotes it to a text colour later.
        ratio = contrast_ratio(palette.BRASS_RULE, palette.SURFACE)

        # Assert
        assert ratio < _AA_SMALL, (
            "BRASS_RULE now passes AA. If it is intended for text, move it to "
            "the text-token list above; otherwise keep it decorative."
        )


class TestStylesheet:
    """The stylesheet must be shipped and wired to the Python tokens."""

    def should_loadStylesheet_withPackagedAsset_returnNonEmptyCss(self):
        # Act
        sheet = theme._stylesheet()

        # Assert
        assert sheet.startswith("<style>")
        assert sheet.rstrip().endswith("</style>")
        assert ".crm-verdict" in sheet

    def should_loadStylesheet_withTokens_declareEveryCustomProperty(self):
        # Act
        sheet = theme._stylesheet()

        # Assert - a missing declaration would silently render as an invalid value
        for name, value in theme._TOKENS.items():
            assert f"{name}: {value};" in sheet

    def should_loadStylesheet_withReferencedVariables_declareAllOfThem(self):
        # Arrange
        sheet = theme._stylesheet()
        # Exclude --crm-band: it is set per-assessment inline, not globally.
        referenced = {
            match for match in re.findall(r"var\((--crm-[a-z-]+)\)", sheet) if match != "--crm-band"
        }

        # Assert - catches a token referenced in CSS but never declared
        assert referenced <= set(theme._TOKENS), referenced - set(theme._TOKENS)

    def should_loadStylesheet_withoutRemoteFonts_avoidNetworkFetch(self):
        # Act
        sheet = theme._stylesheet()

        # Assert - the app must render identically offline
        assert "@import" not in sheet
        assert "fonts.googleapis" not in sheet

    def should_loadStylesheet_withWidgetOverrides_avoidVolatileEmotionClasses(self):
        # Act
        sheet = theme._stylesheet()

        # Assert - `st-emotion-cache-*` hashes change between Streamlit builds,
        # so selectors must never depend on them.
        assert "st-emotion-cache" not in sheet

    def should_loadStylesheet_withInputOverrides_targetFieldWrapper(self):
        # Act
        sheet = theme._stylesheet()

        # Assert - Streamlit 1.62 puts the visible border on this wrapper; if a
        # future upgrade changes the structure, this documents what to re-check.
        assert ".stNumberInput .react-aria-TextField > div" in sheet

    def should_loadStylesheet_withNarrowViewports_declareResponsiveRules(self):
        # Act
        sheet = theme._stylesheet()

        # Assert - regression guard: without these, `st.columns` refuse to stack
        # and field labels collapse to one character per line below ~1100px.
        assert "@media (max-width: 1100px)" in sheet
        assert "@media (max-width: 720px)" in sheet
        assert "flex-wrap: wrap" in sheet
