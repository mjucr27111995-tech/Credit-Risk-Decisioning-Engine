"""Tests for input-side display formatting."""

from __future__ import annotations

import pytest

from credit_risk.ui.inputs import format_ratio

_MAX_DISPLAY_CHARS = 8


class TestFormatRatio:
    """The loan-to-income callout must stay inside its box for any input."""

    @pytest.mark.parametrize(
        ("ratio", "expected"),
        [(0.0, "0.00"), (0.19, "0.19"), (2.13, "2.13"), (5.0, "5.00"), (999.99, "999.99")],
    )
    def should_formatRatio_withPlausibleValue_useTwoDecimals(
        self, ratio: float, expected: str
    ) -> None:
        # Assert
        assert format_ratio(ratio) == expected

    @pytest.mark.parametrize("ratio", [1_000.0, 25_600.0, 2_560_000.0, 1e12])
    def should_formatRatio_withExtremeValue_useCompactNotation(self, ratio: float) -> None:
        # Act
        rendered = format_ratio(ratio)

        # Assert - regression guard: 2560000.00 previously wrapped and broke layout
        assert "e" in rendered
        assert len(rendered) <= _MAX_DISPLAY_CHARS

    @pytest.mark.parametrize("ratio", [0.0, 2.13, 999.99, 1_000.0, 2_560_000.0, 1e12])
    def should_formatRatio_withAnyValue_stayWithinDisplayBudget(self, ratio: float) -> None:
        # Assert
        assert len(format_ratio(ratio)) <= _MAX_DISPLAY_CHARS
