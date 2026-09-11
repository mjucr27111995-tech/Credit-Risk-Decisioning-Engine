"""Scorecard: probability of default -> credit score -> rating band.

Transform (identical to the approved training notebook):

    credit_score = base_score + (1 - PD) * scale_length

With the default calibration (300, 600) this yields the CIBIL-like 300-900
range required by the SOW.
"""

from __future__ import annotations

import math
from typing import Final, NamedTuple

from credit_risk.domain.models import RecommendedAction, RiskRating

# Band edges as fractions of the score range, so the bands scale correctly if
# the calibration is ever changed via configuration.
_BAND_FRACTIONS: Final[tuple[tuple[RiskRating, float, RecommendedAction], ...]] = (
    (RiskRating.POOR, 0.0, RecommendedAction.AUTO_DECLINE),
    (RiskRating.AVERAGE, 1 / 3, RecommendedAction.REFER_TO_UNDERWRITER),
    (RiskRating.GOOD, 7 / 12, RecommendedAction.APPROVE_WITH_REVIEW),
    (RiskRating.EXCELLENT, 3 / 4, RecommendedAction.AUTO_APPROVE),
)


class ScorecardBand(NamedTuple):
    """An inclusive-lower, exclusive-upper score band (upper band is closed)."""

    rating: RiskRating
    lower: int
    upper: int
    action: RecommendedAction

    def contains(self, score: int) -> bool:
        """True when ``score`` falls in this band."""
        return self.lower <= score <= self.upper

    @property
    def range_label(self) -> str:
        """Score range only, e.g. ``650-749``."""
        return f"{self.lower}-{self.upper}"

    @property
    def label(self) -> str:
        """Human-readable band range, e.g. ``Good (650-749)``."""
        return f"{self.rating} ({self.range_label})"


class Scorecard:
    """Maps default probabilities onto the configured score range and bands.

    Args:
        base_score: Lower bound of the score range.
        scale_length: Width of the score range.

    Raises:
        ValueError: If the calibration cannot produce a usable range.
    """

    def __init__(self, base_score: int = 300, scale_length: int = 600) -> None:
        if scale_length < len(_BAND_FRACTIONS):
            raise ValueError(
                f"scale_length must be at least {len(_BAND_FRACTIONS)}, got {scale_length}"
            )
        self._base_score = base_score
        self._scale_length = scale_length
        self._bands = self._build_bands(base_score, scale_length)

    @staticmethod
    def _build_bands(base: int, length: int) -> tuple[ScorecardBand, ...]:
        """Derive concrete, contiguous, non-overlapping bands from the fractions."""
        edges = [base + math.floor(fraction * length) for _, fraction, _ in _BAND_FRACTIONS]
        top = base + length
        bands: list[ScorecardBand] = []
        for index, (rating, _, action) in enumerate(_BAND_FRACTIONS):
            is_last = index == len(_BAND_FRACTIONS) - 1
            upper = top if is_last else edges[index + 1] - 1
            bands.append(
                ScorecardBand(
                    rating=rating,
                    lower=edges[index],
                    upper=upper,
                    action=action,
                )
            )
        return tuple(bands)

    @property
    def bands(self) -> tuple[ScorecardBand, ...]:
        """All bands, ordered from lowest to highest score."""
        return self._bands

    @property
    def min_score(self) -> int:
        """Lowest attainable score."""
        return self._base_score

    @property
    def max_score(self) -> int:
        """Highest attainable score."""
        return self._base_score + self._scale_length

    def to_credit_score(self, probability_of_default: float) -> int:
        """Convert a default probability into a credit score.

        Args:
            probability_of_default: Model output in ``[0, 1]``.

        Returns:
            The credit score, clamped to the configured range.

        Raises:
            ValueError: If the probability is outside ``[0, 1]`` or is NaN.
        """
        if math.isnan(probability_of_default) or not 0.0 <= probability_of_default <= 1.0:
            raise ValueError(
                f"probability_of_default must be within [0, 1], got {probability_of_default}"
            )
        raw = self._base_score + (1.0 - probability_of_default) * self._scale_length
        return int(min(max(int(raw), self.min_score), self.max_score))

    def band_for(self, credit_score: int) -> ScorecardBand:
        """Return the band containing ``credit_score``.

        Scores outside the configured range are clamped to the nearest band,
        which keeps the UI safe if the calibration is changed at runtime.
        """
        for band in self._bands:
            if band.contains(credit_score):
                return band
        return self._bands[0] if credit_score < self.min_score else self._bands[-1]

    def rating_for(self, credit_score: int) -> RiskRating:
        """Return the rating for ``credit_score``."""
        return self.band_for(credit_score).rating

    def action_for(self, credit_score: int) -> RecommendedAction:
        """Return the recommended operational action for ``credit_score``."""
        return self.band_for(credit_score).action
