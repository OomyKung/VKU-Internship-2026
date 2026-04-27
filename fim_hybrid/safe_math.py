"""Small numeric guards for benchmark scoring and diagnostics."""

from __future__ import annotations

import logging
import math
from typing import Iterable


LOGGER = logging.getLogger(__name__)


def safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0,
    *,
    context: str = "division",
) -> float:
    """Return numerator / denominator, or default when the denominator is unusable."""

    try:
        denominator_value = float(denominator)
        numerator_value = float(numerator)
    except (TypeError, ValueError):
        LOGGER.warning("%s skipped because numerator or denominator is non-numeric.", context)
        return float(default)

    if denominator_value == 0.0 or not math.isfinite(denominator_value):
        LOGGER.warning("%s skipped because denominator is %r.", context, denominator)
        return float(default)
    if not math.isfinite(numerator_value):
        LOGGER.warning("%s skipped because numerator is %r.", context, numerator)
        return float(default)
    return float(numerator_value / denominator_value)


def safe_minmax_normalize(
    values: Iterable[float],
    default: float = 0.0,
    *,
    context: str = "min-max normalization",
) -> list[float]:
    """Min-max normalize a vector, returning defaults for empty or constant input."""

    value_list = [float(value) for value in values]
    if not value_list:
        return []

    finite_values = [value for value in value_list if math.isfinite(value)]
    if not finite_values:
        LOGGER.warning("%s is non-finite for all values; using default %.6f.", context, float(default))
        return [float(default) for _ in value_list]

    minimum = min(finite_values)
    maximum = max(finite_values)
    if maximum <= minimum:
        LOGGER.warning(
            "%s is constant at %.6f; using default %.6f.",
            context,
            minimum,
            float(default),
        )
        return [float(default) for _ in value_list]

    denominator = maximum - minimum
    return [
        safe_divide(value - minimum, denominator, default=default, context=context)
        if math.isfinite(value)
        else float(default)
        for value in value_list
    ]
