"""Pure, testable calculator functions demonstrating functional style and good practices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from .logging_config import configure_logging
import logging

configure_logging()
logger = logging.getLogger(__name__)


class CalculationError(ValueError):
    """Raised when an invalid calculation is attempted."""


def _validate_numbers(numbers: Iterable[float]) -> List[float]:
    """Return list of floats and validate input is not empty.

    Args:
        numbers: Iterable of numeric values.

    Returns:
        List[float]: normalized list of floats.

    Raises:
        CalculationError: If numbers is empty.
    """
    nums = [float(n) for n in numbers]
    if not nums:
        raise CalculationError("numbers must contain at least one value")
    return nums


def mean(numbers: Iterable[float]) -> float:
    """Compute arithmetic mean of numbers.

    This function is pure (no side-effects) and raises CalculationError on invalid input.

    Args:
        numbers: Iterable of numeric values.

    Returns:
        float: arithmetic mean.
    """
    nums = _validate_
