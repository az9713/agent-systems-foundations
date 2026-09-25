"""Outcome measures for coding and computer-use agents."""

from dataclasses import dataclass
from math import comb
from typing import Mapping


def normalized_to_pixels(
    u: float, v: float, width: int, height: int
) -> tuple[float, float]:
    """Map normalized GUI coordinates to the declared pixel frame."""
    if not 0.0 <= u <= 1.0 or not 0.0 <= v <= 1.0:
        raise ValueError("normalized coordinates must lie in [0, 1]")
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")
    return u * (width - 1), v * (height - 1)


def pass_at_k(sample_count: int, correct_count: int, k: int) -> float:
    """Estimate the chance that at least one of k samples is correct."""
    if not 0 <= correct_count <= sample_count:
        raise ValueError("correct_count must be within the sample")
    if not 1 <= k <= sample_count:
        raise ValueError("k must be between one and sample_count")
    if sample_count - correct_count < k:
        return 1.0
    return 1.0 - comb(sample_count - correct_count, k) / comb(
        sample_count, k
    )


@dataclass(frozen=True)
class OutcomeVector:
    task_success: bool
    regressions_preserved: bool
    forbidden_effects: int
    cost: float
    latency: float

    @property
    def admissible_success(self) -> bool:
        return (
            self.task_success
            and self.regressions_preserved
            and self.forbidden_effects == 0
        )


def rubric_score(
    weights: Mapping[str, float], checks: Mapping[str, bool]
) -> float:
    """Return achieved rubric weight divided by applicable weight."""
    if set(weights) != set(checks):
        raise ValueError("weights and checks must name the same criteria")
    if any(weight < 0 for weight in weights.values()):
        raise ValueError("rubric weights must be nonnegative")
    total = sum(weights.values())
    if total == 0:
        raise ValueError("rubric must have positive total weight")
    achieved = sum(weights[name] for name, passed in checks.items() if passed)
    return achieved / total
