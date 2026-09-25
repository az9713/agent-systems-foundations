"""Small numerical definitions for Chapters 8 and 9."""

from math import log
from typing import Sequence


def masked_cross_entropy(
    target_probabilities: Sequence[float],
    mask: Sequence[bool],
    weights: Sequence[float] | None = None,
) -> float:
    """Return weighted mean negative log probability on selected tokens."""
    if len(target_probabilities) != len(mask):
        raise ValueError("probabilities and mask must have equal length")
    if weights is None:
        weights = [1.0] * len(mask)
    if len(weights) != len(mask):
        raise ValueError("weights and mask must have equal length")
    if any(not 0.0 < probability <= 1.0 for probability in target_probabilities):
        raise ValueError("target probabilities must lie in (0, 1]")
    if any(weight < 0 for weight in weights):
        raise ValueError("weights must be nonnegative")
    terms = [
        (-log(probability), weight)
        for probability, selected, weight in zip(
            target_probabilities, mask, weights
        )
        if selected
    ]
    if not terms:
        raise ValueError("mask must select at least one token")
    normalizer = sum(weight for _, weight in terms)
    if normalizer == 0:
        raise ValueError("selected weights must have positive sum")
    return sum(loss * weight for loss, weight in terms) / normalizer


def reinforce_estimate(
    score_gradients: Sequence[Sequence[float]],
    rewards: Sequence[float],
    baselines: Sequence[float] | None = None,
) -> tuple[float, ...]:
    """Estimate E[(R-b) grad log p] from sampled trajectories."""
    if len(score_gradients) != len(rewards) or not rewards:
        raise ValueError("one nonempty gradient is required per reward")
    if baselines is None:
        baselines = [0.0] * len(rewards)
    if len(baselines) != len(rewards):
        raise ValueError("one baseline is required per reward")
    width = len(score_gradients[0])
    if any(len(gradient) != width for gradient in score_gradients):
        raise ValueError("all gradients must have equal width")
    return tuple(
        sum(
            (reward - baseline) * gradient[j]
            for gradient, reward, baseline in zip(
                score_gradients, rewards, baselines
            )
        ) / len(rewards)
        for j in range(width)
    )


def centered_group_advantages(rewards: Sequence[float]) -> tuple[float, ...]:
    """Center group rewards without standard-deviation normalization."""
    if not rewards:
        raise ValueError("at least one reward is required")
    mean = sum(rewards) / len(rewards)
    return tuple(reward - mean for reward in rewards)
