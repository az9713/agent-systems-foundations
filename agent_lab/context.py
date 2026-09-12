"""Finite context selection, deterministic compaction, and cost equations."""

from dataclasses import dataclass
from typing import Protocol, Sequence

from .types import Authority, Observation, Prompt


class StateView(Protocol):
    goal: str
    authority: Authority
    hard_constraints: tuple[str, ...]
    observations: list[Observation]


def submitted_input_tokens(calls: int, prefix: int, growth: int) -> int:
    """Sum prefix + (t - 1) * growth over t = 1, ..., calls."""
    if calls < 0 or prefix < 0 or growth < 0:
        raise ValueError("token counts must be nonnegative")
    return prefix * calls + growth * calls * (calls - 1) // 2


def compacted_input_tokens(
    calls: int,
    prefix: int,
    growth: int,
    block: int,
    summary: int,
) -> int:
    """Count blocks when one fixed summary replaces earlier history."""
    if block < 1 or calls < 0 or calls % block:
        raise ValueError("calls must be a nonnegative multiple of block")
    if min(prefix, growth, summary) < 0:
        raise ValueError("token counts must be nonnegative")
    if calls == 0:
        return 0
    return (
        prefix * calls
        + growth * calls * (block - 1) // 2
        + summary * (calls - block)
    )


def kv_cache_bytes(
    layers: int,
    tokens: int,
    kv_heads: int,
    head_width: int,
    bytes_per_number: int,
) -> int:
    """Count key and value entries before allocation overhead."""
    values = (layers, tokens, kv_heads, head_width, bytes_per_number)
    if any(value < 0 for value in values):
        raise ValueError("dimensions must be nonnegative")
    return 2 * layers * tokens * kv_heads * head_width * bytes_per_number


@dataclass(frozen=True)
class Candidate:
    name: str
    tokens: int
    value: float


def select_items(
    candidates: Sequence[Candidate], budget: int
) -> tuple[str, ...]:
    """Solve a small zero-one knapsack by dynamic programming."""
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    best: list[tuple[float, tuple[str, ...]]] = [
        (0.0, ()) for _ in range(budget + 1)
    ]
    for item in candidates:
        if item.tokens < 1:
            raise ValueError("each candidate must cost at least one token")
        for capacity in range(budget, item.tokens - 1, -1):
            old_value, old_names = best[capacity - item.tokens]
            new_value = old_value + item.value
            if new_value > best[capacity][0]:
                best[capacity] = (new_value, old_names + (item.name,))
    return best[budget][1]


@dataclass(frozen=True)
class CompactState:
    goal: str
    hard_constraints: tuple[str, ...]
    recent_observations: tuple[Observation, ...]
    evidence_ids: tuple[str, ...]


def compact_state(state: StateView, keep_recent: int) -> CompactState:
    """Retain constraints exactly and keep handles to the full event record."""
    if keep_recent < 0:
        raise ValueError("keep_recent must be nonnegative")
    recent = tuple(state.observations[-keep_recent:]) if keep_recent else ()
    return CompactState(
        state.goal,
        state.hard_constraints,
        recent,
        tuple(observation.call_id for observation in state.observations),
    )


class ContextBuilder:
    def __init__(self, keep_recent: int = 12) -> None:
        if keep_recent < 0:
            raise ValueError("keep_recent must be nonnegative")
        self.keep_recent = keep_recent

    def build(self, state: StateView, memories: tuple[str, ...]) -> Prompt:
        compact = compact_state(state, self.keep_recent)
        parts = [state.goal, *compact.hard_constraints, *memories]
        parts.extend(repr(item) for item in compact.recent_observations)
        # Whitespace units are a teaching proxy, not a provider token count.
        estimated = sum(len(part.split()) for part in parts)
        return Prompt(
            state.goal,
            compact.hard_constraints,
            compact.recent_observations,
            memories,
            estimated,
        )
