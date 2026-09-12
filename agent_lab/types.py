"""Typed boundaries shared by the model, harness, and tools."""

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol


@dataclass(frozen=True)
class Authority:
    """Rights granted to one run; text returned by tools cannot change them."""

    effects: frozenset[str]
    memory_scopes: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: Mapping[str, Any]
    call_id: str
    idempotency_key: str | None = None


@dataclass(frozen=True)
class FinalAnswer:
    text: str


Decision = ToolCall | FinalAnswer


@dataclass(frozen=True)
class Observation:
    call_id: str
    tool: str
    status: Literal["ok", "error", "unknown", "rejected"]
    data: Mapping[str, Any]


@dataclass(frozen=True)
class Event:
    step: int
    kind: Literal["proposed", "admitted", "rejected", "result", "finish"]
    detail: str


@dataclass(frozen=True)
class Prompt:
    goal: str
    hard_constraints: tuple[str, ...]
    observations: tuple[Observation, ...]
    memories: tuple[str, ...]
    estimated_tokens: int


class Model(Protocol):
    """A model adapter proposes a call or a final answer; it never executes."""

    def propose(self, prompt: Prompt) -> Decision:
        ...
