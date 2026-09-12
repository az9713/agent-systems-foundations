"""The event-recorded proposal, admission, execution, observation loop."""

from dataclasses import dataclass, field
from typing import Callable

from .context import ContextBuilder
from .memory import MemoryStore
from .tools import CallRejected, ToolRegistry
from .types import (
    Authority,
    Event,
    FinalAnswer,
    Model,
    Observation,
    ToolCall,
)


@dataclass
class RunState:
    goal: str
    authority: Authority
    hard_constraints: tuple[str, ...]
    observations: list[Observation] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    submitted_input_tokens: int = 0


@dataclass(frozen=True)
class RunResult:
    answer: str
    events: tuple[Event, ...]
    observations: tuple[Observation, ...]
    steps: int
    submitted_input_tokens: int


Verifier = Callable[[RunState, FinalAnswer], bool]


class Harness:
    def __init__(
        self,
        tools: ToolRegistry,
        context: ContextBuilder,
        verifier: Verifier,
        memory: MemoryStore | None = None,
    ) -> None:
        self.tools = tools
        self.context = context
        self.verifier = verifier
        self.memory = memory

    def run(
        self,
        goal: str,
        model: Model,
        authority: Authority,
        hard_constraints: tuple[str, ...] = (),
        max_steps: int = 12,
    ) -> RunResult:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        state = RunState(goal, authority, hard_constraints)
        for step in range(1, max_steps + 1):
            memories = ()
            if self.memory is not None:
                memories = tuple(
                    item.content
                    for item in self.memory.retrieve(goal, authority)
                )
            prompt = self.context.build(state, memories)
            state.submitted_input_tokens += prompt.estimated_tokens
            proposal = model.propose(prompt)
            label = (
                f"{proposal.name}:{proposal.call_id}"
                if isinstance(proposal, ToolCall)
                else "final answer"
            )
            state.events.append(Event(step, "proposed", label))
            if isinstance(proposal, FinalAnswer):
                if self.verifier(state, proposal):
                    state.events.append(Event(step, "finish", "verified"))
                    return RunResult(
                        proposal.text,
                        tuple(state.events),
                        tuple(state.observations),
                        step,
                        state.submitted_input_tokens,
                    )
                state.events.append(
                    Event(step, "rejected", "unverified answer")
                )
                continue
            if not isinstance(proposal, ToolCall):
                raise TypeError("model returned an unsupported decision")
            try:
                spec = self.tools.admit(
                    proposal, authority, state.observations
                )
            except CallRejected as exc:
                state.events.append(Event(step, "rejected", str(exc)))
                state.observations.append(
                    Observation(
                        proposal.call_id,
                        proposal.name,
                        "rejected",
                        {"error": str(exc)},
                    )
                )
                continue
            state.events.append(Event(step, "admitted", proposal.name))
            result = self.tools.invoke(spec, proposal)
            state.observations.append(result)
            state.events.append(Event(step, "result", result.status))
        raise RuntimeError("step budget exhausted without verified completion")
