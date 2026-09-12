"""Tool schemas, admission checks, and explicit uncertain outcomes."""

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .types import Authority, Observation, ToolCall


class CallRejected(Exception):
    """The harness rejected a proposal before execution."""


class OutcomeUnknown(Exception):
    """A tool may have committed, but the caller did not receive its result."""


class ToolFailure(ValueError):
    """The tool rejected a request before making any external change."""


Precondition = Callable[[ToolCall, Sequence[Observation]], bool]
Handler = Callable[[ToolCall], Mapping[str, Any]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    fields: Mapping[str, type]
    effects: frozenset[str]
    precondition: Precondition
    handler: Handler


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"duplicate tool: {spec.name}")
        self._tools[spec.name] = spec

    def admit(
        self,
        call: ToolCall,
        authority: Authority,
        evidence: Sequence[Observation],
    ) -> ToolSpec:
        spec = self._tools.get(call.name)
        if spec is None:
            raise CallRejected("unknown tool")
        if set(call.arguments) != set(spec.fields):
            raise CallRejected("argument fields do not match schema")
        for field, expected_type in spec.fields.items():
            if type(call.arguments[field]) is not expected_type:
                raise CallRejected(f"wrong type for {field}")
        if not spec.effects <= authority.effects:
            raise CallRejected("effect not authorized")
        if not spec.precondition(call, evidence):
            raise CallRejected("precondition lacks trusted evidence")
        return spec

    @staticmethod
    def invoke(spec: ToolSpec, call: ToolCall) -> Observation:
        try:
            data = spec.handler(call)
            return Observation(call.call_id, call.name, "ok", data)
        except OutcomeUnknown as exc:
            return Observation(
                call.call_id, call.name, "unknown", {"error": str(exc)}
            )
        except ToolFailure as exc:
            return Observation(
                call.call_id, call.name, "error", {"error": str(exc)}
            )
        except Exception as exc:
            return Observation(
                call.call_id, call.name, "unknown", {"error": str(exc)}
            )
