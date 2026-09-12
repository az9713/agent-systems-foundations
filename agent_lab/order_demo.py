"""An in-memory order agent; no network or payment operation is performed."""

from dataclasses import dataclass
from threading import Lock

from .context import ContextBuilder
from .harness import Harness, RunState
from .memory import MemoryStore
from .tools import OutcomeUnknown, ToolFailure, ToolRegistry, ToolSpec
from .types import (
    Authority,
    FinalAnswer,
    Observation,
    Prompt,
    ToolCall,
)


@dataclass
class Order:
    status: str
    revision: int = 0


class OrderService:
    def __init__(self, status: str, fail_after_commit: bool = False) -> None:
        if status not in {"unshipped", "shipped"}:
            raise ValueError("invalid initial status")
        self.order = Order(status)
        self.fail_after_commit = fail_after_commit
        self._lock = Lock()
        self._committed: dict[
            str, tuple[str, int, dict[str, object]]
        ] = {}

    def read_status(self, order_id: str) -> dict[str, object]:
        if order_id != "O-1":
            raise ToolFailure("unknown order")
        with self._lock:
            return {
                "order_id": order_id,
                "status": self.order.status,
                "revision": self.order.revision,
            }

    def ship(self) -> None:
        """Simulate another actor changing the order between calls."""
        with self._lock:
            if self.order.status != "unshipped":
                raise ValueError("order cannot ship from this state")
            self.order.status = "shipped"
            self.order.revision += 1

    def cancel(
        self, order_id: str, expected_revision: int, key: str
    ) -> dict[str, object]:
        if not key:
            raise ToolFailure("idempotency key required")
        with self._lock:
            old = self._committed.get(key)
            if old is not None:
                old_order_id, old_revision, old_result = old
                if (old_order_id, old_revision) != (
                    order_id, expected_revision
                ):
                    raise ToolFailure(
                        "idempotency key reused with new arguments"
                    )
                return old_result
            if order_id != "O-1":
                raise ToolFailure("unknown order")
            if self.order.status != "unshipped":
                raise ToolFailure("order no longer unshipped")
            if self.order.revision != expected_revision:
                raise ToolFailure("status observation is stale")
            self.order.status = "cancelled"
            self.order.revision += 1
            result = {
                "order_id": order_id,
                "status": "cancelled",
                "revision": self.order.revision,
            }
            self._committed[key] = (order_id, expected_revision, result)
        if self.fail_after_commit:
            self.fail_after_commit = False
            raise OutcomeUnknown("response lost after cancellation committed")
        return result

    @staticmethod
    def return_policy(order_id: str) -> dict[str, object]:
        if order_id != "O-1":
            raise ToolFailure("unknown order")
        return {
            "order_id": order_id,
            "policy": "Contact the merchant for current return terms.",
            "source": "simulated merchant policy",
        }


def has_unshipped_evidence(
    call: ToolCall, observations: tuple[Observation, ...] | list[Observation]
) -> bool:
    """The client gate checks recorded evidence; the service checks again."""
    for item in reversed(observations):
        if item.tool != "get_order_status" or item.status != "ok":
            continue
        return (
            item.data.get("order_id") == call.arguments["order_id"]
            and item.data.get("status") == "unshipped"
            and item.data.get("revision")
            == call.arguments["expected_revision"]
            and bool(call.idempotency_key)
        )
    return False


def build_harness(
    service: OrderService, memory: MemoryStore | None = None
) -> Harness:
    tools = ToolRegistry()
    tools.register(
        ToolSpec(
            "get_order_status",
            {"order_id": str},
            frozenset({"order.read"}),
            lambda call, evidence: True,
            lambda call: service.read_status(call.arguments["order_id"]),
        )
    )
    tools.register(
        ToolSpec(
            "cancel_order",
            {"order_id": str, "expected_revision": int},
            frozenset({"order.cancel"}),
            has_unshipped_evidence,
            lambda call: service.cancel(
                call.arguments["order_id"],
                call.arguments["expected_revision"],
                call.idempotency_key or "",
            ),
        )
    )
    tools.register(
        ToolSpec(
            "get_return_policy",
            {"order_id": str},
            frozenset({"policy.read"}),
            lambda call, evidence: True,
            lambda call: service.return_policy(call.arguments["order_id"]),
        )
    )

    def verify(state: RunState, answer: FinalAnswer) -> bool:
        text = answer.text.casefold()
        if "cancelled" in text:
            return service.read_status("O-1")["status"] == "cancelled"
        if "return" in text:
            return service.read_status("O-1")["status"] == "shipped" and any(
                item.tool == "get_return_policy" and item.status == "ok"
                for item in state.observations
            )
        return False

    return Harness(tools, ContextBuilder(), verify, memory)


class ScriptedOrderModel:
    """A deterministic proposal source standing in for an LLM adapter."""

    def __init__(self) -> None:
        self.calls = 0

    def _call(
        self, name: str, arguments: dict[str, object], key: str | None = None
    ) -> ToolCall:
        self.calls += 1
        return ToolCall(name, arguments, f"call-{self.calls}", key)

    def propose(self, prompt: Prompt) -> ToolCall | FinalAnswer:
        observations = prompt.observations
        if any(
            item.tool == "cancel_order" and item.status == "ok"
            for item in observations
        ):
            return FinalAnswer("Order O-1 was cancelled.")
        if any(
            item.tool == "cancel_order" and item.status in {"unknown", "error"}
            for item in observations
        ):
            latest = observations[-1]
            if latest.tool == "cancel_order":
                return self._call("get_order_status", {"order_id": "O-1"})
        status = next(
            (
                item
                for item in reversed(observations)
                if item.tool == "get_order_status" and item.status == "ok"
            ),
            None,
        )
        if status is None:
            return self._call("get_order_status", {"order_id": "O-1"})
        if status.data["status"] == "cancelled":
            return FinalAnswer("Order O-1 was cancelled.")
        if status.data["status"] == "unshipped":
            return self._call(
                "cancel_order",
                {
                    "order_id": "O-1",
                    "expected_revision": status.data["revision"],
                },
                "cancel-O-1",
            )
        if any(
            item.tool == "get_return_policy" and item.status == "ok"
            for item in observations
        ):
            return FinalAnswer("Order O-1 shipped; check its return terms.")
        return self._call("get_return_policy", {"order_id": "O-1"})


def main() -> None:
    goal = "Cancel order O-1 if unshipped; otherwise explain returns."
    authority = Authority(
        frozenset({"order.read", "order.cancel", "policy.read"})
    )
    for status, lose_response in (("unshipped", True), ("shipped", False)):
        service = OrderService(status, lose_response)
        result = build_harness(service).run(
            goal,
            ScriptedOrderModel(),
            authority,
            ("Do not place an order or charge a card.",),
        )
        outcomes = [
            f"{item.tool}:{item.status}" for item in result.observations
        ]
        print(status, "->", result.answer)
        print("  observations:", ", ".join(outcomes))


if __name__ == "__main__":
    main()
