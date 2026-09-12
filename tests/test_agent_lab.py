"""Behavioral checks for the equations and boundaries used in the chapters."""

import tempfile
import unittest
from pathlib import Path

from agent_lab.context import (
    Candidate,
    ContextBuilder,
    compact_state,
    compacted_input_tokens,
    kv_cache_bytes,
    select_items,
    submitted_input_tokens,
)
from agent_lab.harness import RunState
from agent_lab.memory import MemoryItem, MemoryStore
from agent_lab.order_demo import (
    OrderService,
    ScriptedOrderModel,
    build_harness,
)
from agent_lab.tools import CallRejected
from agent_lab.types import Authority, FinalAnswer, Observation, ToolCall


class HarnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.goal = "Cancel order O-1 if unshipped; otherwise explain returns."
        self.rights = Authority(
            frozenset({"order.read", "order.cancel", "policy.read"})
        )

    def test_timeout_after_commit_is_reconciled_by_read(self) -> None:
        service = OrderService("unshipped", fail_after_commit=True)
        result = build_harness(service).run(
            self.goal, ScriptedOrderModel(), self.rights
        )
        self.assertEqual(service.read_status("O-1")["status"], "cancelled")
        self.assertEqual(len(service._committed), 1)
        self.assertEqual(
            [item.status for item in result.observations],
            ["ok", "unknown", "ok"],
        )
        self.assertEqual(result.events[-1].detail, "verified")

    def test_shipped_order_selects_read_only_branch(self) -> None:
        service = OrderService("shipped")
        result = build_harness(service).run(
            self.goal, ScriptedOrderModel(), self.rights
        )
        self.assertIn("return", result.answer)
        self.assertEqual(service.read_status("O-1")["status"], "shipped")
        self.assertNotIn(
            "cancel_order", [item.tool for item in result.observations]
        )

    def test_permission_gate_rejects_call_before_execution(self) -> None:
        service = OrderService("unshipped")
        tools = build_harness(service).tools
        status = Observation(
            "read-1", "get_order_status", "ok", service.read_status("O-1")
        )
        call = ToolCall(
            "cancel_order",
            {"order_id": "O-1", "expected_revision": 0},
            "cancel-1",
            "key-1",
        )
        read_only = Authority(frozenset({"order.read"}))
        with self.assertRaisesRegex(CallRejected, "not authorized"):
            tools.admit(call, read_only, [status])
        self.assertEqual(service.read_status("O-1")["status"], "unshipped")

    def test_service_rechecks_state_after_client_admission(self) -> None:
        service = OrderService("unshipped")
        tools = build_harness(service).tools
        status = Observation(
            "read-1", "get_order_status", "ok", service.read_status("O-1")
        )
        call = ToolCall(
            "cancel_order",
            {"order_id": "O-1", "expected_revision": 0},
            "cancel-1",
            "key-1",
        )
        spec = tools.admit(call, self.rights, [status])
        service.ship()
        result = tools.invoke(spec, call)
        self.assertEqual(result.status, "error")
        self.assertEqual(service.read_status("O-1")["status"], "shipped")

    def test_idempotency_key_deduplicates_same_order(self) -> None:
        service = OrderService("unshipped")
        first = service.cancel("O-1", 0, "one-logical-cancel")
        second = service.cancel("O-1", 0, "one-logical-cancel")
        self.assertEqual(first, second)
        self.assertEqual(service.read_status("O-1")["revision"], 1)
        with self.assertRaisesRegex(ValueError, "new arguments"):
            service.cancel("O-1", 1, "one-logical-cancel")

    def test_final_answer_requires_verification(self) -> None:
        class UnsupportedModel:
            def propose(self, prompt):
                return FinalAnswer("Order O-1 was cancelled.")

        service = OrderService("unshipped")
        harness = build_harness(service)
        with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
            harness.run(
                self.goal, UnsupportedModel(), self.rights, max_steps=2
            )
        self.assertEqual(service.read_status("O-1")["status"], "unshipped")

    def test_proposal_trace_does_not_copy_arguments(self) -> None:
        class SensitiveModel:
            def __init__(self) -> None:
                self.first = True
                self.delegate = ScriptedOrderModel()

            def propose(self, prompt):
                if self.first:
                    self.first = False
                    return ToolCall(
                        "get_order_status",
                        {"order_id": "O-1", "secret": "private-value"},
                        "rejected-1",
                    )
                return self.delegate.propose(prompt)

        result = build_harness(OrderService("shipped")).run(
            self.goal, SensitiveModel(), self.rights
        )
        self.assertNotIn("private-value", repr(result.events))


class ContextTests(unittest.TestCase):
    def test_chapter_three_token_equations(self) -> None:
        self.assertEqual(submitted_input_tokens(100, 2000, 300), 1_685_000)
        self.assertEqual(
            compacted_input_tokens(100, 2000, 300, 20, 1000), 565_000
        )
        self.assertEqual(
            kv_cache_bytes(32, 128_000, 8, 128, 2), 16_777_216_000
        )

    def test_context_knapsack_checks_combinations(self) -> None:
        items = [
            Candidate("large", 6, 8.0),
            Candidate("small-a", 3, 5.0),
            Candidate("small-b", 3, 5.0),
        ]
        self.assertEqual(
            select_items(items, 6), ("small-a", "small-b")
        )

    def test_compaction_preserves_constraints(self) -> None:
        state = RunState(
            "Repair the build",
            Authority(frozenset()),
            ("Do not publish.",),
        )
        state.observations.extend(
            [
                Observation("call-1", "read", "ok", {"value": 1}),
                Observation("call-2", "test", "ok", {"value": 2}),
            ]
        )
        compact = compact_state(state, keep_recent=1)
        self.assertEqual(compact.hard_constraints, ("Do not publish.",))
        self.assertEqual(compact.evidence_ids, ("call-1", "call-2"))
        self.assertEqual(compact.recent_observations[0].call_id, "call-2")
        prompt = ContextBuilder(keep_recent=1).build(state, ())
        self.assertIn("Do not publish.", prompt.hard_constraints)


class MemoryTests(unittest.TestCase):
    def test_skill_enters_prompt_without_granting_effects(self) -> None:
        store = MemoryStore()
        store.write(
            MemoryItem(
                "return-skill", "skill",
                "Read the current return policy before answering.",
                "trusted authored procedure", "shopping", 0.0, None,
                8, 0.3, 0.01, ("return",),
                frozenset({"policy.read"}),
            )
        )

        class CapturingModel:
            def __init__(self) -> None:
                self.prompts = []

            def propose(self, prompt):
                self.prompts.append(prompt)
                return FinalAnswer(
                    "Order O-1 shipped; check its return terms."
                )

        model = CapturingModel()
        service = OrderService("shipped")
        authority = Authority(
            frozenset({"policy.read"}), frozenset({"shopping"})
        )
        with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
            build_harness(service, store).run(
                "Find return terms", model, authority, max_steps=1
            )
        self.assertEqual(
            model.prompts[0].memories,
            ("Read the current return policy before answering.",),
        )
        self.assertEqual(authority.effects, frozenset({"policy.read"}))

    def test_skill_obeys_scope_time_effects_and_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            store = MemoryStore(path)
            store.write(
                MemoryItem(
                    "return-skill",
                    "skill",
                    "Find the merchant's current dated return policy.",
                    "authored and tested procedure",
                    "shopping",
                    0.0,
                    200.0,
                    10,
                    0.3,
                    0.02,
                    ("return",),
                    frozenset({"policy.read"}),
                )
            )
            store.write(
                MemoryItem(
                    "stale-policy",
                    "fact",
                    "Old return deadline",
                    "old merchant page",
                    "shopping",
                    0.0,
                    50.0,
                    5,
                    0.5,
                    0.01,
                    ("return",),
                )
            )
            reopened = MemoryStore(path)
            authority = Authority(
                frozenset({"policy.read"}), frozenset({"shopping"})
            )
            found = reopened.retrieve(
                "Find return terms", authority, budget=10, now=100.0
            )
            self.assertEqual(
                [item.item_id for item in found], ["return-skill"]
            )
            no_right = Authority(frozenset(), frozenset({"shopping"}))
            self.assertEqual(
                reopened.retrieve("Find return terms", no_right, now=100.0),
                (),
            )
            self.assertEqual(
                reopened.retrieve("Find return terms", authority, now=201.0),
                (),
            )


if __name__ == "__main__":
    unittest.main()
