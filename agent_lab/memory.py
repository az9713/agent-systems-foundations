"""File-backed facts and skills with scope, validity, and value selection."""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .context import Candidate, select_items
from .types import Authority


@dataclass(frozen=True)
class MemoryItem:
    item_id: str
    kind: Literal["fact", "skill"]
    content: str
    source: str
    scope: str
    valid_from: float
    valid_until: float | None
    token_cost: int
    estimated_gain: float
    estimated_risk: float
    trigger_terms: tuple[str, ...] = ()
    required_effects: frozenset[str] = frozenset()

    def applicable(self, goal: str, authority: Authority, now: float) -> bool:
        if self.scope not in authority.memory_scopes:
            return False
        if not self.required_effects <= authority.effects:
            return False
        if now < self.valid_from:
            return False
        if self.valid_until is not None and now >= self.valid_until:
            return False
        lowered = goal.casefold()
        return not self.trigger_terms or any(
            term.casefold() in lowered for term in self.trigger_terms
        )


class MemoryStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._items: dict[str, MemoryItem] = {}
        if path is not None and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            for record in data:
                record["trigger_terms"] = tuple(record["trigger_terms"])
                record["required_effects"] = frozenset(
                    record["required_effects"]
                )
                item = MemoryItem(**record)
                self._items[item.item_id] = item

    def write(self, item: MemoryItem) -> None:
        if item.token_cost < 1:
            raise ValueError("token_cost must be positive")
        self._items[item.item_id] = item
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                **vars(value),
                "trigger_terms": list(value.trigger_terms),
                "required_effects": sorted(value.required_effects),
            }
            for value in self._items.values()
        ]
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(temporary, self.path)

    def retrieve(
        self,
        goal: str,
        authority: Authority,
        budget: int = 80,
        token_price: float = 0.001,
        now: float | None = None,
    ) -> tuple[MemoryItem, ...]:
        current = time.time() if now is None else now
        eligible = [
            item
            for item in self._items.values()
            if item.applicable(goal, authority, current)
        ]
        candidates = [
            Candidate(
                item.item_id,
                item.token_cost,
                item.estimated_gain
                - item.estimated_risk
                - token_price * item.token_cost,
            )
            for item in eligible
        ]
        selected = set(select_items(candidates, budget))
        return tuple(
            item for item in eligible if item.item_id in selected
        )
