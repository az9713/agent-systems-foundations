"""Dependency plans and work/span bounds for Chapter 5."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskNode:
    name: str
    duration: float
    dependencies: frozenset[str] = frozenset()


def topological_order(nodes: tuple[TaskNode, ...]) -> tuple[str, ...]:
    """Return a dependency-respecting order or reject an invalid plan."""
    by_name = {node.name: node for node in nodes}
    if len(by_name) != len(nodes):
        raise ValueError("task names must be unique")
    if any(node.duration < 0 for node in nodes):
        raise ValueError("task durations must be nonnegative")
    for node in nodes:
        missing = node.dependencies - by_name.keys()
        if missing:
            raise ValueError(f"unknown dependencies: {sorted(missing)}")
    remaining = set(by_name)
    completed: set[str] = set()
    order: list[str] = []
    while remaining:
        ready = sorted(
            name for name in remaining
            if by_name[name].dependencies <= completed
        )
        if not ready:
            raise ValueError("dependency graph contains a cycle")
        order.extend(ready)
        completed.update(ready)
        remaining.difference_update(ready)
    return tuple(order)


def ready_tasks(
    nodes: tuple[TaskNode, ...], completed: frozenset[str]
) -> tuple[str, ...]:
    """Return unfinished tasks whose dependencies are complete."""
    topological_order(nodes)
    return tuple(sorted(
        node.name for node in nodes
        if node.name not in completed
        and node.dependencies <= completed
    ))


def work_span_bound(
    nodes: tuple[TaskNode, ...], workers: int
) -> tuple[float, float, float]:
    """Return total work, critical-path span, and their runtime bound."""
    if workers < 1:
        raise ValueError("workers must be positive")
    by_name = {node.name: node for node in nodes}
    order = topological_order(nodes)
    finish: dict[str, float] = {}
    for name in order:
        node = by_name[name]
        start = max((finish[dep] for dep in node.dependencies), default=0.0)
        finish[name] = start + node.duration
    work = sum(node.duration for node in nodes)
    span = max(finish.values(), default=0.0)
    return work, span, max(work / workers, span)
