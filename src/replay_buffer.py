from __future__ import annotations

from collections import defaultdict
from random import Random


def _task_seed(seed: int, task_name: str, offset: int = 0) -> int:
    return seed + offset + sum(ord(ch) for ch in task_name)


class RandomReplayBuffer:
    def __init__(self, budget_per_task: int, seed: int = 0):
        if budget_per_task < 0:
            raise ValueError("budget_per_task must be non-negative")
        self.budget_per_task = budget_per_task
        self.seed = seed
        self.examples_by_task: dict[str, list[dict]] = {}

    def add_task_examples(self, task_name: str, examples: list[dict]) -> None:
        rows = [dict(row) for row in examples]
        rng = Random(_task_seed(self.seed, task_name, offset=101))
        rng.shuffle(rows)
        self.examples_by_task[task_name] = rows[: self.budget_per_task]

    def sample_replay(self, task_names: list[str]) -> list[dict]:
        rows: list[dict] = []
        for task_name in task_names:
            rows.extend(dict(row) for row in self.examples_by_task.get(task_name, []))
        rng = Random(self.seed + 29 * len(task_names))
        rng.shuffle(rows)
        return rows


class ClassBalancedReplayBuffer:
    def __init__(self, budget_per_task: int, seed: int = 0):
        if budget_per_task < 0:
            raise ValueError("budget_per_task must be non-negative")
        self.budget_per_task = budget_per_task
        self.seed = seed
        self.examples_by_task: dict[str, list[dict]] = {}

    def add_task_examples(self, task_name: str, examples: list[dict]) -> None:
        self.examples_by_task[task_name] = self._select_balanced(examples, self.budget_per_task, task_name)

    def sample_replay(self, task_names: list[str]) -> list[dict]:
        rows: list[dict] = []
        for task_name in task_names:
            rows.extend(dict(row) for row in self.examples_by_task.get(task_name, []))
        rng = Random(self.seed + 17 * len(task_names))
        rng.shuffle(rows)
        return rows

    def _select_balanced(self, examples: list[dict], budget: int, task_name: str) -> list[dict]:
        if budget == 0 or not examples:
            return []
        rng = Random(_task_seed(self.seed, task_name))
        by_label: dict[int, list[dict]] = defaultdict(list)
        for row in examples:
            by_label[int(row["label"])].append(dict(row))
        for label_rows in by_label.values():
            rng.shuffle(label_rows)

        selected: list[dict] = []
        labels = sorted(by_label)
        while len(selected) < budget and any(by_label.values()):
            for label in labels:
                if by_label[label] and len(selected) < budget:
                    selected.append(by_label[label].pop())
        rng.shuffle(selected)
        return selected


class HardExampleReplayBuffer:
    """Stores the highest-loss examples from each completed task.

    The pilot computes per-example training loss after each task and passes it
    as `hardness_scores`. Tests may provide a precomputed `hardness` field.
    """

    def __init__(self, budget_per_task: int, seed: int = 0):
        if budget_per_task < 0:
            raise ValueError("budget_per_task must be non-negative")
        self.budget_per_task = budget_per_task
        self.seed = seed
        self.examples_by_task: dict[str, list[dict]] = {}

    def add_task_examples(
        self,
        task_name: str,
        examples: list[dict],
        hardness_scores: list[float] | None = None,
    ) -> None:
        if hardness_scores is not None and len(hardness_scores) != len(examples):
            raise ValueError("hardness_scores must match examples length")
        rows: list[dict] = []
        for index, row in enumerate(examples):
            copied = dict(row)
            if hardness_scores is not None:
                copied["hardness"] = float(hardness_scores[index])
            else:
                copied["hardness"] = float(copied.get("hardness", 0.0))
            rows.append(copied)
        rows.sort(key=lambda item: (-float(item["hardness"]), str(item.get("text", ""))))
        self.examples_by_task[task_name] = rows[: self.budget_per_task]

    def sample_replay(self, task_names: list[str]) -> list[dict]:
        rows: list[dict] = []
        for task_name in task_names:
            rows.extend(dict(row) for row in self.examples_by_task.get(task_name, []))
        rng = Random(self.seed + 43 * len(task_names))
        rng.shuffle(rows)
        return rows


def create_replay_buffer(method: str, budget_per_task: int, seed: int = 0):
    if method == "no_replay":
        return None
    if method == "random_replay":
        return RandomReplayBuffer(budget_per_task, seed=seed)
    if method == "class_balanced_replay":
        return ClassBalancedReplayBuffer(budget_per_task, seed=seed)
    if method in {"hard_example_replay", "loss_aware_replay"}:
        return HardExampleReplayBuffer(budget_per_task, seed=seed)
    raise ValueError(f"Unknown replay method: {method}")
