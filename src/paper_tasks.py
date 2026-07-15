from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Callable

from datasets import load_dataset


@dataclass(frozen=True)
class TaskSpec:
    name: str
    dataset_name: str
    config: str | None
    train_split: str
    eval_split: str
    label_names: list[str]
    text_builder: Callable[[dict], str]


def _single_text(row: dict) -> str:
    return str(row["text"])


def _sentence(row: dict) -> str:
    return str(row["sentence"])


def _sentence_pair(row: dict) -> str:
    return f"{row['sentence1']} [SEP] {row['sentence2']}"


TASK_SEQUENCE = ["sst2", "mrpc", "rte", "ag_news"]

TASK_SPECS = {
    "sst2": TaskSpec(
        name="sst2",
        dataset_name="glue",
        config="sst2",
        train_split="train",
        eval_split="validation",
        label_names=["negative", "positive"],
        text_builder=_sentence,
    ),
    "mrpc": TaskSpec(
        name="mrpc",
        dataset_name="glue",
        config="mrpc",
        train_split="train",
        eval_split="validation",
        label_names=["not_equivalent", "equivalent"],
        text_builder=_sentence_pair,
    ),
    "rte": TaskSpec(
        name="rte",
        dataset_name="glue",
        config="rte",
        train_split="train",
        eval_split="validation",
        label_names=["entailment", "not_entailment"],
        text_builder=_sentence_pair,
    ),
    "ag_news": TaskSpec(
        name="ag_news",
        dataset_name="ag_news",
        config=None,
        train_split="train",
        eval_split="test",
        label_names=["World", "Sports", "Business", "Sci/Tech"],
        text_builder=_single_text,
    ),
}


def get_task_spec(task_name: str) -> TaskSpec:
    try:
        return TASK_SPECS[task_name]
    except KeyError as exc:
        raise ValueError(f"Unknown task: {task_name}") from exc


def load_raw_dataset(spec: TaskSpec):
    if spec.config:
        return load_dataset(spec.dataset_name, spec.config)
    return load_dataset(spec.dataset_name)


def select_examples(task_name: str, split: str, limit: int, seed: int) -> list[dict]:
    spec = get_task_spec(task_name)
    dataset = load_raw_dataset(spec)
    rows = list(dataset[split])
    rng = Random(seed)
    by_label: dict[int, list[dict]] = {}
    for row in rows:
        by_label.setdefault(int(row["label"]), []).append(row)
    for label_rows in by_label.values():
        rng.shuffle(label_rows)

    selected: list[dict] = []
    labels = sorted(by_label)
    while len(selected) < limit and any(by_label.values()):
        for label in labels:
            if by_label[label] and len(selected) < limit:
                row = by_label[label].pop()
                selected.append(
                    {
                        "task": task_name,
                        "text": spec.text_builder(row),
                        "label": int(row["label"]),
                    }
                )
    rng.shuffle(selected)
    return selected


def load_task_examples(task_name: str, train_limit: int, eval_limit: int, seed: int) -> tuple[list[dict], list[dict]]:
    spec = get_task_spec(task_name)
    train = select_examples(task_name, spec.train_split, train_limit, seed)
    eval_rows = select_examples(task_name, spec.eval_split, eval_limit, seed + 100_000)
    return train, eval_rows
