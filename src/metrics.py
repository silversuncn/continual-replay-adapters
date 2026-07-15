from __future__ import annotations


def accuracy(predictions: list[int], labels: list[int]) -> float:
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have the same length")
    if not labels:
        return 0.0
    return sum(int(p == y) for p, y in zip(predictions, labels)) / len(labels)


def compute_average_forgetting(history: dict[str, list[float]]) -> dict:
    per_task: dict[str, float] = {}
    for task, values in history.items():
        if len(values) < 2:
            continue
        best_before_final = max(values[:-1])
        final = values[-1]
        per_task[task] = max(0.0, best_before_final - final)
    avg = sum(per_task.values()) / len(per_task) if per_task else 0.0
    return {"average_forgetting": avg, "per_task": per_task}


def final_average_accuracy(final_scores: dict[str, float]) -> float:
    return sum(final_scores.values()) / len(final_scores) if final_scores else 0.0
