from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from continual_runner import (
    build_model_and_tokenizer,
    evaluate_examples,
    score_examples_by_loss,
    set_seed,
    train_on_examples,
)
from metrics import compute_average_forgetting, final_average_accuracy
from paper_tasks import TASK_SEQUENCE, get_task_spec, load_task_examples
from replay_buffer import create_replay_buffer


def parse_task_order(value: str | None) -> list[str]:
    if not value:
        return list(TASK_SEQUENCE)
    tasks = [part.strip() for part in value.split(",") if part.strip()]
    if sorted(tasks) != sorted(TASK_SEQUENCE) or len(tasks) != len(TASK_SEQUENCE):
        raise ValueError(f"task order must contain each task exactly once: {TASK_SEQUENCE}")
    return tasks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--train-limit", type=int, default=24)
    parser.add_argument("--eval-limit", type=int, default=64)
    parser.add_argument("--replay-budget", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--methods", default="no_replay,random_replay,class_balanced_replay,hard_example_replay")
    parser.add_argument("--sequence", dest="task_order", default=",".join(TASK_SEQUENCE))
    parser.add_argument("--order-name", default="O1")
    return parser.parse_args()


def run_method(
    args,
    method: str,
    task_data: dict,
    device: torch.device,
    task_sequence: list[str] | None = None,
    order_name: str | None = None,
) -> dict:
    task_sequence = list(task_sequence or TASK_SEQUENCE)
    set_seed(args.seed)
    task_label_counts = {task: len(get_task_spec(task).label_names) for task in task_sequence}
    model, tokenizer = build_model_and_tokenizer(args.model_name, task_label_counts, device)
    buffer = create_replay_buffer(method, args.replay_budget, seed=args.seed)
    seen: list[str] = []
    history: dict[str, list[float]] = {task: [] for task in task_sequence}
    steps = []
    method_start = time.time()

    for step_index, task in enumerate(task_sequence, start=1):
        train_rows = list(task_data[task]["train"])
        replay_rows = buffer.sample_replay(seen) if buffer is not None else []
        combined = train_rows + replay_rows
        stats = train_on_examples(
            model=model,
            tokenizer=tokenizer,
            examples=combined,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
        )
        hardness_summary = None
        if method in {"hard_example_replay", "loss_aware_replay"}:
            hardness_scores = score_examples_by_loss(
                model=model,
                tokenizer=tokenizer,
                examples=train_rows,
                device=device,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )
            hardness_summary = {
                "min": min(hardness_scores) if hardness_scores else None,
                "max": max(hardness_scores) if hardness_scores else None,
                "mean": sum(hardness_scores) / len(hardness_scores) if hardness_scores else None,
            }
            if buffer is not None:
                buffer.add_task_examples(task, train_rows, hardness_scores=hardness_scores)
        elif buffer is not None:
            buffer.add_task_examples(task, train_rows)
        seen.append(task)

        evals = {}
        for eval_task in seen:
            score = evaluate_examples(
                model=model,
                tokenizer=tokenizer,
                examples=task_data[eval_task]["eval"],
                device=device,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )
            evals[eval_task] = score
            history[eval_task].append(score["accuracy"])

        steps.append(
            {
                "step": step_index,
                "task": task,
                "train_examples_current": len(train_rows),
                "train_examples_replay": len(replay_rows),
                "train_examples_total": len(combined),
                "replay_source_tasks": list(seen[:-1]),
                "hardness_summary": hardness_summary,
                "loss": stats.loss,
                "wall_time_seconds": stats.wall_time_seconds,
                "peak_gpu_memory_mb": stats.peak_gpu_memory_mb,
                "max_rss_mb": stats.max_rss_mb,
                "eval": evals,
            }
        )

    final_scores = {task: values[-1] for task, values in history.items() if values}
    forgetting = compute_average_forgetting(history)
    return {
        "method": method,
        "model_name": args.model_name,
        "seed": args.seed,
        "order_name": order_name,
        "task_sequence": task_sequence,
        "train_limit": args.train_limit,
        "eval_limit": args.eval_limit,
        "replay_budget": args.replay_budget,
        "replay_method_definition": {
            "no_replay": "Train on current task examples only.",
            "random_replay": "Store a seeded random subset of each previous task up to the per-task budget.",
            "class_balanced_replay": "Store a label-balanced subset of each previous task up to the per-task budget.",
            "hard_example_replay": "Store examples with the highest post-training per-example loss for each previous task.",
            "loss_aware_replay": "Alias of hard_example_replay in this Pilot implementation.",
        }.get(method, method),
        "steps": steps,
        "final_scores": final_scores,
        "final_average_accuracy": final_average_accuracy(final_scores),
        "forgetting": forgetting,
        "total_wall_time_seconds": time.time() - method_start,
    }


def main():
    args = parse_args()
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_sequence = parse_task_order(args.task_order)
    task_data = {}
    for task in task_sequence:
        train, eval_rows = load_task_examples(task, args.train_limit, args.eval_limit, args.seed)
        task_data[task] = {"train": train, "eval": eval_rows}
    dataset_summary = {
        task: {
            "train_rows": len(data["train"]),
            "eval_rows": len(data["eval"]),
            "labels": get_task_spec(task).label_names,
            "sample": data["train"][0],
        }
        for task, data in task_data.items()
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2, ensure_ascii=False))

    methods = [part.strip() for part in args.methods.split(",") if part.strip()]
    results = []
    for method in methods:
        result = run_method(args, method, task_data, device, task_sequence=task_sequence, order_name=args.order_name)
        (output_dir / f"{method}_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
        results.append(result)
    summary = {
        "status": "PASS_PILOT_RUN_COMPLETED",
        "device": str(device),
        "methods": methods,
        "results": results,
    }
    (output_dir / "pilot_raw_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
