from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

from paper_tasks import load_task_examples
from run_pilot import run_method


TASK_ORDERS = {
    "O1": ["sst2", "mrpc", "rte", "ag_news"],
    "O2": ["ag_news", "rte", "mrpc", "sst2"],
}
SEEDS = [113, 227, 349]
REPLAY_METHODS = ["random_replay", "class_balanced_replay", "hard_example_replay"]
REPLAY_BUDGETS = [16, 64, 256]


def _cell_id(index: int, order_name: str, seed: int, method: str, budget: int) -> str:
    return f"cell_{index:03d}_{order_name}_seed{seed}_{method}_b{budget}"


def build_core_matrix_plan() -> list[dict]:
    cells: list[dict] = []
    index = 1
    for order_name in ["O1", "O2"]:
        for seed in SEEDS:
            cells.append(
                {
                    "cell_id": _cell_id(index, order_name, seed, "no_replay", 0),
                    "order_name": order_name,
                    "task_sequence": TASK_ORDERS[order_name],
                    "seed": seed,
                    "method": "no_replay",
                    "budget": 0,
                }
            )
            index += 1
            for method in REPLAY_METHODS:
                for budget in REPLAY_BUDGETS:
                    cells.append(
                        {
                            "cell_id": _cell_id(index, order_name, seed, method, budget),
                            "order_name": order_name,
                            "task_sequence": TASK_ORDERS[order_name],
                            "seed": seed,
                            "method": method,
                            "budget": budget,
                        }
                    )
                    index += 1
    return cells


def build_launch_smoke_plan() -> list[dict]:
    return [
        {
            "cell_id": "smoke_001_O1_seed113_no_replay_b0",
            "order_name": "O1",
            "task_sequence": TASK_ORDERS["O1"],
            "seed": 113,
            "method": "no_replay",
            "budget": 0,
        },
        {
            "cell_id": "smoke_002_O1_seed113_class_balanced_replay_b64",
            "order_name": "O1",
            "task_sequence": TASK_ORDERS["O1"],
            "seed": 113,
            "method": "class_balanced_replay",
            "budget": 64,
        },
        {
            "cell_id": "smoke_003_O2_seed113_hard_example_replay_b64",
            "order_name": "O2",
            "task_sequence": TASK_ORDERS["O2"],
            "seed": 113,
            "method": "hard_example_replay",
            "budget": 64,
        },
    ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--plan", choices=["core", "smoke"], default="core")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--train-limit", type=int, default=512)
    parser.add_argument("--eval-limit", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser.parse_args()


def _load_task_data(task_sequence: list[str], train_limit: int, eval_limit: int, seed: int) -> dict:
    task_data = {}
    for task in task_sequence:
        train, eval_rows = load_task_examples(task, train_limit, eval_limit, seed)
        task_data[task] = {"train": train, "eval": eval_rows}
    return task_data


def _run_cell(args, cell: dict, output_dir: Path, device: torch.device) -> dict:
    cell_dir = output_dir / "cells" / cell["cell_id"]
    cell_dir.mkdir(parents=True, exist_ok=True)
    result_path = cell_dir / "result.json"
    error_path = cell_dir / "error.txt"
    if args.resume and result_path.exists():
        result = json.loads(result_path.read_text())
        return {
            **cell,
            "status": "PASS",
            "result_path": str(result_path),
            "total_wall_time_seconds": result.get("total_wall_time_seconds"),
            "resumed": True,
        }

    cell_start = time.time()
    task_data = _load_task_data(cell["task_sequence"], args.train_limit, args.eval_limit, cell["seed"])
    method_args = SimpleNamespace(
        model_name=args.model_name,
        seed=cell["seed"],
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
        replay_budget=cell["budget"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
    )
    try:
        result = run_method(
            method_args,
            cell["method"],
            task_data,
            device,
            task_sequence=cell["task_sequence"],
            order_name=cell["order_name"],
        )
        result.update(
            {
                "cell_id": cell["cell_id"],
                "order_name": cell["order_name"],
                "budget": cell["budget"],
                "phase3_train_limit": args.train_limit,
                "phase3_eval_limit": args.eval_limit,
            }
        )
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        return {
            **cell,
            "status": "PASS",
            "result_path": str(result_path),
            "total_wall_time_seconds": time.time() - cell_start,
            "resumed": False,
        }
    except Exception:
        error_path.write_text(traceback.format_exc())
        if args.stop_on_failure:
            raise
        return {
            **cell,
            "status": "FAIL",
            "error_path": str(error_path),
            "total_wall_time_seconds": time.time() - cell_start,
            "resumed": False,
        }


def run_matrix(args) -> dict:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cells = build_core_matrix_plan() if args.plan == "core" else build_launch_smoke_plan()
    (output_dir / "matrix_plan.json").write_text(json.dumps(cells, indent=2, ensure_ascii=False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.time()
    cell_records = []
    for index, cell in enumerate(cells, start=1):
        progress = {
            "status": "RUNNING",
            "plan": args.plan,
            "current_index": index,
            "expected_run_count": len(cells),
            "current_cell": cell,
            "completed": cell_records,
        }
        (output_dir / "progress.json").write_text(json.dumps(progress, indent=2, ensure_ascii=False))
        cell_records.append(_run_cell(args, cell, output_dir, device))
    failed = [cell for cell in cell_records if cell["status"] != "PASS"]
    summary = {
        "status": "PASS_FULL_MATRIX_COMPLETED" if not failed else "FAIL_FULL_MATRIX_HAS_FAILED_CELLS",
        "plan": args.plan,
        "device": str(device),
        "model_name": args.model_name,
        "train_limit": args.train_limit,
        "eval_limit": args.eval_limit,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "expected_run_count": len(cells),
        "actual_run_count": len(cell_records),
        "failed_run_count": len(failed),
        "started_at_unix": start,
        "ended_at_unix": time.time(),
        "total_wall_time_seconds": time.time() - start,
        "cells": cell_records,
    }
    (output_dir / "matrix_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (output_dir / "progress.json").write_text(json.dumps({**summary, "status": "DONE"}, indent=2, ensure_ascii=False))
    return summary


def main():
    args = parse_args()
    summary = run_matrix(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["failed_run_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
