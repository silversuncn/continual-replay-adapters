from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", required=True)
    return parser.parse_args()


def _method_summary(result: dict) -> dict:
    return {
        "final_average_accuracy": result["final_average_accuracy"],
        "average_forgetting": result["forgetting"]["average_forgetting"],
        "forgetting_per_task": result["forgetting"].get("per_task", {}),
        "final_scores": result["final_scores"],
        "total_wall_time_seconds": result["total_wall_time_seconds"],
        "step_count": len(result.get("steps", [])),
        "step_tasks": [step.get("task") for step in result.get("steps", [])],
        "step_wall_time_seconds": [step.get("wall_time_seconds") for step in result.get("steps", [])],
    }


def summarize_pilot(pilot_dir: str | Path) -> dict:
    pilot_dir = Path(pilot_dir)
    raw = json.loads((pilot_dir / "pilot_raw_summary.json").read_text())
    methods = {}
    for result in raw["results"]:
        methods[result["method"]] = _method_summary(result)
    no_replay = methods.get("no_replay")
    comparisons = {}
    if no_replay:
        for method, info in methods.items():
            if method == "no_replay":
                continue
            comparisons[method] = {
                "delta_final_average_accuracy": info["final_average_accuracy"]
                - no_replay["final_average_accuracy"],
                "delta_average_forgetting": info["average_forgetting"] - no_replay["average_forgetting"],
                "improves_final_average_accuracy": info["final_average_accuracy"]
                > no_replay["final_average_accuracy"],
                "reduces_average_forgetting": info["average_forgetting"] < no_replay["average_forgetting"],
            }

    required_methods_present = (
        {"no_replay", "random_replay", "class_balanced_replay"}.issubset(methods)
        and bool({"hard_example_replay", "loss_aware_replay"} & set(methods))
    )
    all_methods_completed = bool(methods) and all(info["step_count"] == 4 for info in methods.values())
    no_replay_forgetting_values = list(no_replay.get("forgetting_per_task", {}).values()) if no_replay else []
    no_replay_has_observable_forgetting = any(value > 0 for value in no_replay_forgetting_values)
    any_replay_positive = any(
        info["improves_final_average_accuracy"] or info["reduces_average_forgetting"]
        for info in comparisons.values()
    )
    final_acc_values = [info["final_average_accuracy"] for info in methods.values()]
    forgetting_values = [info["average_forgetting"] for info in methods.values()]
    results_not_identical = (
        (max(final_acc_values) - min(final_acc_values) > 1e-12)
        or (max(forgetting_values) - min(forgetting_values) > 1e-12)
        if final_acc_values and forgetting_values
        else False
    )
    method_times = [info["total_wall_time_seconds"] for info in methods.values()]
    mean_method_time = sum(method_times) / len(method_times) if method_times else 0.0
    estimated_full_matrix_hours_range = [
        mean_method_time * 63 / 3600,
        mean_method_time * 69 / 3600,
    ]
    gate_pass = bool(
        required_methods_present
        and all_methods_completed
        and no_replay_has_observable_forgetting
        and any_replay_positive
        and results_not_identical
    )
    full_matrix_gate = {
        "required_methods_present": required_methods_present,
        "all_methods_completed": all_methods_completed,
        "no_replay_has_observable_forgetting": no_replay_has_observable_forgetting,
        "any_replay_method_positive_vs_no_replay": any_replay_positive,
        "results_not_identical": results_not_identical,
        "estimated_63_to_69_sequential_runs_hours": estimated_full_matrix_hours_range,
        "recommendation": "RECOMMEND_MANAGER_REVIEW_FOR_FULL_MATRIX" if gate_pass else "PILOT_V2_INCONCLUSIVE",
    }
    summary = {
        "status": "PILOT_V2_SUPPORTS_FULL_MATRIX_REVIEW" if gate_pass else "PILOT_V2_INCONCLUSIVE",
        "methods": methods,
        "comparisons_vs_no_replay": comparisons,
        "full_matrix_gate": full_matrix_gate,
        "pilot_release_recommendation": "MANAGER_REVIEW_REQUIRED",
    }
    return summary


def main():
    args = parse_args()
    pilot_dir = Path(args.pilot_dir)
    summary = summarize_pilot(pilot_dir)
    (pilot_dir / "pilot_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
