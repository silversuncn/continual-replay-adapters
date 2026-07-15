from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"


def _read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_float(value: str) -> float:
    return float(value)


def _assert_close(actual: float, expected: float, name: str, tol: float = 1e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tol):
        raise AssertionError(f"{name}: expected {expected}, got {actual}")


def main() -> None:
    results = _read_csv("results.csv")
    aggregated = _read_csv("results_aggregated.csv")
    summary = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))
    statistics = json.loads((DATA / "statistics.json").read_text(encoding="utf-8"))

    if len(results) != 60:
        raise AssertionError(f"results.csv row count should be 60, got {len(results)}")
    if len(aggregated) != 10:
        raise AssertionError(f"results_aggregated.csv row count should be 10, got {len(aggregated)}")
    if "result_path" in results[0]:
        raise AssertionError("results.csv should not expose result_path")

    required_methods = {
        ("no_replay", "0"),
        ("random_replay", "16"),
        ("random_replay", "64"),
        ("random_replay", "256"),
        ("class_balanced_replay", "16"),
        ("class_balanced_replay", "64"),
        ("class_balanced_replay", "256"),
        ("hard_example_replay", "16"),
        ("hard_example_replay", "64"),
        ("hard_example_replay", "256"),
    }
    observed_methods = {(row["method"], row["budget"]) for row in aggregated}
    if observed_methods != required_methods:
        raise AssertionError(f"unexpected method-budget rows: {sorted(observed_methods)}")

    best = summary["best_final_average_accuracy"]
    if best["method"] != "random_replay" or best["budget"] != 256:
        raise AssertionError("summary best row should be random_replay budget 256")
    _assert_close(best["mean_final_average_accuracy"], 0.6873283221512883, "best accuracy")
    _assert_close(best["mean_average_forgetting"], 0.01390508005749431, "best forgetting")

    match = None
    for item in statistics["pairwise_vs_no_replay"]:
        if item["method"] == "random_replay" and item["budget"] == 256:
            match = item
            break
    if match is None:
        raise AssertionError("missing random_replay budget 256 pairwise statistics")
    _assert_close(match["accuracy"]["mean_delta"], 0.06195131294094056, "accuracy delta")
    _assert_close(match["accuracy"]["holm_p_value"], 0.375, "accuracy Holm p-value")

    for figure in [
        "figure_method_budget_accuracy.png",
        "figure_forgetting_by_method_budget.png",
        "figure_accuracy_runtime.png",
    ]:
        if not (FIGURES / figure).is_file():
            raise AssertionError(f"missing figure: {figure}")

    combined_text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in DATA.glob("*"))
    blocked_terms = [
        "/home/",
        "/Users/",
        "Evo" + "Scientist",
        "open" + "ai",
        "cla" + "ude",
        "ns" + "ahub",
    ]
    for term in blocked_terms:
        if term in combined_text:
            raise AssertionError(f"blocked internal token in public data: {term}")
    if "sk" + "-" in combined_text:
        raise AssertionError("blocked secret-like token in public data")

    print("PASS: public data checks completed")


if __name__ == "__main__":
    main()
