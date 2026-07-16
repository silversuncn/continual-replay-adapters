#!/usr/bin/env python3
"""Quick consistency checks for the public analysis package."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    summary = json.loads((DATA / "summary.json").read_text())
    cells = load_csv(DATA / "full_matrix_cells.csv")
    aggregates = load_csv(DATA / "aggregate_results.csv")

    assert len(cells) == summary["cell_summary_rows"] == 200
    assert len(aggregates) == summary["aggregate_rows"] == 20
    assert {row["status"] for row in cells} == {"PASS"}
    assert {row["model_name"] for row in cells} == {
        "distilbert-base-uncased",
        "bert-base-uncased",
    }
    assert len({(row["model_name"], row["cell_id"]) for row in cells}) == 200

    distilbert = [row for row in aggregates if row["model_name"] == "distilbert-base-uncased"]
    bertbase = [row for row in aggregates if row["model_name"] == "bert-base-uncased"]
    assert len(distilbert) == 10
    assert len(bertbase) == 10

    distilbert_best = max(distilbert, key=lambda row: float(row["mean_final_average_accuracy"]))
    bertbase_best = max(bertbase, key=lambda row: float(row["mean_final_average_accuracy"]))
    assert distilbert_best["method"] == "random_replay"
    assert distilbert_best["budget"] == "256"
    assert bertbase_best["method"] == "class_balanced_replay"
    assert bertbase_best["budget"] == "256"

    for path in [
        ROOT / "figures/figure_method_budget_accuracy_distilbert.png",
        ROOT / "figures/figure_method_budget_accuracy_bertbase.png",
        ROOT / "figures/figure_stepwise_seen_accuracy_distilbert.png",
        ROOT / "figures/figure_stepwise_seen_accuracy_bertbase.png",
        ROOT / "figures/figure_stepwise_forgetting_distilbert.png",
        ROOT / "figures/figure_stepwise_forgetting_bertbase.png",
    ]:
        assert path.exists(), f"missing figure: {path}"

    print("PASS: public data checks completed")


if __name__ == "__main__":
    main()
