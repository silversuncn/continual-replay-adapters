from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = _mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def _read_matrix_rows(matrix_dir: Path) -> list[dict]:
    summary = json.loads((matrix_dir / "matrix_summary.json").read_text())
    rows = []
    for cell in summary["cells"]:
        if cell.get("status") != "PASS":
            continue
        result_path = Path(cell["result_path"])
        if not result_path.is_absolute():
            candidates = [
                result_path,
                matrix_dir / result_path,
                matrix_dir / "cells" / cell["cell_id"] / "result.json",
            ]
            result_path = next((candidate for candidate in candidates if candidate.exists()), candidates[-1])
        result = json.loads(result_path.read_text())
        peak_gpu_values = [
            step.get("peak_gpu_memory_mb")
            for step in result.get("steps", [])
            if step.get("peak_gpu_memory_mb") is not None
        ]
        max_rss_values = [
            step.get("max_rss_mb")
            for step in result.get("steps", [])
            if step.get("max_rss_mb") is not None
        ]
        rows.append(
            {
                "cell_id": result.get("cell_id", cell["cell_id"]),
                "order_name": result.get("order_name", cell["order_name"]),
                "seed": int(result.get("seed", cell["seed"])),
                "method": result.get("method", cell["method"]),
                "budget": int(result.get("budget", cell["budget"])),
                "final_average_accuracy": float(result["final_average_accuracy"]),
                "average_forgetting": float(result["forgetting"]["average_forgetting"]),
                "total_wall_time_seconds": float(result["total_wall_time_seconds"]),
                "peak_gpu_memory_mb": max(peak_gpu_values) if peak_gpu_values else "",
                "max_rss_mb": max(max_rss_values) if max_rss_values else "",
                "sst2": result["final_scores"].get("sst2", ""),
                "mrpc": result["final_scores"].get("mrpc", ""),
                "rte": result["final_scores"].get("rte", ""),
                "ag_news": result["final_scores"].get("ag_news", ""),
                "result_path": str(result_path),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _aggregate_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["budget"])].append(row)
    out = []
    for (method, budget), group in sorted(grouped.items()):
        acc = [row["final_average_accuracy"] for row in group]
        forgetting = [row["average_forgetting"] for row in group]
        runtime = [row["total_wall_time_seconds"] for row in group]
        out.append(
            {
                "method": method,
                "budget": budget,
                "n": len(group),
                "mean_final_average_accuracy": _mean(acc),
                "std_final_average_accuracy": _std(acc),
                "mean_average_forgetting": _mean(forgetting),
                "std_average_forgetting": _std(forgetting),
                "mean_runtime_seconds": _mean(runtime),
                "std_runtime_seconds": _std(runtime),
            }
        )
    return out


def _wilcoxon_or_descriptive(a: list[float], b: list[float]) -> dict:
    deltas = [x - y for x, y in zip(a, b)]
    result = {
        "n": len(deltas),
        "mean_delta": _mean(deltas),
        "median_delta": sorted(deltas)[len(deltas) // 2] if deltas else 0.0,
        "test": "descriptive",
        "statistic": None,
        "p_value": None,
    }
    try:
        from scipy.stats import wilcoxon

        if len(deltas) >= 2 and any(abs(delta) > 1e-12 for delta in deltas):
            stat, p_value = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            result.update({"test": "wilcoxon_signed_rank", "statistic": float(stat), "p_value": float(p_value)})
    except Exception as exc:
        result["fallback_reason"] = str(exc)
    return result


def _pairwise_statistics(rows: list[dict]) -> list[dict]:
    baseline = {
        (row["order_name"], row["seed"]): row
        for row in rows
        if row["method"] == "no_replay" and row["budget"] == 0
    }
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        if row["method"] != "no_replay":
            grouped[(row["method"], row["budget"])].append(row)
    comparisons = []
    for (method, budget), group in sorted(grouped.items()):
        matched = [
            (row, baseline[(row["order_name"], row["seed"])])
            for row in group
            if (row["order_name"], row["seed"]) in baseline
        ]
        acc_stats = _wilcoxon_or_descriptive(
            [row["final_average_accuracy"] for row, _ in matched],
            [base["final_average_accuracy"] for _, base in matched],
        )
        forgetting_stats = _wilcoxon_or_descriptive(
            [row["average_forgetting"] for row, _ in matched],
            [base["average_forgetting"] for _, base in matched],
        )
        comparisons.append(
            {
                "method": method,
                "budget": budget,
                "matched_pairs": len(matched),
                "accuracy": acc_stats,
                "forgetting": forgetting_stats,
            }
        )
    p_entries = [
        (comparison, metric)
        for comparison in comparisons
        for metric in ["accuracy", "forgetting"]
        if comparison[metric].get("p_value") is not None
    ]
    for rank, (comparison, metric) in enumerate(
        sorted(p_entries, key=lambda item: item[0][item[1]]["p_value"]),
        start=1,
    ):
        m = len(p_entries)
        raw = comparison[metric]["p_value"]
        comparison[metric]["holm_p_value"] = min(1.0, raw * (m - rank + 1))
    return comparisons


def _write_pairwise_tex(path: Path, comparisons: list[dict]) -> None:
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Method & Budget & $\\Delta$ Acc. & $p_{acc}$ & $\\Delta$ Forget & $p_{forget}$ \\\\",
        "\\midrule",
    ]
    for item in comparisons:
        acc = item["accuracy"]
        forgetting = item["forgetting"]
        p_acc = acc.get("holm_p_value", acc.get("p_value"))
        p_forget = forgetting.get("holm_p_value", forgetting.get("p_value"))
        lines.append(
            f"{item['method']} & {item['budget']} & "
            f"{acc['mean_delta']:.4f} & {p_acc if p_acc is not None else 'n/a'} & "
            f"{forgetting['mean_delta']:.4f} & {p_forget if p_forget is not None else 'n/a'} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines))


def _write_figures(output_dir: Path, aggregated: list[dict], rows: list[dict]) -> list[str]:
    figure_paths: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (output_dir / "figure_generation_error.txt").write_text(str(exc))
        return figure_paths

    labels = [f"{row['method']}\nb{row['budget']}" for row in aggregated]
    x = list(range(len(aggregated)))

    def save_current(name: str) -> None:
        for suffix in ["png", "pdf"]:
            path = output_dir / f"{name}.{suffix}"
            plt.savefig(path, bbox_inches="tight", dpi=180)
            figure_paths.append(str(path))
        plt.close()

    plt.figure(figsize=(10, 4.8))
    plt.bar(x, [row["mean_final_average_accuracy"] for row in aggregated], color="#4C78A8")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Final average accuracy")
    plt.title("Method and budget performance")
    save_current("figure_method_budget_accuracy")

    plt.figure(figsize=(10, 4.8))
    plt.bar(x, [row["mean_average_forgetting"] for row in aggregated], color="#F58518")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Average forgetting")
    plt.title("Forgetting by replay allocation")
    save_current("figure_forgetting_by_method_budget")

    plt.figure(figsize=(6.4, 4.8))
    for row in rows:
        plt.scatter(row["total_wall_time_seconds"], row["final_average_accuracy"], label=row["method"], alpha=0.7)
    handles, handle_labels = plt.gca().get_legend_handles_labels()
    dedup = dict(zip(handle_labels, handles))
    plt.legend(dedup.values(), dedup.keys(), fontsize=8)
    plt.xlabel("Runtime seconds")
    plt.ylabel("Final average accuracy")
    plt.title("Performance-runtime context")
    save_current("figure_accuracy_runtime")
    return figure_paths


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze_matrix(matrix_dir: str | Path, output_dir: str | Path) -> dict:
    matrix_dir = Path(matrix_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_matrix_rows(matrix_dir)
    aggregated = _aggregate_rows(rows)
    comparisons = _pairwise_statistics(rows)

    result_fields = [
        "cell_id",
        "order_name",
        "seed",
        "method",
        "budget",
        "final_average_accuracy",
        "average_forgetting",
        "total_wall_time_seconds",
        "peak_gpu_memory_mb",
        "max_rss_mb",
        "sst2",
        "mrpc",
        "rte",
        "ag_news",
        "result_path",
    ]
    aggregate_fields = [
        "method",
        "budget",
        "n",
        "mean_final_average_accuracy",
        "std_final_average_accuracy",
        "mean_average_forgetting",
        "std_average_forgetting",
        "mean_runtime_seconds",
        "std_runtime_seconds",
    ]
    _write_csv(output_dir / "results.csv", rows, result_fields)
    _write_csv(output_dir / "results_aggregated.csv", aggregated, aggregate_fields)
    _write_pairwise_tex(output_dir / "table_pairwise.tex", comparisons)
    figure_paths = _write_figures(output_dir, aggregated, rows)

    best_accuracy = max(aggregated, key=lambda row: row["mean_final_average_accuracy"]) if aggregated else {}
    best_forgetting = min(aggregated, key=lambda row: row["mean_average_forgetting"]) if aggregated else {}
    statistics = {
        "pairwise_vs_no_replay": comparisons,
        "correction": "Holm correction applied where Wilcoxon p-values are available",
    }
    (output_dir / "statistics.json").write_text(json.dumps(statistics, indent=2, ensure_ascii=False))
    summary = {
        "status": "PASS_PHASE4_ANALYSIS_COMPLETED",
        "run_count": len(rows),
        "best_final_average_accuracy": best_accuracy,
        "lowest_average_forgetting": best_forgetting,
        "figure_paths": figure_paths,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    outputs = [
        "results.csv",
        "results_aggregated.csv",
        "summary.json",
        "statistics.json",
        "table_pairwise.tex",
    ]
    outputs.extend(Path(path).name for path in figure_paths)
    provenance = {
        "matrix_dir": str(matrix_dir),
        "input_summary": str(matrix_dir / "matrix_summary.json"),
        "output_hashes": {
            name: _sha256(output_dir / name)
            for name in outputs
            if (output_dir / name).exists()
        },
    }
    (output_dir / "provenance_manifest.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False))
    return summary


def main():
    args = parse_args()
    summary = analyze_matrix(args.matrix_dir, args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
