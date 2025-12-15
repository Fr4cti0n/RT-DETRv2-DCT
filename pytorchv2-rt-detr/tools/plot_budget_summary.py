#!/usr/bin/env python3
"""Generate presentation-style plots relating accuracy to retained DCT coefficients."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delta-csv",
        type=Path,
        default=Path("output/compressed_resnet34/crianns_eval/accuracy_delta_vs_block_stem.csv"),
        help="CSV produced by compute_accuracy_delta.py containing baseline and luma-fusion rows.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where the plots will be saved (defaults beside the CSV).",
    )
    return parser.parse_args()


def _read_rows(csv_path: Path) -> List[Dict[str, float]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    rows: List[Dict[str, float]] = []
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            variant = (raw.get("variant") or "").strip()
            coeff_window_str = raw.get("coeff_window", "")
            acc1_str = raw.get("acc1", "")
            delta_str = raw.get("delta_acc1_pct", "")
            acc5_str = raw.get("acc5", "")
            delta5_str = raw.get("delta_acc5_pct", "")
            if not variant or not coeff_window_str or not acc1_str or not acc5_str:
                continue
            coeff_window = int(coeff_window_str)
            retained = coeff_window * coeff_window
            percent_kept = retained / 64.0 * 100.0
            acc1 = float(acc1_str) * 100.0
            delta = float(delta_str)
            acc5 = float(acc5_str) * 100.0
            delta5 = float(delta5_str)
            rows.append(
                {
                    "variant": variant,
                    "coeff_window": coeff_window,
                    "retained_coeffs": retained,
                    "percent_kept": percent_kept,
                    "acc1": acc1,
                    "delta_acc1": delta,
                    "acc5": acc5,
                    "delta_acc5": delta5,
                }
            )
    if not rows:
        raise ValueError("No valid rows found in the accuracy delta CSV.")
    # Sort from highest budget to lowest for consistent plotting
    rows.sort(key=lambda item: item["retained_coeffs"], reverse=True)
    return rows


def _accuracy_vs_budget(
    rows: List[Dict[str, float]],
    output_path: Path,
    metric_key: str,
    delta_key: str,
    title: str,
    ylabel: str,
    annotation_fmt: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    x_vals = [row["percent_kept"] for row in rows]
    y_vals = [row[metric_key] for row in rows]
    labels = [f"{int(row['retained_coeffs'])} coeffs" for row in rows]

    ax.plot(x_vals, y_vals, marker="o", linewidth=2.0)
    for x, y, label in zip(x_vals, y_vals, labels):
        ax.annotate(label, xy=(x, y), xytext=(5, 5), textcoords="offset points")

    target = next((row for row in rows if row["retained_coeffs"] == 4), None)
    if target is not None:
        ax.axvline(target["percent_kept"], color="#d62728", linestyle="--", linewidth=1.2)
        annotation = annotation_fmt.format(
            delta=target[delta_key],
            pct=target["percent_kept"],
        )
        ax.annotate(
            annotation,
            xy=(target["percent_kept"], target["acc1"]),
            xytext=(10, -30),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="#d62728"),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#d62728", alpha=0.9),
            fontsize=9,
        )

    ax.set_xlabel("Coefficients kept per block (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlim(0, 105)
    ax.set_ylim(min(y_vals) - 2, max(y_vals) + 2)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _pareto_delta_plot(
    rows: List[Dict[str, float]],
    output_path: Path,
    delta_key: str,
    title: str,
    ylabel: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    x_vals = [row["percent_kept"] for row in rows]
    y_vals = [row[delta_key] for row in rows]

    ax.plot(x_vals, y_vals, marker="o", linewidth=2.0)
    for row in rows:
        ax.annotate(f"{int(row['retained_coeffs'])}", xy=(row["percent_kept"], row[delta_key]), xytext=(6, 5), textcoords="offset points")

    target = next((row for row in rows if row["retained_coeffs"] == 4), None)
    if target is not None:
        ax.annotate(
            "4 coeffs",
            xy=(target["percent_kept"], target[delta_key]),
            xytext=(target["percent_kept"] * 1.4, target[delta_key] - 1.0),
            arrowprops=dict(arrowstyle="->", color="#1f77b4"),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1f77b4", alpha=0.9),
            fontsize=10,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Coefficients kept per block (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.set_xlim(min(x_vals) * 0.8, max(x_vals) * 1.05)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _value_for_budget_bar(
    rows: List[Dict[str, float]],
    output_path: Path,
    metric_key: str,
    delta_key: str,
    title: str,
    ylabel: str,
    callout_fmt: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    bars = [row[metric_key] for row in rows]
    labels = [
        f"{int(row['retained_coeffs'])} coeffs\n({row['percent_kept']:.2f}% kept)" for row in rows
    ]
    x_positions = range(len(rows))

    ax.bar(x_positions, bars, color=["#1f77b4", "#ff7f0e", "#d62728"])
    ax.set_xticks(list(x_positions), labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(bars) + 10)

    smallest = rows[-1]
    reduction = 100.0 - smallest["percent_kept"]
    loss = abs(smallest[delta_key])
    ax.text(
        len(rows) - 1,
        bars[-1] + 3,
        callout_fmt.format(reduction=reduction, loss=loss),
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#d62728", alpha=0.9),
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    csv_path = args.delta_csv.expanduser()
    output_dir = args.output_dir.expanduser() if args.output_dir is not None else csv_path.parent

    rows = _read_rows(csv_path)

    # Expect data ordered high → low, but enforce specific order 64,16,4 if available
    expected_order = {64: 0, 16: 1, 4: 2}
    rows.sort(key=lambda row: expected_order.get(row["retained_coeffs"], row["retained_coeffs"]), reverse=False)

    _accuracy_vs_budget(
        rows,
        output_dir / "accuracy_vs_retained_coeffs_top1.png",
        "acc1",
        "delta_acc1",
        "Top-1 accuracy vs. retained DCT coefficients",
        "Top-1 accuracy (%)",
        "4 coeffs → {delta:+.1f} pts at {pct:.2f}% kept",
    )
    _accuracy_vs_budget(
        rows,
        output_dir / "accuracy_vs_retained_coeffs_top5.png",
        "acc5",
        "delta_acc5",
        "Top-5 accuracy vs. retained DCT coefficients",
        "Top-5 accuracy (%)",
        "4 coeffs → {delta:+.1f} pts at {pct:.2f}% kept",
    )

    _pareto_delta_plot(
        rows,
        output_dir / "delta_accuracy_pareto_top1.png",
        "delta_acc1",
        "Top-1 accuracy trade-off vs. retained budget",
        "Δ Top-1 accuracy vs. block-stem (pts)",
    )
    _pareto_delta_plot(
        rows,
        output_dir / "delta_accuracy_pareto_top5.png",
        "delta_acc5",
        "Top-5 accuracy trade-off vs. retained budget",
        "Δ Top-5 accuracy vs. block-stem (pts)",
    )

    _value_for_budget_bar(
        rows,
        output_dir / "accuracy_value_for_budget_top1.png",
        "acc1",
        "delta_acc1",
        "Top-1 accuracy per coefficient budget",
        "Top-1 accuracy (%)",
        "{reduction:.0f}% coefficients removed\n→ only ≈{loss:.1f} pts Top-1 loss",
    )
    _value_for_budget_bar(
        rows,
        output_dir / "accuracy_value_for_budget_top5.png",
        "acc5",
        "delta_acc5",
        "Top-5 accuracy per coefficient budget",
        "Top-5 accuracy (%)",
        "{reduction:.0f}% coefficients removed\n→ only ≈{loss:.1f} pts Top-5 loss",
    )

    print(f"[info] Saved budget plots to {output_dir}")


if __name__ == "__main__":
    main()
