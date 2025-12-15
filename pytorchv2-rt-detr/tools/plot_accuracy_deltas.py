#!/usr/bin/env python3
"""Plot scatter charts for accuracy deltas relative to the block-stem baseline."""

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
        help="CSV file produced by compute_accuracy_delta.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the generated plots (defaults beside the CSV).",
    )
    parser.add_argument(
        "--hide-baseline",
        action="store_true",
        help="Hide the baseline point in the scatter plots.",
    )
    return parser.parse_args()


_ALLOWED_VARIANTS = {"block-stem", "luma-fusion"}


def _read_delta_rows(csv_path: Path, include_baseline: bool) -> List[Dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            variant = row.get("variant")
            if not include_baseline and variant == "block-stem":
                continue
            if variant not in _ALLOWED_VARIANTS:
                continue
            rows.append(row)
    if not rows:
        raise ValueError("No rows available after filtering; check the CSV content.")
    return rows


def _ensure_float(row: Dict[str, str], key: str) -> float:
    value = row.get(key)
    if value is None or not value.strip():
        raise ValueError(f"Missing required value '{key}' in row: {row}")
    return float(value)


def _plot_coeff_vs_delta(rows: List[Dict[str, str]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for variant in sorted({row["variant"] for row in rows}):
        subset = [row for row in rows if row["variant"] == variant]
        x_vals = [int(row["coeff_window"]) for row in subset]
        y_vals = [_ensure_float(row, "delta_acc1_pct") for row in subset]
        ax.scatter(x_vals, y_vals, label=variant, marker="o")
    ax.set_xlabel("Coefficient window")
    ax.set_ylabel("Delta acc@1 vs. block-stem (%)")
    ax.set_title("Accuracy delta by coefficient window")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_delta_acc_plane(rows: List[Dict[str, str]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for variant in sorted({row["variant"] for row in rows}):
        subset = [row for row in rows if row["variant"] == variant]
        x_vals = [_ensure_float(row, "delta_acc1_pct") for row in subset]
        y_vals = [_ensure_float(row, "delta_acc5_pct") for row in subset]
        ax.scatter(x_vals, y_vals, label=variant, marker="o")
    ax.set_xlabel("Delta acc@1 (%)")
    ax.set_ylabel("Delta acc@5 (%)")
    ax.set_title("Point cloud of accuracy deltas")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_coeff_erased_vs_delta(rows: List[Dict[str, str]], metric_key: str, ylabel: str, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for variant in sorted({row["variant"] for row in rows}):
        subset = [row for row in rows if row["variant"] == variant]
        x_vals = [(64 - int(row["coeff_window"]) ** 2) / 64.0 * 100.0 for row in subset]
        y_vals = [_ensure_float(row, metric_key) for row in subset]
        ax.scatter(x_vals, y_vals, label=variant, marker="o")
    ax.set_xlabel("Coefficients erased (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs. coefficients erased")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    csv_path = args.delta_csv.expanduser()
    output_dir = args.output_dir.expanduser() if args.output_dir is not None else csv_path.parent

    rows = _read_delta_rows(csv_path, include_baseline=not args.hide_baseline)

    coeff_plot = output_dir / "accuracy_delta_coeff_scatter.png"
    plane_plot = output_dir / "accuracy_delta_point_cloud.png"
    coeff_erased_acc1_plot = output_dir / "delta_acc1_vs_coeff_erased.png"
    coeff_erased_acc5_plot = output_dir / "delta_acc5_vs_coeff_erased.png"

    _plot_coeff_vs_delta(rows, coeff_plot)
    _plot_delta_acc_plane(rows, plane_plot)
    _plot_coeff_erased_vs_delta(rows, "delta_acc1_pct", "Delta acc@1 (%)", coeff_erased_acc1_plot)
    _plot_coeff_erased_vs_delta(rows, "delta_acc5_pct", "Delta acc@5 (%)", coeff_erased_acc5_plot)

    print(f"[info] Saved delta scatter plots to {output_dir}")


if __name__ == "__main__":
    main()
