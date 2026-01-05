#!/usr/bin/env python3
"""Generate accuracy vs. coefficient-count plots for compressed backbone checkpoints.

This script scans a directory containing multiple training runs (e.g. the
`crianns_training` folder) and extracts the best validation accuracy recorded in
`model_best.pth` for each run. The total retained coefficient count is computed
as the sum of the Y, Cb and Cr coefficients stored inside each checkpoint. A
scatter plot is produced to visualise the relationship between the retained
coefficients and model accuracy, with one point per run.

Example
-------
python tools/plot_accuracy_vs_coeff_sum.py \
    --runs-root src/nn/backbone/crianns_training \
    --output-plot plots/accuracy_vs_coeff_sum.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterator, NamedTuple

import matplotlib.pyplot as plt
import torch


class RunMetrics(NamedTuple):
    run_dir: Path
    variant: str
    coeff_luma: int
    coeff_cb: int
    coeff_cr: int
    total_coeff: int
    best_acc: float
    epoch: int


def _iter_run_directories(root: Path) -> Iterator[Path]:
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            yield entry


def _load_run_metrics(run_dir: Path) -> RunMetrics | None:
    ckpt_path = run_dir / "model_best.pth"
    if not ckpt_path.exists():
        return None
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as exc:  # pragma: no cover - defensive guard for corrupt files
        print(f"[warn] Failed to load {ckpt_path}: {exc}")
        return None

    required_keys = {
        "variant",
        "coeff_count_luma",
        "coeff_count_cb",
        "coeff_count_cr",
        "best_acc",
        "epoch",
    }
    if not required_keys.issubset(checkpoint):
        missing = required_keys.difference(checkpoint)
        print(f"[warn] Checkpoint at {ckpt_path} missing keys: {sorted(missing)}")
        return None

    coeff_luma = int(checkpoint["coeff_count_luma"])
    coeff_cb = int(checkpoint["coeff_count_cb"])
    coeff_cr = int(checkpoint["coeff_count_cr"])
    total_coeff = coeff_luma + coeff_cb + coeff_cr
    best_acc_raw = checkpoint["best_acc"]
    try:
        best_acc = float(best_acc_raw)
    except (TypeError, ValueError):
        print(f"[warn] Invalid best_acc value '{best_acc_raw}' in {ckpt_path}")
        return None

    if math.isnan(best_acc):
        print(f"[warn] best_acc is NaN for {ckpt_path}")
        return None

    variant = str(checkpoint["variant"])
    epoch = int(checkpoint["epoch"])
    return RunMetrics(
        run_dir=run_dir,
        variant=variant,
        coeff_luma=coeff_luma,
        coeff_cb=coeff_cb,
        coeff_cr=coeff_cr,
        total_coeff=total_coeff,
        best_acc=best_acc,
        epoch=epoch,
    )


def _build_plot(metrics: list[RunMetrics], output_path: Path | None, show: bool) -> None:
    if not metrics:
        print("[info] No runs found; nothing to plot.")
        return

    variants = sorted({item.variant for item in metrics})
    colour_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    colour_lookup = {variant: colour_cycle[i % len(colour_cycle)] for i, variant in enumerate(variants)}

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Accuracy vs. Total Retained Coefficients")
    ax.set_xlabel("Total retained coefficients (Y + Cb + Cr)")
    ax.set_ylabel("Best validation accuracy (top-1)")

    for item in metrics:
        colour = colour_lookup.get(item.variant, "C0")
        ax.scatter(item.total_coeff, item.best_acc, color=colour, label=item.variant)
        combo_label = f"Y{item.coeff_luma}-Cb{item.coeff_cb}-Cr{item.coeff_cr}"
        ax.annotate(
            combo_label,
            (item.total_coeff, item.best_acc),
            textcoords="offset points",
            xytext=(5, 5),
            ha="left",
            fontsize=8,
        )

    # Build legend without duplicates by using the colours we assigned.
    handles = []
    labels = []
    for variant in variants:
        handles.append(ax.scatter([], [], color=colour_lookup.get(variant, "C0")))
        labels.append(variant)
    if handles:
        ax.legend(handles, labels, title="Variant", loc="lower right")

    ax.grid(True, linestyle="--", alpha=0.3)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        print(f"[info] Plot written to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _print_table(metrics: list[RunMetrics]) -> None:
    header = f"{'Run Directory':60}  {'Combo':>20}  {'TotalCoeff':>10}  {'BestAcc':>8}"
    print(header)
    print("-" * len(header))
    for item in sorted(metrics, key=lambda r: r.best_acc, reverse=True):
        combo_label = f"Y{item.coeff_luma}-Cb{item.coeff_cb}-Cr{item.coeff_cr}"
        print(
            f"{item.run_dir.name:60}  "
            f"{combo_label:>20}  {item.total_coeff:10d}  {item.best_acc:8.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__ or "Plot accuracy vs coefficient counts")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("src/nn/backbone/crianns_training"),
        help="Directory containing run sub-folders (default: %(default)s)",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=Path("plots/accuracy_vs_coeff_sum.png"),
        help="Where to save the generated plot (default: %(default)s)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively in addition to saving it.",
    )
    args = parser.parse_args()

    runs_root = args.runs_root.expanduser().resolve()
    if not runs_root.exists():
        raise FileNotFoundError(f"Runs root not found: {runs_root}")

    metrics: list[RunMetrics] = []
    for run_dir in _iter_run_directories(runs_root):
        run_metrics = _load_run_metrics(run_dir)
        if run_metrics is not None:
            metrics.append(run_metrics)

    _print_table(metrics)
    _build_plot(metrics, args.output_plot, args.show)


if __name__ == "__main__":
    # Use a non-interactive backend unless the user explicitly asks to show the plot.
    if "--show" not in __import__("sys").argv:
        plt.switch_backend("Agg")
    main()
