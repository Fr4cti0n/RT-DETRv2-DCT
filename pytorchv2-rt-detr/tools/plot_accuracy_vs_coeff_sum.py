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
from matplotlib.lines import Line2D
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
    info_fraction: float | None = None


_EXTRA_RUNS: tuple[RunMetrics, ...] = (
    RunMetrics(
        run_dir=Path("manual/Y64_CbCr64_block-upsamp"),
        variant="block-upsamp",
        coeff_luma=64,
        coeff_cb=64,
        coeff_cr=64,
        total_coeff=64 + 64 + 64,
        best_acc=0.73253,
        epoch=0,
        info_fraction=1.0,
    ),
    RunMetrics(
        run_dir=Path("manual/Y64_CbCr64_rgb-baseline"),
        variant="rgb-baseline",
        coeff_luma=64,
        coeff_cb=64,
        coeff_cr=64,
        total_coeff=64 + 64 + 64,
        best_acc=0.7311,
        epoch=0,
        info_fraction=1.0,
    ),
)


def _format_combo(coeff_luma: int, coeff_cb: int, coeff_cr: int) -> str:
    if coeff_cb == coeff_cr:
        return f"Y{coeff_luma}-CbCr{coeff_cb}"
    return f"Y{coeff_luma}-Cb{coeff_cb}-Cr{coeff_cr}"


def _append_extra_runs(metrics: list[RunMetrics]) -> None:
    metrics.extend(_EXTRA_RUNS)


def _compute_info_fraction(entry: RunMetrics) -> float:
    if entry.info_fraction is not None:
        return entry.info_fraction
    coeff_luma = max(entry.coeff_luma, 0)
    coeff_cb = max(entry.coeff_cb, 0)
    coeff_cr = max(entry.coeff_cr, 0)
    return (
        (coeff_luma / 64.0) * (1.0 / 3.0)
        + (coeff_cb / 64.0) * (1.0 / 6.0)
        + (coeff_cr / 64.0) * (1.0 / 6.0)
    )


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

    combos = sorted({(m.coeff_luma, m.coeff_cb, m.coeff_cr) for m in metrics})
    variants = sorted({item.variant for item in metrics})

    colour_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    marker_cycle = ["o", "^", "s", "D", "P", "X", "*", "v"]

    colour_lookup = {
        combo: colour_cycle[i % len(colour_cycle)]
        for i, combo in enumerate(combos)
    }
    marker_lookup = {
        variant: marker_cycle[i % len(marker_cycle)]
        for i, variant in enumerate(variants)
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(right=0.74)
    ax.set_title("Accuracy vs. Retained Information Fraction")
    ax.set_xlabel("Retained information (fraction of original)")
    ax.set_ylabel("Best validation accuracy (top-1)")

    for index, item in enumerate(metrics, start=1):
        combo_key = (item.coeff_luma, item.coeff_cb, item.coeff_cr)
        colour = colour_lookup.get(combo_key, "C0")
        marker = marker_lookup.get(item.variant, "o")
        x_value = _compute_info_fraction(item)
        ax.scatter(
            x_value,
            item.best_acc,
            color=colour,
            marker=marker,
            edgecolor="black",
            linewidth=0.5,
        )
        right_side = index % 2 == 1
        x_offset = 6 if right_side else -6
        ha = "left" if right_side else "right"
        jitter = (index % 3) - 1  # -1, 0, or +1 for slight vertical variety
        ax.annotate(
            str(index),
            (x_value, item.best_acc),
            textcoords="offset points",
            xytext=(x_offset, 3 + 2 * jitter),
            ha=ha,
            fontsize=7,
            color=colour,
        )

    legend_handles: list[Line2D] = []
    legend_labels: list[str] = []

    for index, item in enumerate(metrics, start=1):
        combo_key = (item.coeff_luma, item.coeff_cb, item.coeff_cr)
        colour = colour_lookup.get(combo_key, "C0")
        marker = marker_lookup.get(item.variant, "o")
        legend_handles.append(
            Line2D(
                [],
                [],
                linestyle="",
                marker=marker,
                color=colour,
                markeredgecolor="black",
                markeredgewidth=0.5,
            )
        )
        legend_labels.append(
            f"{index}: {_format_combo(item.coeff_luma, item.coeff_cb, item.coeff_cr)} ({item.variant})"
        )

    if legend_handles:
        legend = ax.legend(
            legend_handles,
            legend_labels,
                title="Colour, number = variants",
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            fontsize=8,
        )
        ax.add_artist(legend)

    ax.grid(True, linestyle="--", alpha=0.3)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(rect=(0, 0, 0.74, 1))
        fig.savefig(output_path, dpi=200)
        print(f"[info] Plot written to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _print_table(metrics: list[RunMetrics]) -> None:
    header = f"{'Run Directory':60}  {'Combo':>20}  {'InfoFrac':>8}  {'BestAcc':>8}"
    print(header)
    print("-" * len(header))
    for item in sorted(metrics, key=lambda r: r.best_acc, reverse=True):
        combo_label = _format_combo(item.coeff_luma, item.coeff_cb, item.coeff_cr)
        print(
            f"{item.run_dir.name:60}  "
            f"{combo_label:>20}  {_compute_info_fraction(item):8.4f}  {item.best_acc:8.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__ or "Plot accuracy vs coefficient counts")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("output/compressed_resnet34"),
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

    _append_extra_runs(metrics)

    _print_table(metrics)
    _build_plot(metrics, args.output_plot, args.show)


if __name__ == "__main__":
    # Use a non-interactive backend unless the user explicitly asks to show the plot.
    if "--show" not in __import__("sys").argv:
        plt.switch_backend("Agg")
    main()
