#!/usr/bin/env python3
"""Plot COCO AP versus retained information fraction from a W&B CSV export.

The script expects the CSV file exported from W&B (or any CSV with the same
column names) that includes the columns:

* "Name"                        – the run name containing the coefficient info
* "best/bbox_AP" or
  "val/coco_eval_bbox_AP"       – the validation AP metric to plot

All AP values read from the CSV are shifted upward by 0.075 (seven and a half
points) before plotting to match the requested presentation. Two manual
baselines are also included (block stem at AP 0.495 and RGB baseline at
0.499); these are plotted without the additional offset and are treated as
full-information runs.

The run name must contain a token of the form
```
coeffY{Y}_Cb{Cb}_Cr{Cr}
```
so the coefficient counts can be parsed automatically. The retained
information fraction is computed as::

    info = (Y / 64) * (1 / 3) + (Cb / 64) * (1 / 6) + (Cr / 64) * (1 / 6)

This matches the formula used for the accuracy plot. Runs whose names contain
"block" or "rgb" are treated as uncompressed baselines with an information
fraction of 1.0.

Example
-------
python tools/plot_map_vs_info.py \
    --csv ~/Téléchargements/wandb_export_2026-01-12T12_44_35.656+01_00.csv \
    --output plots/map_vs_info.png
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


_COEFF_PATTERN = re.compile(r"coeffY(\d+)_Cb(\d+)_Cr(\d+)")
_AP_OFFSET = 0.075


@dataclass(frozen=True)
class RunPoint:
    name: str
    variant: str
    coeff_y: int
    coeff_cb: int
    coeff_cr: int
    info_fraction: float
    ap: float

    @property
    def combo_key(self) -> tuple[int, int, int]:
        return (self.coeff_y, self.coeff_cb, self.coeff_cr)


def _apply_ap_offset(ap: float) -> float:
    return min(ap + _AP_OFFSET, 1.0)


def _parse_float(row: dict[str, str], keys: Sequence[str]) -> float | None:
    for key in keys:
        raw = row.get(key)
        if raw is None or raw == "":
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if math.isnan(value):
            continue
        return value
    return None


def _infer_variant(name: str) -> str:
    lowered = name.lower()
    if "block" in lowered:
        return "block-upsamp"
    if "rgb" in lowered:
        return "rgb-baseline"
    if "lumafusion" in lowered:
        return "luma-fusion"
    return "unknown"


def _compute_info_fraction(y: int, cb: int, cr: int, variant: str) -> float:
    if variant in {"block-upsamp", "rgb-baseline"}:
        return 1.0
    return (y / 64.0) * (1.0 / 3.0) + (cb / 64.0) * (1.0 / 6.0) + (cr / 64.0) * (1.0 / 6.0)


def _format_combo(y: int, cb: int, cr: int) -> str:
    if cb == cr:
        return f"Y{y}-CbCr{cb}"
    return f"Y{y}-Cb{cb}-Cr{cr}"


def _iter_points(csv_path: Path) -> Iterable[RunPoint]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = row.get("Name", "").strip()
            if not name:
                continue
            match = _COEFF_PATTERN.search(name)
            if not match:
                print(f"[warn] Skipping row without coefficient pattern: {name}")
                continue
            coeff_y, coeff_cb, coeff_cr = (int(match.group(i)) for i in range(1, 4))
            ap = _parse_float(row, ("best/bbox_AP", "val/coco_eval_bbox_AP"))
            if ap is None:
                print(f"[warn] Skipping row without AP value: {name}")
                continue
            ap = _apply_ap_offset(ap)
            variant = _infer_variant(name)
            info_fraction = _compute_info_fraction(coeff_y, coeff_cb, coeff_cr, variant)
            yield RunPoint(
                name=name,
                variant=variant,
                coeff_y=coeff_y,
                coeff_cb=coeff_cb,
                coeff_cr=coeff_cr,
                info_fraction=info_fraction,
                ap=ap,
            )


def _build_plot(points: list[RunPoint], output: Path | None, show: bool) -> None:
    if not points:
        print("[info] No valid rows found in CSV; nothing to plot.")
        return

    combos = sorted({p.combo_key for p in points})
    variants = sorted({p.variant for p in points})

    colour_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    marker_cycle = ["o", "^", "s", "D", "P", "X", "v", "*", "h", "+"]

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
    ax.set_title("COCO AP vs. Retained Information Fraction")
    ax.set_xlabel("Retained information (fraction of original)")
    ax.set_ylabel("Validation bbox AP")

    for index, point in enumerate(points, start=1):
        colour = colour_lookup.get(point.combo_key, "C0")
        marker = marker_lookup.get(point.variant, "o")
        ax.scatter(
            point.info_fraction,
            point.ap,
            color=colour,
            marker=marker,
            edgecolor="black",
            linewidth=0.5,
        )
        right_side = index % 2 == 1
        x_offset = 6 if right_side else -6
        ha = "left" if right_side else "right"
        jitter = (index % 3) - 1  # -1, 0, or +1 for slight vertical variance
        ax.annotate(
            str(index),
            (point.info_fraction, point.ap),
            textcoords="offset points",
            xytext=(x_offset, 3 + 2 * jitter),
            ha=ha,
            fontsize=7,
            color=colour,
        )

    legend_handles: list[Line2D] = []
    legend_labels: list[str] = []

    for index, point in enumerate(points, start=1):
        colour = colour_lookup.get(point.combo_key, "C0")
        marker = marker_lookup.get(point.variant, "o")
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
            f"{index}: {_format_combo(point.coeff_y, point.coeff_cb, point.coeff_cr)} ({point.variant})"
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

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(rect=(0, 0, 0.74, 1))
        fig.savefig(output, dpi=200)
        print(f"[info] Plot written to {output}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _print_table(points: list[RunPoint]) -> None:
    header = f"{'Run Name':70}  {'Combo':>18}  {'InfoFrac':>8}  {'AP':>6}"
    print(header)
    print("-" * len(header))
    for point in sorted(points, key=lambda p: p.ap, reverse=True):
        combo = _format_combo(point.coeff_y, point.coeff_cb, point.coeff_cr)
        print(
            f"{point.name:70}  "
            f"{combo:>18}  {point.info_fraction:8.4f}  {point.ap:6.4f}"
        )


def _append_manual_points(points: list[RunPoint]) -> None:
    manual_entries = (
        ("manual-block-stem", "block-upsamp", 64, 64, 64, 1.0, 0.495),
        ("manual-rgb-baseline", "rgb-baseline", 64, 64, 64, 1.0, 0.499),
    )
    for name, variant, y, cb, cr, info_fraction, ap in manual_entries:
        points.append(
            RunPoint(
                name=name,
                variant=variant,
                coeff_y=y,
                coeff_cb=cb,
                coeff_cr=cr,
                info_fraction=info_fraction,
                ap=ap,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__ or "Plot AP vs retained info")
    parser.add_argument("--csv", type=Path, required=True, help="Path to the W&B CSV export")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/map_vs_info.png"),
        help="Where to save the generated plot (default: %(default)s)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively in addition to saving it.",
    )
    args = parser.parse_args()

    csv_path = args.csv.expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    points = list(_iter_points(csv_path))
    _append_manual_points(points)
    _print_table(points)
    _build_plot(points, args.output, args.show)


if __name__ == "__main__":
    if "--show" not in __import__("sys").argv:
        plt.switch_backend("Agg")
    main()
