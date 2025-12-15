#!/usr/bin/env python3
"""Generate diagnostic plots from exported Weights & Biases CSV summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("output/compressed_resnet34/crianns_eval/res_wand.csv"),
        help="Path to the exported CSV containing W&B run metrics.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where the plots will be written (defaults beside the CSV).",
    )
    parser.add_argument(
        "--include-running",
        action="store_true",
        help="Plot runs that are still marked as running (skips them by default).",
    )
    parser.add_argument(
        "--supplemental-csv",
        type=Path,
        default=None,
        help="Optional CSV carrying additional throughput stats (only used when provided).",
    )
    return parser.parse_args()


def _to_int(value: str) -> Optional[int]:
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _to_float(value: str) -> Optional[float]:
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _coefficients_erased(coeff_window: int) -> int:
    return max(0, 64 - coeff_window * coeff_window)


_ALLOWED_VARIANTS = {"block-stem", "reconstruction", "luma-fusion-pruned"}


def _load_rows(
    csv_path: Path,
    include_running: bool,
    supplemental_throughput: Optional[Dict[Tuple[str, int], float]] = None,
) -> List[Dict[str, object]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: List[Dict[str, object]] = []
        for row in reader:
            state = (row.get("State") or "").strip().lower()
            if not include_running and state != "finished":
                continue
            coeff_window = _to_int(row.get("coeff_window", ""))
            best_acc1 = _to_float(row.get("best/acc1", ""))
            best_acc5 = _to_float(row.get("best/acc5", ""))
            throughput = _to_float(row.get("benchmark/bs128/throughput_img_s", ""))
            variant = (row.get("variant") or "").strip()
            if not variant or coeff_window is None:
                continue
            if variant not in _ALLOWED_VARIANTS:
                continue
            if throughput is None and supplemental_throughput is not None:
                throughput = supplemental_throughput.get((variant, coeff_window))
            rows.append(
                {
                    "variant": variant,
                    "coeff_window": coeff_window,
                    "coefficients_erased": _coefficients_erased(coeff_window),
                    "best_acc1": best_acc1,
                    "best_acc5": best_acc5,
                    "throughput": throughput,
                }
            )
    if not rows:
        raise ValueError("No finished runs found in the CSV.")
    return rows


def _group_by_variant(rows: Iterable[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(row["variant"], []).append(row)
    return grouped


def _plot_accuracy(grouped: Dict[str, List[Dict[str, object]]], output_path: Path, metric_key: str, ylabel: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for variant, entries in grouped.items():
        filtered = [e for e in entries if e.get(metric_key) is not None]
        if not filtered:
            continue
        sorted_entries = sorted(filtered, key=lambda item: item["coefficients_erased"])
        x_vals = [entry["coefficients_erased"] for entry in sorted_entries]
        y_vals = [entry[metric_key] * 100.0 for entry in sorted_entries]
        ax.plot(x_vals, y_vals, marker="o", label=f"{variant}")
    ax.set_xlabel("Coefficients erased (per 8x8 block)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs. coefficient budget")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_throughput(grouped: Dict[str, List[Dict[str, object]]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for variant, entries in grouped.items():
        filtered = [e for e in entries if e.get("throughput") is not None and e.get("best_acc1") is not None]
        if not filtered:
            continue
        sorted_entries = sorted(filtered, key=lambda item: item["coefficients_erased"])
        x_vals = [entry["throughput"] for entry in sorted_entries]
        y_vals = [entry["best_acc1"] * 100.0 for entry in sorted_entries]
        ax.plot(x_vals, y_vals, marker="o", label=f"{variant}")
    ax.set_xlabel("Throughput (img/s @ batch 128)")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("Accuracy vs. throughput")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    csv_path = args.csv_path.expanduser()
    output_dir = args.output_dir.expanduser() if args.output_dir is not None else csv_path.parent
    supplemental_csv = args.supplemental_csv.expanduser() if args.supplemental_csv is not None else None

    supplemental_throughput: Optional[Dict[Tuple[str, int], float]] = None
    if supplemental_csv is not None and supplemental_csv.exists():
        supplemental_throughput = {}
        with supplemental_csv.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                variant = (row.get("variant") or "").strip()
                coeff_window = _to_int(str(row.get("coeff_window", "")))
                throughput = _to_float(str(row.get("throughput_img_s", "")))
                if not variant or coeff_window is None or throughput is None:
                    continue
                supplemental_throughput[(variant, coeff_window)] = throughput

    rows = _load_rows(csv_path, include_running=args.include_running, supplemental_throughput=supplemental_throughput)
    grouped = _group_by_variant(rows)

    acc1_path = output_dir / "res_wand_acc1_vs_coeff.png"
    acc5_path = output_dir / "res_wand_acc5_vs_coeff.png"
    throughput_path = output_dir / "res_wand_acc1_vs_throughput.png"

    _plot_accuracy(grouped, acc1_path, "best_acc1", "Top-1 accuracy (%)")
    _plot_accuracy(grouped, acc5_path, "best_acc5", "Top-5 accuracy (%)")
    _plot_throughput(grouped, throughput_path)

    print(f"[info] Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
