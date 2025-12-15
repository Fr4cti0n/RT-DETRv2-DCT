#!/usr/bin/env python3
"""Compute accuracy deltas relative to the block-stem baseline from a W&B export."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Optional, Set, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("output/compressed_resnet34/crianns_eval/res_wand.csv"),
        help="Path to the W&B export CSV containing accuracy metrics.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("output/compressed_resnet34/crianns_eval/accuracy_delta_vs_block_stem.csv"),
        help="Path where the CSV with accuracy deltas will be written.",
    )
    parser.add_argument(
        "--baseline-variant",
        type=str,
        default="block-stem",
        help="Variant name to use as the baseline when computing accuracy deltas.",
    )
    parser.add_argument(
        "--baseline-coeff-window",
        type=int,
        default=8,
        help="Coefficient window for the baseline variant.",
    )
    return parser.parse_args()


def _to_float(value: str) -> Optional[float]:
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _to_int(value: str) -> Optional[int]:
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _read_rows(csv_path: Path) -> Dict[Tuple[str, int], Dict[str, float]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    results: Dict[Tuple[str, int], Dict[str, float]] = {}
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            state = (row.get("State") or "").strip().lower()
            if state and state != "finished":
                continue
            variant = (row.get("variant") or "").strip()
            coeff_window = _to_int(row.get("coeff_window", ""))
            if not variant or coeff_window is None:
                continue
            acc1 = _to_float(row.get("best/acc1", ""))
            acc5 = _to_float(row.get("best/acc5", ""))
            if acc1 is None and acc5 is None:
                continue
            results[(variant, coeff_window)] = {
                "acc1": acc1 if acc1 is not None else 0.0,
                "acc5": acc5 if acc5 is not None else 0.0,
            }
    if not results:
        raise ValueError("No finished runs with accuracy metrics were found in the CSV.")
    return results


def _write_deltas(
    output_path: Path,
    baseline: Tuple[str, int],
    baseline_metrics: Dict[str, float],
    rows: Dict[Tuple[str, int], Dict[str, float]],
    allowed_variants: Optional[Set[str]] = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "coeff_window",
        "acc1",
        "acc5",
        "delta_acc1_pct",
        "delta_acc5_pct",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for (variant, coeff_window), metrics in sorted(rows.items()):
            if allowed_variants is not None and variant not in allowed_variants:
                continue
            delta_acc1 = (metrics["acc1"] - baseline_metrics["acc1"]) * 100.0
            delta_acc5 = (metrics["acc5"] - baseline_metrics["acc5"]) * 100.0
            writer.writerow(
                {
                    "variant": variant,
                    "coeff_window": coeff_window,
                    "acc1": metrics["acc1"],
                    "acc5": metrics["acc5"],
                    "delta_acc1_pct": delta_acc1,
                    "delta_acc5_pct": delta_acc5,
                }
            )


def main() -> None:
    args = parse_args()
    rows = _read_rows(args.csv_path.expanduser())
    baseline_key = (args.baseline_variant, args.baseline_coeff_window)
    if baseline_key not in rows:
        raise KeyError(
            f"Baseline ({args.baseline_variant}, coeff_window={args.baseline_coeff_window}) not found in CSV."
        )
    baseline_metrics = rows[baseline_key]

    allowed = {args.baseline_variant, "luma-fusion"}
    _write_deltas(
        args.output_path.expanduser(),
        baseline_key,
        baseline_metrics,
        rows,
        allowed_variants=allowed,
    )
    print(f"[info] Wrote accuracy deltas to {args.output_path}")


if __name__ == "__main__":
    main()
