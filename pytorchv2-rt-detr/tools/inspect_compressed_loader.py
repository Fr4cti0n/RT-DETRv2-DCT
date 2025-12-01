#!/usr/bin/env python3
"""Inspect the output of the ImageNet compressed data loader."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import torch

# Ensure project root on sys.path when invoked from the tools/ directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nn.backbone.train_backbones import (  # noqa: E402
    build_dataloaders,
    default_configs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="resnet34",
        choices=list(default_configs().keys()),
        help="Backbone preset used to select image size and augmentations.",
    )
    parser.add_argument(
        "--train-dirs",
        nargs="+",
        required=True,
        help="ImageNet-style directories for the training split.",
    )
    parser.add_argument(
        "--val-dir",
        required=True,
        help="Validation directory following ImageNet layout.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="train",
        help="Choose which loader to inspect.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-images",
        type=int,
        default=8,
        help="Optional cap on the number of samples loaded for the inspected split.",
    )
    parser.add_argument(
        "--coeff-window",
        type=int,
        default=4,
        choices=[1, 2, 4, 8],
        help="Low-frequency window kept per 8x8 block.",
    )
    parser.add_argument(
        "--range-mode",
        choices=["studio", "full"],
        default="studio",
        help="Numeric range mapped before the DCT.",
    )
    parser.add_argument(
        "--keep-original",
        action="store_true",
        help="Retain the original RGB tensor alongside the compressed payload.",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Display a tqdm progress bar while indexing the dataset.",
    )
    return parser.parse_args()


def _format_stats(tensor: torch.Tensor) -> str:
    if tensor.numel() == 0:
        return "empty"
    min_val = tensor.amin().item()
    max_val = tensor.amax().item()
    mean_val = tensor.mean().item()
    return f"min={min_val:.2f}, max={max_val:.2f}, mean={mean_val:.2f}"


def _describe_payload(batch_payload) -> Tuple[str, Iterable[str]]:
    if isinstance(batch_payload, torch.Tensor):
        return "tensor", [
            f"shape={tuple(batch_payload.shape)}",
            f"dtype={batch_payload.dtype}",
            _format_stats(batch_payload),
        ]

    if isinstance(batch_payload, (list, tuple)):
        entries = []
        for idx, item in enumerate(batch_payload):
            kind, stats = _describe_payload(item)
            prefix = f"[{idx}] {kind}"
            entries.append(" - ".join([prefix] + list(stats)))
        return "sequence", entries

    if isinstance(batch_payload, dict):
        entries = []
        for key, value in batch_payload.items():
            kind, stats = _describe_payload(value)
            headline = f"{key} ({kind})"
            entries.append(" - ".join([headline] + list(stats)))
        return "mapping", entries

    return type(batch_payload).__name__, [str(batch_payload)]


def main() -> None:
    args = parse_args()

    cfg = default_configs()[args.model]

    max_images = args.max_images if args.max_images and args.max_images > 0 else None

    train_loader, val_loader = build_dataloaders(
        args.model,
        [Path(p).expanduser().resolve() for p in args.train_dirs],
        Path(args.val_dir).expanduser().resolve(),
        args.batch_size,
        args.num_workers,
        cfg.image_size,
        max_images,
        max_images,
        "compressed",
        {
            "coeff_window": args.coeff_window,
            "range_mode": args.range_mode,
            "dtype": torch.float32,
            "keep_original": args.keep_original,
        },
        args.show_progress,
    )

    loader = train_loader if args.split == "train" else val_loader
    total_batches = 0
    processed_images = 0

    for batch_idx, (images, targets) in enumerate(loader, start=1):
        batch_size = len(targets)
        total_batches = batch_idx
        processed_images += batch_size

        print(f"\n=== Batch {batch_idx} (loader idx {batch_idx - 1}) ===")
        print(f"Batch size: {batch_size}")
        print(f"Total processed: {processed_images}")
        print(f"Targets dtype: {targets.dtype}")
        print(f"Targets sample: {targets[: min(5, targets.numel())].tolist()}")

        payload_type, payload_stats = _describe_payload(images)
        print(f"Payload structure: {payload_type}")
        for entry in payload_stats:
            print(f"  {entry}")

        if isinstance(images, (list, tuple)) and len(images) >= 1:
            primary = images[0]
        else:
            primary = images

        if isinstance(primary, (list, tuple)) and len(primary) == 2:
            y_blocks, cbcr_blocks = primary
            if isinstance(y_blocks, torch.Tensor):
                print("Y blocks:")
                print(f"  shape={tuple(y_blocks.shape)}")
                print(f"  {_format_stats(y_blocks)}")
            if isinstance(cbcr_blocks, torch.Tensor):
                print("Cb/Cr blocks:")
                print(f"  shape={tuple(cbcr_blocks.shape)}")
                print(f"  {_format_stats(cbcr_blocks)}")

        if args.keep_original and isinstance(images, (list, tuple)) and len(images) == 2:
            original = images[1]
            if isinstance(original, torch.Tensor):
                print("Original tensor:")
                print(f"  shape={tuple(original.shape)}")
                print(f"  {_format_stats(original)}")

    print(f"\nProcessed batches: {total_batches}")
    print(f"Processed images: {processed_images}")


if __name__ == "__main__":
    main()
