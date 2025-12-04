#!/usr/bin/env python3
"""Inspect per-coefficient means after applying DCT normalisation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional progress bar
    tqdm = None

from src.data.dataset.imagenet import ImageNetDataset
from src.data.transforms.dct_normalize import NormalizeDCTCoefficients
from src.nn.backbone.train_backbones import build_resnet_transforms


def _maybe_default_stats_path(coeff_window: int, range_mode: str) -> Path:
    candidate = Path("configs/dct_stats") / f"imagenet_coeff{coeff_window}_{range_mode}.pt"
    return candidate if candidate.exists() else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dirs", nargs="+", required=True,
                        help="ImageNet-style folders used to load samples.")
    parser.add_argument("--coeff-window", type=int, choices=[1, 2, 4, 8], default=8,
                        help="Low-frequency window preserved by CompressToDCT.")
    parser.add_argument("--range-mode", choices=["studio", "full"], default="studio",
                        help="Pixel range remapping applied before the DCT.")
    parser.add_argument("--image-size", type=int, default=256,
                        help="Square crop size supplied to build_resnet_transforms.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=None,
                        help="Optionally cap the number of images processed.")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Process at most this many batches (after filtering by --max-images if provided).")
    parser.add_argument("--dct-stats", type=Path, default=None,
                        help="Path to the statistics .pt file. Defaults to configs/dct_stats/imagenet_coeff{window}_{range}.pt if present.")
    parser.add_argument("--progress", action="store_true", help="Display a tqdm progress bar if available.")
    parser.add_argument("--full", action="store_true",
                        help="Print per-coefficient means/stds instead of only summary deviations.")
    return parser.parse_args()


def build_loader(args: argparse.Namespace, normaliser: T.Transform | None) -> DataLoader:
    compression_cfg = {
        "coeff_window": args.coeff_window,
        "range_mode": args.range_mode,
        "dtype": torch.float32,
        "keep_original": False,
    }
    train_tf, _ = build_resnet_transforms(
        args.image_size,
        compression=compression_cfg,
        dct_normalizer_train=normaliser,
        dct_normalizer_val=normaliser,
    )
    dataset = ImageNetDataset(
        [str(Path(p)) for p in args.data_dirs],
        transforms=train_tf,
        max_samples=args.max_images,
        show_progress=args.progress,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    return loader


def accumulate_statistics(
    loader: DataLoader,
    show_progress: bool,
    max_batches: int | None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if show_progress and tqdm is None:
        print("tqdm is not installed; disabling progress bar.")
        show_progress = False

    iterator: Iterable = loader
    if show_progress and tqdm is not None:
        iterator = tqdm(loader, total=len(loader), disable=False)

    sum_luma = torch.zeros(64, dtype=torch.float64)
    sumsq_luma = torch.zeros(64, dtype=torch.float64)
    sum_chroma = torch.zeros(2, 64, dtype=torch.float64)
    sumsq_chroma = torch.zeros(2, 64, dtype=torch.float64)
    count_luma = torch.zeros(1, dtype=torch.float64)
    count_chroma = torch.zeros(1, dtype=torch.float64)

    processed_batches = 0
    try:
        for batch_idx, (payload, _) in enumerate(iterator):
            if isinstance(payload, (tuple, list)) and len(payload) == 2:
                y_blocks, cbcr_blocks = payload
            else:
                raise TypeError("Unexpected payload structure; expected tuple/list of two tensors.")

            y_blocks = y_blocks.to(dtype=torch.float64)
            cbcr_blocks = cbcr_blocks.to(dtype=torch.float64)

            sum_luma += y_blocks.sum(dim=(0, 2, 3))
            sum_chroma += cbcr_blocks.sum(dim=(0, 3, 4))
            sumsq_luma += (y_blocks ** 2).sum(dim=(0, 2, 3))
            sumsq_chroma += (cbcr_blocks ** 2).sum(dim=(0, 3, 4))
            count_luma += y_blocks.shape[0] * y_blocks.shape[2] * y_blocks.shape[3]
            count_chroma += cbcr_blocks.shape[0] * cbcr_blocks.shape[3] * cbcr_blocks.shape[4]

            processed_batches = batch_idx + 1
            if max_batches is not None and processed_batches >= max_batches:
                break
    except KeyboardInterrupt:
        print("Interrupted by user; returning partial statistics.", flush=True)

    if count_luma.item() == 0 or count_chroma.item() == 0:
        raise RuntimeError("No samples were processed; cannot compute statistics.")

    print(f"Processed {processed_batches} batch(es).", flush=True)

    mean_luma = (sum_luma / count_luma).to(dtype=torch.float32)
    mean_chroma = (sum_chroma / count_chroma).to(dtype=torch.float32)
    var_luma = (sumsq_luma / count_luma) - mean_luma.double() ** 2
    var_chroma = (sumsq_chroma / count_chroma) - mean_chroma.double() ** 2
    std_luma = torch.sqrt(torch.clamp(var_luma, min=0.0)).to(dtype=torch.float32)
    std_chroma = torch.sqrt(torch.clamp(var_chroma, min=0.0)).to(dtype=torch.float32)
    return mean_luma, std_luma, mean_chroma, std_chroma


def main() -> None:
    args = parse_args()

    stats_path = args.dct_stats
    if stats_path is None:
        stats_path = _maybe_default_stats_path(args.coeff_window, args.range_mode)
        if stats_path is not None:
            print(f"Using default stats file: {stats_path}", flush=True)
    normaliser = None
    if stats_path is not None:
        stats_path = stats_path.expanduser().resolve()
        if not stats_path.exists():
            raise FileNotFoundError(f"Statistics file not found: {stats_path}")
        normaliser = NormalizeDCTCoefficients.from_file(
            stats_path,
            coeff_window=args.coeff_window,
        )
        print(f"Loaded DCT stats from {stats_path}", flush=True)
    else:
        print("No stats file supplied; normalisation will be skipped.", flush=True)

    loader = build_loader(args, normaliser)
    print("Accumulating statistics…", flush=True)
    mean_luma, std_luma, mean_chroma, std_chroma = accumulate_statistics(
        loader,
        args.progress,
        args.max_batches,
    )

    ref_stats = None
    if normaliser is not None:
        ref_stats = {
            "luma_mean": normaliser.mean_luma.view(-1).detach().cpu().to(torch.float32),
            "luma_std": normaliser.std_luma.view(-1).detach().cpu().to(torch.float32),
            "chroma_mean": normaliser.mean_chroma.view(2, -1).detach().cpu().to(torch.float32),
            "chroma_std": normaliser.std_chroma.view(2, -1).detach().cpu().to(torch.float32),
        }

    def _print_block(
        label: str,
        means: torch.Tensor,
        stds: torch.Tensor,
        ref_means: torch.Tensor | None,
        ref_stds: torch.Tensor | None,
        indent: str = "  ",
    ) -> None:
        flat_mean = means.view(-1)
        flat_std = stds.view(-1)
        max_mean = flat_mean.abs().max().item()
        max_std_delta = (flat_std - 1.0).abs().max().item()
        print(f"{label}: max|mean|={max_mean:.6f} max|std-1|={max_std_delta:.6f}", flush=True)
        if args.full:
            for idx in range(flat_mean.numel()):
                print(f"{indent}{label}[{idx:02d}] mean={flat_mean[idx]:.6f} std={flat_std[idx]:.6f}", flush=True)
                if ref_means is not None and ref_stds is not None:
                    print(f"{indent}  ref mean={ref_means[idx]:.6f} std={ref_stds[idx]:.6f}", flush=True)

    _print_block(
        "Luma",
        mean_luma,
        std_luma,
        ref_stats["luma_mean"] if ref_stats is not None else None,
        ref_stats["luma_std"] if ref_stats is not None else None,
    )

    print("Chroma channels:", flush=True)
    for channel, prefix in enumerate(("Cb", "Cr")):
        _print_block(
            prefix,
            mean_chroma[channel],
            std_chroma[channel],
            ref_stats["chroma_mean"][channel] if ref_stats is not None else None,
            ref_stats["chroma_std"][channel] if ref_stats is not None else None,
            indent="    ",
        )

    print("Normalisation check complete.", flush=True)

if __name__ == "__main__":
    main()
