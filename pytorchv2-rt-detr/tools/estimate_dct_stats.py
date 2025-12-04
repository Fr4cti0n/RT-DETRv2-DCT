#!/usr/bin/env python3
"""Estimate per-coefficient DCT statistics for compressed ImageNet samples."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional progress bar
    tqdm = None

from src.data.dataset.imagenet import ImageNetDataset
from src.data.transforms.compress_reference_images import CompressToDCT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dirs", nargs="+", required=True,
                        help="One or more ImageNet-style folders used for statistics estimation.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Destination path for the statistics file (.pt). If omitted, a name is generated under configs/dct_stats/.")
    parser.add_argument("--coeff-window", type=int, default=8, choices=[1, 2, 4, 8],
                        help="Low-frequency window preserved during compression.")
    parser.add_argument("--range-mode", choices=["studio", "full"], default="studio",
                        help="Pixel range remapping applied prior to DCT.")
    parser.add_argument("--image-size", type=int, default=256,
                        help="Spatial size (pixels) for the resized square crop prior to compression.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=None,
                        help="Optionally cap the number of images processed for a quick estimate.")
    parser.add_argument("--device", default="cpu", help="Device used for accumulation (cpu or cuda:N).")
    parser.add_argument("--progress", action="store_true", help="Display a tqdm progress bar.")
    return parser.parse_args()


def build_dataloader(paths: Sequence[str], image_size: int, coeff_window: int, range_mode: str,
                     batch_size: int, workers: int, max_images: int | None, show_progress: bool) -> DataLoader:
    transform = T.Compose([
        T.ToImage(),
        T.Resize(image_size + 32),
        T.CenterCrop(image_size),
        T.ToDtype(torch.float32, scale=True),
        CompressToDCT(coeff_window=coeff_window, range_mode=range_mode, dtype=torch.float32, keep_original=False),
    ])
    dataset = ImageNetDataset([str(Path(p)) for p in paths], transforms=transform, max_samples=max_images,
                               show_progress=show_progress)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)


def accumulate_stats(loader: DataLoader, device: torch.device, show_progress: bool) -> dict[str, torch.Tensor]:
    if show_progress and tqdm is None:
        print("tqdm is not installed; disabling progress bar.")
        show_progress = False
    sum_luma = torch.zeros(64, dtype=torch.float64, device=device)
    sumsq_luma = torch.zeros(64, dtype=torch.float64, device=device)
    sum_chroma = torch.zeros(2, 64, dtype=torch.float64, device=device)
    sumsq_chroma = torch.zeros(2, 64, dtype=torch.float64, device=device)
    count_luma = torch.zeros(1, dtype=torch.float64, device=device)
    count_chroma = torch.zeros(1, dtype=torch.float64, device=device)

    iterator = loader
    if tqdm is not None and show_progress:
        iterator = tqdm(loader, total=len(loader), disable=False)

    for payload, _ in iterator:
        if isinstance(payload, (tuple, list)) and len(payload) == 2:
            y_blocks, cbcr_blocks = payload
        else:
            raise TypeError("Unexpected payload structure from dataloader; expected tuple/list of tensors.")

        y_blocks = y_blocks.to(device=device, dtype=torch.float64)
        cbcr_blocks = cbcr_blocks.to(device=device, dtype=torch.float64)

        sum_luma += y_blocks.sum(dim=(0, 2, 3))
        sumsq_luma += (y_blocks ** 2).sum(dim=(0, 2, 3))
        count_luma += y_blocks.shape[0] * y_blocks.shape[2] * y_blocks.shape[3]

        sum_chroma += cbcr_blocks.sum(dim=(0, 3, 4))
        sumsq_chroma += (cbcr_blocks ** 2).sum(dim=(0, 3, 4))
        count_chroma += cbcr_blocks.shape[0] * cbcr_blocks.shape[3] * cbcr_blocks.shape[4]

    mean_luma = (sum_luma / count_luma).cpu()
    mean_chroma = (sum_chroma / count_chroma).cpu()
    var_luma = (sumsq_luma / count_luma) - mean_luma.double() ** 2
    var_chroma = (sumsq_chroma / count_chroma) - mean_chroma.double() ** 2

    mean_luma = mean_luma.float()
    mean_chroma = mean_chroma.float()
    std_luma = torch.sqrt(torch.clamp(var_luma, min=0.0)).float()
    std_chroma = torch.sqrt(torch.clamp(var_chroma, min=0.0)).float()

    return {
        "mean_luma": mean_luma,
        "std_luma": std_luma,
        "mean_chroma": mean_chroma,
        "std_chroma": std_chroma,
        "count_luma": count_luma.item(),
        "count_chroma": count_chroma.item(),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    loader = build_dataloader(
        args.data_dirs,
        args.image_size,
        args.coeff_window,
        args.range_mode,
        args.batch_size,
        args.workers,
        args.max_images,
        args.progress,
    )
    stats = accumulate_stats(loader, device, args.progress)
    stats.update({
        "coeff_window": args.coeff_window,
        "range_mode": args.range_mode,
        "image_size": args.image_size,
    })

    if args.output is None:
        output_dir = Path("configs/dct_stats")
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"imagenet_coeff{args.coeff_window}_{args.range_mode}.pt"
        output_path = output_dir / filename
    else:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(stats, output_path)
    print(f"Saved DCT statistics -> {output_path}")


if __name__ == "__main__":
    main()
