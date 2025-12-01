#!/usr/bin/env python3
"""Inspect compressed ImageNet samples by reconstructing them back to RGB."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch
from torchvision.utils import save_image

from ...data.dataset.imagenet import ImageNetDataset
from ...data.transforms.compress_reference_images import CompressToDCT
from .compressed_presnet import _upsample_chroma
from .train_backbones import (
    build_cspdarknet_transforms,
    build_efficientvit_transforms,
    build_resnet_transforms,
    default_configs,
)


def _ensure_multiple(values: Iterable[int], divisor: int = 8) -> List[int]:
    result: List[int] = []
    for value in values:
        if value % divisor != 0:
            raise ValueError(f"Value {value} must be divisible by {divisor}.")
        result.append(value)
    return result


def _ycbcr_to_rgb(y: torch.Tensor, cb: torch.Tensor, cr: torch.Tensor, range_mode: str) -> torch.Tensor:
    if range_mode == "studio":
        y_shift = (y - 16.0).clamp(min=0.0)
        cb_shift = cb - 128.0
        cr_shift = cr - 128.0
        scale = 255.0 / 219.0
        r = scale * y_shift + 1.596027 * cr_shift
        g = scale * y_shift - 0.391762 * cb_shift - 0.812968 * cr_shift
        b = scale * y_shift + 2.017232 * cb_shift
    else:
        cb_shift = cb - 128.0
        cr_shift = cr - 128.0
        r = y + 1.402 * cr_shift
        g = y - 0.344136 * cb_shift - 0.714136 * cr_shift
        b = y + 1.772 * cb_shift
    rgb = torch.stack((r, g, b), dim=0)
    rgb = rgb.clamp(0.0, 255.0) / 255.0
    return rgb


class DCTReconstructor:
    def __init__(self, range_mode: str) -> None:
        self.range_mode = range_mode
        self.block_size = 8
        basis = self._build_dct_matrix(self.block_size)
        self._basis = basis
        self._basis_t = basis.t()

    @staticmethod
    def _build_dct_matrix(size: int) -> torch.Tensor:
        rows = []
        scale0 = math.sqrt(1.0 / size)
        scale = math.sqrt(2.0 / size)
        for k in range(size):
            alpha = scale0 if k == 0 else scale
            row = [alpha * math.cos(math.pi * (n + 0.5) * k / size) for n in range(size)]
            rows.append(row)
        return torch.tensor(rows, dtype=torch.float32)

    def _decode_plane(self, blocks: torch.Tensor) -> torch.Tensor:
        if blocks.dim() != 3 or blocks.size(0) != self.block_size * self.block_size:
            raise ValueError("Blocks tensor must have shape [64, By, Bx].")
        device = blocks.device
        dtype = blocks.dtype
        basis = self._basis.to(device=device, dtype=dtype)
        basis_t = self._basis_t.to(device=device, dtype=dtype)
        by = blocks.size(1)
        bx = blocks.size(2)
        rows: list[torch.Tensor] = []
        for by_idx in range(by):
            cols: list[torch.Tensor] = []
            for bx_idx in range(bx):
                coeff_vec = blocks[:, by_idx, bx_idx]
                coeff_block = coeff_vec.view(self.block_size, self.block_size).t()
                spatial = basis_t @ coeff_block @ basis
                cols.append(spatial)
            rows.append(torch.cat(cols, dim=1))
        return torch.cat(rows, dim=0)

    def _decode_planes(
        self,
        y_blocks: torch.Tensor,
        cbcr_blocks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = y_blocks.device
        y_blocks = y_blocks.to(device=device, dtype=torch.float32)
        cbcr_blocks = cbcr_blocks.to(device=device, dtype=torch.float32)
        y_plane = self._decode_plane(y_blocks)
        cb_lr = self._decode_plane(cbcr_blocks[0])
        cr_lr = self._decode_plane(cbcr_blocks[1])
        target_hw = y_plane.shape[-2:]
        cb_plane = _upsample_chroma(cb_lr.unsqueeze(0).unsqueeze(0), target_hw).squeeze(0).squeeze(0)
        cr_plane = _upsample_chroma(cr_lr.unsqueeze(0).unsqueeze(0), target_hw).squeeze(0).squeeze(0)
        return y_plane, cb_plane, cr_plane

    def __call__(
        self,
        y_blocks: torch.Tensor,
        cbcr_blocks: torch.Tensor,
        *,
        crop_hw: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        y_plane, cb_plane, cr_plane = self._decode_planes(y_blocks, cbcr_blocks)
        if crop_hw is not None:
            height, width = crop_hw
            y_plane = y_plane[:height, :width]
            cb_plane = cb_plane[:height, :width]
            cr_plane = cr_plane[:height, :width]
        rgb = _ycbcr_to_rgb(y_plane, cb_plane, cr_plane, self.range_mode)
        return rgb, y_plane, cb_plane, cr_plane


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(default_configs().keys()), default="resnet34")
    parser.add_argument("--train-dirs", nargs="+", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--indices", type=int, nargs="+", default=[0])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--coeff-window", type=int, choices=[1, 2, 4, 8], default=8)
    parser.add_argument("--range-mode", choices=["studio", "full"], default="studio")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--flip-prob", type=float, default=0.5, help="Override probability for RandomHorizontalFlip in the training transform.")
    parser.add_argument(
        "--crop-scale",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(0.08, 1.0),
        help="Override (min, max) scale for RandomResizedCrop in the training transform.",
    )
    parser.add_argument("--show-progress", action="store_true")
    return parser.parse_args()


def _build_transforms(
    model: str,
    image_size: int,
    compression_cfg,
    augmentation_overrides: Optional[dict[str, object]],
) -> tuple:
    if model == "resnet34":
        return build_resnet_transforms(
            image_size,
            compression_cfg,
            augmentation_overrides=augmentation_overrides,
        )
    if model == "cspdarknet53":
        return build_cspdarknet_transforms(
            image_size,
            compression_cfg,
            augmentation_overrides=augmentation_overrides,
        )
    if model == "efficientvit_m4":
        return build_efficientvit_transforms(
            image_size,
            compression_cfg,
            augmentation_overrides=augmentation_overrides,
        )
    raise ValueError(model)


def main() -> None:
    args = parse_args()
    cfg = default_configs()[args.model]
    torch.manual_seed(args.seed)

    compression_cfg = {
        "coeff_window": args.coeff_window,
        "range_mode": args.range_mode,
        "dtype": torch.float32,
        "keep_original": True,
    }

    crop_min, crop_max = args.crop_scale
    if crop_min <= 0.0 or crop_max <= 0.0 or crop_min > crop_max or crop_max > 1.0:
        raise ValueError("crop-scale must satisfy 0 < min <= max <= 1.0")
    if not 0.0 <= args.flip_prob <= 1.0:
        raise ValueError("flip-prob must be within [0, 1].")
    augmentation_overrides: dict[str, object] = {
        "flip_prob": args.flip_prob,
        "crop_scale": (crop_min, crop_max),
    }

    train_tf, val_tf = _build_transforms(
        args.model,
        cfg.image_size,
        compression_cfg,
        augmentation_overrides,
    )
    transforms = train_tf if args.split == "train" else val_tf

    if args.split == "train":
        dataset = ImageNetDataset(
            [str(Path(p)) for p in args.train_dirs],
            transforms=transforms,
            show_progress=args.show_progress,
        )
    else:
        dataset = ImageNetDataset(
            [str(Path(args.val_dir))],
            transforms=transforms,
            show_progress=args.show_progress,
        )

    reconstructor = DCTReconstructor(args.range_mode)
    out_dir = None
    if args.output_dir is not None:
        out_dir = args.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

    for idx in _ensure_multiple(args.indices, 1):
        sample, target = dataset[idx]
        original = None
        if isinstance(sample, tuple) and len(sample) == 2 and isinstance(sample[0], tuple):
            (y_blocks, cbcr_blocks), original = sample
        else:
            y_blocks, cbcr_blocks = sample

        crop_hw = None
        if original is not None:
            original = original.clamp(0.0, 1.0)
            crop_hw = (original.shape[-2], original.shape[-1])

        recon, y_plane, cb_plane, cr_plane = reconstructor(y_blocks, cbcr_blocks, crop_hw=crop_hw)
        recon = recon.clamp(0.0, 1.0)
        print(f"index={idx} class={target} recon min={recon.min():.3f} max={recon.max():.3f}")
        if original is not None:
            delta = float((recon - original).abs().mean())
            print(
                f"  original min={original.min():.3f} max={original.max():.3f} "
                f"delta={delta:.4f}"
            )
        if out_dir is not None:
            base = f"{args.split}_{idx:05d}"
            save_image(recon, out_dir / f"{base}_recon.png")
            y_vis = y_plane.unsqueeze(0).clamp(0.0, 255.0) / 255.0
            save_image(y_vis, out_dir / f"{base}_luma.png")
            cb_vis = ((cb_plane - 128.0) / 255.0 + 0.5).clamp(0.0, 1.0)
            cr_vis = ((cr_plane - 128.0) / 255.0 + 0.5).clamp(0.0, 1.0)
            save_image(cb_vis.unsqueeze(0), out_dir / f"{base}_chroma_cb.png")
            save_image(cr_vis.unsqueeze(0), out_dir / f"{base}_chroma_cr.png")
            if original is not None:
                save_image(original, out_dir / f"{base}_original.png")


if __name__ == "__main__":
    main()
