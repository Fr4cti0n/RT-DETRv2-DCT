#!/usr/bin/env python3
"""Compare backbone and detector DCT pipelines on a single sample.

The script pulls one image from the configured detection dataset, applies the
same augmentation / preprocessing sequence that was used to train the
compressed backbone, and runs it through both the backbone and detection
pipelines.  The resulting DCT coefficient tensors are compared element-wise to
confirm parity.

Example usage:

```
python tools/compare_backbone_detection_aug.py \
    --config configs/rtdetrv2/rtdetrv2_r34vd_lumafusion_coeff8_120e_coco.yml \
    --index 0 --seed 42 --image-size 640
```
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as T
from PIL import Image

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.yaml_config import YAMLConfig
from src.data.transforms import (
    Compose,
    CompressToDCT,
    NormalizeDCTCoefficients,
    NormalizeDCTCoefficientsFromFile,
)
from src.nn.backbone.compressed_presnet import _DCTBlockDecoder


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_detection_transform(
    image_size: int,
    coeff_window: int,
    range_mode: str,
    normalizer,
) -> Compose:
    transforms = [
        T.ToImage(),
        T.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToDtype(torch.float32, scale=True),
        CompressToDCT(
            coeff_window=coeff_window,
            range_mode=range_mode,
            dtype=torch.float32,
            keep_original=False,
        ),
        normalizer,
    ]
    return Compose(transforms)


def build_backbone_transform(
    image_size: int,
    coeff_window: int,
    range_mode: str,
    normalizer,
) -> T.Compose:
    transforms = [
        T.ToImage(),
        T.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToDtype(torch.float32, scale=True),
        CompressToDCT(
            coeff_window=coeff_window,
            range_mode=range_mode,
            dtype=torch.float32,
            keep_original=False,
        ),
        normalizer,
    ]
    return T.Compose(transforms)


def compare_payloads(det_payload, backbone_payload) -> dict[str, float]:
    det_y, det_cbcr = det_payload
    back_y, back_cbcr = backbone_payload

    diff_y = (det_y - back_y).abs()
    diff_cb = (det_cbcr - back_cbcr).abs()

    metrics = {
        "max_diff_y": diff_y.max().item(),
        "mean_diff_y": diff_y.mean().item(),
        "max_diff_cbcr": diff_cb.max().item(),
        "mean_diff_cbcr": diff_cb.mean().item(),
    }
    metrics["allclose_y"] = bool(torch.allclose(det_y, back_y))
    metrics["allclose_cbcr"] = bool(torch.allclose(det_cbcr, back_cbcr))
    return metrics


def _denormalize_coefficients(
    payload: Tuple[torch.Tensor, torch.Tensor],
    normalizer: NormalizeDCTCoefficients,
) -> Tuple[torch.Tensor, torch.Tensor]:
    y_blocks, cbcr_blocks = payload
    eps = normalizer.eps

    mean_luma = normalizer.mean_luma.to(device=y_blocks.device, dtype=y_blocks.dtype)
    std_luma = normalizer.std_luma.to(device=y_blocks.device, dtype=y_blocks.dtype)
    mask_luma = normalizer.mask_luma.to(device=y_blocks.device, dtype=y_blocks.dtype)
    y_denorm = torch.where(
        mask_luma > 0.5,
        y_blocks * (std_luma + eps) + mean_luma,
        y_blocks,
    )

    mean_chroma = normalizer.mean_chroma.to(device=cbcr_blocks.device, dtype=cbcr_blocks.dtype)
    std_chroma = normalizer.std_chroma.to(device=cbcr_blocks.device, dtype=cbcr_blocks.dtype)
    mask_chroma = normalizer.mask_chroma.to(device=cbcr_blocks.device, dtype=cbcr_blocks.dtype)
    mask_chroma = mask_chroma.expand_as(cbcr_blocks)
    cbcr_denorm = torch.where(
        mask_chroma > 0.5,
        cbcr_blocks * (std_chroma + eps) + mean_chroma,
        cbcr_blocks,
    )

    return y_denorm, cbcr_denorm


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
    rgb = torch.stack((r, g, b), dim=1)
    return (rgb / 255.0).clamp(0.0, 1.0)


def reconstruct_image(
    payload: Tuple[torch.Tensor, torch.Tensor],
    normalizer: NormalizeDCTCoefficients,
    range_mode: str,
) -> torch.Tensor:
    decoder = _DCTBlockDecoder().to(payload[0].device)
    y_blocks, cbcr_blocks = _denormalize_coefficients(payload, normalizer)

    y_plane = decoder(y_blocks.unsqueeze(0))
    cb_plane = decoder(cbcr_blocks[0].unsqueeze(0))
    cr_plane = decoder(cbcr_blocks[1].unsqueeze(0))

    target_hw = y_plane.shape[-2:]
    cb_up = F.interpolate(cb_plane.unsqueeze(1), size=target_hw, mode="bilinear", align_corners=False).squeeze(1)
    cr_up = F.interpolate(cr_plane.unsqueeze(1), size=target_hw, mode="bilinear", align_corners=False).squeeze(1)

    rgb = _ycbcr_to_rgb(y_plane, cb_up, cr_up, range_mode)
    return rgb.squeeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rtdetrv2/rtdetrv2_r34vd_lumafusion_coeff8_120e_coco.yml"),
        help="Detector YAML configuration path.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Dataset index to sample (zero-based).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed applied before each pipeline run.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="Augmented spatial size (must be divisible by 8).",
    )
    parser.add_argument(
        "--coeff-window",
        type=int,
        default=8,
        choices=[1, 2, 4, 8],
        help="Low-frequency coefficient window.",
    )
    parser.add_argument(
        "--range-mode",
        type=str,
        default="studio",
        choices=["studio", "full"],
        help="Input range passed to the DCT compressor.",
    )
    parser.add_argument(
        "--normalizer",
        type=Path,
        default=Path("configs/dct_stats/imagenet_coeff8_studio.pt"),
        help="Path to the pre-computed DCT statistics file.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display the reconstructed images using matplotlib.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to save reconstructed images.",
    )
    args = parser.parse_args()

    cfg = YAMLConfig(args.config)
    dataset = cfg.train_dataloader.dataset
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(0)

    raw_image, raw_target = dataset.load_item(args.index)

    normalizer_kwargs = dict(path=args.normalizer, coeff_window=args.coeff_window)
    det_normalizer = NormalizeDCTCoefficientsFromFile(**normalizer_kwargs)
    back_normalizer = NormalizeDCTCoefficientsFromFile(**normalizer_kwargs)
    recon_normalizer = NormalizeDCTCoefficientsFromFile(**normalizer_kwargs)

    det_transform = build_detection_transform(
        image_size=args.image_size,
        coeff_window=args.coeff_window,
        range_mode=args.range_mode,
        normalizer=det_normalizer,
    )

    seed_everything(args.seed)
    det_image, det_target, _ = det_transform(raw_image, raw_target, dataset)
    det_payload = det_image  # first element holds the DCT coefficients

    backbone_transform = build_backbone_transform(
        image_size=args.image_size,
        coeff_window=args.coeff_window,
        range_mode=args.range_mode,
        normalizer=back_normalizer,
    )

    seed_everything(args.seed)
    backbone_payload = backbone_transform(raw_image)

    if isinstance(backbone_payload, tuple) and len(backbone_payload) == 2:
        pass
    else:
        raise RuntimeError("Backbone transform did not return a DCT coefficient pair.")

    metrics = compare_payloads(det_payload, backbone_payload)

    print("=== Comparison Result ===")
    for key, value in metrics.items():
        if isinstance(value, bool):
            print(f"{key:>16}: {value}")
        else:
            print(f"{key:>16}: {value:.6f}")

    num_boxes = det_target["boxes"].shape[0] if "boxes" in det_target else 0
    print(f"boxes after aug    : {num_boxes}")
    print(f"image tensor shape  : {det_payload[0].shape}")
    print(f"chroma tensor shape : {det_payload[1].shape}")

    det_rgb = reconstruct_image(det_payload, recon_normalizer, args.range_mode)
    back_rgb = reconstruct_image(backbone_payload, recon_normalizer, args.range_mode)

    diff_rgb = (det_rgb - back_rgb).abs().mean(dim=0)

    det_np = det_rgb.permute(1, 2, 0).cpu().numpy()
    back_np = back_rgb.permute(1, 2, 0).cpu().numpy()
    diff_np = diff_rgb.cpu().numpy()

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        det_img = Image.fromarray((det_np * 255.0).clip(0, 255).astype(np.uint8))
        back_img = Image.fromarray((back_np * 255.0).clip(0, 255).astype(np.uint8))
        if diff_np.max() > 0:
            diff_norm = (diff_np / diff_np.max()).clip(0, 1)
        else:
            diff_norm = diff_np
        diff_img = Image.fromarray((diff_norm * 255.0).astype(np.uint8))
        diff_img = diff_img.convert("L")
        det_img.save(args.output_dir / "detector.png")
        back_img.save(args.output_dir / "backbone.png")
        diff_img.convert("L").save(args.output_dir / "difference.png")

    if args.display:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(det_np)
        axes[0].set_title("Detector pipeline")
        axes[1].imshow(back_np)
        axes[1].set_title("Backbone pipeline")
        im = axes[2].imshow(diff_np, cmap="magma")
        axes[2].set_title("Abs diff (mean)")
        for ax in axes:
            ax.axis("off")
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
