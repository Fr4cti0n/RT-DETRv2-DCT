#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compress reference (I-frame) images into DCT block coefficients.

The script takes one or multiple image paths, converts them to YCbCr 4:2:0,
then computes 8x8 DCT blocks for luminance and chrominance planes. The
resulting block tensors can be saved as `.npz` archives for downstream use.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torchvision.transforms import v2 as T

from .frame_dct import FrameDCT
from ...core import register

RangeMode = tuple[str, ...]

TensorLike = Union[torch.Tensor, np.ndarray]


def _ensure_macroblock_multiple(image: np.ndarray) -> np.ndarray:
    """Pad the image so width and height become multiples of 16."""
    height, width = image.shape[:2]
    pad_bottom = (16 - height % 16) % 16
    pad_right = (16 - width % 16) % 16
    if pad_bottom == 0 and pad_right == 0:
        return image
    return cv2.copyMakeBorder(
        image,
        top=0,
        bottom=pad_bottom,
        left=0,
        right=pad_right,
        borderType=cv2.BORDER_REPLICATE,
    )


def _to_ycbcr_420(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a BGR image to Y, Cb, Cr planes using 4:2:0 subsampling."""
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    y_plane = ycrcb[:, :, 0].astype(np.float32)
    cr_full = ycrcb[:, :, 1].astype(np.float32)
    cb_full = ycrcb[:, :, 2].astype(np.float32)
    height, width = y_plane.shape
    half_size = (width // 2, height // 2)
    cb_plane = cv2.resize(cb_full, half_size, interpolation=cv2.INTER_AREA)
    cr_plane = cv2.resize(cr_full, half_size, interpolation=cv2.INTER_AREA)
    return y_plane, cb_plane, cr_plane


def _map_range(plane: np.ndarray, mode: str, chroma: bool = False) -> np.ndarray:
    """Adjust the numeric range of a plane depending on the requested mode."""
    studio_max = 240.0 if chroma else 235.0
    studio_span = 224.0 if chroma else 219.0

    if mode == "studio":
        # If the plane is already in studio range, keep it as-is.
        if plane.min() >= 16.0 and plane.max() <= studio_max:
            return plane
        return np.clip(plane * (studio_span / 255.0) + 16.0, 16.0, studio_max)

    if mode == "full":
        # Convert studio-range values back to full range only when needed.
        if plane.min() >= 16.0 and plane.max() <= studio_max:
            return np.clip((plane - 16.0) * (255.0 / studio_span), 0.0, 255.0)
        return plane

    raise ValueError(f"Unsupported range mode: {mode}")


def _plane_to_blocks(dct_plane: np.ndarray) -> np.ndarray:
    """Reshape a 2D DCT plane into (nb_y, nb_x, 64) blocks using column-major order."""
    height, width = dct_plane.shape
    blocks_y = height // 8
    blocks_x = width // 8
    blocks = np.zeros((blocks_y, blocks_x, 64), dtype=np.float32)
    for by in range(blocks_y):
        top = by * 8
        for bx in range(blocks_x):
            left = bx * 8
            block = dct_plane[top:top + 8, left:left + 8]
            blocks[by, bx] = block.reshape(64, order="F")
    return blocks


def _build_active_index(coeff_window: int) -> np.ndarray | None:
    if coeff_window >= 8:
        return None
    indices = [row + col * 8 for col in range(coeff_window) for row in range(coeff_window)]
    return np.array(indices, dtype=np.int64)


def _apply_frequency_window(blocks: np.ndarray, coeff_window: int) -> np.ndarray:
    """Zero out high-frequency coefficients outside a size×size window per block."""
    if coeff_window not in (1, 2, 4, 8):
        raise ValueError(f"Unsupported coefficient window: {coeff_window}")
    if coeff_window == 8:
        return blocks
    mask = np.zeros((8, 8), dtype=np.float32)
    mask[:coeff_window, :coeff_window] = 1.0
    reshaped = blocks.reshape((-1, 8, 8), order="F")
    reshaped *= mask
    return reshaped.reshape(blocks.shape, order="F")


def _forward_dct(plane: np.ndarray) -> np.ndarray:
    """Run the forward block-wise DCT using the existing project helper."""
    dct_plane = FrameDCT._forward_dct_plane(plane)
    if dct_plane is None:
        raise RuntimeError("Forward DCT returned None despite valid input plane.")
    return dct_plane.astype(np.float32)


def _compress_image_array(
    image_bgr: np.ndarray,
    range_mode: str,
    coeff_window: int,
    round_blocks: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Core helper that converts a BGR image into DCT coefficient blocks."""

    image = _ensure_macroblock_multiple(image_bgr)
    y_plane, cb_plane, cr_plane = _to_ycbcr_420(image)
    y_plane = _map_range(y_plane, range_mode, chroma=False)
    cb_plane = _map_range(cb_plane, range_mode, chroma=True)
    cr_plane = _map_range(cr_plane, range_mode, chroma=True)

    y_dct = _forward_dct(y_plane)
    cb_dct = _forward_dct(cb_plane)
    cr_dct = _forward_dct(cr_plane)

    luma_blocks = _plane_to_blocks(y_dct)
    cb_blocks = _plane_to_blocks(cb_dct)
    cr_blocks = _plane_to_blocks(cr_dct)

    luma_blocks = _apply_frequency_window(luma_blocks, coeff_window)
    cb_blocks = _apply_frequency_window(cb_blocks, coeff_window)
    cr_blocks = _apply_frequency_window(cr_blocks, coeff_window)

    if round_blocks:
        luma_blocks = np.rint(luma_blocks).astype(np.int16, copy=False)
        cb_blocks = np.rint(cb_blocks).astype(np.int16, copy=False)
        cr_blocks = np.rint(cr_blocks).astype(np.int16, copy=False)
    else:
        luma_blocks = luma_blocks.astype(np.float32, copy=False)
        cb_blocks = cb_blocks.astype(np.float32, copy=False)
        cr_blocks = cr_blocks.astype(np.float32, copy=False)

    luma_grid_shape = np.array(luma_blocks.shape[:2], dtype=np.int32)
    chroma_grid_shape = np.array(cb_blocks.shape[:2], dtype=np.int32)

    metadata = {
        "spatial_shape": np.array(y_plane.shape, dtype=np.int32),
        "chroma_shape": np.array(cb_plane.shape, dtype=np.int32),
        "coeff_window": np.array(coeff_window, dtype=np.int32),
        "coefficients_per_block": np.array(64, dtype=np.int32),
        "active_coefficients": np.array(coeff_window * coeff_window, dtype=np.int32),
        "luma_grid_shape": luma_grid_shape,
        "chroma_grid_shape": chroma_grid_shape,
    }

    return luma_blocks, cb_blocks, cr_blocks, metadata


def compress_reference_image(path: Path, range_mode: str, coeff_window: int) -> dict[str, np.ndarray]:
    """Compute Y, Cb, Cr DCT blocks for a single reference image."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")

    luma_blocks, cb_blocks, cr_blocks, metadata = _compress_image_array(
        image, range_mode, coeff_window, round_blocks=True
    )

    return {
        "source_path": np.array(str(path), dtype=np.string_),
        "range_mode": np.array(range_mode, dtype=np.string_),
        "luma_blocks": luma_blocks,
        "cb_blocks": cb_blocks,
        "cr_blocks": cr_blocks,
        **metadata,
    }


def _resolve_torch_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        key = dtype.lower()
        aliases = {
            "float": torch.float32,
            "float32": torch.float32,
            "fp32": torch.float32,
            "half": torch.float16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
        }
        if key in aliases:
            return aliases[key]
        candidate = getattr(torch, key, None)
        if isinstance(candidate, torch.dtype):
            return candidate
    raise ValueError(f"Unsupported dtype specification: {dtype!r}")


@register()
class CompressToDCT(T.Transform):
    """Transform that converts an RGB tensor into DCT coefficient blocks.

    Returns a tuple ``(y_blocks, cbcr_blocks)`` where ``y_blocks`` has shape
    ``(64, nb_y, nb_x)`` and ``cbcr_blocks`` has shape ``(2, 64, nb_y//2, nb_x//2)``
    for 4:2:0 chroma subsampling. When ``keep_original`` is ``True`` the original
    input is appended as a second element.
    """

    def __init__(
        self,
        coeff_window: int = 8,
        range_mode: str = "studio",
        dtype: Union[str, torch.dtype] = torch.float32,
        keep_original: bool = False,
    ) -> None:
        super().__init__()
        if range_mode not in {"studio", "full"}:
            raise ValueError(f"Unsupported range_mode: {range_mode}")
        if coeff_window not in (1, 2, 4, 8):
            raise ValueError(f"Unsupported coeff_window: {coeff_window}")

        self.coeff_window = coeff_window
        self.range_mode = range_mode
        self.dtype = _resolve_torch_dtype(dtype)
        self.keep_original = keep_original

    @staticmethod
    def _to_bgr_array(image: TensorLike) -> np.ndarray:
        if isinstance(image, torch.Tensor):
            tensor = image.detach().cpu()
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim != 3:
                raise ValueError("Expected image tensor with 3 dimensions for compression.")

            if tensor.shape[0] in (1, 3):
                channel_first = tensor
            elif tensor.shape[-1] in (1, 3):
                # torchvision v2 tv_tensors keep channels in the last dimension.
                channel_first = tensor.permute(2, 0, 1)
            else:
                raise ValueError(
                    "Unsupported channel layout for compression; expected 1 or 3 channels."
                )

            if channel_first.shape[0] == 1:
                channel_first = channel_first.repeat(3, 1, 1)

            if channel_first.dtype.is_floating_point:
                channel_first = channel_first.clamp(0.0, 1.0) * 255.0
                channel_first = channel_first.round()
            else:
                channel_first = channel_first.to(torch.float32)

            array = channel_first.permute(1, 2, 0).numpy()
        elif isinstance(image, np.ndarray):
            array = np.asarray(image)
            if array.ndim == 2:
                array = np.stack([array, array, array], axis=-1)
            if array.ndim != 3 or array.shape[2] != 3:
                raise ValueError("NumPy inputs must have shape (H, W, 3) for compression.")
            if array.dtype != np.float32:
                array = array.astype(np.float32)
            if array.max() <= 1.0 + 1e-3:
                array *= 255.0
        else:
            array = np.asarray(image, dtype=np.float32)
            if array.ndim == 2:
                array = np.stack([array, array, array], axis=-1)
            if array.ndim != 3 or array.shape[2] != 3:
                raise ValueError("Inputs must be convertible to an (H, W, 3) array.")
            if array.max() <= 1.0 + 1e-3:
                array *= 255.0

        if array.dtype != np.uint8:
            array = array.astype(np.uint8, copy=False)
        return array[..., ::-1]

    @staticmethod
    def _blocks_to_tensor(blocks: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        tensor = torch.from_numpy(blocks).permute(2, 0, 1)
        return tensor.to(dtype)

    def _compress(self, image: TensorLike, target=None):
        original = image if self.keep_original else None
        bgr_image = self._to_bgr_array(image)
        luma_blocks, cb_blocks, cr_blocks, metadata = _compress_image_array(
            bgr_image, self.range_mode, self.coeff_window, round_blocks=False
        )

        y_tensor = self._blocks_to_tensor(luma_blocks, self.dtype)
        cb_tensor = self._blocks_to_tensor(cb_blocks, self.dtype)
        cr_tensor = self._blocks_to_tensor(cr_blocks, self.dtype)
        cbcr_tensor = torch.stack((cb_tensor, cr_tensor), dim=0)

        payload = (y_tensor, cbcr_tensor)

        # Attach useful metadata to the target when it is a mutable mapping.
        if target is not None and hasattr(target, "update") and callable(target.update):
            target.update({
                "dct_spatial_shape": torch.tensor(metadata["spatial_shape"], dtype=torch.int32),
                "dct_chroma_shape": torch.tensor(metadata["chroma_shape"], dtype=torch.int32),
                "dct_luma_grid": torch.tensor(metadata["luma_grid_shape"], dtype=torch.int32),
                "dct_chroma_grid": torch.tensor(metadata["chroma_grid_shape"], dtype=torch.int32),
                "dct_coeff_window": int(metadata["coeff_window"]),
            })

        if self.keep_original:
            payload = (payload, original)

        return payload, target

    def forward(self, inputs: TensorLike | tuple, target=None):
        if target is None and isinstance(inputs, (tuple, list)):
            if len(inputs) == 3:
                image, tgt, dataset = inputs
                payload, tgt = self._compress(image, tgt)
                return payload, tgt, dataset
            if len(inputs) == 2:
                image, tgt = inputs
                payload, tgt = self._compress(image, tgt)
                return payload, tgt

        payload, target = self._compress(inputs, target)
        if target is None:
            return payload
        return payload, target


class TrimDCTCoefficients(T.Transform):
    """Reduce DCT payload depth to coeff_window² active coefficients."""

    def __init__(self, coeff_window: int) -> None:
        super().__init__()
        if coeff_window not in (1, 2, 4, 8):
            raise ValueError(f"Unsupported coeff_window: {coeff_window}")
        self.coeff_window = coeff_window
        active_idx = _build_active_index(coeff_window)
        self.active_idx = active_idx

    def _trim_pair(self, y_blocks: torch.Tensor, cbcr_blocks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.active_idx is None or y_blocks.size(0) <= self.coeff_window * self.coeff_window:
            return y_blocks, cbcr_blocks
        idx = torch.as_tensor(self.active_idx, dtype=torch.long, device=y_blocks.device)
        y_trim = torch.index_select(y_blocks, 0, idx)
        cb = torch.index_select(cbcr_blocks[0], 0, idx)
        cr = torch.index_select(cbcr_blocks[1], 0, idx)
        cbcr_trim = torch.stack((cb, cr), dim=0)
        return y_trim, cbcr_trim

    def _trim_payload(self, payload):
        if isinstance(payload, tuple) and len(payload) == 2:
            first, second = payload
            if torch.is_tensor(first) and torch.is_tensor(second):
                return self._trim_pair(first, second)
            trimmed_first = self._trim_payload(first)
            return trimmed_first, second
        if isinstance(payload, tuple) and len(payload) > 2:
            trimmed_first = self._trim_payload(payload[0])
            return (trimmed_first, *payload[1:])
        raise TypeError("TrimDCTCoefficients expects payloads produced by CompressToDCT")

    def forward(self, inputs, target=None):
        trimmed = self._trim_payload(inputs)
        if target is None:
            return trimmed
        return trimmed, target


def _save_result(result: dict[str, np.ndarray], output_dir: Path) -> Path:
    """Persist the compressed blocks into an `.npz` archive."""
    source_path = Path(result["source_path"].item().decode("utf-8"))
    stem = source_path.stem
    target = output_dir / f"{stem}_dct.npz"
    np.savez_compressed(target, **result)
    return target


def _iter_image_paths(entries: Iterable[str]) -> Iterable[Path]:
    """Yield every image path, expanding directories into sorted file lists."""
    for entry in entries:
        path = Path(entry)
        if path.is_dir():
            for candidate in sorted(path.glob("*")):
                if candidate.is_file():
                    yield candidate
        else:
            yield path


def main(argv: RangeMode | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", help="Paths to images or folders to compress.")
    parser.add_argument(
        "--range",
        dest="range_mode",
        choices=["studio", "full"],
        default="studio",
        help="Pixel range to use before computing the DCT (default: studio).",
    )
    parser.add_argument(
        "--coeff-window",
        dest="coeff_window",
        type=int,
        choices=[1, 2, 4, 8],
        default=8,
        help="Size of the preserved low-frequency window per block (default: 8).",
    )
    parser.add_argument(
        "--print-only",
        dest="print_only",
        action="store_true",
        help="Do not write `.npz` files; only display the resulting tensor shapes.",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        help="Directory where `.npz` archives are stored (default: alongside each image).",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    compressed_paths: list[Path] = []
    for image_path in _iter_image_paths(args.images):
        result = compress_reference_image(image_path, args.range_mode, args.coeff_window)
        luma_shape = result["luma_blocks"].shape
        cb_shape = result["cb_blocks"].shape
        cr_shape = result["cr_blocks"].shape
        coeff_per_block = int(result["coefficients_per_block"].item())
        active_coeff = int(result.get("active_coefficients", args.coeff_window * args.coeff_window))
        print(
            f"{image_path}: coeff_window={args.coeff_window}, coeffs/block={coeff_per_block}, "
            f"active={active_coeff}, Y{luma_shape}, Cb{cb_shape}, Cr{cr_shape}"
        )
        if args.print_only:
            continue
        dest_dir = output_dir if output_dir is not None else image_path.parent
        archive_path = _save_result(result, dest_dir)
        compressed_paths.append(archive_path)
        print(f"Saved blocks for {image_path} -> {archive_path}")

    if not args.print_only:
        print(f"Processed {len(compressed_paths)} reference image(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
