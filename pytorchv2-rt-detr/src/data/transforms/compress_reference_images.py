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
from ...misc.dct_coefficients import (
    build_active_indices,
    build_active_mask,
    count_to_window,
    resolve_coefficient_counts,
    validate_coeff_count,
)

RangeMode = tuple[str, ...]

TensorLike = Union[torch.Tensor, np.ndarray]


def _parse_coeff_count(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:  # pragma: no cover - argparse already reports format errors
        raise argparse.ArgumentTypeError("coefficient count must be an integer") from exc
    if not 0 <= parsed <= 64:
        raise argparse.ArgumentTypeError("coefficient count must be within [0, 64]")
    return parsed


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


def _build_active_index(coeff_count: int) -> np.ndarray | None:
    count = validate_coeff_count(coeff_count, name="coeff_count")
    if count >= 64:
        return None
    indices = build_active_indices(count)
    return np.array(indices, dtype=np.int64)


def _apply_frequency_selection(blocks: np.ndarray, coeff_count: int) -> np.ndarray:
    """Zero out high-frequency coefficients outside the selected subset per block."""
    count = validate_coeff_count(coeff_count, name="coeff_count")
    if count >= 64:
        return blocks
    mask_vec = np.array(build_active_mask(count), dtype=np.float32)
    reshaped = blocks.reshape((-1, 64), order="F")
    reshaped *= mask_vec
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
    coeff_count_luma: int,
    coeff_count_cb: int,
    coeff_count_cr: int,
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

    luma_blocks = _apply_frequency_selection(luma_blocks, coeff_count_luma)
    cb_blocks = _apply_frequency_selection(cb_blocks, coeff_count_cb)
    cr_blocks = _apply_frequency_selection(cr_blocks, coeff_count_cr)

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

    window_luma = count_to_window(coeff_count_luma)
    window_cb = count_to_window(coeff_count_cb)
    window_cr = count_to_window(coeff_count_cr)
    window_chroma = window_cb if window_cb == window_cr else None
    coeff_count_chroma = coeff_count_cb if coeff_count_cb == coeff_count_cr else max(coeff_count_cb, coeff_count_cr)
    metadata = {
        "spatial_shape": np.array(y_plane.shape, dtype=np.int32),
        "chroma_shape": np.array(cb_plane.shape, dtype=np.int32),
        "coeff_window": np.array(window_luma if window_luma is not None else -1, dtype=np.int32),
        "coeff_window_luma": np.array(window_luma if window_luma is not None else -1, dtype=np.int32),
        "coeff_window_chroma": np.array(window_chroma if window_chroma is not None else -1, dtype=np.int32),
        "coeff_window_cb": np.array(window_cb if window_cb is not None else -1, dtype=np.int32),
        "coeff_window_cr": np.array(window_cr if window_cr is not None else -1, dtype=np.int32),
        "coeff_count": np.array(coeff_count_luma, dtype=np.int32),
        "coeff_count_luma": np.array(coeff_count_luma, dtype=np.int32),
        "coeff_count_chroma": np.array(coeff_count_chroma, dtype=np.int32),
        "coeff_count_cb": np.array(coeff_count_cb, dtype=np.int32),
        "coeff_count_cr": np.array(coeff_count_cr, dtype=np.int32),
        "coefficients_per_block": np.array(64, dtype=np.int32),
        "active_coefficients": np.array(coeff_count_luma, dtype=np.int32),
        "active_coefficients_luma": np.array(coeff_count_luma, dtype=np.int32),
        "active_coefficients_chroma": np.array(coeff_count_chroma, dtype=np.int32),
        "active_coefficients_cb": np.array(coeff_count_cb, dtype=np.int32),
        "active_coefficients_cr": np.array(coeff_count_cr, dtype=np.int32),
        "luma_grid_shape": luma_grid_shape,
        "chroma_grid_shape": chroma_grid_shape,
    }

    return luma_blocks, cb_blocks, cr_blocks, metadata


def compress_reference_image(
    path: Path,
    range_mode: str,
    coeff_count_luma: int,
    coeff_count_cb: int | None = None,
    coeff_count_cr: int | None = None,
) -> dict[str, np.ndarray]:
    """Compute Y, Cb, Cr DCT blocks for a single reference image."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")

    coeff_count_luma = validate_coeff_count(coeff_count_luma, name="coeff_count_luma")
    chroma_count_cb = coeff_count_luma if coeff_count_cb is None else validate_coeff_count(
        coeff_count_cb, name="coeff_count_cb"
    )
    chroma_count_cr = coeff_count_luma if coeff_count_cr is None else validate_coeff_count(
        coeff_count_cr, name="coeff_count_cr"
    )

    luma_blocks, cb_blocks, cr_blocks, metadata = _compress_image_array(
        image,
        range_mode,
        coeff_count_luma,
        chroma_count_cb,
        chroma_count_cr,
        round_blocks=True,
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

    Returns a tuple ``(y_blocks, (cb_blocks, cr_blocks))`` where ``y_blocks`` has
    shape ``(64, nb_y, nb_x)`` while the chroma planes retain their independent
    coefficient depths without padding. When ``keep_original`` is ``True`` the original
    input is appended as a second element.
    """

    def __init__(
        self,
        coeff_window: int = 8,
        range_mode: str = "studio",
        dtype: Union[str, torch.dtype] = torch.float32,
        keep_original: bool = False,
        *,
        coeff_window_luma: int | None = None,
        coeff_window_chroma: int | None = None,
        coeff_window_cb: int | None = None,
        coeff_window_cr: int | None = None,
        coeff_count: int | None = None,
        coeff_count_luma: int | None = None,
        coeff_count_chroma: int | None = None,
        coeff_count_cb: int | None = None,
        coeff_count_cr: int | None = None,
        selection_order: str = "zigzag",
    ) -> None:
        super().__init__()
        if range_mode not in {"studio", "full"}:
            raise ValueError(f"Unsupported range_mode: {range_mode}")
        if selection_order != "zigzag":
            raise ValueError(f"Unsupported coefficient selection order: {selection_order}")

        base_count, luma_count, cb_count, cr_count = resolve_coefficient_counts(
            coeff_window=coeff_window,
            coeff_count=coeff_count,
            coeff_window_luma=coeff_window_luma,
            coeff_count_luma=coeff_count_luma,
            coeff_window_chroma=coeff_window_chroma,
            coeff_count_chroma=coeff_count_chroma,
            coeff_window_cb=coeff_window_cb,
            coeff_window_cr=coeff_window_cr,
            coeff_count_cb=coeff_count_cb,
            coeff_count_cr=coeff_count_cr,
        )

        self.coeff_count = base_count
        self.coeff_count_luma = luma_count
        self.coeff_count_cb = cb_count
        self.coeff_count_cr = cr_count
        self.coeff_count_chroma = cb_count if cb_count == cr_count else max(cb_count, cr_count)
        self.coeff_window = count_to_window(luma_count)
        self.coeff_window_luma = count_to_window(luma_count)
        self.coeff_window_cb = count_to_window(cb_count)
        self.coeff_window_cr = count_to_window(cr_count)
        self.coeff_window_chroma = (
            self.coeff_window_cb if self.coeff_window_cb == self.coeff_window_cr else None
        )
        self.range_mode = range_mode
        self.dtype = _resolve_torch_dtype(dtype)
        self.keep_original = keep_original
        self.selection_order = selection_order

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
            bgr_image,
            self.range_mode,
            self.coeff_count_luma,
            self.coeff_count_cb,
            self.coeff_count_cr,
            round_blocks=False,
        )

        y_tensor = self._blocks_to_tensor(luma_blocks, self.dtype)
        cb_tensor = self._blocks_to_tensor(cb_blocks, self.dtype)
        cr_tensor = self._blocks_to_tensor(cr_blocks, self.dtype)
        cbcr_payload = (cb_tensor, cr_tensor)

        payload = (y_tensor, cbcr_payload)

        # Attach useful metadata to the target when it is a mutable mapping.
        if target is not None and hasattr(target, "update") and callable(target.update):
            coeff_window = int(metadata.get("coeff_window", -1))
            coeff_window_luma = int(metadata.get("coeff_window_luma", -1))
            coeff_window_chroma = int(metadata.get("coeff_window_chroma", -1))
            coeff_window_cb = int(metadata.get("coeff_window_cb", -1))
            coeff_window_cr = int(metadata.get("coeff_window_cr", -1))
            coeff_count = int(metadata.get("coeff_count", self.coeff_count_luma))
            coeff_count_luma = int(metadata.get("coeff_count_luma", self.coeff_count_luma))
            coeff_count_chroma = int(metadata.get("coeff_count_chroma", self.coeff_count_chroma))
            coeff_count_cb = int(metadata.get("coeff_count_cb", self.coeff_count_cb))
            coeff_count_cr = int(metadata.get("coeff_count_cr", self.coeff_count_cr))
            target.update({
                "dct_spatial_shape": torch.tensor(metadata["spatial_shape"], dtype=torch.int32),
                "dct_chroma_shape": torch.tensor(metadata["chroma_shape"], dtype=torch.int32),
                "dct_luma_grid": torch.tensor(metadata["luma_grid_shape"], dtype=torch.int32),
                "dct_chroma_grid": torch.tensor(metadata["chroma_grid_shape"], dtype=torch.int32),
                "dct_coeff_window": coeff_window,
                "dct_coeff_window_luma": coeff_window_luma,
                "dct_coeff_window_chroma": coeff_window_chroma,
                "dct_coeff_window_cb": coeff_window_cb,
                "dct_coeff_window_cr": coeff_window_cr,
                "dct_coeff_count": coeff_count,
                "dct_coeff_count_luma": coeff_count_luma,
                "dct_coeff_count_chroma": coeff_count_chroma,
                "dct_coeff_count_cb": coeff_count_cb,
                "dct_coeff_count_cr": coeff_count_cr,
                "dct_active_coeff_luma": int(metadata["active_coefficients_luma"]),
                "dct_active_coeff_chroma": int(metadata["active_coefficients_chroma"]),
                "dct_active_coeff_cb": int(metadata["active_coefficients_cb"]),
                "dct_active_coeff_cr": int(metadata["active_coefficients_cr"]),
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
    """Reduce DCT payload depth to the requested active coefficients per plane."""

    def __init__(
        self,
        coeff_count_luma: int,
        coeff_count_chroma: int | None = None,
        *,
        coeff_count_cb: int | None = None,
        coeff_count_cr: int | None = None,
    ) -> None:
        super().__init__()
        self.coeff_count_luma = validate_coeff_count(coeff_count_luma, name="coeff_count_luma")
        chroma_fallback = coeff_count_luma if coeff_count_chroma is None else coeff_count_chroma
        cb_count = chroma_fallback if coeff_count_cb is None else coeff_count_cb
        cr_count = chroma_fallback if coeff_count_cr is None else coeff_count_cr
        self.coeff_count_cb = validate_coeff_count(cb_count, name="coeff_count_cb")
        self.coeff_count_cr = validate_coeff_count(cr_count, name="coeff_count_cr")
        self.coeff_count_chroma = (
            self.coeff_count_cb
            if self.coeff_count_cb == self.coeff_count_cr
            else max(self.coeff_count_cb, self.coeff_count_cr)
        )
        self.luma_channels = self.coeff_count_luma
        self.chroma_channels_cb = self.coeff_count_cb
        self.chroma_channels_cr = self.coeff_count_cr
        self.chroma_channels = max(self.coeff_count_cb, self.coeff_count_cr)
        self.active_idx_luma = _build_active_index(self.coeff_count_luma)
        self.active_idx_cb = _build_active_index(self.coeff_count_cb)
        self.active_idx_cr = _build_active_index(self.coeff_count_cr)

    def _trim_pair(
        self,
        y_blocks: torch.Tensor,
        cbcr_blocks: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        y_trim = y_blocks
        if self.active_idx_luma is not None and y_blocks.size(0) > self.luma_channels:
            idx = torch.as_tensor(self.active_idx_luma, dtype=torch.long, device=y_blocks.device)
            y_trim = torch.index_select(y_blocks, 0, idx)

        cb, cr = cbcr_blocks
        if self.active_idx_cb is not None and cb.size(0) > self.chroma_channels_cb:
            idx_cb = torch.as_tensor(self.active_idx_cb, dtype=torch.long, device=cb.device)
            cb = torch.index_select(cb, 0, idx_cb)
        if self.active_idx_cr is not None and cr.size(0) > self.chroma_channels_cr:
            idx_cr = torch.as_tensor(self.active_idx_cr, dtype=torch.long, device=cr.device)
            cr = torch.index_select(cr, 0, idx_cr)

        if y_trim.size(0) != self.luma_channels:
            raise ValueError(
                f"Expected {self.luma_channels} luma coefficients, received {y_trim.size(0)}."
            )
        if cb.size(0) != self.chroma_channels_cb:
            raise ValueError(
                f"Expected {self.chroma_channels_cb} Cb coefficients, received {cb.size(0)}."
            )
        if cr.size(0) != self.chroma_channels_cr:
            raise ValueError(
                f"Expected {self.chroma_channels_cr} Cr coefficients, received {cr.size(0)}."
            )

        if self.chroma_channels_cb != self.chroma_channels_cr:
            cb = cb[: self.chroma_channels_cb]
            cr = cr[: self.chroma_channels_cr]
        else:
            cb = cb[: self.chroma_channels]
            cr = cr[: self.chroma_channels]

        return y_trim, (cb, cr)

    def _trim_payload(self, payload):
        if isinstance(payload, tuple) and len(payload) == 2:
            first, second = payload
            if torch.is_tensor(first) and isinstance(second, tuple):
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
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
        default=None,
        help="Deprecated square window of low-frequency coefficients to retain (defaults to all coefficients).",
    )
    parser.add_argument(
        "--coeff-window-luma",
        dest="coeff_window_luma",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
        default=None,
        help="Deprecated override for the luma window (defaults to --coeff-window).",
    )
    parser.add_argument(
        "--coeff-window-chroma",
        dest="coeff_window_chroma",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
        default=None,
        help="Deprecated override for the chroma window (defaults to --coeff-window).",
    )
    parser.add_argument(
        "--coeff-window-cb",
        dest="coeff_window_cb",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
        default=None,
        help="Override the Cb coefficient window (defaults to chroma window).",
    )
    parser.add_argument(
        "--coeff-window-cr",
        dest="coeff_window_cr",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
        default=None,
        help="Override the Cr coefficient window (defaults to chroma window).",
    )
    parser.add_argument(
        "--coeff-count",
        dest="coeff_count",
        type=_parse_coeff_count,
        default=None,
        help="Total number of coefficients per luma block to preserve (default: 64).",
    )
    parser.add_argument(
        "--coeff-count-luma",
        dest="coeff_count_luma",
        type=_parse_coeff_count,
        default=None,
        help="Override the luma coefficient count (defaults to --coeff-count).",
    )
    parser.add_argument(
        "--coeff-count-chroma",
        dest="coeff_count_chroma",
        type=_parse_coeff_count,
        default=None,
        help="Override the chroma coefficient count (defaults to luma count).",
    )
    parser.add_argument(
        "--coeff-count-cb",
        dest="coeff_count_cb",
        type=_parse_coeff_count,
        default=None,
        help="Override the Cb coefficient count (defaults to chroma count).",
    )
    parser.add_argument(
        "--coeff-count-cr",
        dest="coeff_count_cr",
        type=_parse_coeff_count,
        default=None,
        help="Override the Cr coefficient count (defaults to chroma count).",
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
        try:
            _, coeff_count_luma, coeff_count_cb, coeff_count_cr = resolve_coefficient_counts(
                coeff_window=args.coeff_window,
                coeff_count=args.coeff_count,
                coeff_window_luma=args.coeff_window_luma,
                coeff_count_luma=args.coeff_count_luma,
                coeff_window_chroma=args.coeff_window_chroma,
                coeff_count_chroma=args.coeff_count_chroma,
                coeff_window_cb=args.coeff_window_cb,
                coeff_window_cr=args.coeff_window_cr,
                coeff_count_cb=args.coeff_count_cb,
                coeff_count_cr=args.coeff_count_cr,
            )
        except ValueError as exc:
            raise SystemExit(f"Invalid coefficient selection: {exc}") from exc

        result = compress_reference_image(
            image_path,
            args.range_mode,
            coeff_count_luma,
            coeff_count_cb,
            coeff_count_cr,
        )
        luma_shape = result["luma_blocks"].shape
        cb_shape = result["cb_blocks"].shape
        cr_shape = result["cr_blocks"].shape
        coeff_per_block = int(result["coefficients_per_block"].item())
        active_luma = int(result.get("active_coefficients_luma", coeff_count_luma))
        active_cb = int(result.get("active_coefficients_cb", coeff_count_cb))
        active_cr = int(result.get("active_coefficients_cr", coeff_count_cr))
        window_luma = count_to_window(coeff_count_luma)
        window_cb = count_to_window(coeff_count_cb)
        window_cr = count_to_window(coeff_count_cr)
        window_luma_str = f"{window_luma}x{window_luma}" if window_luma is not None else "n/a"
        window_cb_str = f"{window_cb}x{window_cb}" if window_cb is not None else "n/a"
        window_cr_str = f"{window_cr}x{window_cr}" if window_cr is not None else "n/a"
        print(
            f"{image_path}: coeff_count_luma={coeff_count_luma} (window {window_luma_str}), "
            f"coeff_count_cb={coeff_count_cb} (window {window_cb_str}), "
            f"coeff_count_cr={coeff_count_cr} (window {window_cr_str}), "
            f"coeffs/block={coeff_per_block}, active(Y)={active_luma}, active(Cb)={active_cb}, active(Cr)={active_cr}, "
            f"Y{luma_shape}, Cb{cb_shape}, Cr{cr_shape}"
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
