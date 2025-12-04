from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch
from torchvision.transforms import v2 as T


def _as_tensor(data: Iterable[float] | torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data.to(dtype=dtype)
    return torch.tensor(list(data), dtype=dtype)


def _build_frequency_mask(coeff_window: int) -> torch.Tensor:
    if coeff_window not in (1, 2, 4, 8):
        raise ValueError("coeff_window must be one of {1, 2, 4, 8}.")
    mask = torch.zeros(64, dtype=torch.float32)
    for row in range(8):
        for col in range(8):
            idx = col * 8 + row
            if row < coeff_window and col < coeff_window:
                mask[idx] = 1.0
    return mask


class NormalizeDCTCoefficients(T.Transform):
    """Normalise DCT coefficient tensors using pre-computed statistics.

    The transform expects payloads emitted by :class:`CompressToDCT`, namely a
    ``(y_blocks, cbcr_blocks)`` tuple where ``y_blocks`` has shape ``[64, By, Bx]``
    and ``cbcr_blocks`` has shape ``[2, 64, By/2, Bx/2]`` for 4:2:0 subsampling.
    When the compressor is configured with ``keep_original=True`` the payload will
    be ``((y_blocks, cbcr_blocks), original_tensor)`` and the original tensor is
    forwarded untouched.
    """

    def __init__(
        self,
        mean_luma: Iterable[float] | torch.Tensor,
        std_luma: Iterable[float] | torch.Tensor,
        mean_chroma: Iterable[Iterable[float]] | torch.Tensor,
        std_chroma: Iterable[Iterable[float]] | torch.Tensor,
        *,
        eps: float = 1e-6,
        coeff_window: int | None = None,
    ) -> None:
        super().__init__()
        mean_luma_tensor = _as_tensor(mean_luma, dtype=torch.float32)
        std_luma_tensor = _as_tensor(std_luma, dtype=torch.float32)
        if mean_luma_tensor.ndim != 1 or mean_luma_tensor.numel() != 64:
            raise ValueError("mean_luma must contain 64 coefficients.")
        if std_luma_tensor.ndim != 1 or std_luma_tensor.numel() != 64:
            raise ValueError("std_luma must contain 64 coefficients.")

        mean_chroma_tensor = torch.as_tensor(mean_chroma, dtype=torch.float32)
        std_chroma_tensor = torch.as_tensor(std_chroma, dtype=torch.float32)
        if mean_chroma_tensor.shape != (2, 64):
            raise ValueError("mean_chroma must have shape [2, 64].")
        if std_chroma_tensor.shape != (2, 64):
            raise ValueError("std_chroma must have shape [2, 64].")

        self.register_buffer("mean_luma", mean_luma_tensor.view(64, 1, 1), persistent=False)
        self.register_buffer("std_luma", std_luma_tensor.view(64, 1, 1), persistent=False)
        self.register_buffer("mean_chroma", mean_chroma_tensor.view(2, 64, 1, 1), persistent=False)
        self.register_buffer("std_chroma", std_chroma_tensor.view(2, 64, 1, 1), persistent=False)
        self.eps = float(eps)

        window = 8 if coeff_window is None else int(coeff_window)
        mask_vec = _build_frequency_mask(window)
        self.register_buffer("mask_luma", mask_vec.view(64, 1, 1), persistent=False)
        self.register_buffer("mask_chroma", mask_vec.view(1, 64, 1, 1), persistent=False)

    def _normalise_pair(
        self,
        y_blocks: torch.Tensor,
        cbcr_blocks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean_luma = self.mean_luma.to(device=y_blocks.device, dtype=y_blocks.dtype)
        std_luma = self.std_luma.to(device=y_blocks.device, dtype=y_blocks.dtype)
        norm_y = (y_blocks - mean_luma) / (std_luma + self.eps)
        mask_luma = self.mask_luma.to(device=y_blocks.device, dtype=y_blocks.dtype)
        y_norm = mask_luma * norm_y + (1.0 - mask_luma) * y_blocks

        mean_chroma = self.mean_chroma.to(device=cbcr_blocks.device, dtype=cbcr_blocks.dtype)
        std_chroma = self.std_chroma.to(device=cbcr_blocks.device, dtype=cbcr_blocks.dtype)
        norm_cbcr = (cbcr_blocks - mean_chroma) / (std_chroma + self.eps)
        mask_chroma = self.mask_chroma.to(device=cbcr_blocks.device, dtype=cbcr_blocks.dtype)
        cbcr_norm = mask_chroma * norm_cbcr + (1.0 - mask_chroma) * cbcr_blocks
        return y_norm, cbcr_norm

    def forward(self, inputs: Any, target: Any = None):  # noqa: D401 - signature mirrors torchvision
        if isinstance(inputs, tuple) and len(inputs) == 2:
            first, second = inputs
            # Case 1: (y_blocks, cbcr_blocks)
            if torch.is_tensor(first) and torch.is_tensor(second):
                return self._normalise_pair(first, second)
            # Case 2: ((y_blocks, cbcr_blocks), original)
            if (
                isinstance(first, tuple)
                and len(first) == 2
                and torch.is_tensor(first[0])
                and torch.is_tensor(first[1])
            ):
                y_norm, cbcr_norm = self._normalise_pair(first[0], first[1])
                return (y_norm, cbcr_norm), second
        raise TypeError(
            "NormalizeDCTCoefficients expects payloads produced by CompressToDCT"
        )

    @classmethod
    def from_file(
        cls,
        path: Path | str,
        *,
        eps: float = 1e-6,
        coeff_window: int | None = None,
    ) -> "NormalizeDCTCoefficients":
        data: Dict[str, Any] = torch.load(Path(path), map_location="cpu")
        required_keys = {"mean_luma", "std_luma", "mean_chroma", "std_chroma"}
        missing = required_keys.difference(data)
        if missing:
            raise KeyError(f"Missing keys in DCT stats file {path}: {sorted(missing)}")
        return cls(
            data["mean_luma"],
            data["std_luma"],
            data["mean_chroma"],
            data["std_chroma"],
            eps=eps,
            coeff_window=coeff_window,
        )

__all__ = ["NormalizeDCTCoefficients"]
