from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch
from torchvision.transforms import v2 as T
from ...core import register
from ...misc.dct_coefficients import build_active_mask, resolve_coefficient_counts


def _as_tensor(data: Iterable[float] | torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data.to(dtype=dtype)
    return torch.tensor(list(data), dtype=dtype)


def _build_frequency_mask_from_count(coeff_count: int) -> torch.Tensor:
    mask_values = build_active_mask(coeff_count)
    return torch.tensor(mask_values, dtype=torch.float32)


@register()
class NormalizeDCTCoefficients(T.Transform):
    """Normalise DCT coefficient tensors using pre-computed statistics.

    The transform expects payloads emitted by :class:`CompressToDCT`, namely a
    ``(y_blocks, (cb_blocks, cr_blocks))`` tuple where ``y_blocks`` has shape
    ``[64, By, Bx]`` and the chroma planes keep their own coefficient depths
    (no padding is introduced) for 4:2:0 subsampling.
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
        coeff_window_luma: int | None = None,
        coeff_window_chroma: int | None = None,
        coeff_window_cb: int | None = None,
        coeff_window_cr: int | None = None,
        coeff_count: int | None = None,
        coeff_count_luma: int | None = None,
        coeff_count_chroma: int | None = None,
        coeff_count_cb: int | None = None,
        coeff_count_cr: int | None = None,
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

        _, count_luma, count_cb, count_cr = resolve_coefficient_counts(
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
        mask_luma_vec = _build_frequency_mask_from_count(count_luma)
        mask_cb_vec = _build_frequency_mask_from_count(count_cb)
        mask_cr_vec = _build_frequency_mask_from_count(count_cr)
        self.register_buffer("mask_luma", mask_luma_vec.view(64, 1, 1), persistent=False)
        mask_chroma = torch.stack((mask_cb_vec, mask_cr_vec), dim=0)
        self.register_buffer("mask_chroma", mask_chroma.view(2, 64, 1, 1), persistent=False)
        self.coeff_count_luma = count_luma
        self.coeff_count_cb = count_cb
        self.coeff_count_cr = count_cr
        self.coeff_count_chroma = count_cb if count_cb == count_cr else max(count_cb, count_cr)

    def _normalise_pair(
        self,
        y_blocks: torch.Tensor,
        cbcr_blocks: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        channels_luma = int(y_blocks.shape[-3])
        mean_luma_full = self.mean_luma.to(device=y_blocks.device, dtype=y_blocks.dtype)
        std_luma_full = self.std_luma.to(device=y_blocks.device, dtype=y_blocks.dtype)
        mean_luma = mean_luma_full[:channels_luma]
        std_luma = std_luma_full[:channels_luma]
        if y_blocks.dim() > 3:
            mean_luma = mean_luma.unsqueeze(0)
            std_luma = std_luma.unsqueeze(0)
        norm_y = (y_blocks - mean_luma) / (std_luma + self.eps)
        mask_luma_full = self.mask_luma.to(device=y_blocks.device, dtype=y_blocks.dtype)
        mask_luma = mask_luma_full[:channels_luma]
        if y_blocks.dim() > 3:
            mask_luma = mask_luma.unsqueeze(0)
        y_norm = mask_luma * norm_y + (1.0 - mask_luma) * y_blocks

        cb_blocks, cr_blocks = cbcr_blocks
        mean_chroma_full = self.mean_chroma.to(device=cb_blocks.device, dtype=cb_blocks.dtype)
        std_chroma_full = self.std_chroma.to(device=cb_blocks.device, dtype=cb_blocks.dtype)
        mask_chroma_full = self.mask_chroma.to(device=cb_blocks.device, dtype=cb_blocks.dtype)

        def _normalise_plane(plane: torch.Tensor, plane_idx: int) -> torch.Tensor:
            channels = int(plane.shape[-3])
            mean_plane = mean_chroma_full[plane_idx, :channels]
            std_plane = std_chroma_full[plane_idx, :channels]
            mask_plane = mask_chroma_full[plane_idx, :channels]
            if plane.dim() > 3:
                mean_plane = mean_plane.unsqueeze(0)
                std_plane = std_plane.unsqueeze(0)
                mask_plane = mask_plane.unsqueeze(0)
            norm_plane = (plane - mean_plane) / (std_plane + self.eps)
            return mask_plane * norm_plane + (1.0 - mask_plane) * plane

        cb_norm = _normalise_plane(cb_blocks, 0)
        cr_norm = _normalise_plane(cr_blocks, 1).to(device=cr_blocks.device, dtype=cr_blocks.dtype)
        return y_norm, (cb_norm, cr_norm)

    def _normalise_payload(self, payload: Any) -> Any:
        if isinstance(payload, tuple) and len(payload) == 2:
            first, second = payload
            if torch.is_tensor(first) and isinstance(second, tuple):
                return self._normalise_pair(first, second)
            if (
                isinstance(first, tuple)
                and len(first) == 2
                and torch.is_tensor(first[0])
                and isinstance(first[1], tuple)
            ):
                y_norm, cbcr_norm = self._normalise_pair(first[0], first[1])
                return (y_norm, cbcr_norm), second
        raise TypeError(
            "NormalizeDCTCoefficients expects payloads produced by CompressToDCT"
        )

    def forward(self, inputs: Any, target: Any = None):  # noqa: D401 - signature mirrors torchvision
        payload = self._normalise_payload(inputs)
        if target is None:
            return payload
        return payload, target

    @classmethod
    def from_file(
        cls,
        path: Path | str,
        *,
        eps: float = 1e-6,
        coeff_window: int | None = None,
        coeff_window_luma: int | None = None,
        coeff_window_chroma: int | None = None,
        coeff_window_cb: int | None = None,
        coeff_window_cr: int | None = None,
        coeff_count: int | None = None,
        coeff_count_luma: int | None = None,
        coeff_count_chroma: int | None = None,
        coeff_count_cb: int | None = None,
        coeff_count_cr: int | None = None,
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
            coeff_window_luma=coeff_window_luma,
            coeff_window_chroma=coeff_window_chroma,
            coeff_window_cb=coeff_window_cb,
            coeff_window_cr=coeff_window_cr,
            coeff_count=coeff_count,
            coeff_count_luma=coeff_count_luma,
            coeff_count_chroma=coeff_count_chroma,
            coeff_count_cb=coeff_count_cb,
            coeff_count_cr=coeff_count_cr,
        )

@register()
def NormalizeDCTCoefficientsFromFile(
    path: Path | str,
    *,
    eps: float = 1e-6,
    coeff_window: int | None = None,
    coeff_window_luma: int | None = None,
    coeff_window_chroma: int | None = None,
    coeff_window_cb: int | None = None,
    coeff_window_cr: int | None = None,
    coeff_count: int | None = None,
    coeff_count_luma: int | None = None,
    coeff_count_chroma: int | None = None,
    coeff_count_cb: int | None = None,
    coeff_count_cr: int | None = None,
) -> NormalizeDCTCoefficients:
    return NormalizeDCTCoefficients.from_file(
        path,
        eps=eps,
        coeff_window=coeff_window,
        coeff_window_luma=coeff_window_luma,
        coeff_window_chroma=coeff_window_chroma,
        coeff_window_cb=coeff_window_cb,
        coeff_window_cr=coeff_window_cr,
        coeff_count=coeff_count,
        coeff_count_luma=coeff_count_luma,
        coeff_count_chroma=coeff_count_chroma,
        coeff_count_cb=coeff_count_cb,
        coeff_count_cr=coeff_count_cr,
    )


__all__ = ["NormalizeDCTCoefficients", "NormalizeDCTCoefficientsFromFile"]
