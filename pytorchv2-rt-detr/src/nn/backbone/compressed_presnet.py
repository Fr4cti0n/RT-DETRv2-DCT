"""Compressed-input variants of the ResNet backbone."""

from __future__ import annotations

import math
from typing import Literal, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .presnet import (
    PResNet,
    Blocks,
    BasicBlock,
    BottleNeck,
)
from ...core import register
from ...misc.dct_coefficients import (
    build_active_indices,
    count_to_window,
    resolve_coefficient_counts,
    validate_coeff_count,
)


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class _DCTBlockDecoder(nn.Module):
    def __init__(self, block_size: int = 8) -> None:
        super().__init__()
        if block_size != 8:
            raise ValueError("Only 8x8 DCT blocks are supported.")
        basis = self._build_dct_matrix(block_size)
        self.register_buffer("dct_basis", basis, persistent=False)
        self.block_size = block_size

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

    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        if coeffs.dim() != 4:
            raise ValueError("Expected coefficient tensor with shape [B, 64, By, Bx].")
        bsz, channels, by, bx = coeffs.shape
        if channels != self.block_size * self.block_size:
            raise ValueError("Coefficient depth does not match an 8x8 block layout.")

        patches = coeffs.permute(0, 2, 3, 1).contiguous()
        patches = patches.view(bsz, by, bx, self.block_size, self.block_size)
        patches = patches.permute(0, 1, 2, 4, 3).contiguous()
        patches = patches.view(-1, self.block_size, self.block_size)

        basis = self.dct_basis.to(dtype=patches.dtype, device=patches.device)
        tmp = torch.matmul(basis.t(), patches)
        spatial = torch.matmul(tmp, basis)

        spatial = spatial.view(bsz, by, bx, self.block_size, self.block_size)
        spatial = spatial.permute(0, 1, 3, 2, 4).contiguous()
        spatial = spatial.view(bsz, by * self.block_size, bx * self.block_size)
        return spatial


def _unpack_payload(payload) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if isinstance(payload, (list, tuple)):
        if len(payload) == 2 and isinstance(payload[0], (list, tuple)) and not torch.is_tensor(payload[0]):
            return _unpack_payload(payload[0])
        if (
            len(payload) == 2
            and torch.is_tensor(payload[0])
            and isinstance(payload[1], (list, tuple))
            and len(payload[1]) == 2
            and all(torch.is_tensor(item) for item in payload[1])
        ):
            return payload[0], (payload[1][0], payload[1][1])
    raise TypeError("Expected (y_blocks, (cb_blocks, cr_blocks)) tuple from CompressToDCT.")


def _upsample_chroma(chroma: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(chroma, size=target_hw, mode="bilinear", align_corners=False)


def _build_active_index_from_count(coeff_count: int) -> torch.Tensor | None:
    count = validate_coeff_count(coeff_count, name="coeff_count")
    if count <= 0 or count >= 64:
        return None
    indices = build_active_indices(count)
    return torch.tensor(indices, dtype=torch.long)


class _BackboneAdapter(nn.Module):
    def __init__(self, backbone: PResNet) -> None:
        super().__init__()
        if not isinstance(backbone, PResNet):
            raise TypeError("Compressed backbone helpers require a PResNet instance.")
        self.backbone = backbone
        self.return_idx = list(backbone.return_idx)
        self.out_channels = backbone.out_channels
        self.out_strides = backbone.out_strides
        conv1 = list(backbone.conv1.children())[-1]
        self.target_channels = conv1.norm.num_features if hasattr(conv1, "norm") else 64

    def _forward_residual_stages(self, conv1_out: torch.Tensor, *, skip_pool: bool = False):
        if skip_pool:
            x = conv1_out
        else:
            x = F.max_pool2d(conv1_out, kernel_size=3, stride=2, padding=1)
        outs = []
        for idx, stage in enumerate(self.backbone.res_layers):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs

    @staticmethod
    def _select_chroma_plane(
        plane: torch.Tensor,
        target_count: int,
        active_idx: torch.Tensor | None,
        *,
        name: str,
    ) -> torch.Tensor:
        if target_count <= 0:
            return plane.new_empty((plane.size(0), 0, plane.size(2), plane.size(3)))
        if plane.size(1) > target_count:
            if active_idx is not None and active_idx.numel() > 0:
                max_idx = int(active_idx.max().item())
                if max_idx < plane.size(1):
                    idx = active_idx.to(device=plane.device)
                    plane = torch.index_select(plane, 1, idx)
                else:
                    plane = plane[:, :target_count]
            else:
                plane = plane[:, :target_count]
        if plane.size(1) > target_count:
            plane = plane[:, :target_count]
        if plane.size(1) != target_count:
            raise ValueError(f"Expected {target_count} {name} coefficients, received {plane.size(1)}.")
        return plane


class CompressedResNetReconstruction(_BackboneAdapter):
    def __init__(
        self,
        backbone: PResNet,
        *,
        range_mode: str,
        mean: Sequence[float],
        std: Sequence[float],
        refine_width: int = 32,
    ) -> None:
        super().__init__(backbone)
        if range_mode not in {"studio", "full"}:
            raise ValueError(f"Unsupported range_mode: {range_mode}")
        self.range_mode = range_mode
        self.decoder = _DCTBlockDecoder()
        self.refine = nn.Sequential(
            nn.Conv2d(3, refine_width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(refine_width),
            nn.SiLU(inplace=True),
            nn.Conv2d(refine_width, 3, kernel_size=3, padding=1, bias=False),
        )
        mean_tensor = torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
        std_tensor = torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1)
        self.register_buffer("mean", mean_tensor)
        self.register_buffer("std", std_tensor)

    def _decode_planes(
        self,
        y_blocks: torch.Tensor,
        cbcr_blocks: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y_plane = self.decoder(y_blocks)
        cb_blocks, cr_blocks = cbcr_blocks
        if cb_blocks.dim() != 4 or cr_blocks.dim() != 4:
            raise ValueError("Chrominance tensors must have shape [B, C, By, Bx].")
        cb_plane = self.decoder(cb_blocks)
        cr_plane = self.decoder(cr_blocks)
        target_hw = y_plane.shape[-2:]
        cb_plane = _upsample_chroma(cb_plane.unsqueeze(1), target_hw).squeeze(1)
        cr_plane = _upsample_chroma(cr_plane.unsqueeze(1), target_hw).squeeze(1)
        return y_plane, cb_plane, cr_plane

    def _ycbcr_to_rgb(self, y: torch.Tensor, cb: torch.Tensor, cr: torch.Tensor) -> torch.Tensor:
        if self.range_mode == "studio":
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
        return rgb / 255.0

    def forward(self, inputs):
        y_blocks, (cb_blocks, cr_blocks) = _unpack_payload(inputs)
        device = self.mean.device
        y_blocks = y_blocks.to(device=device, dtype=torch.float32)
        cb_blocks = cb_blocks.to(device=device, dtype=torch.float32)
        cr_blocks = cr_blocks.to(device=device, dtype=torch.float32)
        y_plane, cb_plane, cr_plane = self._decode_planes(y_blocks, (cb_blocks, cr_blocks))
        rgb = self._ycbcr_to_rgb(y_plane, cb_plane, cr_plane)
        refined = self.refine(rgb)
        normalized = (refined - self.mean.to(refined.dtype)) / self.std.to(refined.dtype)
        return self.backbone(normalized)


class CompressedResNetBlockStem(_BackboneAdapter):
    def __init__(
        self,
        backbone: PResNet,
        *,
        coeff_window_luma: int | None = None,
        coeff_window_chroma: int | None = None,
        coeff_count_luma: int | None = None,
        coeff_count_chroma: int | None = None,
        coeff_window_cb: int | None = None,
        coeff_window_cr: int | None = None,
        coeff_count_cb: int | None = None,
        coeff_count_cr: int | None = None,
    ) -> None:
        super().__init__(backbone)
        _, luma_count, cb_count, cr_count = resolve_coefficient_counts(
            coeff_window=coeff_window_luma,
            coeff_count=coeff_count_luma,
            coeff_window_luma=coeff_window_luma,
            coeff_count_luma=coeff_count_luma,
            coeff_window_chroma=coeff_window_chroma,
            coeff_count_chroma=coeff_count_chroma,
            coeff_window_cb=coeff_window_cb,
            coeff_window_cr=coeff_window_cr,
            coeff_count_cb=coeff_count_cb,
            coeff_count_cr=coeff_count_cr,
        )
        self.coeff_count_luma = luma_count
        self.coeff_count_cb = cb_count
        self.coeff_count_cr = cr_count
        self.coeff_count_chroma = cb_count if cb_count == cr_count else max(cb_count, cr_count)
        self.coeff_window_luma = count_to_window(luma_count)
        window_cb = count_to_window(cb_count)
        window_cr = count_to_window(cr_count)
        self.coeff_window_chroma = window_cb if window_cb == window_cr else None
        self.coeff_window = self.coeff_window_luma
        self.luma_channels = self.coeff_count_luma
        self.chroma_channels_cb = self.coeff_count_cb
        self.chroma_channels_cr = self.coeff_count_cr
        self.chroma_channels = self.chroma_channels_cb + self.chroma_channels_cr
        self.has_luma = self.luma_channels > 0
        self.has_cb = self.chroma_channels_cb > 0
        self.has_cr = self.chroma_channels_cr > 0
        self.has_chroma = self.has_cb or self.has_cr
        if not (self.has_luma or self.has_chroma):
            raise ValueError("At least one DCT plane must remain active for block-stem variant.")
        active_idx_luma = _build_active_index_from_count(self.coeff_count_luma)
        active_idx_cb = _build_active_index_from_count(self.coeff_count_cb)
        active_idx_cr = _build_active_index_from_count(self.coeff_count_cr)
        if active_idx_luma is not None and self.has_luma:
            self.register_buffer("active_idx_luma", active_idx_luma, persistent=False)
        else:
            self.active_idx_luma = None
        if active_idx_cb is not None and self.has_cb:
            self.register_buffer("active_idx_cb", active_idx_cb, persistent=False)
        else:
            self.active_idx_cb = None
        if active_idx_cr is not None and self.has_cr:
            self.register_buffer("active_idx_cr", active_idx_cr, persistent=False)
        else:
            self.active_idx_cr = None
        mid_channels = max(self.target_channels, 64)
        self.luma_width = mid_channels if self.has_luma else 0
        self.chroma_width = mid_channels // 2 if self.has_chroma else 0
        if self.has_luma:
            self.luma_proj = nn.Sequential(
                nn.Conv2d(self.luma_channels, self.luma_width, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(self.luma_width),
                nn.SiLU(inplace=True),
            )
        else:
            self.luma_proj = None
        if self.has_chroma:
            self.chroma_proj = nn.Sequential(
                nn.Conv2d(self.chroma_channels, self.chroma_width, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(self.chroma_width),
                nn.SiLU(inplace=True),
            )
        else:
            self.chroma_proj = None
        fusion_in = (self.luma_width if self.has_luma else 0) + (self.chroma_width if self.has_chroma else 0)
        if fusion_in == self.target_channels:
            self.fusion = nn.Identity()
        else:
            self.fusion = nn.Sequential(
                nn.Conv2d(fusion_in, self.target_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.target_channels),
                nn.SiLU(inplace=True),
            )

    def forward(self, inputs):
        y_blocks, (cb_blocks, cr_blocks) = _unpack_payload(inputs)
        device = next(self.parameters()).device
        y_blocks = y_blocks.to(device=device, dtype=torch.float32)
        cb_blocks = cb_blocks.to(device=device, dtype=torch.float32)
        cr_blocks = cr_blocks.to(device=device, dtype=torch.float32)

        features: list[torch.Tensor] = []

        if self.has_luma:
            if self.active_idx_luma is not None and y_blocks.size(1) > self.luma_channels:
                idx_luma = self.active_idx_luma.to(device)
                y_blocks = torch.index_select(y_blocks, 1, idx_luma)
            elif y_blocks.size(1) != self.luma_channels:
                raise ValueError(
                    f"Expected {self.luma_channels} luma coefficients, received {y_blocks.size(1)}."
                )
            y_feat = self.luma_proj(y_blocks) if self.luma_proj is not None else y_blocks
            features.append(y_feat)
        else:
            if y_blocks.size(1) != 0:
                raise ValueError(
                    "Luma coefficients were disabled but input still provides non-zero channels."
                )

        cb = cb_blocks
        cr = cr_blocks
        if self.chroma_channels_cb > 0:
            cb = self._select_chroma_plane(cb, self.chroma_channels_cb, getattr(self, "active_idx_cb", None), name="Cb")
        else:
            cb = cb.new_empty((cb.size(0), 0, cb.size(2), cb.size(3)))
        if self.chroma_channels_cr > 0:
            cr = self._select_chroma_plane(cr, self.chroma_channels_cr, getattr(self, "active_idx_cr", None), name="Cr")
        else:
            cr = cr.new_empty((cr.size(0), 0, cr.size(2), cr.size(3)))

        if self.has_chroma:
            chroma_parts = []
            if self.chroma_channels_cb > 0:
                chroma_parts.append(cb)
            if self.chroma_channels_cr > 0:
                chroma_parts.append(cr)
            chroma = torch.cat(chroma_parts, dim=1)
            if chroma.size(1) != self.chroma_channels:
                raise ValueError(
                    f"Expected concatenated chroma tensor with {self.chroma_channels} channels, "
                    f"received {chroma.size(1)}."
                )
            chroma = F.interpolate(chroma, size=y_blocks.shape[-2:], mode="nearest")
            chroma_feat = self.chroma_proj(chroma) if self.chroma_proj is not None else chroma
            features.append(chroma_feat)
        else:
            if cb.size(1) != 0 or cr.size(1) != 0:
                raise ValueError("Chroma coefficients were disabled but payload still includes channels.")

        if not features:
            raise RuntimeError("No active coefficient branches available for fusion.")
        if len(features) == 1:
            fusion_input = features[0]
        else:
            fusion_input = torch.cat(features, dim=1)
        fused = fusion_input if isinstance(self.fusion, nn.Identity) else self.fusion(fusion_input)
        conv1_like = F.interpolate(fused, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self._forward_residual_stages(conv1_like, skip_pool=True)


class CompressedResNetLumaFusion(_BackboneAdapter):
    def __init__(
        self,
        backbone: PResNet,
        *,
        coeff_window_luma: int | None = None,
        coeff_window_chroma: int | None = None,
        coeff_count_luma: int | None = None,
        coeff_count_chroma: int | None = None,
        coeff_window_cb: int | None = None,
        coeff_window_cr: int | None = None,
        coeff_count_cb: int | None = None,
        coeff_count_cr: int | None = None,
    ) -> None:
        super().__init__(backbone)
        _, luma_count, cb_count, cr_count = resolve_coefficient_counts(
            coeff_window=coeff_window_luma,
            coeff_count=coeff_count_luma,
            coeff_window_luma=coeff_window_luma,
            coeff_count_luma=coeff_count_luma,
            coeff_window_chroma=coeff_window_chroma,
            coeff_count_chroma=coeff_count_chroma,
            coeff_window_cb=coeff_window_cb,
            coeff_window_cr=coeff_window_cr,
            coeff_count_cb=coeff_count_cb,
            coeff_count_cr=coeff_count_cr,
        )
        self.coeff_count_luma = luma_count
        self.coeff_count_cb = cb_count
        self.coeff_count_cr = cr_count
        self.coeff_count_chroma = cb_count if cb_count == cr_count else max(cb_count, cr_count)
        self.coeff_window_luma = count_to_window(luma_count)
        window_cb = count_to_window(cb_count)
        window_cr = count_to_window(cr_count)
        self.coeff_window_chroma = window_cb if window_cb == window_cr else None
        self.coeff_window = self.coeff_window_luma
        self.luma_channels = self.coeff_count_luma
        self.chroma_channels_cb = self.coeff_count_cb
        self.chroma_channels_cr = self.coeff_count_cr
        self.chroma_channels = self.chroma_channels_cb + self.chroma_channels_cr
        self.has_luma = self.luma_channels > 0
        self.has_cb = self.chroma_channels_cb > 0
        self.has_cr = self.chroma_channels_cr > 0
        self.has_chroma = self.has_cb or self.has_cr
        if not (self.has_luma or self.has_chroma):
            raise ValueError("At least one DCT plane must remain active for luma-fusion variant.")
        active_idx_luma = _build_active_index_from_count(self.coeff_count_luma)
        active_idx_cb = _build_active_index_from_count(self.coeff_count_cb)
        active_idx_cr = _build_active_index_from_count(self.coeff_count_cr)
        if active_idx_luma is not None and self.has_luma:
            self.register_buffer("active_idx_luma", active_idx_luma, persistent=False)
        else:
            self.active_idx_luma = None
        if active_idx_cb is not None and self.has_cb:
            self.register_buffer("active_idx_cb", active_idx_cb, persistent=False)
        else:
            self.active_idx_cb = None
        if active_idx_cr is not None and self.has_cr:
            self.register_buffer("active_idx_cr", active_idx_cr, persistent=False)
        else:
            self.active_idx_cr = None
        base_width = max(self.target_channels, 64)
        self.luma_width = base_width if self.has_luma else 0
        self.chroma_width = max(base_width // 2, 1) if self.has_chroma else 0
        if self.has_luma:
            self.luma_down = nn.Sequential(
                nn.Conv2d(self.luma_channels, self.luma_width, kernel_size=1, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(self.luma_width),
                nn.SiLU(inplace=True),
            )
        else:
            self.luma_down = None
        if self.has_chroma:
            self.chroma_proj = nn.Sequential(
                nn.Conv2d(self.chroma_channels, self.chroma_width, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(self.chroma_width),
                nn.SiLU(inplace=True),
            )
        else:
            self.chroma_proj = None
        fusion_in = (self.luma_width if self.has_luma else 0) + (self.chroma_width if self.has_chroma else 0)
        if fusion_in == 0:
            raise ValueError("Luma-fusion adapter received empty feature configuration.")
        if fusion_in == self.target_channels:
            self.fusion = nn.Identity()
        else:
            self.fusion = nn.Sequential(
                nn.Conv2d(fusion_in, self.target_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.target_channels),
                nn.SiLU(inplace=True),
            )

    def forward(self, inputs):
        y_blocks, (cb_blocks, cr_blocks) = _unpack_payload(inputs)
        device = next(self.parameters()).device
        y_blocks = y_blocks.to(device=device, dtype=torch.float32)
        cb_blocks = cb_blocks.to(device=device, dtype=torch.float32)
        cr_blocks = cr_blocks.to(device=device, dtype=torch.float32)

        features: list[torch.Tensor] = []

        if self.has_luma:
            if self.active_idx_luma is not None and y_blocks.size(1) > self.luma_channels:
                idx_luma = self.active_idx_luma.to(device)
                y_blocks = torch.index_select(y_blocks, 1, idx_luma)
            elif y_blocks.size(1) != self.luma_channels:
                raise ValueError(
                    f"Expected {self.luma_channels} luma coefficients, received {y_blocks.size(1)}."
                )
            luma_feat = self.luma_down(y_blocks) if self.luma_down is not None else y_blocks
            features.append(luma_feat)
        else:
            if y_blocks.size(1) != 0:
                raise ValueError(
                    "Luma coefficients were disabled but input still provides non-zero channels."
                )

        cb = cb_blocks
        cr = cr_blocks
        if self.chroma_channels_cb > 0:
            cb = self._select_chroma_plane(cb, self.chroma_channels_cb, getattr(self, "active_idx_cb", None), name="Cb")
        else:
            cb = cb.new_empty((cb.size(0), 0, cb.size(2), cb.size(3)))
        if self.chroma_channels_cr > 0:
            cr = self._select_chroma_plane(cr, self.chroma_channels_cr, getattr(self, "active_idx_cr", None), name="Cr")
        else:
            cr = cr.new_empty((cr.size(0), 0, cr.size(2), cr.size(3)))

        if self.has_chroma:
            chroma_parts = []
            if self.chroma_channels_cb > 0:
                chroma_parts.append(cb)
            if self.chroma_channels_cr > 0:
                chroma_parts.append(cr)
            chroma = torch.cat(chroma_parts, dim=1)
            if chroma.size(1) != self.chroma_channels:
                raise ValueError(
                    f"Expected concatenated chroma tensor with {self.chroma_channels} channels, "
                    f"received {chroma.size(1)}."
                )
            chroma_feat = self.chroma_proj(chroma) if self.chroma_proj is not None else chroma
            features.append(chroma_feat)
        else:
            if cb.size(1) != 0 or cr.size(1) != 0:
                raise ValueError("Chroma coefficients were disabled but payload still includes channels.")

        if not features:
            raise RuntimeError("No active coefficient branches available for fusion.")
        if len(features) == 1:
            fusion_input = features[0]
        else:
            fusion_input = torch.cat(features, dim=1)
        fused = fusion_input if isinstance(self.fusion, nn.Identity) else self.fusion(fusion_input)
        conv1_like = F.interpolate(fused, scale_factor=4.0, mode="bilinear", align_corners=False)
        return self._forward_residual_stages(conv1_like, skip_pool=True)


class CompressedResNetLumaFusionPruned(_BackboneAdapter):
    def __init__(
        self,
        backbone: PResNet,
        *,
        coeff_window_luma: int | None = None,
        coeff_window_chroma: int | None = None,
        coeff_count_luma: int | None = None,
        coeff_count_chroma: int | None = None,
        coeff_window_cb: int | None = None,
        coeff_window_cr: int | None = None,
        coeff_count_cb: int | None = None,
        coeff_count_cr: int | None = None,
    ) -> None:
        super().__init__(backbone)
        _, luma_count, cb_count, cr_count = resolve_coefficient_counts(
            coeff_window=coeff_window_luma,
            coeff_count=coeff_count_luma,
            coeff_window_luma=coeff_window_luma,
            coeff_count_luma=coeff_count_luma,
            coeff_window_chroma=coeff_window_chroma,
            coeff_count_chroma=coeff_count_chroma,
            coeff_window_cb=coeff_window_cb,
            coeff_window_cr=coeff_window_cr,
            coeff_count_cb=coeff_count_cb,
            coeff_count_cr=coeff_count_cr,
        )
        self.coeff_count_luma = luma_count
        self.coeff_count_cb = cb_count
        self.coeff_count_cr = cr_count
        self.coeff_count_chroma = cb_count if cb_count == cr_count else max(cb_count, cr_count)
        self.coeff_window_luma = count_to_window(luma_count)
        window_cb = count_to_window(cb_count)
        window_cr = count_to_window(cr_count)
        self.coeff_window_chroma = window_cb if window_cb == window_cr else None
        scale_window = max(math.sqrt(luma_count), 1.0)
        scale = max(scale_window / 8.0, 1.0 / 8.0)
        self.coeff_window = self.coeff_window_luma
        self.luma_channels = self.coeff_count_luma
        self.chroma_channels_cb = self.coeff_count_cb
        self.chroma_channels_cr = self.coeff_count_cr
        self.chroma_channels = self.chroma_channels_cb + self.chroma_channels_cr
        self.has_luma = self.luma_channels > 0
        self.has_cb = self.chroma_channels_cb > 0
        self.has_cr = self.chroma_channels_cr > 0
        self.has_chroma = self.has_cb or self.has_cr
        if not (self.has_luma or self.has_chroma):
            raise ValueError("At least one DCT plane must remain active for pruned luma-fusion variant.")
        active_idx_luma = _build_active_index_from_count(self.coeff_count_luma)
        active_idx_cb = _build_active_index_from_count(self.coeff_count_cb)
        active_idx_cr = _build_active_index_from_count(self.coeff_count_cr)
        if active_idx_luma is not None and self.has_luma:
            self.register_buffer("active_idx_luma", active_idx_luma, persistent=False)
        else:
            self.active_idx_luma = None
        if active_idx_cb is not None and self.has_cb:
            self.register_buffer("active_idx_cb", active_idx_cb, persistent=False)
        else:
            self.active_idx_cb = None
        if active_idx_cr is not None and self.has_cr:
            self.register_buffer("active_idx_cr", active_idx_cr, persistent=False)
        else:
            self.active_idx_cr = None
        self._shrink_backbone(scale)
        self.luma_width = max(self.target_channels, 8) if self.has_luma else 0
        self.chroma_width = max(self.luma_width // 2, 4) if self.has_chroma else 0
        if self.has_luma:
            self.luma_down = nn.Sequential(
                nn.Conv2d(self.luma_channels, self.luma_width, kernel_size=1, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(self.luma_width),
                nn.SiLU(inplace=True),
            )
        else:
            self.luma_down = None
        if self.has_chroma:
            self.chroma_proj = nn.Sequential(
                nn.Conv2d(self.chroma_channels, self.chroma_width, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(self.chroma_width),
                nn.SiLU(inplace=True),
            )
        else:
            self.chroma_proj = None
        fusion_in = (self.luma_width if self.has_luma else 0) + (self.chroma_width if self.has_chroma else 0)
        if fusion_in == 0:
            raise ValueError("Pruned luma-fusion adapter received empty feature configuration.")
        if fusion_in == self.target_channels:
            self.fusion = nn.Identity()
        else:
            self.fusion = nn.Sequential(
                nn.Conv2d(fusion_in, self.target_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.target_channels),
                nn.SiLU(inplace=True),
            )

    def _shrink_backbone(self, scale: float) -> None:
        block_cls = type(self.backbone.res_layers[0].blocks[0])
        if block_cls not in {BasicBlock, BottleNeck}:
            raise TypeError("Unsupported block type for pruning.")
        block_nums = [len(stage.blocks) for stage in self.backbone.res_layers]
        base_channels: list[int] = []
        for stage in self.backbone.res_layers:
            first_block = stage.blocks[0]
            base_channels.append(first_block.branch2a.conv.out_channels)

        scaled_initial = max(8, int(round(64 * scale)))
        self.target_channels = scaled_initial
        new_layers = nn.ModuleList()
        in_channels = scaled_initial
        new_stage_outputs: list[int] = []
        for idx, (block_count, base_out) in enumerate(zip(block_nums, base_channels)):
            scaled_out = max(8, int(round(base_out * scale)))
            stage = Blocks(
                block_cls,
                in_channels,
                scaled_out,
                block_count,
                stage_num=idx + 2,
                act="relu",
                variant="d",
            )
            new_layers.append(stage)
            in_channels = scaled_out * block_cls.expansion
            new_stage_outputs.append(in_channels)

        self.backbone.res_layers = new_layers
        self.backbone.out_channels = [new_stage_outputs[i] for i in self.return_idx]
        self.out_channels = self.backbone.out_channels
        self.backbone.conv1 = nn.Identity()

    def forward(self, inputs):
        y_blocks, (cb_blocks, cr_blocks) = _unpack_payload(inputs)
        device = next(self.parameters()).device
        y_blocks = y_blocks.to(device=device, dtype=torch.float32)
        cb_blocks = cb_blocks.to(device=device, dtype=torch.float32)
        cr_blocks = cr_blocks.to(device=device, dtype=torch.float32)

        features: list[torch.Tensor] = []

        if self.has_luma:
            if self.active_idx_luma is not None and y_blocks.size(1) > self.luma_channels:
                idx_luma = self.active_idx_luma.to(device)
                y_blocks = torch.index_select(y_blocks, 1, idx_luma)
            elif y_blocks.size(1) != self.luma_channels:
                raise ValueError(
                    f"Expected {self.luma_channels} luma coefficients, received {y_blocks.size(1)}."
                )
            luma_feat = self.luma_down(y_blocks) if self.luma_down is not None else y_blocks
            features.append(luma_feat)
        else:
            if y_blocks.size(1) != 0:
                raise ValueError(
                    "Luma coefficients were disabled but input still provides non-zero channels."
                )

        cb = cb_blocks
        cr = cr_blocks
        if self.chroma_channels_cb > 0:
            cb = self._select_chroma_plane(cb, self.chroma_channels_cb, getattr(self, "active_idx_cb", None), name="Cb")
        else:
            cb = cb.new_empty((cb.size(0), 0, cb.size(2), cb.size(3)))
        if self.chroma_channels_cr > 0:
            cr = self._select_chroma_plane(cr, self.chroma_channels_cr, getattr(self, "active_idx_cr", None), name="Cr")
        else:
            cr = cr.new_empty((cr.size(0), 0, cr.size(2), cr.size(3)))

        if self.has_chroma:
            chroma_parts = []
            if self.chroma_channels_cb > 0:
                chroma_parts.append(cb)
            if self.chroma_channels_cr > 0:
                chroma_parts.append(cr)
            chroma = torch.cat(chroma_parts, dim=1)
            if chroma.size(1) != self.chroma_channels:
                raise ValueError(
                    f"Expected concatenated chroma tensor with {self.chroma_channels} channels, "
                    f"received {chroma.size(1)}."
                )
            chroma_feat = self.chroma_proj(chroma) if self.chroma_proj is not None else chroma
            features.append(chroma_feat)
        else:
            if cb.size(1) != 0 or cr.size(1) != 0:
                raise ValueError("Chroma coefficients were disabled but payload still includes channels.")

        if not features:
            raise RuntimeError("No active coefficient branches available for fusion.")
        if len(features) == 1:
            fusion_input = features[0]
        else:
            fusion_input = torch.cat(features, dim=1)
        fused = fusion_input if isinstance(self.fusion, nn.Identity) else self.fusion(fusion_input)
        conv1_like = F.interpolate(fused, scale_factor=4.0, mode="bilinear", align_corners=False)
        return self._forward_residual_stages(conv1_like, skip_pool=True)


@register()
class CompressedPResNet(nn.Module):
    """Factory module that wraps :class:`PResNet` with a compressed-input adapter."""

    def __init__(
        self,
        compression_variant: Literal[
            "reconstruction",
            "block-stem",
            "luma-fusion",
            "luma-fusion-pruned",
        ],
        coeff_window: int = 8,
        coeff_window_luma: int | None = None,
        coeff_window_chroma: int | None = None,
        coeff_window_cb: int | None = None,
        coeff_window_cr: int | None = None,
        coeff_count: int | None = None,
        coeff_count_luma: int | None = None,
        coeff_count_chroma: int | None = None,
        coeff_count_cb: int | None = None,
        coeff_count_cr: int | None = None,
        range_mode: str = "studio",
        mean: Sequence[float] = _IMAGENET_MEAN,
        std: Sequence[float] = _IMAGENET_STD,
        strict_load: bool = False,
        **backbone_kwargs,
    ) -> None:
        super().__init__()

        compressed_pretrained = backbone_kwargs.pop("compressed_pretrained", None)
        base_pretrained = backbone_kwargs.pop("pretrained", False)

        backbone = PResNet(pretrained=base_pretrained, **backbone_kwargs)
        _, luma_count, cb_count, cr_count = resolve_coefficient_counts(
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
        window_luma = count_to_window(luma_count)
        window_cb = count_to_window(cb_count)
        window_cr = count_to_window(cr_count)
        window_chroma = window_cb if window_cb == window_cr else None
        chroma_count = cb_count if cb_count == cr_count else max(cb_count, cr_count)
        self.backbone = build_compressed_backbone(
            compression_variant,
            backbone,
            range_mode=range_mode,
            mean=mean,
            std=std,
            coeff_window_luma=window_luma,
            coeff_window_chroma=window_chroma,
            coeff_count_luma=luma_count,
            coeff_count_chroma=chroma_count,
            coeff_window_cb=window_cb,
            coeff_window_cr=window_cr,
            coeff_count_cb=cb_count,
            coeff_count_cr=cr_count,
        )
        self.coeff_count_luma = luma_count
        self.coeff_count_cb = cb_count
        self.coeff_count_cr = cr_count
        self.coeff_count_chroma = chroma_count
        self.coeff_window_luma = window_luma
        self.coeff_window_cb = window_cb
        self.coeff_window_cr = window_cr
        self.coeff_window_chroma = window_chroma
        self.coeff_window = self.coeff_window_luma

        if compressed_pretrained:
            state = torch.load(compressed_pretrained, map_location="cpu")
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            filtered: dict[str, torch.Tensor] = {}
            for key, value in state.items():
                if not key.startswith("backbone"):
                    continue
                trimmed = key[len("backbone.") :]
                filtered[trimmed] = value
            missing, unexpected = self.backbone.load_state_dict(filtered, strict=strict_load)
            if missing or unexpected:
                print(
                    "Warning: loading compressed backbone weights, "
                    f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
                )

        self.return_idx = list(self.backbone.return_idx)
        self.out_channels = list(self.backbone.out_channels)
        self.out_strides = list(self.backbone.out_strides)

    def forward(self, inputs):
        return self.backbone(inputs)


def build_compressed_backbone(
    variant: Literal[
        "reconstruction",
        "block-stem",
        "luma-fusion",
        "luma-fusion-pruned",
    ],
    backbone: PResNet,
    *,
    range_mode: str,
    mean: Sequence[float],
    std: Sequence[float],
    coeff_window_luma: int | None,
    coeff_window_chroma: int | None,
    coeff_count_luma: int,
    coeff_count_chroma: int,
    coeff_window_cb: int | None = None,
    coeff_window_cr: int | None = None,
    coeff_count_cb: int | None = None,
    coeff_count_cr: int | None = None,
) -> nn.Module:
    if coeff_window_cb is None:
        coeff_window_cb = coeff_window_chroma
    if coeff_window_cr is None:
        coeff_window_cr = coeff_window_chroma
    if coeff_count_cb is None:
        coeff_count_cb = coeff_count_chroma
    if coeff_count_cr is None:
        coeff_count_cr = coeff_count_chroma
    if variant == "reconstruction":
        return CompressedResNetReconstruction(
            backbone,
            range_mode=range_mode,
            mean=mean,
            std=std,
        )
    if variant == "block-stem":
        return CompressedResNetBlockStem(
            backbone,
            coeff_window_luma=coeff_window_luma,
            coeff_window_chroma=coeff_window_chroma,
            coeff_count_luma=coeff_count_luma,
            coeff_count_chroma=coeff_count_chroma,
            coeff_window_cb=coeff_window_cb,
            coeff_window_cr=coeff_window_cr,
            coeff_count_cb=coeff_count_cb,
            coeff_count_cr=coeff_count_cr,
        )
    if variant == "luma-fusion":
        return CompressedResNetLumaFusion(
            backbone,
            coeff_window_luma=coeff_window_luma,
            coeff_window_chroma=coeff_window_chroma,
            coeff_count_luma=coeff_count_luma,
            coeff_count_chroma=coeff_count_chroma,
            coeff_window_cb=coeff_window_cb,
            coeff_window_cr=coeff_window_cr,
            coeff_count_cb=coeff_count_cb,
            coeff_count_cr=coeff_count_cr,
        )
    if variant == "luma-fusion-pruned":
        return CompressedResNetLumaFusionPruned(
            backbone,
            coeff_window_luma=coeff_window_luma,
            coeff_window_chroma=coeff_window_chroma,
            coeff_count_luma=coeff_count_luma,
            coeff_count_chroma=coeff_count_chroma,
            coeff_window_cb=coeff_window_cb,
            coeff_window_cr=coeff_window_cr,
            coeff_count_cb=coeff_count_cb,
            coeff_count_cr=coeff_count_cr,
        )
    raise ValueError(f"Unsupported compressed backbone variant: {variant}")
