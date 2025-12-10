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


def _unpack_payload(payload) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(payload, (list, tuple)):
        if len(payload) == 2 and isinstance(payload[0], (list, tuple)):
            payload = payload[0]
        if len(payload) == 2 and all(isinstance(p, torch.Tensor) for p in payload):
            return payload[0], payload[1]
    raise TypeError("Expected (y_blocks, cbcr_blocks) tuple from CompressToDCT.")


def _upsample_chroma(chroma: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(chroma, size=target_hw, mode="bilinear", align_corners=False)


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


def _build_active_index(coeff_window: int) -> torch.Tensor | None:
    if coeff_window >= 8:
        return None
    indices = [row + col * 8 for col in range(coeff_window) for row in range(coeff_window)]
    return torch.tensor(indices, dtype=torch.long)


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

    def _decode_planes(self, y_blocks: torch.Tensor, cbcr_blocks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y_plane = self.decoder(y_blocks)
        if cbcr_blocks.dim() != 5 or cbcr_blocks.size(1) != 2:
            raise ValueError("Chrominance tensor must have shape [B, 2, 64, By, Bx].")
        cb_plane = self.decoder(cbcr_blocks[:, 0])
        cr_plane = self.decoder(cbcr_blocks[:, 1])
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
        y_blocks, cbcr_blocks = _unpack_payload(inputs)
        device = self.mean.device
        y_blocks = y_blocks.to(device=device, dtype=torch.float32)
        cbcr_blocks = cbcr_blocks.to(device=device, dtype=torch.float32)
        y_plane, cb_plane, cr_plane = self._decode_planes(y_blocks, cbcr_blocks)
        rgb = self._ycbcr_to_rgb(y_plane, cb_plane, cr_plane)
        refined = self.refine(rgb)
        normalized = (refined - self.mean.to(refined.dtype)) / self.std.to(refined.dtype)
        return self.backbone(normalized)


class CompressedResNetBlockStem(_BackboneAdapter):
    def __init__(self, backbone: PResNet, coeff_window: int) -> None:
        super().__init__(backbone)
        if coeff_window not in {1, 2, 4, 8}:
            raise ValueError("coeff_window must be one of {1, 2, 4, 8}.")
        self.coeff_window = coeff_window
        self.luma_channels = coeff_window * coeff_window
        self.chroma_channels = 2 * self.luma_channels
        active_idx = _build_active_index(coeff_window)
        if active_idx is not None:
            self.register_buffer("active_idx", active_idx, persistent=False)
        else:
            self.active_idx = None
        mid_channels = max(self.target_channels, 64)
        self.luma_proj = nn.Sequential(
            nn.Conv2d(self.luma_channels, mid_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
        )
        self.chroma_proj = nn.Sequential(
            nn.Conv2d(self.chroma_channels, mid_channels // 2, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels // 2),
            nn.SiLU(inplace=True),
        )
        fusion_in = mid_channels + mid_channels // 2
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_in, self.target_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.target_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs):
        y_blocks, cbcr_blocks = _unpack_payload(inputs)
        device = next(self.parameters()).device
        y_blocks = y_blocks.to(device=device, dtype=torch.float32)
        cbcr_blocks = cbcr_blocks.to(device=device, dtype=torch.float32)

        if self.active_idx is not None:
            idx = self.active_idx.to(device)
            y_blocks = torch.index_select(y_blocks, 1, idx)
            cb = torch.index_select(cbcr_blocks[:, 0], 1, idx)
            cr = torch.index_select(cbcr_blocks[:, 1], 1, idx)
        else:
            cb = cbcr_blocks[:, 0]
            cr = cbcr_blocks[:, 1]

        chroma = torch.cat((cb, cr), dim=1)
        chroma = F.interpolate(chroma, size=y_blocks.shape[-2:], mode="nearest")

        y_feat = self.luma_proj(y_blocks)
        chroma_feat = self.chroma_proj(chroma)
        fused = self.fusion(torch.cat((y_feat, chroma_feat), dim=1))
        conv1_like = F.interpolate(fused, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self._forward_residual_stages(conv1_like, skip_pool=True)


class CompressedResNetLumaFusion(_BackboneAdapter):
    def __init__(self, backbone: PResNet, coeff_window: int) -> None:
        super().__init__(backbone)
        if coeff_window not in {1, 2, 4, 8}:
            raise ValueError("coeff_window must be one of {1, 2, 4, 8}.")
        self.coeff_window = coeff_window
        self.luma_channels = coeff_window * coeff_window
        self.chroma_channels = 2 * self.luma_channels
        active_idx = _build_active_index(coeff_window)
        if active_idx is not None:
            self.register_buffer("active_idx", active_idx, persistent=False)
        else:
            self.active_idx = None
        luma_width = max(self.target_channels, 64)
        chroma_width = luma_width // 2
        self.luma_down = nn.Sequential(
            nn.Conv2d(self.luma_channels, luma_width, kernel_size=1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(luma_width),
            nn.SiLU(inplace=True),
        )
        self.chroma_proj = nn.Sequential(
            nn.Conv2d(self.chroma_channels, chroma_width, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(chroma_width),
            nn.SiLU(inplace=True),
        )
        fusion_in = luma_width + chroma_width
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_in, self.target_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.target_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs):
        y_blocks, cbcr_blocks = _unpack_payload(inputs)
        device = next(self.parameters()).device
        y_blocks = y_blocks.to(device=device, dtype=torch.float32)
        cbcr_blocks = cbcr_blocks.to(device=device, dtype=torch.float32)

        if self.active_idx is not None:
            idx = self.active_idx.to(device)
            y_blocks = torch.index_select(y_blocks, 1, idx)
            cb = torch.index_select(cbcr_blocks[:, 0], 1, idx)
            cr = torch.index_select(cbcr_blocks[:, 1], 1, idx)
        else:
            cb = cbcr_blocks[:, 0]
            cr = cbcr_blocks[:, 1]

        luma_feat = self.luma_down(y_blocks)
        chroma = torch.cat((cb, cr), dim=1)
        chroma_feat = self.chroma_proj(chroma)
        fused = self.fusion(torch.cat((luma_feat, chroma_feat), dim=1))
        conv1_like = F.interpolate(fused, scale_factor=4.0, mode="bilinear", align_corners=False)
        return self._forward_residual_stages(conv1_like, skip_pool=True)


class CompressedResNetLumaFusionPruned(_BackboneAdapter):
    def __init__(self, backbone: PResNet, coeff_window: int) -> None:
        super().__init__(backbone)
        if coeff_window not in {1, 2, 4, 8}:
            raise ValueError("coeff_window must be one of {1, 2, 4, 8}.")
        scale = max(coeff_window / 8.0, 1.0 / 8.0)
        self.coeff_window = coeff_window
        self.luma_channels = coeff_window * coeff_window
        self.chroma_channels = 2 * self.luma_channels
        active_idx = _build_active_index(coeff_window)
        if active_idx is not None:
            self.register_buffer("active_idx", active_idx, persistent=False)
        else:
            self.active_idx = None
        self._shrink_backbone(scale)
        luma_width = max(self.target_channels, 8)
        chroma_width = max(luma_width // 2, 4)
        self.luma_down = nn.Sequential(
            nn.Conv2d(self.luma_channels, luma_width, kernel_size=1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(luma_width),
            nn.SiLU(inplace=True),
        )
        self.chroma_proj = nn.Sequential(
            nn.Conv2d(self.chroma_channels, chroma_width, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(chroma_width),
            nn.SiLU(inplace=True),
        )
        fusion_in = luma_width + chroma_width
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
        y_blocks, cbcr_blocks = _unpack_payload(inputs)
        device = next(self.parameters()).device
        y_blocks = y_blocks.to(device=device, dtype=torch.float32)
        cbcr_blocks = cbcr_blocks.to(device=device, dtype=torch.float32)

        if self.active_idx is not None:
            idx = self.active_idx.to(device)
            y_blocks = torch.index_select(y_blocks, 1, idx)
            cb = torch.index_select(cbcr_blocks[:, 0], 1, idx)
            cr = torch.index_select(cbcr_blocks[:, 1], 1, idx)
        else:
            cb = cbcr_blocks[:, 0]
            cr = cbcr_blocks[:, 1]

        luma_feat = self.luma_down(y_blocks)
        chroma = torch.cat((cb, cr), dim=1)
        chroma_feat = self.chroma_proj(chroma)
        fused = self.fusion(torch.cat((luma_feat, chroma_feat), dim=1))
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
        self.backbone = build_compressed_backbone(
            compression_variant,
            backbone,
            range_mode=range_mode,
            mean=mean,
            std=std,
            coeff_window=coeff_window,
        )

        if compressed_pretrained:
            state = torch.load(compressed_pretrained, map_location="cpu")
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            missing, unexpected = self.backbone.load_state_dict(state, strict=strict_load)
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
    coeff_window: int,
) -> nn.Module:
    if variant == "reconstruction":
        return CompressedResNetReconstruction(
            backbone,
            range_mode=range_mode,
            mean=mean,
            std=std,
        )
    if variant == "block-stem":
        return CompressedResNetBlockStem(backbone, coeff_window=coeff_window)
    if variant == "luma-fusion":
        return CompressedResNetLumaFusion(backbone, coeff_window=coeff_window)
    if variant == "luma-fusion-pruned":
        return CompressedResNetLumaFusionPruned(backbone, coeff_window=coeff_window)
    raise ValueError(f"Unsupported compressed backbone variant: {variant}")
