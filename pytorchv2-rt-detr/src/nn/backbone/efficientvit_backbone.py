"""Wrapper that exposes EfficientViT feature extractor as a backbone."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .EfficientViT.classification.model.efficientvit import EfficientViT
from .EfficientViT.classification.model.build import (
    EfficientViT_m0,
    EfficientViT_m1,
    EfficientViT_m2,
    EfficientViT_m3,
    EfficientViT_m4,
    EfficientViT_m5,
)
from .common import freeze_batch_norm2d
from ...core import register


_VARIANTS: Dict[str, Dict] = {
    "m0": EfficientViT_m0,
    "m1": EfficientViT_m1,
    "m2": EfficientViT_m2,
    "m3": EfficientViT_m3,
    "m4": EfficientViT_m4,
    "m5": EfficientViT_m5,
}


@register()
class EfficientViTBackbone(nn.Module):
    """EfficientViT feature extractor returning the final stage activation."""

    def __init__(
        self,
        variant: str = "m4",
        pretrained: str | bool = False,
        freeze_norm: bool = False,
    ) -> None:
        super().__init__()

        key = variant.lower()
        if key not in _VARIANTS:
            raise ValueError(f"Unsupported EfficientViT variant '{variant}'.")

        cfg = dict(_VARIANTS[key])
        # Instantiate backbone without classification head
        self.backbone = EfficientViT(num_classes=0, **cfg)

        if pretrained:
            if isinstance(pretrained, str):
                state_dict = torch.load(pretrained, map_location="cpu")
                if isinstance(state_dict, dict) and "model" in state_dict:
                    state_dict = state_dict["model"]
                # Drop classifier weights if present
                filtered = {
                    k: v for k, v in state_dict.items() if not k.startswith("head") and not k.startswith("head_dist")
                }
                missing, unexpected = self.backbone.load_state_dict(filtered, strict=False)
                if missing:
                    print(f"EfficientViTBackbone: missing keys ignored {missing}")
                if unexpected:
                    print(f"EfficientViTBackbone: unexpected keys ignored {unexpected}")
            else:
                raise ValueError(
                    "Set 'pretrained' to a checkpoint path or False. Automatic download is not supported."
                )

        if freeze_norm:
            freeze_batch_norm2d(self.backbone)

        self.out_channels = [cfg["embed_dim"][-1]]
        self.out_strides = [cfg["patch_size"]]

    def forward(self, x):
        x = self.backbone.patch_embed(x)
        x = self.backbone.blocks1(x)
        x = self.backbone.blocks2(x)
        x = self.backbone.blocks3(x)
        return [x]
