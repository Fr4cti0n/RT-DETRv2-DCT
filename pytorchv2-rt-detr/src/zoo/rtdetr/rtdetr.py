"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from collections import OrderedDict

import torch
import torch.nn as nn

from ...core import register


__all__ = ['RTDETR', ]


@register()
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        backbone: nn.Module, 
        encoder: nn.Module, 
        decoder: nn.Module, 
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self._match_encoder_to_backbone()
        
    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)

        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 

    def adapt_to_backbone(self) -> None:
        """Public hook that ensures encoder stem matches backbone outputs."""
        self._match_encoder_to_backbone()

    def _match_encoder_to_backbone(self) -> None:
        encoder = getattr(self, 'encoder', None)
        backbone = getattr(self, 'backbone', None)
        if encoder is None or backbone is None:
            return
        backbone_channels = getattr(backbone, 'out_channels', None)
        if not backbone_channels:
            return
        if not hasattr(encoder, 'input_proj'):
            return
        hidden_dim = getattr(encoder, 'hidden_dim', None)
        if hidden_dim is None:
            return

        try:
            desired: List[int] = [int(ch) for ch in backbone_channels]
        except TypeError:
            return
        current = getattr(encoder, 'in_channels', None)
        if current is not None and list(current) == desired:
            return

        new_proj = nn.ModuleList()
        for in_ch in desired:
            block = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_ch, hidden_dim, kernel_size=1, bias=False)),
                ('norm', nn.BatchNorm2d(hidden_dim)),
            ]))
            new_proj.append(block)

        encoder.input_proj = new_proj
        encoder.in_channels = desired
