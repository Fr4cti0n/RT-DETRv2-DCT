"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data import default_collate

import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as VT
from torchvision.transforms.v2 import functional as VF, InterpolationMode

import random
from functools import partial
from typing import Any, Optional, Sequence

from .transforms.compress_reference_images import (
    CompressToDCT,
    TrimDCTCoefficients,
)

from ..core import register


__all__ = [
    'DataLoader',
    'BaseCollateFunction', 
    'BatchImageCollateFuncion',
    'batch_image_collate_fn'
]


@register()
class DataLoader(data.DataLoader):
    __inject__ = ['dataset', 'collate_fn']

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for n in ['dataset', 'batch_size', 'num_workers', 'drop_last', 'collate_fn']:
            format_string += "\n"
            format_string += "    {0}: {1}".format(n, getattr(self, n))
        format_string += "\n)"
        return format_string

    def set_epoch(self, epoch):
        self._epoch = epoch 
        self.dataset.set_epoch(epoch)
        self.collate_fn.set_epoch(epoch)
    
    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle):
        assert isinstance(shuffle, bool), 'shuffle must be a boolean'
        self._shuffle = shuffle


def _normalize_dct_payload(sample):
    """Return a `(y_blocks, cbcr_blocks)` tuple regardless of nested layout."""
    if not isinstance(sample, (tuple, list)) or len(sample) != 2:
        raise TypeError(
            "Expected a tuple/list of length 2 representing DCT payload, "
            f"but received {type(sample).__name__}."
        )

    y_blocks, chroma = sample

    if not torch.is_tensor(y_blocks):
        raise TypeError("First element of DCT payload must be a tensor of luminance blocks.")

    if torch.is_tensor(chroma):
        cbcr_blocks = chroma
    elif isinstance(chroma, (tuple, list)) and len(chroma) == 2 and all(torch.is_tensor(x) for x in chroma):
        cbcr_blocks = torch.stack([chroma[0], chroma[1]], dim=0)
    else:
        raise TypeError("Second element of DCT payload must be a tensor or tuple of (cb, cr) tensors.")

    return y_blocks, cbcr_blocks


@register()
def batch_image_collate_fn(items):
    """Collate helper that supports tensor images and DCT payloads."""
    if not items:
        raise ValueError("batch_image_collate_fn received an empty batch")

    first_sample = items[0][0]

    if torch.is_tensor(first_sample):
        images = torch.cat([sample[0][None] for sample in items], dim=0)
        targets = [sample[1] for sample in items]
        return images, targets

    if isinstance(first_sample, (tuple, list)) and len(first_sample) == 2:
        y_blocks, cbcr_blocks = zip(*[_normalize_dct_payload(sample[0]) for sample in items])
        y_blocks = torch.stack(list(y_blocks), dim=0)
        cbcr_blocks = torch.stack(list(cbcr_blocks), dim=0)
        targets = [sample[1] for sample in items]
        return (y_blocks, cbcr_blocks), targets

    raise TypeError(
        "batch_image_collate_fn expects a tensor or (y_blocks, cbcr_blocks) payload, "
        f"but received {type(first_sample).__name__}."
    )


class BaseCollateFunction(object):
    def set_epoch(self, epoch):
        self._epoch = epoch 

    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1

    def __call__(self, items):
        raise NotImplementedError('')


@register()
class BatchImageCollateFuncion(BaseCollateFunction):
    def __init__(
        self, 
        scales=None, 
        stop_epoch=None,
        *,
        require_multiple_of: int | None = None,
        compress_to_dct: dict[str, Any] | None = None,
        trim_dct: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.scales = scales
        self.stop_epoch = stop_epoch if stop_epoch is not None else 100000000
        self.require_multiple_of = require_multiple_of
        self._compress_params = compress_to_dct or None
        self._trim_params = trim_dct or None
        self._compress_transform: Optional[CompressToDCT] = None
        self._trim_transform: Optional[TrimDCTCoefficients] = None
        # self.interpolation = interpolation

        if self.require_multiple_of is not None:
            divisor = int(self.require_multiple_of)
            if divisor <= 0:
                raise ValueError("require_multiple_of must be a positive integer")
            if self.scales is not None:
                for scale in self.scales:
                    if isinstance(scale, Sequence):
                        dims = list(scale)
                    else:
                        dims = [scale]
                    if any(int(dim) % divisor for dim in dims):
                        raise ValueError(
                            f"Scale {scale} is not divisible by {divisor}; "
                            "update the configuration to use multiples of the compression block size."
                        )

    def _ensure_transforms(self) -> None:
        if self._compress_params is not None and self._compress_transform is None:
            self._compress_transform = CompressToDCT(**self._compress_params)
        if self._trim_params is not None and self._trim_transform is None:
            if self._trim_params.get("coeff_count_luma") is None:
                raise ValueError("trim_dct requires coeff_count_luma to be specified")
            self._trim_transform = TrimDCTCoefficients(**self._trim_params)

    def _apply_compression(self, images: torch.Tensor, targets: list[dict]) -> tuple[tuple[torch.Tensor, torch.Tensor], list[dict]]:
        self._ensure_transforms()
        if self._compress_transform is None:
            raise RuntimeError("Compression parameters were not provided but compression was requested.")

        payloads = []
        updated_targets: list[dict] = []
        for image, target in zip(images, targets, strict=True):
            payload, tgt = self._compress_transform((image, target))
            if self._trim_transform is not None:
                payload, tgt = self._trim_transform((payload, tgt))
            payloads.append(payload)
            updated_targets.append(tgt)

        y_blocks, cbcr_blocks = zip(*[_normalize_dct_payload(payload) for payload in payloads])
        y_tensor = torch.stack(list(y_blocks), dim=0)
        cbcr_tensor = torch.stack(list(cbcr_blocks), dim=0)
        return (y_tensor, cbcr_tensor), updated_targets

    def __call__(self, items):
        first_sample = items[0][0]

        if torch.is_tensor(first_sample):
            images = torch.cat([sample[0][None] for sample in items], dim=0)
        elif isinstance(first_sample, (tuple, list)) and len(first_sample) == 2:
            y_blocks, cbcr_blocks = zip(*[_normalize_dct_payload(sample[0]) for sample in items])
            y_blocks = torch.stack(list(y_blocks), dim=0)
            cbcr_blocks = torch.stack(list(cbcr_blocks), dim=0)
            images = (y_blocks, cbcr_blocks)
        else:
            raise TypeError(
                "BatchImageCollateFuncion expects each sample to be a tensor or a (y_blocks, cbcr_blocks) tuple, "
                f"but received {type(first_sample).__name__}."
            )

        targets = [x[1] for x in items]

        if (
            self.scales is not None
            and self.epoch < self.stop_epoch
            and torch.is_tensor(images)
        ):
            # sz = random.choice(self.scales)
            # sz = [sz] if isinstance(sz, int) else list(sz)
            # VF.resize(inpt, sz, interpolation=self.interpolation)

            sz = random.choice(self.scales)
            images = F.interpolate(images, size=sz)
            if self.require_multiple_of is not None and (sz % self.require_multiple_of):
                raise ValueError(
                    f"Selected scale {sz} is not divisible by {self.require_multiple_of}. "
                    "Update the configuration to use sizes aligned with the compression block size."
                )
            if 'masks' in targets[0]:
                for tg in targets:
                    tg['masks'] = F.interpolate(tg['masks'], size=sz, mode='nearest')
                raise NotImplementedError('')

        if (
            self.scales is not None
            and self.epoch < self.stop_epoch
            and isinstance(images, tuple)
        ):

            if self._compress_params is None:
                raise NotImplementedError(
                    "Multiscale interpolation is not supported for pre-compressed payloads. "
                    "Provide compress_to_dct parameters in the collate_fn configuration "
                    "to enable compression after resizing."
                )

        if torch.is_tensor(images) and self._compress_params is not None:
            if self.require_multiple_of is not None:
                height, width = images.shape[-2:]
                if height % self.require_multiple_of or width % self.require_multiple_of:
                    raise ValueError(
                        "Image resolution after resizing must be divisible by the compression block size."
                    )
            images, targets = self._apply_compression(images, targets)

        return images, targets

