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
    ) -> None:
        super().__init__()
        self.scales = scales
        self.stop_epoch = stop_epoch if stop_epoch is not None else 100000000
        # self.interpolation = interpolation

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
            if 'masks' in targets[0]:
                for tg in targets:
                    tg['masks'] = F.interpolate(tg['masks'], size=sz, mode='nearest')
                raise NotImplementedError('')

        if (
            self.scales is not None
            and self.epoch < self.stop_epoch
            and isinstance(images, tuple)
        ):
            raise NotImplementedError(
                "Multiscale interpolation is not supported for DCT payloads. "
                "Set collate_fn.scales to null when training with compressed inputs."
            )

        return images, targets

