"""Standalone training script for backbone classification models.

This module lives under ``src.nn.backbone`` so that experimentation with
backbone-only ImageNet training does not interfere with the main RT-DETR
training entrypoints.  It supports three presets that mirror the usual
hyper-parameters from the literature:

* ``resnet34``  – He et al. (2015) baseline with step LR decay.
* ``cspdarknet53`` – YOLOv4 backbone with heavy augmentation (mosaic + cutmix).
* ``efficientvit_m4`` – EfficientNet/Vit hybrid with RMSProp, exponential decay,
  label smoothing, and mixup.

Usage example (single GPU):

.. code-block:: bash

    python -m src.nn.backbone.train_backbones \
        --model resnet34 \
        --train-dirs ./dataset/classification/imagenet1k0 \
                     ./dataset/classification/imagenet1k1 \
                     ./dataset/classification/imagenet1k2 \
                     ./dataset/classification/imagenet1k3 \
        --val-dir ./dataset/classification/imagenet1kvalid \
        --output-dir ./output/imagenet_resnet34_backbone

The defaults follow the cited papers, but every hyper-parameter can be
customised via CLI flags.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, MultiStepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import v2 as T

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

from ...data.dataset.imagenet import ImageNetDataset
from ...data.dataset.subset import limit_per_class, limit_total
from ...data.transforms.compress_reference_images import CompressToDCT, TrimDCTCoefficients
from ..arch.classification import Classification, ClassHead
from .presnet import PResNet
from .compressed_presnet import build_compressed_backbone
from .csp_darknet import CSPDarkNet
from .efficientvit_backbone import EfficientViTBackbone


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _ensure_multiple_of(value: int, divisor: int) -> int:
    if value % divisor == 0:
        return value
    return ((value + divisor - 1) // divisor) * divisor


def _resolve_flip_prob(
    overrides: Optional[Dict[str, object]],
    default: float,
) -> float:
    if not overrides or "flip_prob" not in overrides:
        return default
    prob = float(overrides["flip_prob"])
    if not 0.0 <= prob <= 1.0:
        raise ValueError(f"flip_prob must be within [0, 1]; received {prob}.")
    return prob


def _resolve_crop_scale(
    overrides: Optional[Dict[str, object]],
    default: Tuple[float, float],
) -> Tuple[float, float]:
    if not overrides or "crop_scale" not in overrides:
        return default
    value = overrides["crop_scale"]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(
            "crop_scale override must be a 2-tuple or list specifying (min, max)."
        )
    min_scale = float(value[0])
    max_scale = float(value[1])
    if not 0.0 < min_scale <= max_scale <= 1.0:
        raise ValueError(
            "crop_scale values must satisfy 0 < min <= max <= 1."
        )
    return min_scale, max_scale


@dataclass
class TrainConfig:
    name: str
    epochs: int
    batch_size: int
    lr: float
    optimizer: str
    weight_decay: float
    momentum: float
    use_amp: bool
    num_workers: int
    image_size: int
    warmup_epochs: float = 0.0
    lr_schedule: str = "step"
    lr_milestones: Optional[List[int]] = None
    cosine_min_lr: float = 0.0
    ema_decay: Optional[float] = None
    label_smoothing: float = 0.0
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    drop_connect: float = 0.0
    burn_in_iters: int = 0
    multiscale: Optional[Tuple[int, int]] = None
    input_format: str = "rgb"


def _move_to_device(obj, device: torch.device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_to_device(item, device) for item in obj)
    if isinstance(obj, dict):
        return {key: _move_to_device(value, device) for key, value in obj.items()}
    return obj


def build_model(model_name: str, num_classes: int) -> Tuple[nn.Module, int]:
    model_name = model_name.lower()
    if model_name == "resnet34":
        backbone = PResNet(depth=34, return_idx=[3], pretrained=False,
                           freeze_at=-1, freeze_norm=False)
        head = ClassHead(hidden_dim=512, num_classes=num_classes)
        model = Classification(backbone=backbone, head=head)
        return model, 512
    if model_name == "cspdarknet53":
        backbone = CSPDarkNet(width_multi=1.0, depth_multi=1.0,
                              return_idx=[-1], act="silu")
        head = ClassHead(hidden_dim=backbone.out_channels[0], num_classes=num_classes)
        model = Classification(backbone=backbone, head=head)
        return model, backbone.out_channels[0]
    if model_name == "efficientvit_m4":
        backbone = EfficientViTBackbone(variant="m4", pretrained=False)
        head = ClassHead(hidden_dim=backbone.out_channels[0], num_classes=num_classes)
        model = Classification(backbone=backbone, head=head)
        return model, backbone.out_channels[0]
    raise ValueError(f"Unsupported model '{model_name}'.")


def kaiming_initialisation(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def normalise_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.dim() != 2:
        raise ValueError("Expected logits with shape [batch, num_classes].")
    return logits


def ensure_cuda_available() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not detected. Please ensure a compatible device is available to run this script."
        )
    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    print(f"Detected {device_count} CUDA device(s); primary device: {device_name}")


class SoftTargetCrossEntropy(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        loss = (-targets * log_probs).sum(dim=-1)
        return loss.mean()


def one_hot(targets: torch.Tensor, num_classes: int, smoothing: float = 0.0) -> torch.Tensor:
    off_value = smoothing / num_classes
    on_value = 1.0 - smoothing + off_value
    y = torch.full((targets.size(0), num_classes), off_value, device=targets.device)
    y.scatter_(1, targets.unsqueeze(1), on_value)
    return y


def build_resnet_transforms(
    image_size: int,
    compression: Optional[Dict[str, object]] = None,
    augmentation_overrides: Optional[Dict[str, object]] = None,
    dct_normalizer_train: Optional[T.Transform] = None,
    dct_normalizer_val: Optional[T.Transform] = None,
    trim_coefficients: bool = False,
) -> Tuple[T.Transform, T.Transform]:
    image_size = _ensure_multiple_of(image_size, 8)
    crop_scale = _resolve_crop_scale(augmentation_overrides, (0.08, 1.0))
    flip_prob = _resolve_flip_prob(augmentation_overrides, 0.5)
    train_ops: List[T.Transform] = [
        T.ToImage(),
        T.RandomResizedCrop(image_size, scale=crop_scale),
        T.RandomHorizontalFlip(p=flip_prob),
    ]
    val_resize = _ensure_multiple_of(int(image_size * 256 / 224), 8)
    val_ops: List[T.Transform] = [
        T.ToImage(),
        T.Resize(val_resize),
        T.CenterCrop(image_size),
    ]

    if compression is None:
        train_ops.extend([
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])
        val_ops.extend([
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])
    else:
        train_ops.append(T.ToDtype(torch.float32, scale=True))
        trim_coeffs = bool(trim_coefficients)
        coeff_window = int(compression.get("coeff_window", 8))
        train_ops.append(CompressToDCT(**compression))
        if trim_coeffs and coeff_window < 8:
            train_ops.append(TrimDCTCoefficients(coeff_window))
        if dct_normalizer_train is not None:
            train_ops.append(dct_normalizer_train)
        val_ops.append(T.ToDtype(torch.float32, scale=True))
        val_ops.append(CompressToDCT(**compression))
        if trim_coeffs and coeff_window < 8:
            val_ops.append(TrimDCTCoefficients(coeff_window))
        if dct_normalizer_val is not None:
            val_ops.append(dct_normalizer_val)

    return T.Compose(train_ops), T.Compose(val_ops)


def build_cspdarknet_transforms(
    image_size: int,
    compression: Optional[Dict[str, object]] = None,
    augmentation_overrides: Optional[Dict[str, object]] = None,
) -> Tuple[T.Transform, T.Transform]:
    image_size = _ensure_multiple_of(image_size, 8)
    crop_scale = _resolve_crop_scale(augmentation_overrides, (0.2, 1.0))
    flip_prob = _resolve_flip_prob(augmentation_overrides, 0.5)
    train_ops: List[T.Transform] = [
        T.ToImage(),
        T.RandomResizedCrop(image_size, scale=crop_scale),
        T.RandomHorizontalFlip(p=flip_prob),
        T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)], p=0.8),
        T.RandomAdjustSharpness(0.9, p=0.2),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.5, 1.2), shear=0, fill=114),
    ]
    val_resize = _ensure_multiple_of(image_size + 32, 8)
    val_ops: List[T.Transform] = [
        T.ToImage(),
        T.Resize(val_resize),
        T.CenterCrop(image_size),
    ]

    if compression is None:
        train_ops.extend([
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])
        val_ops.extend([
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])
    else:
        train_ops.append(T.ToDtype(torch.float32, scale=True))
        train_ops.append(CompressToDCT(**compression))
        val_ops.append(T.ToDtype(torch.float32, scale=True))
        val_ops.append(CompressToDCT(**compression))

    return T.Compose(train_ops), T.Compose(val_ops)


def build_efficientvit_transforms(
    image_size: int,
    compression: Optional[Dict[str, object]] = None,
    augmentation_overrides: Optional[Dict[str, object]] = None,
) -> Tuple[T.Transform, T.Transform]:
    image_size = _ensure_multiple_of(image_size, 8)
    flip_prob = _resolve_flip_prob(augmentation_overrides, 0.5)
    train_ops: List[T.Transform] = [
        T.ToImage(),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
        T.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
        T.RandomHorizontalFlip(p=flip_prob),
    ]
    val_resize = _ensure_multiple_of(int(image_size * 256 / 224), 8)
    val_ops: List[T.Transform] = [
        T.ToImage(),
        T.Resize(val_resize),
        T.CenterCrop(image_size),
    ]

    if compression is None:
        train_ops.extend([
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])
        val_ops.extend([
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])
    else:
        train_ops.append(T.ToDtype(torch.float32, scale=True))
        train_ops.append(CompressToDCT(**compression))
        val_ops.append(T.ToDtype(torch.float32, scale=True))
        val_ops.append(CompressToDCT(**compression))

    return T.Compose(train_ops), T.Compose(val_ops)


def mosaic_quad(images: torch.Tensor) -> torch.Tensor:
    b, c, h, w = images.shape
    perm1 = torch.randperm(b, device=images.device)
    perm2 = torch.randperm(b, device=images.device)
    perm3 = torch.randperm(b, device=images.device)

    top = torch.cat([images, images[perm1]], dim=3)
    bottom = torch.cat([images[perm2], images[perm3]], dim=3)
    mosaic = torch.cat([top, bottom], dim=2)
    mosaic = F.interpolate(mosaic, size=(h, w), mode="bilinear", align_corners=False)
    return mosaic


def apply_csp_mix(images: torch.Tensor, targets: torch.Tensor, num_classes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    mosaic_images = mosaic_quad(images)
    mix_targets = torch.zeros((images.size(0), num_classes), device=images.device)
    perms = [torch.arange(images.size(0), device=images.device),
             torch.randperm(images.size(0), device=images.device),
             torch.randperm(images.size(0), device=images.device),
             torch.randperm(images.size(0), device=images.device)]
    weights = torch.tensor([0.25, 0.25, 0.25, 0.25], device=images.device)
    for perm, weight in zip(perms, weights):
        mix_targets += weight * one_hot(targets[perm], num_classes, smoothing=0.0)

    # CutMix: blend mosaic result with another shuffled batch
    perm_cutmix = torch.randperm(images.size(0), device=images.device)
    beta = torch.distributions.Beta(concentration1=1.0, concentration0=1.0).sample((images.size(0),)).to(images.device)
    lambda_ = beta.view(-1, 1, 1, 1)
    cutmix_images = lambda_ * mosaic_images + (1 - lambda_) * mosaic_images[perm_cutmix]
    cutmix_targets = lambda_.squeeze()[:, None] * mix_targets + (1 - lambda_.squeeze())[:, None] * mix_targets[perm_cutmix]

    return cutmix_images, cutmix_targets


def apply_mixup_cutmix(
    images: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    mixup_alpha: float,
    cutmix_alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if mixup_alpha <= 0.0 and cutmix_alpha <= 0.0:
        return images, one_hot(targets, num_classes)

    perm = torch.randperm(images.size(0), device=images.device)
    if cutmix_alpha > 0.0 and random.random() < 0.5:
        lam_value = torch.distributions.Beta(cutmix_alpha, cutmix_alpha).sample((1,)).item()
        bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam_value)
        new_images = images.clone()
        new_images[:, :, bby1:bby2, bbx1:bbx2] = images[perm, :, bby1:bby2, bbx1:bbx2]
        lam_scalar = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
        lam = torch.full((images.size(0), 1), lam_scalar, device=images.device)
    else:
        lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample((images.size(0),)).to(images.device)
        lam_x = lam.view(-1, 1, 1, 1)
        new_images = lam_x * images + (1 - lam_x) * images[perm]
        lam = lam.view(-1, 1)

    soft = lam * one_hot(targets, num_classes) + (1 - lam) * one_hot(targets[perm], num_classes)
    return new_images, soft


def rand_bbox(size: torch.Size, lam: float) -> Tuple[int, int, int, int]:
    w = size[-1]
    h = size[-2]
    cut_ratio = math.sqrt(1.0 - lam)
    cut_w = int(w * cut_ratio)
    cut_h = int(h * cut_ratio)

    cx = random.randint(0, w)
    cy = random.randint(0, h)

    bbx1 = max(cx - cut_w // 2, 0)
    bby1 = max(cy - cut_h // 2, 0)
    bbx2 = min(cx + cut_w // 2, w)
    bby2 = min(cy + cut_h // 2, h)
    return bbx1, bby1, bbx2, bby2


def build_dataloaders(
    model_name: str,
    train_dirs: Sequence[Path],
    val_dir: Path,
    batch_size: int,
    num_workers: int,
    image_size: int,
    max_train: Optional[int],
    max_val: Optional[int],
    input_format: str = "rgb",
    compression: Optional[Dict[str, object]] = None,
    show_progress: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    compression_cfg = compression if input_format == "compressed" else None
    if model_name == "resnet34":
        train_tf, val_tf = build_resnet_transforms(image_size, compression_cfg)
    elif model_name == "cspdarknet53":
        train_tf, val_tf = build_cspdarknet_transforms(image_size, compression_cfg)
    elif model_name == "efficientvit_m4":
        train_tf, val_tf = build_efficientvit_transforms(image_size, compression_cfg)
    else:
        raise ValueError(model_name)

    train_ds = ImageNetDataset(
        [str(p) for p in train_dirs],
        transforms=train_tf,
        max_samples=max_train,
        show_progress=show_progress,
    )
    if max_train is not None:
        per_class = max(1, max_train // max(1, len(train_ds.class_to_idx)))
        train_ds = limit_per_class(train_ds, per_class)

    val_ds = ImageNetDataset(
        [str(val_dir)],
        transforms=val_tf,
        max_samples=max_val,
        show_progress=show_progress,
    )
    if max_val is not None:
        val_ds = limit_total(val_ds, max_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def adjust_learning_rate(base_lr: float, batch_size: int, reference_bs: int = 256) -> float:
    return base_lr * batch_size / reference_bs


def create_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=False,
        )
    if cfg.optimizer == "rmsprop":
        return torch.optim.RMSprop(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            alpha=0.9,
        )
    raise ValueError(f"Unsupported optimizer {cfg.optimizer}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    if cfg.lr_schedule == "step":
        milestones = cfg.lr_milestones or [30, 60, 90]
        return MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    if cfg.lr_schedule == "cosine":
        return CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.cosine_min_lr)

    if cfg.lr_schedule == "exponential":
        def lr_lambda(step: int) -> float:
            epoch = step / steps_per_epoch
            if cfg.warmup_epochs > 0 and epoch < cfg.warmup_epochs:
                return epoch / cfg.warmup_epochs
            decay_rate = 0.97
            effective_epoch = max(0.0, epoch - cfg.warmup_epochs)
            return decay_rate ** effective_epoch
        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    raise ValueError(cfg.lr_schedule)


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1, 5)) -> List[torch.Tensor]:
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k * (100.0 / target.size(0)))
        return res


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    epoch: int,
    cfg: TrainConfig,
    num_classes: int,
    scaler: Optional[torch.cuda.amp.GradScaler],
    loss_fn: nn.Module,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    steps = 0
    steps_per_epoch = len(train_loader)
    epoch_start = time.time()
    last_log_time = epoch_start

    for step, (images, targets) in enumerate(train_loader):
        images = _move_to_device(images, device)
        targets = targets.to(device)

        if cfg.name == "cspdarknet53":
            images, soft_targets = apply_csp_mix(images, targets, num_classes)
            loss_targets = soft_targets
        elif cfg.mixup_alpha > 0.0 or cfg.cutmix_alpha > 0.0:
            images, soft_targets = apply_mixup_cutmix(images, targets, num_classes, cfg.mixup_alpha, cfg.cutmix_alpha)
            loss_targets = soft_targets
        elif cfg.label_smoothing > 0.0 and isinstance(loss_fn, SoftTargetCrossEntropy):
            loss_targets = one_hot(targets, num_classes, smoothing=cfg.label_smoothing)
        else:
            loss_targets = targets

        with torch.amp.autocast('cuda', enabled=cfg.use_amp):
            logits = normalise_logits(model(images))
            if isinstance(loss_fn, SoftTargetCrossEntropy) and loss_targets.dim() == 1:
                loss_targets = one_hot(loss_targets, num_classes, smoothing=cfg.label_smoothing)
                loss = loss_fn(logits, loss_targets)
            else:
                loss = loss_fn(logits, loss_targets)

        optimizer.zero_grad()
        if scaler is not None and cfg.use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if cfg.name == "cspdarknet53" and cfg.multiscale:
            if (step + 1) % 10 == 0:
                new_size = random.randint(cfg.multiscale[0] // 32, cfg.multiscale[1] // 32) * 32
                train_loader.dataset.transforms = build_cspdarknet_transforms(new_size)[0]

        running_loss += loss.item()
        steps += 1

        if (step + 1) % max(1, steps_per_epoch // 20) == 0:
            avg_loss = running_loss / max(steps, 1)
            percent = 100.0 * (step + 1) / steps_per_epoch
            now = time.time()
            step_elapsed = now - last_log_time
            mean_step = (now - epoch_start) / max(steps, 1)
            remaining_steps = steps_per_epoch - (step + 1)
            eta_epoch = mean_step * remaining_steps
            print(
                f"  step {step+1:04}/{steps_per_epoch} ({percent:5.1f}%) | "
                f"loss {avg_loss:.4f} | step time {step_elapsed:.2f}s | ETA {eta_epoch/60:.1f} min",
                flush=True,
            )
            last_log_time = now

        global_step = epoch * steps_per_epoch + step
        if cfg.burn_in_iters > 0 and global_step < cfg.burn_in_iters:
            burn_factor = (global_step + 1) / cfg.burn_in_iters
            for param_group in optimizer.param_groups:
                param_group["lr"] = cfg.lr * burn_factor
        elif cfg.warmup_epochs > 0 and epoch + step / steps_per_epoch < cfg.warmup_epochs:
            warmup_factor = (global_step + 1) / (cfg.warmup_epochs * steps_per_epoch)
            for param_group in optimizer.param_groups:
                param_group["lr"] = cfg.lr * warmup_factor

    scheduler.step()

    return {
        "loss": running_loss / max(steps, 1),
        "avg_step_time": (time.time() - epoch_start) / max(steps, 1),
    }


@torch.no_grad()
def evaluate(model: nn.Module, val_loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    criterion = nn.CrossEntropyLoss()
    steps = 0
    for images, targets in val_loader:
        images = _move_to_device(images, device)
        targets = targets.to(device)
        logits = normalise_logits(model(images))
        loss = criterion(logits, targets)
        top1, top5 = accuracy(logits, targets, topk=(1, 5))
        total_loss += loss.item()
        total_top1 += top1.item()
        total_top5 += top5.item()
        steps += 1
    return {
        "loss": total_loss / max(steps, 1),
        "top1": total_top1 / max(steps, 1),
        "top5": total_top5 / max(steps, 1),
    }


def default_configs() -> Dict[str, TrainConfig]:
    return {
        "resnet34": TrainConfig(
            name="resnet34",
            epochs=100,
            batch_size=256,
            lr=0.1,
            optimizer="sgd",
            weight_decay=1e-4,
            momentum=0.9,
            use_amp=True,
            num_workers=8,
            image_size=224,
            lr_schedule="step",
            lr_milestones=[30, 60, 90],
        ),
        "cspdarknet53": TrainConfig(
            name="cspdarknet53",
            epochs=350,
            batch_size=32,
            lr=0.01,
            optimizer="sgd",
            weight_decay=5e-4,
            momentum=0.937,
            use_amp=True,
            num_workers=8,
            image_size=256,
            lr_schedule="cosine",
            cosine_min_lr=1e-4,
            burn_in_iters=1000,
            multiscale=[320,608],
        ),
        "efficientvit_m4": TrainConfig(
            name="efficientvit_m4",
            epochs=350,
            batch_size=256,
            lr=0.256,
            optimizer="rmsprop",
            weight_decay=1e-5,
            momentum=0.9,
            use_amp=True,
            num_workers=8,
            image_size=224,
            lr_schedule="exponential",
            warmup_epochs=5,
            label_smoothing=0.1,
            mixup_alpha=0.2,
            cutmix_alpha=0.0,
            drop_connect=0.2,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backbone classification trainer")
    parser.add_argument("--model", required=True, choices=list(default_configs().keys()))
    parser.add_argument("--train-dirs", nargs="+", required=True,
                        help="One or more ImageNet-style folders for training.")
    parser.add_argument("--val-dir", required=True,
                        help="Validation folder in ImageNet layout.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="./output/backbone_runs")
    parser.add_argument("--max-train-images", type=int, default=None,
                        help="Limit the number of training samples (approximate).")
    parser.add_argument("--max-val-images", type=int, default=None,
                        help="Limit the number of validation samples (approximate).")
    parser.add_argument("--amp", action="store_true", help="Force AMP on.")
    parser.add_argument("--no-amp", action="store_true", help="Force AMP off.")
    parser.add_argument("--input-format", choices=["rgb", "compressed"], default="rgb",
                        help="Choose between raw RGB tensors and DCT-compressed blocks.")
    parser.add_argument("--compression-range-mode", choices=["studio", "full"], default="studio",
                        help="Pixel range used before DCT when --input-format=compressed.")
    parser.add_argument("--compression-coeff-window", type=int, choices=[1, 2, 4, 8], default=8,
                        help="Low-frequency window size kept per block when using compression.")
    parser.add_argument("--compression-keep-original", action="store_true",
                        help="Alongside DCT blocks, keep the original tensor in the sample payload.")
    parser.add_argument(
        "--compressed-backbone",
        choices=["reconstruction", "block-stem", "luma-fusion", "luma-fusion-pruned"],
        default="reconstruction",
        help="Architecture variant to use when --input-format=compressed.",
    )
    parser.add_argument("--dataset-progress", action="store_true",
                        help="Show a tqdm progress bar while scanning dataset files.")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging (requires wandb package).")
    parser.add_argument("--wandb-project", default="rtdetr-backbones",
                        help="Weights & Biases project name.")
    parser.add_argument("--wandb-entity", default=None,
                        help="Optional Weights & Biases entity/team.")
    parser.add_argument("--wandb-run-name", default=None,
                        help="Optional run name shown in the Weights & Biases UI.")
    parser.add_argument("--wandb-tags", nargs="*", default=None,
                        help="Optional list of tags for the Weights & Biases run.")
    return parser.parse_args()


def main() -> None:
    ensure_cuda_available()
    args = parse_args()
    cfg_map = default_configs()
    cfg = cfg_map[args.model]
    wandb_run = None

    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.amp:
        cfg.use_amp = True
    if args.no_amp:
        cfg.use_amp = False
    cfg.input_format = args.input_format

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dirs = [Path(p).expanduser().resolve() for p in args.train_dirs]
    val_dir = Path(args.val_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model, feat_dim = build_model(cfg.name, num_classes=1000)
    if cfg.name == "resnet34":
        model.apply(kaiming_initialisation)
    model.to(device)

    compression_cfg: Optional[Dict[str, object]] = None
    if cfg.input_format == "compressed":
        if cfg.name != "resnet34":
            raise NotImplementedError("Compressed input path currently supports only ResNet34.")
        compression_cfg = {
            "coeff_window": args.compression_coeff_window,
            "range_mode": args.compression_range_mode,
            "dtype": torch.float32,
            "keep_original": args.compression_keep_original,
        }
        model.backbone = build_compressed_backbone(
            args.compressed_backbone,
            model.backbone,
            range_mode=args.compression_range_mode,
            mean=_IMAGENET_MEAN,
            std=_IMAGENET_STD,
            coeff_window=args.compression_coeff_window,
        )
        if args.compressed_backbone == "luma-fusion-pruned":
            hidden_dim = model.backbone.out_channels[0]
            model.head = ClassHead(hidden_dim=hidden_dim, num_classes=1000).to(device)
        model.to(device)

    train_loader, val_loader = build_dataloaders(
        cfg.name,
        train_dirs,
        val_dir,
        cfg.batch_size,
        cfg.num_workers,
        cfg.image_size,
        args.max_train_images,
        args.max_val_images,
        cfg.input_format,
        compression_cfg,
        args.dataset_progress,
    )

    cfg.lr = adjust_learning_rate(cfg.lr, cfg.batch_size)
    optimizer = create_optimizer(model, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    steps_per_epoch = len(train_loader)
    scheduler = create_scheduler(optimizer, cfg, steps_per_epoch)

    if cfg.label_smoothing > 0.0 or cfg.mixup_alpha > 0.0 or cfg.name == "cspdarknet53":
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss()

    print("Configured run:")
    print(f"  model: {cfg.name}")
    print(f"  epochs: {cfg.epochs}")
    print(f"  batch size: {cfg.batch_size}")
    print(f"  input format: {cfg.input_format}")
    if cfg.input_format == "compressed":
        print(f"    coeff window: {args.compression_coeff_window}")
        print(f"    range mode: {args.compression_range_mode}")
        print(f"    keep original: {args.compression_keep_original}")
        print(f"    backbone variant: {args.compressed_backbone}")
    if args.dataset_progress:
        print("  dataset progress: enabled")
    if args.max_train_images:
        print(f"  capped train samples: ~{len(train_loader.dataset)}")
    if args.max_val_images:
        print(f"  capped val samples: ~{len(val_loader.dataset)}")
    print(f"  base lr: {cfg.lr}")
    print(f"  optimizer: {cfg.optimizer} (momentum={cfg.momentum}, weight_decay={cfg.weight_decay})")
    print(f"  lr schedule: {cfg.lr_schedule}")
    if cfg.lr_schedule == "step":
        print(f"    milestones: {cfg.lr_milestones}")
    elif cfg.lr_schedule == "cosine":
        print(f"    eta_min: {cfg.cosine_min_lr}")
    elif cfg.lr_schedule == "exponential":
        print(f"    warmup_epochs: {cfg.warmup_epochs}")
    if cfg.burn_in_iters > 0:
        print(f"  burn-in iterations: {cfg.burn_in_iters}")
    if cfg.multiscale:
        print(f"  multiscale range: {cfg.multiscale}")
    if cfg.label_smoothing > 0.0:
        print(f"  label smoothing: {cfg.label_smoothing}")
    if cfg.mixup_alpha > 0.0:
        print(f"  mixup alpha: {cfg.mixup_alpha}")
    if cfg.cutmix_alpha > 0.0:
        print(f"  cutmix alpha: {cfg.cutmix_alpha}")
    print(f"  amp: {cfg.use_amp}")
    print(f"  num workers: {cfg.num_workers}")

    best_top1 = 0.0
    for epoch in range(cfg.epochs):
        start = time.time()
        train_stats = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, cfg, 1000, scaler, criterion
        )
        val_stats = evaluate(model, val_loader, device)
        elapsed = time.time() - start

        lr = optimizer.param_groups[0]["lr"]
        eta_epochs = cfg.epochs - epoch - 1
        eta_minutes = eta_epochs * (elapsed / 60.0)
        print(
            f"Epoch {epoch+1:03}/{cfg.epochs} | "
            f"train loss {train_stats['loss']:.4f} | "
            f"val loss {val_stats['loss']:.4f} top1 {val_stats['top1']:.2f} top5 {val_stats['top5']:.2f} | "
            f"lr {lr:.5f} | epoch time {elapsed/60:.2f} min | ETA {eta_minutes:.1f} min"
        )

        if wandb_run is not None:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_stats["loss"],
                "train/avg_step_time": train_stats["avg_step_time"],
                "val/loss": val_stats["loss"],
                "val/top1": val_stats["top1"],
                "val/top5": val_stats["top5"],
                "lr": lr,
                "epoch_time_min": elapsed / 60.0,
            })

        is_best = val_stats["top1"] > best_top1
        if is_best:
            best_top1 = val_stats["top1"]
            if wandb_run is not None:
                wandb_run.summary["best/top1"] = best_top1
                wandb_run.summary["best/epoch"] = epoch + 1

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if cfg.use_amp else None,
            "best_top1": best_top1,
            "config": cfg,
        }
        torch.save(checkpoint, output_dir / "checkpoint_last.pth")
        torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch+1:04}.pth")
        if is_best:
            torch.save(checkpoint, output_dir / "checkpoint_best.pth")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
