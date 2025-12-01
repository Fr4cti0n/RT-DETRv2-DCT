#!/usr/bin/env python3
"""Train a ResNet34 backbone directly on DCT-compressed inputs."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from ...data.dataset.imagenet import ImageNetDataset
from ...data.dataset.subset import limit_total
from ..arch.classification import ClassHead
from .compressed_presnet import build_compressed_backbone
from .train_backbones import (
    _IMAGENET_MEAN,
    _IMAGENET_STD,
    _move_to_device,
    build_model,
    build_resnet_transforms,
    kaiming_initialisation,
    normalise_logits,
)

_NUM_CLASSES = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dirs", nargs="+", required=True,
                        help="ImageNet-style folders used for training.")
    parser.add_argument("--val-dir", required=True,
                        help="Validation folder with ImageNet layout.")
    parser.add_argument("--output-dir", type=Path, default=Path("output/compressed_resnet34"))
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--milestones", type=int, nargs="*", default=[30, 60, 80])
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--coeff-window", type=int, choices=[1, 2, 4, 8], default=4)
    parser.add_argument("--range-mode", choices=["studio", "full"], default="studio")
    parser.add_argument(
        "--variant",
        choices=["reconstruction", "block-stem", "luma-fusion", "luma-fusion-pruned"],
        default="reconstruction",
        help="Compressed backbone adapter to employ.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--device", default=None, help="Optional explicit device string, e.g. cuda:0")
    parser.add_argument("--print-freq", type=int, default=50, help="Steps between training log lines.")
    parser.add_argument("--max-train-images", type=int, default=None,
                        help="Limit number of training samples for quick sanity checks.")
    parser.add_argument("--max-val-images", type=int, default=None,
                        help="Limit number of validation samples for quick sanity checks.")
    return parser.parse_args()


def build_dataloaders(
    train_dirs: list[str],
    val_dir: str,
    image_size: int,
    batch_size: int,
    workers: int,
    compression_cfg: dict,
    show_progress: bool,
    max_train: int | None,
    max_val: int | None,
) -> tuple[DataLoader, DataLoader]:
    train_tf, val_tf = build_resnet_transforms(image_size, compression=compression_cfg)
    train_set = ImageNetDataset(train_dirs, transforms=train_tf, show_progress=show_progress)
    val_set = ImageNetDataset([val_dir], transforms=val_tf, show_progress=show_progress)
    if max_train is not None:
        train_set = limit_total(train_set, max_train)
    if max_val is not None:
        val_set = limit_total(val_set, max_val)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, workers // 2),
        pin_memory=True,
    )
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    use_amp: bool,
    scaler: torch.cuda.amp.GradScaler | None,
    print_freq: int = 50,
) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    last_log = time.time()
    for step, (inputs, targets) in enumerate(loader):
        inputs = _move_to_device(inputs, device)
        targets = targets.to(device)
        optimizer.zero_grad()
        if use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = normalise_logits(model(inputs))
                loss = criterion(outputs, targets)
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = normalise_logits(model(inputs))
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * targets.size(0)
        correct += outputs.argmax(dim=1).eq(targets).sum().item()
        total += targets.size(0)
        if (step + 1) % print_freq == 0 or (step + 1) == len(loader):
            avg_loss = running_loss / max(total, 1)
            acc = 100.0 * correct / max(total, 1)
            now = time.time()
            print(
                f"  epoch {epoch:03d} step {step+1:04}/{len(loader)} | "
                f"loss={avg_loss:.4f} acc@1={acc:.2f}% time={now - last_log:.1f}s"
            )
            last_log = now
    return {
        "loss": running_loss / max(total, 1),
        "acc1": 100.0 * correct / max(total, 1),
    }


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = _move_to_device(inputs, device)
            targets = targets.to(device)
            outputs = normalise_logits(model(inputs))
            loss = criterion(outputs, targets)
            running_loss += loss.item() * targets.size(0)
            correct += outputs.argmax(dim=1).eq(targets).sum().item()
            total += targets.size(0)
    return {
        "loss": running_loss / max(total, 1),
        "acc1": 100.0 * correct / max(total, 1),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    compression_cfg = {
        "coeff_window": args.coeff_window,
        "range_mode": args.range_mode,
        "dtype": torch.float32,
        "keep_original": False,
    }

    train_loader, val_loader = build_dataloaders(
        [str(Path(p)) for p in args.train_dirs],
        str(Path(args.val_dir)),
        args.image_size,
        args.batch_size,
        args.workers,
        compression_cfg,
        args.show_progress,
        args.max_train_images,
        args.max_val_images,
    )

    model, _ = build_model("resnet34", _NUM_CLASSES)
    model.apply(kaiming_initialisation)
    model.backbone = build_compressed_backbone(
        args.variant,
        model.backbone,
        range_mode=args.range_mode,
        mean=_IMAGENET_MEAN,
        std=_IMAGENET_STD,
        coeff_window=args.coeff_window,
    )
    if args.variant == "luma-fusion-pruned":
        hidden_dim = model.backbone.out_channels[0]
        model.head = ClassHead(hidden_dim=hidden_dim, num_classes=_NUM_CLASSES)
        model.head.apply(kaiming_initialisation)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            use_amp=args.amp,
            scaler=scaler,
            print_freq=args.print_freq,
        )
        val_stats = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        acc1 = val_stats["acc1"]
        print(
            f"  train loss={stats['loss']:.4f} acc@1={stats['acc1']:.2f}% | "
            f"val loss={val_stats['loss']:.4f} acc@1={val_stats['acc1']:.2f}%"
        )
        if acc1 > best_acc:
            best_acc = acc1
            if args.save_best:
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "variant": args.variant,
                        "coeff_window": args.coeff_window,
                        "range_mode": args.range_mode,
                        "best_acc": best_acc,
                    },
                    output_dir / "model_best.pth",
                )
                print(f"  Saved new best checkpoint acc@1={best_acc:.2f}%")
        if args.save_every and epoch % args.save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "variant": args.variant,
                    "coeff_window": args.coeff_window,
                    "range_mode": args.range_mode,
                    "best_acc": best_acc,
                },
                output_dir / f"checkpoint_{epoch:04d}.pth",
            )

    print(f"Training complete. Best val acc@1={best_acc:.2f}%")


if __name__ == "__main__":
    main()
