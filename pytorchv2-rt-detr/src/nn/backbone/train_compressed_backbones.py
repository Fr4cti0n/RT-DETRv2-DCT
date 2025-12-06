#!/usr/bin/env python3
"""Train a ResNet34 backbone directly on DCT-compressed inputs."""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T

from ...data.dataset.imagenet import ImageNetDataset
from ...data.dataset.subset import limit_total
from ...data.transforms.dct_normalize import NormalizeDCTCoefficients
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
try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    wandb = None
_NUM_CLASSES = 1000
_LR_REFERENCE_BATCH = 256
_LR_CAP = 0.4


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
    parser.add_argument("--eta-min", type=float, default=0.0,
                        help="Minimum learning rate placeholder (unused with step schedule; retained for CLI compatibility).")
    parser.add_argument("--warmup-epochs", type=int, default=0,
                        help="Number of linear warmup epochs applied before cosine decay.")
    parser.add_argument("--coeff-window", type=int, choices=[1, 2, 4, 8], default=4)
    parser.add_argument("--range-mode", choices=["studio", "full"], default="studio")
    parser.add_argument(
        "--variant",
        choices=["reconstruction", "block-stem", "luma-fusion", "luma-fusion-pruned"],
        default="reconstruction",
        help="Compressed backbone adapter to employ.",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument("--save-last", action=argparse.BooleanOptionalAction, default=True,
                        help="Persist checkpoint_last.pth each epoch (use --no-save-last to disable).")
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
                        help="Optional directory dedicated to checkpoints (defaults to output dir).")
    parser.add_argument("--auto-resume", action="store_true",
                        help="Resume automatically from checkpoint_dir/checkpoint_last.pth when available.")
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--device", default=None, help="Optional explicit device string, e.g. cuda:0")
    parser.add_argument("--print-freq", type=int, default=50, help="Steps between training log lines.")
    parser.add_argument("--max-train-images", type=int, default=None,
                        help="Limit number of training samples for quick sanity checks.")
    parser.add_argument("--max-val-images", type=int, default=None,
                        help="Limit number of validation samples for quick sanity checks.")
    parser.add_argument("--preview-sample", action="store_true",
                        help="Decode and save a reconstruction preview before training (batch size 1 recommended).")
    parser.add_argument("--preview-output", type=Path, default=None,
                        help="Optional filepath for the reconstruction preview image.")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging (requires wandb package).")
    parser.add_argument("--wandb-project", default="rtdetr-compressed",
                        help="Weights & Biases project name.")
    parser.add_argument("--wandb-entity", default=None,
                        help="Optional Weights & Biases entity/team.")
    parser.add_argument("--wandb-run-name", default=None,
                        help="Optional run name shown in the Weights & Biases UI.")
    parser.add_argument("--wandb-tags", nargs="*", default=None,
                        help="Optional list of tags for the Weights & Biases run.")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Path to a checkpoint to resume training from.")
    parser.add_argument("--dct-stats", type=Path, default=None,
                        help="Optional path to a .pt file with per-coefficient DCT statistics for normalisation.")
    parser.add_argument("--lr-milestones", type=int, nargs="*", default=[30, 60, 90],
                        help="Epoch milestones (1-indexed) where the learning rate is multiplied by lr_gamma.")
    parser.add_argument("--lr-gamma", type=float, default=0.1,
                        help="Multiplicative factor applied to the learning rate at each milestone (default: 0.1).")
    parser.add_argument("--time-limit-hours", type=float, default=1.0,
                        help="Maximum wall-clock time for training (hours, 0 disables the limit).")
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
    dct_normalizer_train: T.Transform | None = None,
    dct_normalizer_val: T.Transform | None = None,
) -> tuple[DataLoader, DataLoader]:
    train_tf, val_tf = build_resnet_transforms(
        image_size,
        compression=compression_cfg,
        dct_normalizer_train=dct_normalizer_train,
        dct_normalizer_val=dct_normalizer_val,
    )
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


def preview_reconstruction_sample(
    backbone: nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_path: Path,
) -> None:
    try:
        batch = next(iter(loader))
    except StopIteration:
        print("[preview] Training dataset is empty; skipping reconstruction preview.")
        return

    inputs, _ = batch
    if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
        print("[preview] Unexpected payload structure; reconstruction preview is only supported for DCT inputs.")
        return

    y_blocks, cbcr_blocks = inputs
    if y_blocks.dim() != 4 or cbcr_blocks.dim() != 5:
        print("[preview] Unexpected coefficient tensor shape; skipping reconstruction preview.")
        return

    y_blocks = y_blocks.to(device=device, dtype=torch.float32)
    cbcr_blocks = cbcr_blocks.to(device=device, dtype=torch.float32)

    backbone = backbone.to(device)
    backbone.eval()
    with torch.no_grad():
        try:
            y_plane, cb_plane, cr_plane = backbone._decode_planes(y_blocks, cbcr_blocks)  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - defensive logging path
            print(f"[preview] Failed to decode DCT payload: {exc}")
            return
        rgb = backbone._ycbcr_to_rgb(y_plane, cb_plane, cr_plane)  # type: ignore[attr-defined]
        refined = backbone.refine(rgb)  # type: ignore[attr-defined]

    rgb_cpu = rgb[0].detach().cpu().clamp(0.0, 1.0)
    refined_cpu = refined[0].detach().cpu()
    refined_clamped = refined_cpu.clamp(0.0, 1.0)
    diff = (refined_cpu - rgb_cpu).abs().mean(dim=0)
    diff_max = float(diff.max().item() if diff.numel() > 0 else 0.0)
    if diff_max > 0.0:
        diff_vis = (diff / diff_max).clamp(0.0, 1.0)
    else:
        diff_vis = diff

    # Lazy import so matplotlib is only required when previewing.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(rgb_cpu.permute(1, 2, 0).numpy())
    axes[0].set_title("Decoded RGB (pre-refine)")
    axes[1].imshow(refined_clamped.permute(1, 2, 0).numpy())
    axes[1].set_title("Refine Output (clamped)")
    axes[2].imshow(diff_vis.numpy(), cmap="magma")
    axes[2].set_title("|Refine - Decoded| (mean)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[preview] Saved reconstruction preview to {save_path}")
    print(
        "[preview] decoded_rgb: min={:.4f} max={:.4f} mean={:.4f}".format(
            float(rgb_cpu.min()), float(rgb_cpu.max()), float(rgb_cpu.mean())
        )
    )
    print(
        "[preview] refined   : min={:.4f} max={:.4f} mean={:.4f}".format(
            float(refined_cpu.min()), float(refined_cpu.max()), float(refined_cpu.mean())
        )
    )
    print(
        "[preview] |refined-decoded| mean={:.4f} max={:.4f}".format(
            float(diff.mean().item()), diff_max
        )
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    run_start_time: float,
    use_amp: bool,
    scaler: torch.cuda.amp.GradScaler | None,
    print_freq: int = 50,
    time_limit_seconds: float | None = None,
) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    correct1 = 0
    correct5 = 0
    total = 0
    last_log = time.time()
    epoch_start_time = time.time()
    time_limit_reached = False
    for step, (inputs, targets) in enumerate(loader):
        if time_limit_seconds and time.time() - run_start_time >= time_limit_seconds:
            time_limit_reached = True
            break
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
        total += targets.size(0)

        maxk = min(5, outputs.size(1))
        _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct_matrix = pred.eq(targets.view(1, -1).expand_as(pred))
        correct1 += correct_matrix[:1].reshape(-1).float().sum().item()
        correct5 += correct_matrix[:5].reshape(-1).float().sum().item()
        if (step + 1) % print_freq == 0 or (step + 1) == len(loader):
            avg_loss = running_loss / max(total, 1)
            acc1 = correct1 / max(total, 1)
            acc5 = correct5 / max(total, 1)
            now = time.time()
            step_time = now - last_log
            elapsed = now - epoch_start_time
            progress = (step + 1) / len(loader)
            remaining_epoch = (elapsed / max(progress, 1e-6)) - elapsed
            lr = optimizer.param_groups[0]["lr"]
            gpu_mem = None
            if device.type == "cuda":
                gpu_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            global_elapsed = now - run_start_time
            steps_per_epoch = len(loader)
            total_steps = total_epochs * steps_per_epoch
            completed_steps = (epoch - 1) * steps_per_epoch + (step + 1)
            progress_total = completed_steps / max(total_steps, 1)
            remaining_total = (global_elapsed / max(progress_total, 1e-6)) - global_elapsed
            msg = (
                f"  epoch {epoch:03d} step {step+1:04}/{len(loader)} | "
                f"loss={avg_loss:.4f} acc@1={acc1:.4f} acc@5={acc5:.4f} "
                f"lr={lr:.5f} step={step_time:.1f}s "
                f"eta_epoch={remaining_epoch/60:.1f}m eta_total={remaining_total/3600:.2f}h"
            )
            if gpu_mem is not None:
                msg += f" gpu={gpu_mem:.1f}MB"
            print(msg)
            last_log = now
    return {
        "loss": running_loss / max(total, 1),
        "acc1": correct1 / max(total, 1),
        "acc5": correct5 / max(total, 1),
        "time_limit_reached": time_limit_reached,
    }


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, float]:
    model.eval()
    running_loss = 0.0
    correct1 = 0
    correct5 = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = _move_to_device(inputs, device)
            targets = targets.to(device)
            outputs = normalise_logits(model(inputs))
            loss = criterion(outputs, targets)
            running_loss += loss.item() * targets.size(0)
            maxk = min(5, outputs.size(1))
            _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
            pred = pred.t()
            correct_matrix = pred.eq(targets.view(1, -1).expand_as(pred))
            correct1 += correct_matrix[:1].reshape(-1).float().sum().item()
            correct5 += correct_matrix[:5].reshape(-1).float().sum().item()
            total += targets.size(0)
    return {
        "loss": running_loss / max(total, 1),
        "acc1": correct1 / max(total, 1),
        "acc5": correct5 / max(total, 1),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    wandb_run = None

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

    normalizer_train: T.Transform | None = None
    normalizer_val: T.Transform | None = None
    stats_path_input: Path | None = args.dct_stats
    if args.variant == "reconstruction":
        if stats_path_input is not None:
            print("Reconstruction variant ignores --dct-stats; proceeding without coefficient normalisation.")
        stats_path_input = None
    else:
        if stats_path_input is None:
            default_stats = Path("configs/dct_stats") / f"imagenet_coeff{args.coeff_window}_{args.range_mode}.pt"
            if default_stats.exists():
                stats_path_input = default_stats
    stats_path_resolved: Path | None = None
    if stats_path_input is not None:
        try:
            stats_path_resolved = stats_path_input.expanduser().resolve()
            normalizer_train = NormalizeDCTCoefficients.from_file(
                stats_path_resolved,
                coeff_window=args.coeff_window,
            )
            normalizer_val = NormalizeDCTCoefficients.from_file(
                stats_path_resolved,
                coeff_window=args.coeff_window,
            )
            print(f"Loaded DCT coefficient statistics from {stats_path_resolved}")
        except Exception as exc:  # pragma: no cover - safeguard for user-provided files
            print(f"Failed to load DCT stats from {stats_path_input}: {exc}")
            stats_path_resolved = None
    if stats_path_resolved is None:
        print("Proceeding without DCT coefficient normalisation.")

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
        normalizer_train,
        normalizer_val,
    )

    effective_lr = args.lr * args.batch_size / _LR_REFERENCE_BATCH
    if effective_lr > _LR_CAP:
        effective_lr = _LR_CAP
    if effective_lr != args.lr:
        print(
            f"Adjusted learning rate from {args.lr:.5f} (base) to {effective_lr:.5f} "
            f"for batch size {args.batch_size} (cap={_LR_CAP:.5f})."
        )
    milestones = sorted(m for m in args.lr_milestones if m > 0)
    if not milestones:
        milestones = [30, 60, 80]
    warmup_epochs = max(0, args.warmup_epochs)
    if warmup_epochs > 0:
        print(f"Applying {warmup_epochs} warmup epoch(s) before step LR milestones {milestones}.")
    time_limit_hours = args.time_limit_hours if args.time_limit_hours > 0 else None
    if time_limit_hours is not None:
        print(f"Enforcing wall-clock time limit of {time_limit_hours:.2f}h.")
    time_limit_seconds = time_limit_hours * 3600.0 if time_limit_hours is not None else None

    if args.wandb:
        if wandb is None:
            raise RuntimeError(
                "Weights & Biases is not installed. Run 'pip install wandb' to enable logging."
            )
        wandb_config = {
            "variant": args.variant,
            "coeff_window": args.coeff_window,
            "range_mode": args.range_mode,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "lr_effective": effective_lr,
            "lr_reference_batch": _LR_REFERENCE_BATCH,
            "lr_cap": _LR_CAP,
            "warmup_epochs": warmup_epochs,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "eta_min": args.eta_min,
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
            "seed": args.seed,
            "train_dirs": [str(p) for p in args.train_dirs],
            "val_dir": str(args.val_dir),
            "channels_last": args.channels_last,
            "dct_stats_path": str(stats_path_resolved) if stats_path_resolved is not None else None,
            "time_limit_hours": time_limit_hours,
        }
        default_name = f"{args.variant}_{args.coeff_window}"
        run_name = args.wandb_run_name or default_name
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=run_name,
            tags=args.wandb_tags,
            config=wandb_config,
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
        lr=effective_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_gamma)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    run_subdir_name = f"{args.variant}_coeff{args.coeff_window}"
    base_output_dir = args.output_dir
    base_checkpoint_dir = args.checkpoint_dir or base_output_dir

    resume_path: Path | None = None
    if args.resume is not None:
        resume_path = args.resume.expanduser()
        if resume_path.is_dir():
            resume_path = resume_path / "checkpoint_last.pth"
        resume_dir = resume_path.parent
        output_dir = resume_dir
        checkpoint_dir = resume_dir
    else:
        output_dir = base_output_dir / run_subdir_name
        checkpoint_dir = base_checkpoint_dir / run_subdir_name

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    csv_path = checkpoint_dir / "training_params.csv"
    best_acc = 0.0
    start_epoch = 1

    if resume_path is None and args.auto_resume:
        candidate = checkpoint_dir / "checkpoint_last.pth"
        if candidate.exists():
            resume_path = candidate
    if resume_path is not None:
        if resume_path.is_dir():
            resume_path = resume_path / "checkpoint_last.pth"
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if scaler is not None and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        best_acc = float(checkpoint.get("best_acc", 0.0))
        last_epoch = int(checkpoint.get("epoch", 0))
        start_epoch = last_epoch + 1
        print(f"Resumed training from epoch {last_epoch} with best acc@1={best_acc:.4f} ({resume_path})")
        if start_epoch > args.epochs:
            print("Checkpoint epoch exceeds requested total epochs; nothing to train.")
            if wandb_run is not None:
                wandb_run.finish()
            return

    timestamp_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    params_to_record = {
        "variant": args.variant,
        "coeff_window": args.coeff_window,
        "range_mode": args.range_mode,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "start_epoch": start_epoch,
        "lr": args.lr,
        "effective_lr": effective_lr,
        "lr_milestones": " ".join(str(m) for m in milestones),
        "lr_gamma": args.lr_gamma,
        "warmup_epochs": warmup_epochs,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "eta_min": args.eta_min,
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "seed": args.seed,
        "train_dirs": ";".join(str(p) for p in args.train_dirs),
        "val_dir": str(args.val_dir),
        "channels_last": args.channels_last,
        "dct_stats_path": str(stats_path_resolved) if stats_path_resolved is not None else "",
        "time_limit_hours": time_limit_hours if time_limit_hours is not None else "none",
        "output_dir": str(output_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "resume_checkpoint": str(resume_path) if resume_path is not None else "",
        "timestamp": timestamp_now,
    }
    csv_exists = csv_path.exists()
    with csv_path.open("a" if csv_exists else "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if not csv_exists:
            writer.writerow(["parameter", "value"])
            for key, value in params_to_record.items():
                writer.writerow([key, value])
        else:
            writer.writerow(["resume_timestamp", timestamp_now])
            writer.writerow(["resume_start_epoch", start_epoch])
            if resume_path is not None:
                writer.writerow(["resume_checkpoint", str(resume_path)])

    run_start_time = time.time()

    time_limit_triggered = False

    if args.preview_sample:
        if args.variant != "reconstruction":
            print("[preview] Reconstruction preview currently supports only the reconstruction variant.")
        else:
            preview_path = args.preview_output or (output_dir / "preview_reconstruction.png")
            preview_reconstruction_sample(model.backbone, train_loader, device, preview_path)
        if args.epochs <= 0:
            print("Preview complete and epochs<=0; exiting without training.")
            if wandb_run is not None:
                wandb_run.finish()
            return

    for epoch in range(start_epoch, args.epochs + 1):
        if time_limit_seconds and time.time() - run_start_time >= time_limit_seconds:
            print("Time limit reached before starting new epoch; stopping training loop.")
            time_limit_triggered = True
            break
        print(f"Epoch {epoch}/{args.epochs}")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        warmup_active = False
        warmup_factor = 1.0
        if warmup_epochs > 0:
            warmup_idx = epoch - 1
            if warmup_idx < warmup_epochs:
                warmup_active = True
                warmup_factor = (warmup_idx + 1) / warmup_epochs
                warmup_lr = effective_lr * warmup_factor
                for param_group in optimizer.param_groups:
                    param_group["lr"] = warmup_lr
        stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            args.epochs,
            run_start_time,
            use_amp=args.amp,
            scaler=scaler,
            print_freq=args.print_freq,
            time_limit_seconds=time_limit_seconds,
        )
        if stats.get("time_limit_reached"):
            print("Time limit reached during training phase; terminating without evaluation.")
            time_limit_triggered = True
            break
        val_stats = evaluate(model, val_loader, criterion, device)
        if time_limit_seconds and time.time() - run_start_time >= time_limit_seconds:
            print("Time limit reached after evaluation; stopping training loop.")
            time_limit_triggered = True
            current_lr = optimizer.param_groups[0]["lr"]
            next_lr = current_lr
            acc1 = val_stats["acc1"]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                gpu_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            else:
                gpu_mem = None
            print(
                f"  train loss={stats['loss']:.4f} acc@1={stats['acc1']:.4f} acc@5={stats['acc5']:.4f} | "
                f"val loss={val_stats['loss']:.4f} acc@1={val_stats['acc1']:.4f} acc@5={val_stats['acc5']:.4f}"
                + (f" gpu_mem={gpu_mem:.1f}MB" if gpu_mem is not None else "")
            )
            break
        current_lr = optimizer.param_groups[0]["lr"]
        if not warmup_active:
            scheduler.step()
        next_lr = optimizer.param_groups[0]["lr"]
        acc1 = val_stats["acc1"]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            gpu_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        else:
            gpu_mem = None
        print(
            f"  train loss={stats['loss']:.4f} acc@1={stats['acc1']:.4f} acc@5={stats['acc5']:.4f} | "
            f"val loss={val_stats['loss']:.4f} acc@1={val_stats['acc1']:.4f} acc@5={val_stats['acc5']:.4f}"
            + (f" gpu_mem={gpu_mem:.1f}MB" if gpu_mem is not None else "")
        )
        if wandb_run is not None:
            wandb.log({
                "epoch": epoch,
                "train/loss": stats["loss"],
                "train/acc1": stats["acc1"],
                "train/acc5": stats["acc5"],
                "val/loss": val_stats["loss"],
                "val/acc1": val_stats["acc1"],
                "val/acc5": val_stats["acc5"],
                "lr": current_lr,
                "lr_next": next_lr,
                "warmup_factor": warmup_factor,
                **({"gpu/peak_mem_mb": gpu_mem} if gpu_mem is not None else {}),
            })
        if acc1 > best_acc:
            best_acc = acc1
            if wandb_run is not None:
                wandb_run.summary["best/acc1"] = best_acc
                wandb_run.summary["best/acc5"] = val_stats["acc5"]
                wandb_run.summary["best/epoch"] = epoch
            if args.save_best:
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict() if scaler is not None else None,
                        "variant": args.variant,
                        "coeff_window": args.coeff_window,
                        "range_mode": args.range_mode,
                        "best_acc": best_acc,
                    },
                    checkpoint_dir / "model_best.pth",
                )
                print(f"  Saved new best checkpoint acc@1={best_acc:.4f}")
        if args.save_last:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "variant": args.variant,
                    "coeff_window": args.coeff_window,
                    "range_mode": args.range_mode,
                    "best_acc": best_acc,
                },
                checkpoint_dir / "checkpoint_last.pth",
            )
        if args.save_every and epoch % args.save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict() if scaler is not None else None,
                    "variant": args.variant,
                    "coeff_window": args.coeff_window,
                    "range_mode": args.range_mode,
                    "best_acc": best_acc,
                },
                checkpoint_dir / f"checkpoint_{epoch:04d}.pth",
            )

    print(f"Training complete. Best val acc@1={best_acc:.4f}")
    if time_limit_triggered:
        print("Stopped early due to the configured time limit.")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
