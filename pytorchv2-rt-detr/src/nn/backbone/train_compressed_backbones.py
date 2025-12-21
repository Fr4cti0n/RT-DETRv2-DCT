#!/usr/bin/env python3
"""Train a ResNet34 backbone directly on DCT-compressed inputs."""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import time
from pathlib import Path

import torch
from torch import nn
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torchvision.transforms import v2 as T

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

from ...data.dataset.imagenet import ImageNetDataset
from ...data.dataset.subset import limit_total
from ...data.transforms.dct_normalize import NormalizeDCTCoefficients
from ...misc.dct_coefficients import resolve_coefficient_config
from ..arch.classification import ClassHead
from .compressed_presnet import build_compressed_backbone
from .inference_benchmark import (
    BenchmarkResult,
    build_trimmed_eval_loader,
    run_trimmed_inference_benchmark,
)
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
_LR_REFERENCE_BATCH = 512
_LR_CAP = 0.1

_PRINTED_INPUT_SHAPES = {"train": False, "eval": False}
_WARNED_NO_TQDM = False

def _describe_tensor(tensor: torch.Tensor) -> str:
    shape = "x".join(str(dim) for dim in tensor.shape)
    return f"Tensor[{shape}]({tensor.dtype})"


def _describe_inputs(nested) -> str:
    if torch.is_tensor(nested):
        return _describe_tensor(nested)
    if isinstance(nested, (list, tuple)):
        inner = ", ".join(_describe_inputs(item) for item in nested)
        bracket_open, bracket_close = ("(", ")") if isinstance(nested, tuple) else ("[", "]")
        return f"{bracket_open}{inner}{bracket_close}"
    if isinstance(nested, dict):
        items = ", ".join(f"{key}: {_describe_inputs(value)}" for key, value in nested.items())
        return f"{{{items}}}"
    return repr(type(nested))


def _parse_coeff_window(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:  # pragma: no cover - argparse already handles format errors
        raise argparse.ArgumentTypeError("coefficient window must be an integer") from exc
    if not 1 <= parsed <= 8:
        raise argparse.ArgumentTypeError("coefficient window must be within [1, 8]")
    return parsed


def _parse_coeff_count(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("coefficient count must be an integer") from exc
    if not 0 <= parsed <= 64:
        raise argparse.ArgumentTypeError("coefficient count must be within [0, 64]")
    return parsed


def _parse_positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _init_distributed_mode(args: argparse.Namespace) -> None:
    env_has_dist = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    requested_dist = getattr(args, "distributed", False)
    if not (requested_dist or env_has_dist):
        args.distributed = False
        args.world_size = 1
        args.rank = 0
        args.local_rank = 0
        return

    if args.dist_url == "env://" and env_has_dist:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        args.rank = int(getattr(args, "rank", 0))
        args.world_size = int(getattr(args, "world_size", 1))
        args.local_rank = int(getattr(args, "local_rank", 0))
    args.distributed = True

    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)

    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    dist.barrier()
    args.world_size = dist.get_world_size()
    args.rank = dist.get_rank()


def _cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _is_main_process(args: argparse.Namespace) -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return not getattr(args, "distributed", False) or getattr(args, "rank", 0) == 0


def _resolve_coefficient_args(args: argparse.Namespace) -> None:
    coeff_cfg = resolve_coefficient_config(
        coeff_window=args.coeff_window,
        coeff_count=args.coeff_count,
        coeff_window_luma=args.coeff_window_luma,
        coeff_count_luma=args.coeff_count_luma,
        coeff_window_chroma=args.coeff_window_chroma,
        coeff_count_chroma=args.coeff_count_chroma,
        coeff_window_cb=args.coeff_window_cb,
        coeff_window_cr=args.coeff_window_cr,
        coeff_count_cb=args.coeff_count_cb,
        coeff_count_cr=args.coeff_count_cr,
    )
    args.coeff_count = coeff_cfg["coeff_count"]
    args.coeff_count_luma = coeff_cfg["coeff_count_luma"]
    args.coeff_count_chroma = coeff_cfg["coeff_count_chroma"]
    args.coeff_count_cb = coeff_cfg["coeff_count_cb"]
    args.coeff_count_cr = coeff_cfg["coeff_count_cr"]
    args.coeff_window = coeff_cfg["coeff_window"]
    args.coeff_window_luma = coeff_cfg["coeff_window_luma"]
    args.coeff_window_chroma = coeff_cfg["coeff_window_chroma"]
    args.coeff_window_cb = coeff_cfg["coeff_window_cb"]
    args.coeff_window_cr = coeff_cfg["coeff_window_cr"]


def _load_checkpoint_compression_config(checkpoint_path: Path) -> dict[str, object]:
    keys_to_extract = {
        "coeff_count",
        "coeff_count_luma",
        "coeff_count_chroma",
        "coeff_count_cb",
        "coeff_count_cr",
        "coeff_window",
        "coeff_window_luma",
        "coeff_window_chroma",
        "coeff_window_cb",
        "coeff_window_cr",
        "trim_coefficients",
    }
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:  # pragma: no cover - best effort metadata load
        print(f"[resume] Warning: failed to inspect checkpoint at {checkpoint_path}: {exc}")
        return {}
    return {key: checkpoint[key] for key in keys_to_extract if key in checkpoint}


def _apply_checkpoint_compression_config(args: argparse.Namespace, config: dict[str, object]) -> None:
    def _assign_optional_int(attr: str) -> None:
        if attr not in config:
            return
        value = config[attr]
        if value is None:
            return
        try:
            stored = int(value)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return
        current = getattr(args, attr, None)
        if current is None:
            setattr(args, attr, stored)
        elif current != stored:
            print(
                f"[resume] Warning: CLI {attr}={current} differs from checkpoint value {stored}; "
                "proceeding with CLI value."
            )

    def _assign_optional_window(attr: str) -> None:
        if attr not in config:
            return
        value = config[attr]
        if value is None:
            return
        try:
            stored = int(value)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return
        current = getattr(args, attr, None)
        if current is None:
            setattr(args, attr, stored)
        elif current != stored:
            print(
                f"[resume] Warning: CLI {attr}={current} differs from checkpoint value {stored}; "
                "proceeding with CLI value."
            )

    def _assign_bool(attr: str) -> None:
        if attr not in config:
            return
        value = config[attr]
        if value is None:
            return
        stored = bool(value)
        current = getattr(args, attr, None)
        if current != stored:
            print(
                f"[resume] Info: aligning {attr} from {current} to checkpoint value {stored} for resume compatibility."
            )
            setattr(args, attr, stored)

    for attr in (
        "coeff_count",
        "coeff_count_luma",
        "coeff_count_chroma",
        "coeff_count_cb",
        "coeff_count_cr",
    ):
        _assign_optional_int(attr)
    for attr in (
        "coeff_window",
        "coeff_window_luma",
        "coeff_window_chroma",
        "coeff_window_cb",
        "coeff_window_cr",
    ):
        _assign_optional_window(attr)
    _assign_bool("trim_coefficients")


def _find_latest_checkpoint(base_dir: Path, variant: str) -> Path | None:
    if not base_dir.exists():
        return None
    candidates: list[tuple[float, Path]] = []
    prefix = f"{variant}_"
    for item in base_dir.iterdir():
        if not item.is_dir():
            continue
        if not item.name.startswith(prefix):
            continue
        checkpoint_path = item / "checkpoint_last.pth"
        if not checkpoint_path.exists():
            continue
        try:
            mtime = checkpoint_path.stat().st_mtime
        except OSError:
            continue
        candidates.append((mtime, checkpoint_path))
    if not candidates:
        return None
    candidates.sort(key=lambda entry: entry[0], reverse=True)
    return candidates[0][1]


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
    parser.add_argument("--prefetch-factor", type=_parse_positive_int, default=None,
                        help="Override DataLoader prefetch factor when workers > 0 (default: framework default).")
    parser.add_argument("--persistent-workers", action="store_true",
                        help="Enable persistent worker processes for the train/val DataLoaders (requires workers > 0).")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eta-min", type=float, default=0.0,
                        help="Minimum learning rate placeholder (unused with step schedule; retained for CLI compatibility).")
    parser.add_argument("--warmup-epochs", type=int, default=0,
                        help="Number of linear warmup epochs applied before cosine decay.")
    parser.add_argument("--coeff-window", type=_parse_coeff_window, default=None,
                        help="Deprecated square window of lowest-frequency coefficients (defaults to all coefficients).")
    parser.add_argument("--coeff-window-luma", type=_parse_coeff_window, default=None,
                        help="Deprecated override for the luma window (defaults to --coeff-window).")
    parser.add_argument("--coeff-window-chroma", type=_parse_coeff_window, default=None,
                        help="Deprecated override for the chroma window (defaults to --coeff-window).")
    parser.add_argument("--coeff-window-cb", type=_parse_coeff_window, default=None,
                        help="Override the Cb coefficient window (defaults to chroma window).")
    parser.add_argument("--coeff-window-cr", type=_parse_coeff_window, default=None,
                        help="Override the Cr coefficient window (defaults to chroma window).")
    parser.add_argument("--coeff-count", type=_parse_coeff_count, default=None,
                        help="Total number of luma coefficients per block to retain (default: 64).")
    parser.add_argument("--coeff-count-luma", type=_parse_coeff_count, default=None,
                        help="Override the luma coefficient count (defaults to --coeff-count).")
    parser.add_argument("--coeff-count-chroma", type=_parse_coeff_count, default=None,
                        help="Override the chroma coefficient count (defaults to luma count).")
    parser.add_argument("--coeff-count-cb", type=_parse_coeff_count, default=None,
                        help="Override the Cb coefficient count (defaults to chroma count).")
    parser.add_argument("--coeff-count-cr", type=_parse_coeff_count, default=None,
                        help="Override the Cr coefficient count (defaults to chroma count).")
    parser.add_argument("--trim-coefficients", action=argparse.BooleanOptionalAction, default=True,
                        help="Reduce DCT payload depth to the configured coefficient counts (use --no-trim-coefficients to keep 64).")
    parser.add_argument("--range-mode", choices=["studio", "full"], default="studio")
    parser.add_argument("--dct-device", default="cpu",
                        help="Device used for the DCT compression step (e.g. 'cpu', 'cuda', 'cuda:0', 'auto').")
    parser.add_argument(
        "--variant",
        choices=["reconstruction", "block-stem", "luma-fusion", "luma-fusion-pruned"],
        default="reconstruction",
        help="Compressed backbone adapter to employ.",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--profile-steps",
        action="store_true",
        help="Collect per-batch timing for data loading, device transfer, forward, and backward passes.",
    )
    parser.add_argument("--compile-model", action="store_true",
                        help="Compile the model with torch.compile for potential speedups (PyTorch 2.0+).")
    parser.add_argument("--compile-mode", choices=["default", "reduce-overhead", "max-autotune"], default="reduce-overhead",
                        help="Torch compile mode to use when --compile-model is set.")
    parser.add_argument("--matmul-precision", choices=["high", "highest", "medium"], default=None,
                        help="Override torch.set_float32_matmul_precision for matmul kernels.")
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
    parser.add_argument("--benchmark", dest="benchmark_enabled", action=argparse.BooleanOptionalAction, default=True,
                        help="Run trimmed-input inference benchmarking after training (use --no-benchmark to skip).")
    parser.add_argument("--benchmark-validate", action=argparse.BooleanOptionalAction, default=False,
                        help="Collect top-1/top-5 accuracy and loss during benchmarking (use --no-benchmark-validate to skip metrics).")
    parser.add_argument("--benchmark-disable", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--benchmark-max-samples", type=int, default=128,
                        help="Maximum validation samples used during trimmed-input benchmarking (default: 128).")
    parser.add_argument("--benchmark-warmup-batches", type=int, default=5,
                        help="Warmup batches discarded before timing inference (default: 5).")
    parser.add_argument("--benchmark-measure-batches", type=int, default=20,
                        help="Number of timed batches during trimmed-input benchmarking (default: 20).")
    parser.add_argument("--benchmark-workers", type=int, default=2,
                        help="Number of workers for trimmed-input benchmarking (default: 2).")
    parser.add_argument("--trimmed-val-disable", action="store_true",
                        help="Skip accuracy evaluation using the trimmed-input dataloader.")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training and only run validation/benchmarking (requires --resume for pretrained weights).")
    parser.add_argument("--distributed", action="store_true",
                        help="Enable multi-GPU distributed data parallel training (torch.distributed).")
    parser.add_argument("--dist-backend", default="nccl",
                        help="Distributed backend to use (default: nccl).")
    parser.add_argument("--dist-url", default="env://",
                        help="URL used to set up distributed training (default: env:// for torchrun).")
    parser.add_argument("--world-size", type=int, default=1,
                        help="Number of processes participating in the job.")
    parser.add_argument("--rank", type=int, default=0,
                        help="Rank of the current process.")
    parser.add_argument("--local-rank", type=int, default=0,
                        help="Local rank passed by torchrun/torch.distributed.launch.")
    parser.add_argument("--ddp-static-graph", action="store_true",
                        help="Enable DistributedDataParallel static_graph optimisation for stable workloads.")
    args = parser.parse_args()
    if args.resume is not None:
        resume_config_path = args.resume.expanduser()
        if resume_config_path.is_dir():
            resume_config_path = resume_config_path / "checkpoint_last.pth"
        if resume_config_path.exists():
            resume_config = _load_checkpoint_compression_config(resume_config_path)
            if resume_config:
                _apply_checkpoint_compression_config(args, resume_config)
                if args.variant == "reconstruction" and args.trim_coefficients:
                    print(
                        "[resume] Reconstruction variant requires full coefficient depth; overriding stored trim setting."
                    )
                    args.trim_coefficients = False
    if getattr(args, "benchmark_disable", False):
        print("[cli] --benchmark-disable is deprecated; use --no-benchmark instead.")
        args.benchmark_enabled = False
    if not hasattr(args, "benchmark_enabled"):
        args.benchmark_enabled = True
    if not hasattr(args, "benchmark_validate"):
        args.benchmark_validate = False
    if not args.benchmark_enabled:
        args.benchmark_validate = False
    if args.variant == "reconstruction" and args.trim_coefficients:
        print("[cli] Reconstruction variant requires full coefficient depth; disabling coefficient trimming.")
        args.trim_coefficients = False
    _resolve_coefficient_args(args)
    return args


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
    trim_coefficients: bool = False,
    prefetch_factor: int | None = None,
    persistent_workers: bool = False,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, DataLoader]:
    dct_device_spec = compression_cfg.get("dct_device") if isinstance(compression_cfg, dict) else None
    dct_device_name = None
    if isinstance(dct_device_spec, torch.device):
        dct_device_name = dct_device_spec.type
    elif isinstance(dct_device_spec, str):
        dct_device_name = dct_device_spec.lower()
    needs_spawn = False
    use_cuda_dct = False
    if isinstance(dct_device_spec, torch.device) and dct_device_spec.type == "cuda":
        use_cuda_dct = True
    elif isinstance(dct_device_name, str) and dct_device_name.startswith("cuda"):
        use_cuda_dct = True
    elif dct_device_name == "auto":
        use_cuda_dct = torch.cuda.is_available()

    if use_cuda_dct or (dct_device_name == "auto" and torch.cuda.is_available()):
        needs_spawn = True
    multiprocessing_context = mp.get_context("spawn") if needs_spawn else None

    train_tf, val_tf = build_resnet_transforms(
        image_size,
        compression=compression_cfg,
        dct_normalizer_train=dct_normalizer_train,
        dct_normalizer_val=dct_normalizer_val,
        trim_coefficients=trim_coefficients,
    )
    train_set = ImageNetDataset(train_dirs, transforms=train_tf, show_progress=show_progress)
    val_set = ImageNetDataset([val_dir], transforms=val_tf, show_progress=show_progress)
    if max_train is not None:
        train_set = limit_total(train_set, max_train)
    if max_val is not None:
        val_set = limit_total(val_set, max_val)
    train_loader_kwargs: dict[str, object] = {}
    if workers > 0:
        if prefetch_factor is not None:
            train_loader_kwargs["prefetch_factor"] = prefetch_factor
        train_loader_kwargs["persistent_workers"] = persistent_workers
    train_sampler = None
    val_sampler = None
    if distributed and world_size > 1:
        train_sampler = DistributedSampler(
            train_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        val_sampler = DistributedSampler(
            val_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=workers,
        pin_memory=not use_cuda_dct,
        drop_last=True,
        multiprocessing_context=multiprocessing_context if workers > 0 else None,
        **train_loader_kwargs,
    )
    val_workers = max(1, workers // 2)
    val_loader_kwargs: dict[str, object] = {}
    if val_workers > 0:
        if prefetch_factor is not None:
            val_loader_kwargs["prefetch_factor"] = prefetch_factor
        val_loader_kwargs["persistent_workers"] = persistent_workers
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=val_sampler is None,
        sampler=val_sampler,
        num_workers=val_workers,
        pin_memory=not use_cuda_dct,
        multiprocessing_context=multiprocessing_context if val_workers > 0 else None,
        **val_loader_kwargs,
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
    profile_steps: bool = False,
    distributed: bool = False,
    world_size: int = 1,
    is_main_process: bool = True,
) -> dict[str, object]:
    model.train()
    running_loss = 0.0
    correct1 = 0
    correct5 = 0
    total = 0
    last_log = time.time()
    epoch_start_time = time.time()
    time_limit_reached = False
    data_time_total = 0.0
    to_device_time_total = 0.0
    forward_time_total = 0.0
    backward_time_total = 0.0
    step_time_total = 0.0
    steps_processed = 0
    batch_fetch_start = time.time() if profile_steps else 0.0
    for step, (inputs, targets) in enumerate(loader):
        step_loop_start = time.time()
        if profile_steps:
            data_time_total += step_loop_start - batch_fetch_start
        if time_limit_seconds and time.time() - run_start_time >= time_limit_seconds:
            time_limit_reached = True
            break
        if profile_steps:
            to_device_start = time.time()
        inputs = _move_to_device(inputs, device)
        if not _PRINTED_INPUT_SHAPES["train"]:
            if is_main_process:
                print(f"[inputs][train] { _describe_inputs(inputs) }")
            _PRINTED_INPUT_SHAPES["train"] = True
        targets = targets.to(device)
        if profile_steps:
            after_move = time.time()
            to_device_time_total += after_move - to_device_start
            forward_start = after_move
        else:
            forward_start = time.time()
        optimizer.zero_grad()
        if use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = normalise_logits(model(inputs))
                loss = criterion(outputs, targets)
        else:
            outputs = normalise_logits(model(inputs))
            loss = criterion(outputs, targets)
        forward_end = time.time()
        if profile_steps:
            forward_time_total += forward_end - forward_start
            backward_start_time = forward_end
        if use_amp and device.type == "cuda":
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if profile_steps:
            backward_end = time.time()
            backward_time_total += backward_end - backward_start_time
            step_time_total += backward_end - step_loop_start
            batch_fetch_start = time.time()
        running_loss += loss.item() * targets.size(0)
        total += targets.size(0)
        steps_processed += 1

        maxk = min(5, outputs.size(1))
        _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct_matrix = pred.eq(targets.view(1, -1).expand_as(pred))
        correct1 += correct_matrix[:1].reshape(-1).float().sum().item()
        correct5 += correct_matrix[:5].reshape(-1).float().sum().item()
        if is_main_process and ((step + 1) % print_freq == 0 or (step + 1) == len(loader)):
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
    profile_stats: dict[str, float] | None = None
    if profile_steps and steps_processed > 0:
        total_profile_time = data_time_total + to_device_time_total + forward_time_total + backward_time_total
        avg_step_time = step_time_total / steps_processed if steps_processed > 0 else 0.0
        profile_stats = {
            "data_time": data_time_total,
            "to_device_time": to_device_time_total,
            "forward_time": forward_time_total,
            "backward_time": backward_time_total,
            "total": total_profile_time,
            "steps": float(steps_processed),
            "avg_step_time": avg_step_time,
        }
    result: dict[str, object] = {
        "loss": running_loss / max(total, 1),
        "acc1": correct1 / max(total, 1),
        "acc5": correct5 / max(total, 1),
        "time_limit_reached": time_limit_reached,
    }
    if profile_stats is not None:
        result["profile"] = profile_stats if is_main_process else None

    if distributed and world_size > 1:
        sums = torch.tensor([running_loss, correct1, correct5], device=device, dtype=torch.float64)
        counts = torch.tensor([total], device=device, dtype=torch.float64)
        limit_flag = torch.tensor([1.0 if time_limit_reached else 0.0], device=device, dtype=torch.float64)
        dist.all_reduce(sums)
        dist.all_reduce(counts)
        dist.all_reduce(limit_flag, op=dist.ReduceOp.SUM)
        running_loss, correct1, correct5 = sums.tolist()
        total = int(counts.item())
        time_limit_reached = limit_flag.item() > 0.0
        result.update({
            "loss": running_loss / max(total, 1),
            "acc1": correct1 / max(total, 1),
            "acc5": correct5 / max(total, 1),
            "time_limit_reached": time_limit_reached,
        })
    return result


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    show_progress: bool = False,
    progress_desc: str | None = None,
    distributed: bool = False,
    world_size: int = 1,
    is_main_process: bool = True,
) -> dict[str, float]:
    model.eval()
    running_loss = 0.0
    correct1 = 0
    correct5 = 0
    total = 0
    iterator = loader
    progress = None
    global _WARNED_NO_TQDM
    show_progress = show_progress and is_main_process
    if show_progress:
        if tqdm is not None:
            progress = tqdm(loader, desc=progress_desc or "eval", leave=False)
            iterator = progress
        else:
            if not _WARNED_NO_TQDM:
                print("[eval] tqdm not installed; proceeding without progress bar.")
                _WARNED_NO_TQDM = True
    with torch.no_grad():
        for inputs, targets in iterator:
            inputs = _move_to_device(inputs, device)
            if not _PRINTED_INPUT_SHAPES["eval"]:
                if is_main_process:
                    print(f"[inputs][eval] { _describe_inputs(inputs) }")
                _PRINTED_INPUT_SHAPES["eval"] = True
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
    if progress is not None:
        progress.close()
    if distributed and world_size > 1:
        sums = torch.tensor([running_loss, correct1, correct5], device=device, dtype=torch.float64)
        counts = torch.tensor([total], device=device, dtype=torch.float64)
        dist.all_reduce(sums)
        dist.all_reduce(counts)
        running_loss, correct1, correct5 = sums.tolist()
        total = int(counts.item())
    return {
        "loss": running_loss / max(total, 1),
        "acc1": correct1 / max(total, 1),
        "acc5": correct5 / max(total, 1),
    }


def main() -> None:
    args = parse_args()
    _init_distributed_mode(args)
    is_main = _is_main_process(args)

    seed = args.seed + (args.rank if getattr(args, "distributed", False) else 0)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    wandb_run = None

    if args.matmul_precision is not None:
        try:
            torch.set_float32_matmul_precision(args.matmul_precision)
        except AttributeError as exc:
            if is_main:
                print(f"[cli] Matmul precision override not supported: {exc}.")

    if args.prefetch_factor is not None and args.workers == 0 and is_main:
        print("[cli] --prefetch-factor has no effect when --workers is 0.")
    if args.persistent_workers and args.workers == 0 and is_main:
        print("[cli] --persistent-workers has no effect when --workers is 0.")

    base_output_dir = args.output_dir
    base_checkpoint_dir = args.checkpoint_dir or base_output_dir

    resume_path: Path | None = None
    resume_run_dir: Path | None = None
    auto_resume_source_checkpoint: Path | None = None
    using_existing_run_dir = False

    if args.resume is not None:
        resume_path = args.resume.expanduser()
        if resume_path.is_dir():
            resume_run_dir = resume_path
            resume_path = resume_path / "checkpoint_last.pth"
        else:
            resume_run_dir = resume_path.parent
        using_existing_run_dir = True
    elif args.auto_resume:
        candidate = _find_latest_checkpoint(base_checkpoint_dir, args.variant)
        if candidate is not None:
            resume_path = candidate
            resume_run_dir = candidate.parent
            auto_resume_source_checkpoint = candidate
            using_existing_run_dir = True
            if is_main:
                print(f"[auto-resume] Found checkpoint {candidate}")

    if resume_path is not None and resume_path.exists():
        resume_config = _load_checkpoint_compression_config(resume_path)
        if resume_config:
            _apply_checkpoint_compression_config(args, resume_config)
            if args.variant == "reconstruction" and args.trim_coefficients:
                if is_main:
                    print(
                        "[resume] Reconstruction variant requires full coefficient depth; overriding stored trim setting."
                    )
                args.trim_coefficients = False
            _resolve_coefficient_args(args)

    coeff_descriptor = f"coeffY{args.coeff_count_luma}_Cb{args.coeff_count_cb}_Cr{args.coeff_count_cr}"

    if args.distributed:
        if args.device is not None and is_main:
            print("[cli] Ignoring --device because distributed training is enabled; using local rank instead.")
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.local_rank}")
        else:
            device = torch.device("cpu")
        args.device = str(device)
    else:
        if args.device is not None:
            device = torch.device(args.device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    compression_cfg = {
        "coeff_window": args.coeff_window,
        "coeff_window_luma": args.coeff_window_luma,
        "coeff_window_chroma": args.coeff_window_chroma,
        "coeff_window_cb": args.coeff_window_cb,
        "coeff_window_cr": args.coeff_window_cr,
        "coeff_count": args.coeff_count,
        "coeff_count_luma": args.coeff_count_luma,
        "coeff_count_chroma": args.coeff_count_chroma,
        "coeff_count_cb": args.coeff_count_cb,
        "coeff_count_cr": args.coeff_count_cr,
        "range_mode": args.range_mode,
        "dct_device": args.dct_device,
        "dtype": torch.float32,
        "keep_original": False,
    }

    normalizer_train: T.Transform | None = None
    normalizer_val: T.Transform | None = None
    stats_path_input: Path | None = args.dct_stats
    if args.variant == "reconstruction":
        if stats_path_input is not None and is_main:
            print("Reconstruction variant ignores --dct-stats; proceeding without coefficient normalisation.")
        stats_path_input = None
    else:
        if stats_path_input is None:
            candidate_windows: list[int] = []
            for window in (
                args.coeff_window_luma,
                args.coeff_window_cb,
                args.coeff_window_cr,
                args.coeff_window_chroma,
            ):
                if isinstance(window, int) and 1 <= window <= 8 and window not in candidate_windows:
                    candidate_windows.append(window)
            if 8 not in candidate_windows:
                candidate_windows.append(8)
            for candidate_window in candidate_windows:
                default_stats = Path("configs/dct_stats") / f"imagenet_coeff{candidate_window}_{args.range_mode}.pt"
                if default_stats.exists():
                    requested_windows = {
                        w
                        for w in (
                            args.coeff_window_luma,
                            args.coeff_window_cb,
                            args.coeff_window_cr,
                            args.coeff_window_chroma,
                        )
                        if isinstance(w, int)
                    }
                    if any(w != candidate_window for w in requested_windows):
                        if is_main:
                            print(
                                f"Using DCT stats computed for coeff window {candidate_window}; "
                                "requested windows will slice the relevant coefficients."
                            )
                    stats_path_input = default_stats
                    break
            if stats_path_input is None:
                if is_main:
                    print("No default DCT stats file found; skipping normalisation auto-detection.")
    stats_path_resolved: Path | None = None
    if stats_path_input is not None:
        try:
            stats_path_resolved = stats_path_input.expanduser().resolve()
            normalizer_train = NormalizeDCTCoefficients.from_file(
                stats_path_resolved,
                coeff_window=args.coeff_window,
                coeff_window_luma=args.coeff_window_luma,
                coeff_window_chroma=args.coeff_window_chroma,
                coeff_window_cb=args.coeff_window_cb,
                coeff_window_cr=args.coeff_window_cr,
                coeff_count=args.coeff_count,
                coeff_count_luma=args.coeff_count_luma,
                coeff_count_chroma=args.coeff_count_chroma,
                coeff_count_cb=args.coeff_count_cb,
                coeff_count_cr=args.coeff_count_cr,
            )
            normalizer_val = NormalizeDCTCoefficients.from_file(
                stats_path_resolved,
                coeff_window=args.coeff_window,
                coeff_window_luma=args.coeff_window_luma,
                coeff_window_chroma=args.coeff_window_chroma,
                coeff_window_cb=args.coeff_window_cb,
                coeff_window_cr=args.coeff_window_cr,
                coeff_count=args.coeff_count,
                coeff_count_luma=args.coeff_count_luma,
                coeff_count_chroma=args.coeff_count_chroma,
                coeff_count_cb=args.coeff_count_cb,
                coeff_count_cr=args.coeff_count_cr,
            )
            if is_main:
                print(f"Loaded DCT coefficient statistics from {stats_path_resolved}")
        except Exception as exc:  # pragma: no cover - safeguard for user-provided files
            if is_main:
                print(f"Failed to load DCT stats from {stats_path_input}: {exc}")
            stats_path_resolved = None
    if stats_path_resolved is None and is_main:
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
        trim_coefficients=args.trim_coefficients,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        distributed=args.distributed,
        rank=args.rank,
        world_size=args.world_size,
    )

    effective_lr = args.lr
    milestones = sorted(m for m in args.lr_milestones if m > 0)
    if not milestones:
        milestones = [30, 60, 80]
    warmup_epochs = max(0, args.warmup_epochs)
    if warmup_epochs > 0 and is_main:
        print(f"Applying {warmup_epochs} warmup epoch(s) before step LR milestones {milestones}.")
    time_limit_hours = args.time_limit_hours if args.time_limit_hours > 0 else None
    if time_limit_hours is not None and is_main:
        print(f"Enforcing wall-clock time limit of {time_limit_hours:.2f}h.")
    time_limit_seconds = time_limit_hours * 3600.0 if time_limit_hours is not None else None

    timestamp_tag = time.strftime("%Y%m%d-%H%M%S")
    window_tag = f"coeffY{args.coeff_count_luma}"
    if args.coeff_count_cb == args.coeff_count_cr:
        window_tag += f"_CbCr{args.coeff_count_cb}"
    else:
        window_tag += f"_Cb{args.coeff_count_cb}_Cr{args.coeff_count_cr}"
    proposed_run_subdir = f"{args.variant}_{window_tag}_{timestamp_tag}"
    run_base_subdir = proposed_run_subdir
    run_subdir_name = proposed_run_subdir
    if resume_run_dir is not None:
        existing_name = resume_run_dir.name
        prefix, sep, _ = existing_name.rpartition("_epoch")
        run_base_subdir = prefix if sep else existing_name
        run_subdir_name = existing_name

    wandb_run = None

    model, _ = build_model("resnet34", _NUM_CLASSES)
    model.apply(kaiming_initialisation)
    model.backbone = build_compressed_backbone(
        args.variant,
        model.backbone,
        range_mode=args.range_mode,
        mean=_IMAGENET_MEAN,
        std=_IMAGENET_STD,
        coeff_window_luma=args.coeff_window_luma,
        coeff_window_chroma=args.coeff_window_chroma,
        coeff_window_cb=args.coeff_window_cb,
        coeff_window_cr=args.coeff_window_cr,
        coeff_count_luma=args.coeff_count_luma,
        coeff_count_chroma=args.coeff_count_chroma,
        coeff_count_cb=args.coeff_count_cb,
        coeff_count_cr=args.coeff_count_cr,
    )
    if args.variant == "luma-fusion-pruned":
        hidden_dim = model.backbone.out_channels[0]
        model.head = ClassHead(hidden_dim=hidden_dim, num_classes=_NUM_CLASSES)
        model.head.apply(kaiming_initialisation)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.to(device)

    if args.compile_model:
        try:
            model = torch.compile(model, mode=args.compile_mode)
            if is_main:
                print(f"[compile] Enabled torch.compile with mode={args.compile_mode}.")
        except AttributeError as exc:
            if is_main:
                print(f"[compile] Torch compile not available: {exc}. Proceeding without compilation.")
        except Exception as exc:  # pragma: no cover - defensive path for compile runtime failures
            if is_main:
                print(f"[compile] Failed to compile model ({exc}); continuing with eager execution.")

    model_without_ddp = model

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=effective_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_gamma)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    if resume_run_dir is not None:
        output_dir = resume_run_dir
        checkpoint_dir = resume_run_dir
    else:
        output_dir = base_output_dir / run_subdir_name
        checkpoint_dir = base_checkpoint_dir / run_subdir_name

    best_acc = 0.0
    start_epoch = 1

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
        if is_main:
            print(f"Resumed training from epoch {last_epoch} with best acc@1={best_acc:.4f} ({resume_path})")
        if start_epoch > args.epochs and not args.eval_only:
            if is_main:
                print("Checkpoint epoch exceeds requested total epochs; nothing to train.")
            if wandb_run is not None and is_main:
                wandb_run.finish()
            wandb_run = None
            if args.distributed:
                _cleanup_distributed()
            return

    if args.distributed:
        ddp_kwargs: dict[str, object] = {"static_graph": args.ddp_static_graph}
        if device.type == "cuda":
            ddp_kwargs["device_ids"] = [args.local_rank]
            ddp_kwargs["output_device"] = args.local_rank
        model = DistributedDataParallel(model, **ddp_kwargs)
        model_without_ddp = model.module

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    output_root = output_dir.resolve()
    checkpoint_root = checkpoint_dir.resolve()
    if is_main:
        print(f"[setup] outputs -> {output_root}")
        print(f"[setup] checkpoints -> {checkpoint_root}")

    if auto_resume_source_checkpoint is not None and not using_existing_run_dir:
        resume_copy_target = checkpoint_dir / "checkpoint_resumed_from.pth"
        if not resume_copy_target.exists():
            shutil.copy2(auto_resume_source_checkpoint, resume_copy_target)
            if is_main:
                print(f"[setup] copied resume checkpoint -> {resume_copy_target}")

    wandb_run_id_path = checkpoint_dir / "wandb_run_id.txt"
    if args.wandb and is_main:
        if wandb is None:
            raise RuntimeError(
                "Weights & Biases is not installed. Run 'pip install wandb' to enable logging."
            )
        wandb_resume_id: str | None = None
        wandb_resume_mode: str | None = None
        wandb_id_sources: list[Path] = []
        if auto_resume_source_checkpoint is not None:
            wandb_id_sources.append(auto_resume_source_checkpoint.parent / "wandb_run_id.txt")
        wandb_id_sources.append(wandb_run_id_path)
        for candidate in wandb_id_sources:
            if candidate.exists():
                run_id_candidate = candidate.read_text().strip()
                if run_id_candidate:
                    wandb_resume_id = run_id_candidate
                    break
        if wandb_resume_id is not None:
            wandb_resume_mode = "allow"
            print(f"[wandb] Resuming run id={wandb_resume_id}")

        wandb_config = {
            "variant": args.variant,
            "coeff_window": args.coeff_window_luma,
            "coeff_window_luma": args.coeff_window_luma,
            "coeff_window_chroma": args.coeff_window_chroma,
            "coeff_window_cb": args.coeff_window_cb,
            "coeff_window_cr": args.coeff_window_cr,
            "coeff_count": args.coeff_count,
            "coeff_count_luma": args.coeff_count_luma,
            "coeff_count_chroma": args.coeff_count_chroma,
            "coeff_count_cb": args.coeff_count_cb,
            "coeff_count_cr": args.coeff_count_cr,
            "coeff_descriptor": coeff_descriptor,
            "trim_coefficients": args.trim_coefficients,
            "range_mode": args.range_mode,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "prefetch_factor": args.prefetch_factor,
            "persistent_workers": args.persistent_workers,
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
            "compile_model": args.compile_model,
            "compile_mode": args.compile_mode,
            "matmul_precision": args.matmul_precision,
            "dct_stats_path": str(stats_path_resolved) if stats_path_resolved is not None else None,
            "time_limit_hours": time_limit_hours,
        }
        default_name = run_subdir_name
        run_name = args.wandb_run_name or default_name
        auto_tags = [args.variant, coeff_descriptor, f"trim_{'on' if args.trim_coefficients else 'off'}"]
        custom_tags = list(args.wandb_tags) if args.wandb_tags is not None else []
        combined_tags: list[str] = []
        for tag in (*auto_tags, *custom_tags):
            if tag and tag not in combined_tags:
                combined_tags.append(tag)
        print(f"[wandb] run name -> {run_name}")
        if combined_tags:
            print(f"[wandb] tags -> {combined_tags}")
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=run_name,
            tags=combined_tags if combined_tags else None,
            config=wandb_config,
            id=wandb_resume_id,
            resume=wandb_resume_mode,
        )
        resolved_run_id = wandb_run.id
        try:
            wandb_run_id_path.write_text(resolved_run_id)
        except OSError as exc:
            if is_main:
                print(f"[wandb] Warning: failed to persist run id to {wandb_run_id_path}: {exc}")

    last_epoch_completed = start_epoch - 1

    if not args.eval_only and is_main:
        csv_path = checkpoint_dir / "training_params.csv"
        timestamp_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        params_to_record = {
            "variant": args.variant,
            "coeff_window": args.coeff_window_luma,
            "coeff_window_luma": args.coeff_window_luma,
            "coeff_window_chroma": args.coeff_window_chroma,
            "coeff_window_cb": args.coeff_window_cb,
            "coeff_window_cr": args.coeff_window_cr,
            "coeff_count": args.coeff_count,
            "coeff_count_luma": args.coeff_count_luma,
            "coeff_count_chroma": args.coeff_count_chroma,
            "coeff_count_cb": args.coeff_count_cb,
            "coeff_count_cr": args.coeff_count_cr,
            "range_mode": args.range_mode,
            "trim_coefficients": args.trim_coefficients,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "prefetch_factor": args.prefetch_factor,
            "persistent_workers": args.persistent_workers,
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
            "compile_model": args.compile_model,
            "compile_mode": args.compile_mode,
            "matmul_precision": args.matmul_precision,
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
        if is_main:
            if args.variant != "reconstruction":
                print("[preview] Reconstruction preview currently supports only the reconstruction variant.")
            else:
                preview_path = args.preview_output or (output_dir / "preview_reconstruction.png")
                preview_reconstruction_sample(model.backbone, train_loader, device, preview_path)
        if args.epochs <= 0:
            if is_main:
                print("Preview complete and epochs<=0; exiting without training.")
            if wandb_run is not None and is_main:
                wandb_run.finish()
            wandb_run = None
            if args.distributed:
                _cleanup_distributed()
            return

    last_val_stats: dict[str, float] | None = None

    if args.eval_only:
        if is_main:
            print("[eval-only] Skipping training; running validation.")
        last_val_stats = evaluate(
            model,
            val_loader,
            criterion,
            device,
            show_progress=True,
            progress_desc="val",
            distributed=args.distributed,
            world_size=args.world_size,
            is_main_process=is_main,
        )
        if last_val_stats is not None:
            best_acc = max(best_acc, last_val_stats["acc1"])
            if is_main:
                print(
                    "  val loss={loss:.4f} acc@1={acc1:.4f} acc@5={acc5:.4f}".format(
                        loss=last_val_stats["loss"],
                        acc1=last_val_stats["acc1"],
                        acc5=last_val_stats["acc5"],
                    )
                )
            if wandb_run is not None and is_main:
                wandb_run.log({
                    "val/loss": last_val_stats["loss"],
                    "val/acc1": last_val_stats["acc1"],
                    "val/acc5": last_val_stats["acc5"],
                })
    else:
        for epoch in range(start_epoch, args.epochs + 1):
            if time_limit_seconds and time.time() - run_start_time >= time_limit_seconds:
                if is_main:
                    print("Time limit reached before starting new epoch; stopping training loop.")
                time_limit_triggered = True
                break
            if is_main:
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
            if args.distributed:
                train_sampler = getattr(train_loader, "sampler", None)
                if isinstance(train_sampler, DistributedSampler):
                    train_sampler.set_epoch(epoch)
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
                profile_steps=args.profile_steps,
                distributed=args.distributed,
                world_size=args.world_size,
                is_main_process=is_main,
            )
            profile_stats = stats.get("profile") if args.profile_steps else None
            if stats.get("time_limit_reached"):
                if is_main:
                    print("Time limit reached during training phase; terminating without evaluation.")
                time_limit_triggered = True
                break
            val_stats = evaluate(
                model,
                val_loader,
                criterion,
                device,
                show_progress=True,
                progress_desc=f"val epoch {epoch}",
                distributed=args.distributed,
                world_size=args.world_size,
                is_main_process=is_main,
            )
            last_val_stats = val_stats
            if time_limit_seconds and time.time() - run_start_time >= time_limit_seconds:
                if is_main:
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
                if is_main:
                    print(
                        f"  train loss={stats['loss']:.4f} acc@1={stats['acc1']:.4f} acc@5={stats['acc5']:.4f} | "
                        f"val loss={val_stats['loss']:.4f} acc@1={val_stats['acc1']:.4f} acc@5={val_stats['acc5']:.4f}"
                        + (f" gpu_mem={gpu_mem:.1f}MB" if gpu_mem is not None else "")
                    )
                last_epoch_completed = epoch
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
            if is_main:
                print(
                    f"  train loss={stats['loss']:.4f} acc@1={stats['acc1']:.4f} acc@5={stats['acc5']:.4f} | "
                    f"val loss={val_stats['loss']:.4f} acc@1={val_stats['acc1']:.4f} acc@5={val_stats['acc5']:.4f}"
                    + (f" gpu_mem={gpu_mem:.1f}MB" if gpu_mem is not None else "")
                )
            if args.profile_steps and isinstance(profile_stats, dict) and is_main:
                total_profile_time = float(profile_stats.get("total", 0.0))
                if total_profile_time > 0.0:
                    data_time = float(profile_stats.get("data_time", 0.0))
                    to_device_time = float(profile_stats.get("to_device_time", 0.0))
                    forward_time = float(profile_stats.get("forward_time", 0.0))
                    backward_time = float(profile_stats.get("backward_time", 0.0))
                    avg_step_time = float(profile_stats.get("avg_step_time", 0.0))
                    data_pct = 100.0 * data_time / total_profile_time
                    to_device_pct = 100.0 * to_device_time / total_profile_time
                    forward_pct = 100.0 * forward_time / total_profile_time
                    backward_pct = 100.0 * backward_time / total_profile_time
                    steps_recorded = int(profile_stats.get("steps", 0.0))
                    print(
                        "    [profile] data_load={:.2f}s ({:.1f}%) | to_device={:.2f}s ({:.1f}%) | forward={:.2f}s ({:.1f}%) | backward/optim={:.2f}s ({:.1f}%) | avg_step={:.3f}s over {} step(s)".format(
                            data_time,
                            data_pct,
                            to_device_time,
                            to_device_pct,
                            forward_time,
                            forward_pct,
                            backward_time,
                            backward_pct,
                            avg_step_time,
                            steps_recorded,
                        )
                    )
            if wandb_run is not None and is_main:
                wandb_payload = {
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
                }
                if args.profile_steps and isinstance(profile_stats, dict):
                    wandb_payload.update(
                        {
                            "profile/data_time": float(profile_stats.get("data_time", 0.0)),
                            "profile/to_device_time": float(profile_stats.get("to_device_time", 0.0)),
                            "profile/forward_time": float(profile_stats.get("forward_time", 0.0)),
                            "profile/backward_time": float(profile_stats.get("backward_time", 0.0)),
                            "profile/total_time": float(profile_stats.get("total", 0.0)),
                            "profile/avg_step_time": float(profile_stats.get("avg_step_time", 0.0)),
                            "profile/steps": float(profile_stats.get("steps", 0.0)),
                        }
                    )
                wandb.log(wandb_payload)
            if acc1 > best_acc:
                best_acc = acc1
                if wandb_run is not None and is_main:
                    wandb_run.summary["best/acc1"] = best_acc
                    wandb_run.summary["best/acc5"] = val_stats["acc5"]
                    wandb_run.summary["best/epoch"] = epoch
                if args.save_best and is_main:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model": model_without_ddp.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "scaler": scaler.state_dict() if scaler is not None else None,
                            "variant": args.variant,
                            "coeff_window": args.coeff_window_luma,
                            "coeff_window_luma": args.coeff_window_luma,
                            "coeff_window_chroma": args.coeff_window_chroma,
                            "coeff_window_cb": args.coeff_window_cb,
                            "coeff_window_cr": args.coeff_window_cr,
                            "coeff_count": args.coeff_count,
                            "coeff_count_luma": args.coeff_count_luma,
                            "coeff_count_chroma": args.coeff_count_chroma,
                            "coeff_count_cb": args.coeff_count_cb,
                            "coeff_count_cr": args.coeff_count_cr,
                            "trim_coefficients": args.trim_coefficients,
                            "range_mode": args.range_mode,
                            "best_acc": best_acc,
                        },
                        checkpoint_dir / "model_best.pth",
                    )
                    print(f"  Saved new best checkpoint acc@1={best_acc:.4f}")
            if args.save_last and is_main:
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict() if scaler is not None else None,
                        "variant": args.variant,
                        "coeff_window": args.coeff_window_luma,
                        "coeff_window_luma": args.coeff_window_luma,
                        "coeff_window_chroma": args.coeff_window_chroma,
                        "coeff_window_cb": args.coeff_window_cb,
                        "coeff_window_cr": args.coeff_window_cr,
                        "coeff_count": args.coeff_count,
                        "coeff_count_luma": args.coeff_count_luma,
                        "coeff_count_chroma": args.coeff_count_chroma,
                        "coeff_count_cb": args.coeff_count_cb,
                        "coeff_count_cr": args.coeff_count_cr,
                        "trim_coefficients": args.trim_coefficients,
                        "range_mode": args.range_mode,
                        "best_acc": best_acc,
                    },
                    checkpoint_dir / "checkpoint_last.pth",
                )
            if args.save_every and epoch % args.save_every == 0 and is_main:
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict() if scaler is not None else None,
                        "variant": args.variant,
                        "coeff_window": args.coeff_window_luma,
                        "coeff_window_luma": args.coeff_window_luma,
                        "coeff_window_chroma": args.coeff_window_chroma,
                        "coeff_window_cb": args.coeff_window_cb,
                        "coeff_window_cr": args.coeff_window_cr,
                        "coeff_count": args.coeff_count,
                        "coeff_count_luma": args.coeff_count_luma,
                        "coeff_count_chroma": args.coeff_count_chroma,
                        "coeff_count_cb": args.coeff_count_cb,
                        "coeff_count_cr": args.coeff_count_cr,
                        "trim_coefficients": args.trim_coefficients,
                        "range_mode": args.range_mode,
                        "best_acc": best_acc,
                    },
                    checkpoint_dir / f"checkpoint_{epoch:04d}.pth",
                )

            last_epoch_completed = epoch

    trimmed_val_stats: dict[str, float] | None = None
    if not args.trimmed_val_disable and is_main:
        try:
            trimmed_loader = build_trimmed_eval_loader(
                val_dirs=[str(Path(args.val_dir))],
                image_size=args.image_size,
                batch_size=args.batch_size,
                workers=args.workers,
                compression_cfg=compression_cfg,
                coeff_count_luma=args.coeff_count_luma,
                coeff_count_cb=args.coeff_count_cb,
                coeff_count_cr=args.coeff_count_cr,
                max_samples=args.max_val_images,
                dct_normalizer=normalizer_val,
                show_progress=args.show_progress,
                trim_coefficients=args.trim_coefficients,
            )
            if len(trimmed_loader) == 0:
                print("[trimmed-val] Validation dataset is empty; skipping trimmed evaluation.")
            else:
                trimmed_val_stats = evaluate(
                    model,
                    trimmed_loader,
                    criterion,
                    device,
                    show_progress=True,
                    progress_desc="trimmed-val",
                    distributed=False,
                    world_size=1,
                    is_main_process=True,
                )
        except Exception as exc:  # pragma: no cover - trimmed validation is best-effort
            print(f"[trimmed-val] Failed to evaluate trimmed dataloader: {exc}")

    benchmark_results: list[tuple[int, BenchmarkResult]] = []
    benchmark_csv_rows: list[dict[str, object]] = []
    if args.benchmark_enabled and args.benchmark_measure_batches > 0 and is_main:
        benchmark_batch_sizes = [1, 8, 32, 64, 128, 256]
        benchmark_max_samples = args.benchmark_max_samples if args.benchmark_max_samples > 0 else None
        benchmark_timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        for bench_bs in benchmark_batch_sizes:
            try:
                result = run_trimmed_inference_benchmark(
                    model,
                    device=device,
                    coeff_window_luma=args.coeff_window_luma,
                    coeff_window_chroma=args.coeff_window_chroma,
                    image_size=args.image_size,
                    batch_size=bench_bs,
                    val_dirs=[str(Path(args.val_dir))],
                    compression_cfg=compression_cfg,
                    dct_normalizer=normalizer_val,
                    max_samples=benchmark_max_samples,
                    workers=args.benchmark_workers,
                    warmup_batches=max(0, args.benchmark_warmup_batches),
                    measure_batches=max(0, args.benchmark_measure_batches),
                    show_progress=args.show_progress,
                    collect_metrics=args.benchmark_validate,
                    trim_coefficients=args.trim_coefficients,
                )
            except Exception as exc:  # pragma: no cover - benchmark is best-effort
                print(f"[benchmark] Failed for batch size {bench_bs}: {exc}")
                continue
            if result is not None:
                benchmark_results.append((bench_bs, result))
                benchmark_csv_rows.append({
                    "timestamp": benchmark_timestamp,
                    "variant": args.variant,
                    "coeff_window": args.coeff_window_luma,
                    "coeff_window_luma": args.coeff_window_luma,
                    "coeff_window_chroma": args.coeff_window_chroma,
                    "coeff_window_cb": args.coeff_window_cb,
                    "coeff_window_cr": args.coeff_window_cr,
                    "coeff_count": args.coeff_count,
                    "coeff_count_luma": args.coeff_count_luma,
                    "coeff_count_chroma": args.coeff_count_chroma,
                    "coeff_count_cb": args.coeff_count_cb,
                    "coeff_count_cr": args.coeff_count_cr,
                    "image_size": args.image_size,
                    "batch_size": bench_bs,
                    "samples": result.samples,
                    "measured_batches": result.measured_batches,
                    "throughput_img_s": result.throughput_img_s,
                    "mean_latency_ms": result.mean_latency_ms,
                    "input_mb_per_batch": result.input_mb_per_batch,
                    "peak_memory_mb": result.peak_memory_mb,
                    "coeff_channels": result.coeff_channels,
                    "coeff_channels_cb": result.coeff_channels_cb,
                    "coeff_channels_cr": result.coeff_channels_cr,
                    "coeff_channels_chroma": result.coeff_channels_chroma,
                    "val_loss": result.loss if result.loss is not None else "",
                    "val_acc1": result.acc1 if result.acc1 is not None else "",
                    "val_acc5": result.acc5 if result.acc5 is not None else "",
                })

    if benchmark_csv_rows and not args.eval_only and is_main:
        benchmark_csv_path = checkpoint_dir / "benchmark_metrics.csv"
        csv_exists = benchmark_csv_path.exists()
        fieldnames = [
            "timestamp",
            "variant",
            "coeff_window",
            "coeff_window_luma",
            "coeff_window_chroma",
            "coeff_window_cb",
            "coeff_window_cr",
            "coeff_count",
            "coeff_count_luma",
            "coeff_count_chroma",
            "coeff_count_cb",
            "coeff_count_cr",
            "image_size",
            "batch_size",
            "samples",
            "measured_batches",
            "throughput_img_s",
            "mean_latency_ms",
            "input_mb_per_batch",
            "peak_memory_mb",
            "coeff_channels",
            "coeff_channels_cb",
            "coeff_channels_cr",
            "coeff_channels_chroma",
            "val_loss",
            "val_acc1",
            "val_acc5",
        ]
        with benchmark_csv_path.open("a" if csv_exists else "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if not csv_exists:
                writer.writeheader()
            for row in benchmark_csv_rows:
                writer.writerow(row)
    elif benchmark_csv_rows and args.eval_only and is_main:
        print("[benchmark] Skipping CSV write in eval-only mode (read-only checkpoint directory).")

    if trimmed_val_stats is not None and is_main:
        print(
            "[trimmed-val] acc@1={acc1:.4f} acc@5={acc5:.4f} loss={loss:.4f}".format(
                acc1=trimmed_val_stats["acc1"],
                acc5=trimmed_val_stats["acc5"],
                loss=trimmed_val_stats["loss"],
            )
        )
        if wandb_run is not None and is_main:
            wandb_run.summary.update({
                "trimmed_val/acc1": trimmed_val_stats["acc1"],
                "trimmed_val/acc5": trimmed_val_stats["acc5"],
                "trimmed_val/loss": trimmed_val_stats["loss"],
            })

    for batch_size, benchmark_result in benchmark_results:
        peak_text = (
            f" | peak_mem={benchmark_result.peak_memory_mb:.1f}MB"
            if benchmark_result.peak_memory_mb is not None
            else ""
        )
        metrics_text = ""
        if benchmark_result.acc1 is not None and benchmark_result.acc5 is not None:
            metrics_text = (
                f" | acc@1={benchmark_result.acc1:.4f} acc@5={benchmark_result.acc5:.4f}"
            )
        if benchmark_result.loss is not None:
            metrics_text += f" | loss={benchmark_result.loss:.4f}"
        if is_main:
            print(
                f"[benchmark] trimmed inference (bs={batch_size}): "
                f"{benchmark_result.throughput_img_s:.2f} img/s | "
                f"{benchmark_result.mean_latency_ms:.2f} ms/batch | "
                f"coeffs=Y{benchmark_result.coeff_channels}/Cb{benchmark_result.coeff_channels_cb}/Cr{benchmark_result.coeff_channels_cr} | "
                f"input={benchmark_result.input_mb_per_batch:.2f} MB/batch"
                f"{peak_text}{metrics_text}"
            )
        if wandb_run is not None and is_main:
            wandb_run.summary.update({
                f"benchmark/bs{batch_size}/samples": benchmark_result.samples,
                f"benchmark/bs{batch_size}/measured_batches": benchmark_result.measured_batches,
                f"benchmark/bs{batch_size}/throughput_img_s": benchmark_result.throughput_img_s,
                f"benchmark/bs{batch_size}/mean_latency_ms": benchmark_result.mean_latency_ms,
                f"benchmark/bs{batch_size}/input_mb_per_batch": benchmark_result.input_mb_per_batch,
                f"benchmark/bs{batch_size}/coeff_channels": benchmark_result.coeff_channels,
                f"benchmark/bs{batch_size}/coeff_channels_cb": benchmark_result.coeff_channels_cb,
                f"benchmark/bs{batch_size}/coeff_channels_cr": benchmark_result.coeff_channels_cr,
                f"benchmark/bs{batch_size}/coeff_channels_chroma": benchmark_result.coeff_channels_chroma,
                f"benchmark/bs{batch_size}/peak_memory_mb": benchmark_result.peak_memory_mb,
                f"benchmark/bs{batch_size}/val_loss": benchmark_result.loss,
                f"benchmark/bs{batch_size}/val_acc1": benchmark_result.acc1,
                f"benchmark/bs{batch_size}/val_acc5": benchmark_result.acc5,
            })

    if args.eval_only:
        if last_val_stats is not None and is_main:
            print(
                "Evaluation complete. val acc@1={acc1:.4f} acc@5={acc5:.4f} loss={loss:.4f}".format(
                    acc1=last_val_stats["acc1"],
                    acc5=last_val_stats["acc5"],
                    loss=last_val_stats["loss"],
                )
            )
        elif is_main:
            print("Evaluation complete.")
        if last_val_stats is not None and wandb_run is not None and is_main:
            wandb_run.summary.update({
                "val/acc1": last_val_stats["acc1"],
                "val/acc5": last_val_stats["acc5"],
                "val/loss": last_val_stats["loss"],
            })
    else:
        if is_main:
            epoch_suffix = f"epoch{max(last_epoch_completed, 0):04d}"
            final_run_subdir = f"{run_base_subdir}_{epoch_suffix}"
            if final_run_subdir != run_subdir_name:
                final_output_dir = base_output_dir / final_run_subdir
                final_checkpoint_dir = base_checkpoint_dir / final_run_subdir
                final_output_dir.parent.mkdir(parents=True, exist_ok=True)
                final_checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
                same_storage = checkpoint_dir.resolve() == output_dir.resolve()
                if same_storage:
                    if final_output_dir.exists():
                        print(
                            f"[setup] Target directory {final_output_dir} already exists; retaining {output_dir}."
                        )
                    else:
                        output_dir.rename(final_output_dir)
                        output_dir = final_output_dir
                        checkpoint_dir = final_output_dir
                        run_subdir_name = final_run_subdir
                        print(f"[setup] renamed run directory -> {output_dir}")
                else:
                    if final_output_dir.exists() or final_checkpoint_dir.exists():
                        print(
                            "[setup] Target directories already exist; retaining original run directories."
                        )
                    else:
                        output_dir.rename(final_output_dir)
                        checkpoint_dir.rename(final_checkpoint_dir)
                        output_dir = final_output_dir
                        checkpoint_dir = final_checkpoint_dir
                        run_subdir_name = final_run_subdir
                        print(f"[setup] renamed run directory -> {output_dir}")
            print(f"Training complete. Best val acc@1={best_acc:.4f}")
            if time_limit_triggered:
                print("Stopped early due to the configured time limit.")
    if wandb_run is not None and is_main:
        wandb_run.finish()
    if args.distributed:
        _cleanup_distributed()


if __name__ == "__main__":
    main()
