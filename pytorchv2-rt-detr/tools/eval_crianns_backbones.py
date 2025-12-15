#!/usr/bin/env python3
"""Evaluate compressed ResNet-34 checkpoints and aggregate metrics."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import sys

import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset.imagenet import ImageNetDataset
from src.nn.backbone.compressed_presnet import build_compressed_backbone
from src.nn.backbone.inference_benchmark import run_trimmed_inference_benchmark
from src.nn.backbone.train_backbones import (
    _IMAGENET_MEAN,
    _IMAGENET_STD,
    _move_to_device,
    build_model,
    build_resnet_transforms,
    normalise_logits,
)
from src.data.transforms.dct_normalize import NormalizeDCTCoefficients
from src.nn.arch.classification import ClassHead

_NUM_CLASSES = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("output/compressed_resnet34/crianns_weights/checkpoints"),
        help="Directory containing per-variant checkpoint subfolders.",
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=Path("dataset/classification/imagenet1kvalid"),
        help="ImageNet validation directory (ImageNet folder layout).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size used during accuracy evaluation.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of dataloader workers for accuracy evaluation.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Validation crop size (should match training setup).",
    )
    parser.add_argument(
        "--range-mode",
        choices=["studio", "full"],
        default="studio",
        help="Pixel range fed into the compression pipeline.",
    )
    parser.add_argument(
        "--dct-stats",
        type=Path,
        default=None,
        help="Optional override for DCT normalisation stats (.pt file).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device string; defaults to CUDA if available.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Where to write the aggregated CSV (defaults beside checkpoint root).",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="Location for the coeff-erasure accuracy plot (defaults beside CSV).",
    )
    parser.add_argument(
        "--benchmark-batch-size",
        type=int,
        default=128,
        help="Batch size used when timing inference throughput.",
    )
    parser.add_argument(
        "--benchmark-workers",
        type=int,
        default=2,
        help="Number of dataloader workers for the benchmark loader.",
    )
    parser.add_argument(
        "--benchmark-warmup",
        type=int,
        default=5,
        help="Warmup batches ignored before measuring latency.",
    )
    parser.add_argument(
        "--benchmark-measure",
        type=int,
        default=20,
        help="Number of timed batches when measuring throughput.",
    )
    parser.add_argument(
        "--benchmark-max-samples",
        type=int,
        default=256,
        help="Maximum validation samples pulled when benchmarking speed.",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Display dataset discovery progress bars if available.",
    )
    return parser.parse_args()


def _discover_checkpoint_dirs(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Checkpoint root not found: {root}")
    return sorted([p for p in root.iterdir() if p.is_dir()])


def _select_checkpoint_file(directory: Path) -> Optional[Path]:
    preferred = [
        "model_best.pth",
        "checkpoint_best.pth",
        "checkpoint_last.pth",
    ]
    for name in preferred:
        candidate = directory / name
        if candidate.exists():
            return candidate
    epoch_ckpts = sorted(directory.glob("checkpoint_*.pth"))
    if epoch_ckpts:
        return epoch_ckpts[-1]
    return None


def _parse_variant(directory_name: str) -> Tuple[str, int]:
    if "_coeff" not in directory_name:
        raise ValueError(f"Cannot parse coeff window from directory name '{directory_name}'.")
    prefix, suffix = directory_name.split("_coeff", maxsplit=1)
    try:
        coeff = int(suffix)
    except ValueError as exc:
        raise ValueError(f"Invalid coeff window in '{directory_name}'.") from exc
    return prefix, coeff


def _load_dct_normaliser(
    coeff_window: int,
    range_mode: str,
    override: Optional[Path],
) -> Optional[NormalizeDCTCoefficients]:
    if override is not None:
        stats_path = override.expanduser()
        if not stats_path.exists():
            raise FileNotFoundError(f"DCT stats override does not exist: {stats_path}")
    else:
        stats_path = Path("configs/dct_stats") / f"imagenet_coeff{coeff_window}_{range_mode}.pt"
        if not stats_path.exists():
            return None
    return NormalizeDCTCoefficients.from_file(stats_path, coeff_window=coeff_window)


def _build_val_loader(
    val_dir: Path,
    image_size: int,
    batch_size: int,
    workers: int,
    compression_cfg: Dict[str, object],
    normaliser: Optional[NormalizeDCTCoefficients],
    trim_coefficients: bool,
    show_progress: bool,
    use_cuda: bool,
) -> DataLoader:
    _, val_tf = build_resnet_transforms(
        image_size,
        compression=compression_cfg,
        dct_normalizer_val=normaliser,
        trim_coefficients=trim_coefficients,
    )
    dataset = ImageNetDataset([str(val_dir)], transforms=val_tf, show_progress=show_progress)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, workers),
        pin_memory=use_cuda,
        drop_last=False,
    )
    return loader


def _clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def _load_model(
    variant: str,
    coeff_window: int,
    checkpoint_path: Path,
    device: torch.device,
    range_mode: str,
) -> nn.Module:
    model, _ = build_model("resnet34", _NUM_CLASSES)
    model.backbone = build_compressed_backbone(
        variant,
        model.backbone,
        range_mode=range_mode,
        mean=_IMAGENET_MEAN,
        std=_IMAGENET_STD,
        coeff_window=coeff_window,
    )
    if variant == "luma-fusion-pruned":
        hidden_dim = model.backbone.out_channels[0]
        model.head = ClassHead(hidden_dim=hidden_dim, num_classes=_NUM_CLASSES)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model") or checkpoint.get("state_dict") or checkpoint
    state_dict = _clean_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct1 = 0.0
    correct5 = 0.0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = _move_to_device(inputs, device)
            targets = targets.to(device)
            logits = normalise_logits(model(inputs))
            loss = criterion(logits, targets)
            total_loss += float(loss.item()) * targets.size(0)
            maxk = min(5, logits.size(1))
            _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred))
            correct1 += correct[:1].reshape(-1).float().sum().item()
            correct5 += correct[:5].reshape(-1).float().sum().item()
            total += targets.size(0)
    if total == 0:
        return {"loss": 0.0, "acc1": 0.0, "acc5": 0.0}
    return {
        "loss": total_loss / total,
        "acc1": correct1 / total,
        "acc5": correct5 / total,
    }


def _coefficients_erased(coeff_window: int) -> int:
    return max(0, 64 - coeff_window * coeff_window)


def _write_csv(csv_path: Path, rows: Iterable[Dict[str, object]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["variant", "coeff_window", "coefficients_erased", "throughput_img_s", "acc1", "acc5", "checkpoint"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_curves(plot_path: Path, grouped: Dict[str, List[Tuple[int, float]]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, points in grouped.items():
        if not points:
            continue
        points_sorted = sorted(points, key=lambda item: item[0])
        x_vals = [p[0] for p in points_sorted]
        y_vals = [p[1] * 100.0 for p in points_sorted]
        ax.plot(x_vals, y_vals, marker="o", label=label)
    ax.set_xlabel("Coefficients erased (per 8x8 block)")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("Compressed backbone accuracy vs. coefficient budget")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dirs = _discover_checkpoint_dirs(args.checkpoint_root)

    if args.csv_path is None:
        default_dir = args.checkpoint_root.resolve().parents[1] / "crianns_eval"
        csv_path = default_dir / "crianns_checkpoint_eval.csv"
    else:
        csv_path = args.csv_path.expanduser()
        default_dir = csv_path.parent
    plot_path = (args.plot_path.expanduser() if args.plot_path is not None else default_dir / "coeff_vs_accuracy.png")

    results: List[Dict[str, object]] = []
    curve_data: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    loader_cache: Dict[Tuple[int, bool, bool], DataLoader] = {}
    normaliser_cache: Dict[int, Optional[NormalizeDCTCoefficients]] = {}

    for directory in checkpoint_dirs:
        checkpoint = _select_checkpoint_file(directory)
        if checkpoint is None:
            print(f"[warn] No checkpoint found in {directory}; skipping.")
            continue
        try:
            variant, coeff_window = _parse_variant(directory.name)
        except ValueError as exc:
            print(f"[warn] {exc}; skipping directory {directory}.")
            continue
        trim_coefficients = coeff_window < 8 and variant != "reconstruction"
        compression_cfg = {
            "coeff_window": coeff_window,
            "range_mode": args.range_mode,
            "dtype": torch.float32,
            "keep_original": False,
        }
        if variant == "reconstruction":
            normaliser = None
        else:
            if coeff_window not in normaliser_cache:
                normaliser_cache[coeff_window] = _load_dct_normaliser(coeff_window, args.range_mode, args.dct_stats)
            normaliser = normaliser_cache[coeff_window]
        loader_key = (coeff_window, trim_coefficients, normaliser is not None)
        if loader_key not in loader_cache:
            loader_cache[loader_key] = _build_val_loader(
                args.val_dir,
                args.image_size,
                args.batch_size,
                args.workers,
                compression_cfg,
                normaliser,
                trim_coefficients,
                args.show_progress,
                use_cuda=device.type == "cuda",
            )
        val_loader = loader_cache[loader_key]
        print(f"[info] Evaluating {directory.name} ({checkpoint.name}) on {device}...")
        model = _load_model(variant, coeff_window, checkpoint, device, args.range_mode)
        metrics = _evaluate_accuracy(model, val_loader, device)
        benchmark = run_trimmed_inference_benchmark(
            model,
            device=device,
            coeff_window=coeff_window,
            image_size=args.image_size,
            batch_size=args.benchmark_batch_size,
            val_dirs=[str(args.val_dir)],
            compression_cfg=compression_cfg,
            dct_normalizer=normaliser,
            max_samples=args.benchmark_max_samples,
            workers=args.benchmark_workers,
            warmup_batches=args.benchmark_warmup,
            measure_batches=args.benchmark_measure,
            show_progress=args.show_progress,
            collect_metrics=False,
            trim_coefficients=trim_coefficients,
        )
        throughput = benchmark.throughput_img_s if benchmark is not None else None
        result_row = {
            "variant": variant,
            "coeff_window": coeff_window,
            "coefficients_erased": _coefficients_erased(coeff_window),
            "throughput_img_s": throughput,
            "acc1": metrics["acc1"],
            "acc5": metrics["acc5"],
            "checkpoint": str(checkpoint),
        }
        results.append(result_row)
        if variant in {"block-stem", "luma-fusion"}:
            curve_data[variant].append((_coefficients_erased(coeff_window), metrics["acc1"]))
        print(
            "  acc@1={acc1:.4%} acc@5={acc5:.4%} speed={speed}".format(
                acc1=metrics["acc1"],
                acc5=metrics["acc5"],
                speed="{:.2f} img/s".format(throughput) if throughput is not None else "n/a",
            )
        )

    if not results:
        print("[error] No evaluation results were produced.")
        return

    _write_csv(csv_path, results)
    print(f"[info] Wrote aggregated metrics to {csv_path}")

    _plot_curves(plot_path, curve_data)
    print(f"[info] Saved coefficient-erasure plot to {plot_path}")


if __name__ == "__main__":
    main()
