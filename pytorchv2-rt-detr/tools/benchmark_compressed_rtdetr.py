#!/usr/bin/env python3
"""Benchmark compressed RT-DETR variants and record basic stats."""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core import YAMLConfig, yaml_utils


_BLOCK_SIZE = 8


def _build_active_indices(coeff_window: Optional[int]) -> Optional[List[int]]:
    if coeff_window is None or coeff_window >= _BLOCK_SIZE:
        return None
    indices: List[int] = []
    for col in range(coeff_window):
        for row in range(coeff_window):
            indices.append(row + col * _BLOCK_SIZE)
    return indices


def _make_compressed_dummy(
    batch_size: int,
    image_size: int,
    coeff_window: Optional[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    grid = image_size // _BLOCK_SIZE
    if grid <= 0:
        raise ValueError("image_size must be divisible by 8 for compressed inputs")
    chroma_grid = grid // 2
    channels = _BLOCK_SIZE * _BLOCK_SIZE
    dtype = torch.float32

    y_blocks = torch.zeros((batch_size, channels, grid, grid), device=device, dtype=dtype)
    cbcr_blocks = torch.zeros((batch_size, 2, channels, chroma_grid, chroma_grid), device=device, dtype=dtype)

    active_idx = _build_active_indices(coeff_window)
    if active_idx is None:
        y_blocks.normal_()
        cbcr_blocks.normal_()
        return y_blocks, cbcr_blocks

    idx_tensor = torch.tensor(active_idx, device=device, dtype=torch.long)
    y_random = torch.randn((batch_size, len(active_idx), grid, grid), device=device, dtype=dtype)
    cbcr_random = torch.randn((batch_size, 2, len(active_idx), chroma_grid, chroma_grid), device=device, dtype=dtype)
    y_blocks.index_copy_(1, idx_tensor, y_random)
    cbcr_blocks.index_copy_(2, idx_tensor, cbcr_random)
    return y_blocks, cbcr_blocks


def _make_dummy_input(
    metadata: Dict[str, object],
    batch_size: int,
    image_size: int,
    device: torch.device,
) -> object:
    variant = metadata.get("compression_variant")
    if variant:
        coeff = metadata.get("coeff_window")
        # tuple expected by compressed backbones: (y_blocks, cbcr_blocks)
        return _make_compressed_dummy(batch_size, image_size, int(coeff) if coeff is not None else None, device)

    return torch.randn(batch_size, 3, image_size, image_size, device=device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "configs",
        nargs="*",
        type=Path,
        help="Explicit list of config files to benchmark.",
    )
    parser.add_argument(
        "--config-glob",
        type=str,
        default="configs/rtdetrv2/*coeff*_120e_coco.yml",
        help="Glob pattern (relative to repo root) used when no explicit configs are given.",
    )
    parser.add_argument(
        "--weights-root",
        type=Path,
        default=None,
        help="If provided, search this directory for checkpoints matching each config stem.",
    )
    parser.add_argument(
        "--weights-name",
        type=str,
        default="model_best.pth",
        help="Checkpoint filename to look for under each weights directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device string (defaults to CUDA if available).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Synthetic batch size used when benchmarking throughput.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="Input resolution (square) used for synthetic benchmarking.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warm-up iterations before timing.",
    )
    parser.add_argument(
        "--measure",
        type=int,
        default=20,
        help="Number of timed iterations used to compute throughput.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable autocast (mixed precision) during benchmarking.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/compressed_resnet34/crianns_eval/rtdetr_benchmark.csv"),
        help="CSV path where benchmark results will be written.",
    )
    parser.add_argument(
        "--extra-update",
        nargs="*",
        default=None,
        help="Optional key=value overrides applied when instantiating each config.",
    )
    return parser.parse_args()


def _discover_configs(args: argparse.Namespace) -> List[Path]:
    if args.configs:
        return [path.expanduser() for path in args.configs]
    return sorted(Path(REPO_ROOT).glob(args.config_glob))


def _maybe_load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> bool:
    if not checkpoint_path.exists():
        return False
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model") or checkpoint.get("state_dict") or checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] Missing keys when loading {checkpoint_path}: {missing}")
    if unexpected:
        print(f"[warn] Unexpected keys when loading {checkpoint_path}: {unexpected}")
    return True


def _count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _format_variant(metadata: Dict[str, object]) -> str:
    variant = metadata.get("compression_variant", "unknown")
    coeff = metadata.get("coeff_window")
    if coeff is None:
        return str(variant)
    return f"{variant}_coeff{coeff}"


def _lookup_metadata(cfg: YAMLConfig) -> Dict[str, object]:
    meta: Dict[str, object] = {}
    compressed = cfg.yaml_cfg.get("CompressedPResNet", {})
    meta["compression_variant"] = compressed.get("compression_variant")
    meta["coeff_window"] = compressed.get("coeff_window")
    meta["range_mode"] = compressed.get("range_mode")
    meta["depth"] = compressed.get("depth")
    return meta


def _benchmark_model(
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    image_size: int,
    warmup: int,
    measure: int,
    use_amp: bool,
    metadata: Dict[str, object],
) -> Dict[str, float]:
    model.eval()
    dummy = _make_dummy_input(metadata, batch_size, image_size, device)
    amp_ctx = torch.cuda.amp.autocast if use_amp and device.type == "cuda" else torch.cuda.amp.autocast
    # A no-op context manager for CPU when AMP requested
    class _CpuNoOp:
        def __enter__(self):
            return None
        def __exit__(self, exc_type, exc_val, exc_tb):
            return False
    autocast_ctx = amp_ctx(enabled=use_amp and device.type == "cuda") if device.type == "cuda" else _CpuNoOp()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for _ in range(max(0, warmup)):
            with autocast_ctx:
                _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(max(1, measure)):
            with autocast_ctx:
                _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
    iterations = max(1, measure)
    latency_ms = (elapsed / iterations) * 1000.0
    throughput = (batch_size * iterations) / elapsed if elapsed > 0 else math.inf
    peak_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda"
        else 0.0
    )
    return {
        "latency_ms": latency_ms,
        "throughput_img_s": throughput,
        "peak_memory_mb": peak_mb,
    }


def _resolve_weights_path(weights_root: Optional[Path], config_path: Path, weights_name: str) -> Optional[Path]:
    if weights_root is None:
        return None
    candidate = weights_root / config_path.stem / weights_name
    if candidate.exists():
        return candidate
    return None


def _write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    configs = _discover_configs(args)
    if not configs:
        raise SystemExit("No configuration files found for benchmarking.")

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[info] Using device: {device}")
    results: List[Dict[str, object]] = []

    cli_update = yaml_utils.parse_cli(args.extra_update) if args.extra_update else {}

    for config_path in configs:
        config_path = config_path.expanduser()
        if not config_path.is_absolute():
            config_path = (REPO_ROOT / config_path).resolve()
        config_rel = config_path.relative_to(REPO_ROOT)
        print(f"[info] Benchmarking config: {config_rel}")
        cfg = YAMLConfig(str(config_path), **cli_update)
        model = cfg.model
        adapt = getattr(model, "adapt_to_backbone", None)
        if callable(adapt):
            adapt()
        model = model.to(device)
        metadata = _lookup_metadata(cfg)
        variant_name = _format_variant(metadata)
        checkpoint = _resolve_weights_path(args.weights_root, config_path, args.weights_name)
        if checkpoint is not None:
            loaded = _maybe_load_checkpoint(model, checkpoint)
            status = "loaded" if loaded else "missing"
            print(f"    checkpoint: {checkpoint} ({status})")
        num_params = _count_parameters(model)
        bench = _benchmark_model(
            model,
            device=device,
            batch_size=args.batch_size,
            image_size=args.image_size,
            warmup=args.warmup,
            measure=args.measure,
            use_amp=args.amp,
            metadata=metadata,
        )
        results.append(
            {
                "config": str(config_rel),
                "variant": variant_name,
                "coeff_window": metadata.get("coeff_window"),
                "params_million": num_params / 1_000_000.0,
                "latency_ms": bench["latency_ms"],
                "throughput_img_s": bench["throughput_img_s"],
                "peak_memory_mb": bench["peak_memory_mb"],
                "device": str(device),
            }
        )
        # Free GPU memory before moving on to next model to avoid OOM.
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    for row in results:
        print(
            " - {variant}: {params:.2f}M params, {thr:.1f} img/s, {lat:.2f} ms, {mem:.1f} MB peak".format(
                variant=row["variant"],
                params=row["params_million"],
                thr=row["throughput_img_s"],
                lat=row["latency_ms"],
                mem=row["peak_memory_mb"],
            )
        )

    _write_csv(args.output.expanduser(), results)
    print(f"[info] Wrote benchmark results to {args.output}")


if __name__ == "__main__":
    main()
