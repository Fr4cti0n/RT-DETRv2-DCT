#!/usr/bin/env python3
"""Benchmark VRAM usage for RT-DETR/RT-DETRv2 inference on synthetic inputs.

- Loads a model from a YAML config (and optional checkpoint).
- Runs forward passes on random tensors at a fixed resolution.
- Reports peak GPU memory for batch sizes in a user-specified list.
- Supports both RGB backbones and compressed DCT backbones. When a compressed
    backbone is detected (``CompressedPResNet`` in the config), synthetic DCT
    payloads are generated with shapes matching the configured coefficient
    counts: ``(y, (cb, cr))`` where each tensor is ``(B, C, H/8, W/8)``.

Example:
    python tools/benchmark_inference_vram.py \
        --config configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml \
        --checkpoint output/detr_compressed34/rtdetrv2_r34vd_120e_coco_20260115-105920/best.pth \
        --batch-sizes 1 4 16 32 64 \
        --height 640 --width 640
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import torch

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.core import YAMLConfig  # noqa: E402


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = None
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "model_state", "module"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                break
    if state is None and isinstance(ckpt, dict):
        # Fall back to whole dict if it looks like a state_dict.
        if all(isinstance(k, str) for k in ckpt.keys()):
            state = ckpt
    if state is None:
        raise RuntimeError(f"Could not find model weights in checkpoint: {checkpoint_path}")

    missing = model.load_state_dict(state, strict=False)
    if missing.missing_keys:
        print(f"[warn] Missing keys when loading: {missing.missing_keys}")
    if missing.unexpected_keys:
        print(f"[warn] Unexpected keys when loading: {missing.unexpected_keys}")


def _measure_vram(
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    height: int,
    width: int,
    *,
    compressed: bool,
    coeff_luma: int,
    coeff_cb: int,
    coeff_cr: int,
    iterations: int,
    warmup: int,
) -> Tuple[float, float, float]:
    model.eval()

    if compressed:
        h_blocks = max(1, height // 8)
        w_blocks = max(1, width // 8)
        h_chroma = max(1, h_blocks // 2)
        w_chroma = max(1, w_blocks // 2)
        y = torch.randn(batch_size, coeff_luma, h_blocks, w_blocks, device=device)
        cb = (
            torch.randn(batch_size, coeff_cb, h_chroma, w_chroma, device=device)
            if coeff_cb > 0
            else y.new_zeros((batch_size, 0, h_chroma, w_chroma))
        )
        cr = (
            torch.randn(batch_size, coeff_cr, h_chroma, w_chroma, device=device)
            if coeff_cr > 0
            else y.new_zeros((batch_size, 0, h_chroma, w_chroma))
        )
        dummy = (y, (cb, cr))
    else:
        dummy = torch.randn(batch_size, 3, height, width, device=device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    latencies_ms: list[float] = []
    total = warmup + iterations
    with torch.no_grad():
        for i in range(total):
            start = time.perf_counter()
            try:
                _ = model(dummy)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            except RuntimeError as exc:  # likely OOM or shape mismatch
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                raise exc
            end = time.perf_counter()
            if i >= warmup:
                latencies_ms.append((end - start) * 1000.0)

    if device.type != "cuda":
        peak_mb = 0.0
    else:
        peak_bytes = torch.cuda.max_memory_allocated(device)
        peak_mb = peak_bytes / (1024 ** 2)

    mean_latency = sum(latencies_ms) / len(latencies_ms) if latencies_ms else float("nan")
    fps = batch_size / (mean_latency / 1000.0) if mean_latency == mean_latency else float("nan")
    return peak_mb, mean_latency, fps


def run_benchmark(
    config_path: Path,
    checkpoint_path: Path | None,
    batch_sizes: Sequence[int],
    height: int,
    width: int,
    device_str: str | None,
    iterations: int,
    warmup: int,
) -> list[tuple[int, float | str, float | str]]:
    cfg = YAMLConfig(str(config_path))
    device = torch.device(device_str) if device_str else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = cfg.model
    if checkpoint_path:
        print(f"[info] Loading checkpoint: {checkpoint_path}")
        _load_checkpoint(model, checkpoint_path)
    model.to(device)

    compressed_cfg = cfg.yaml_cfg.get("CompressedPResNet") if isinstance(cfg.yaml_cfg, dict) else None
    compressed = isinstance(compressed_cfg, dict) and compressed_cfg.get("compression_variant") is not None
    coeff_luma = int(compressed_cfg.get("coeff_count_luma", compressed_cfg.get("coeff_count", 64))) if compressed_cfg else 0
    coeff_cb = int(compressed_cfg.get("coeff_count_cb", compressed_cfg.get("coeff_count_chroma", coeff_luma))) if compressed_cfg else 0
    coeff_cr = int(compressed_cfg.get("coeff_count_cr", compressed_cfg.get("coeff_count_chroma", coeff_cb))) if compressed_cfg else 0
    if compressed:
        print(f"[info] Detected compressed backbone: coeffs Y={coeff_luma}, Cb={coeff_cb}, Cr={coeff_cr}")

    results: list[tuple[int, float | str, float | str]] = []
    for bs in batch_sizes:
        try:
            peak_mb, latency_ms, fps = _measure_vram(
                model,
                device,
                bs,
                height,
                width,
                compressed=compressed,
                coeff_luma=coeff_luma,
                coeff_cb=coeff_cb,
                coeff_cr=coeff_cr,
                iterations=iterations,
                warmup=warmup,
            )
            results.append((bs, round(peak_mb, 2), round(latency_ms, 3)))
            print(
                f"batch={bs:<3d} peak_vram={peak_mb:.2f} MB  "
                f"latency={latency_ms:.3f} ms  fps={fps:.2f}"
            )
        except RuntimeError as exc:
            results.append((bs, f"OOM: {exc}", "nan"))
            print(f"batch={bs:<3d} OOM: {exc}")
    return results


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional checkpoint to load")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 32, 64], help="Batch sizes to benchmark")
    parser.add_argument("--height", type=int, default=640, help="Input height")
    parser.add_argument("--width", type=int, default=640, help="Input width")
    parser.add_argument("--device", type=str, default=None, help="Device string (e.g., cuda, cuda:0, cpu)")
    parser.add_argument("--iters", type=int, default=5, help="Measured iterations per batch size")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations (not timed)")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    run_benchmark(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        batch_sizes=args.batch_sizes,
        height=args.height,
        width=args.width,
        device_str=args.device,
        iterations=max(1, int(args.iters)),
        warmup=max(0, int(args.warmup)),
    )


if __name__ == "__main__":
    main()
