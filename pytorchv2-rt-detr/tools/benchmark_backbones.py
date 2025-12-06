#!/usr/bin/env python3
"""Backbone benchmarking utility.

Computes parameter counts and FLOPs for the standard RGB ResNet34 backbone and
all compressed-input variants at a set of spatial resolutions (default: 640x640,
matching the RT-DETR backbone input).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
import time

import torch

from src.nn.backbone.train_backbones import build_model, _IMAGENET_MEAN, _IMAGENET_STD
from src.nn.backbone.compressed_presnet import build_compressed_backbone
from src.data.transforms.compress_reference_images import CompressToDCT
from src.nn.arch.classification import ClassHead

try:  # Optional dependency for FLOP accounting
    from fvcore.nn import FlopCountAnalysis  # type: ignore
except ImportError:  # pragma: no cover
    FlopCountAnalysis = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device used to instantiate the models and run FLOP analysis (default: cpu).",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[256],
        help="Spatial resolutions (multiples of 8) to benchmark. Default uses 256x256.",
    )
    parser.add_argument(
        "--coeff-windows",
        type=int,
        nargs="+",
        default=[8],
        choices=[1, 2, 4, 8],
        help="Low-frequency window(s) used for compressed variants.",
    )
    parser.add_argument(
        "--range-mode",
        default="studio",
        choices=["studio", "full"],
        help="Pixel range used before the DCT transform for compressed variants.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "rgb-standard",
            "reconstruction",
            "block-stem",
            "luma-fusion",
            "luma-fusion-pruned",
        ],
        help="Backbone variants to benchmark. Include 'rgb-standard' for the baseline RGB model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for synthetic inputs.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Optional path to export benchmark results as CSV.",
    )
    return parser.parse_args()


def _ensure_multiple_of(values: Iterable[int], divisor: int = 8) -> List[int]:
    cleaned: List[int] = []
    for value in values:
        if value % divisor != 0:
            raise ValueError(f"Spatial size {value} is not divisible by {divisor}.")
        cleaned.append(value)
    return cleaned


def _build_model(variant: str, range_mode: str, coeff_window: int | None) -> torch.nn.Module:
    model, _ = build_model("resnet34", num_classes=1000)
    if variant == "rgb-standard":
        return model
    compressed_variant = variant
    model.backbone = build_compressed_backbone(
        compressed_variant,
        model.backbone,
        range_mode=range_mode,
        mean=_IMAGENET_MEAN,
        std=_IMAGENET_STD,
        coeff_window=coeff_window or 8,
    )
    if compressed_variant == "luma-fusion-pruned":
        hidden_dim = model.backbone.out_channels[0]
        model.head = ClassHead(hidden_dim=hidden_dim, num_classes=1000)
    return model


def _make_input(
    variant: str,
    size: int,
    device: torch.device,
    compressor: CompressToDCT | None,
) -> Sequence[torch.Tensor] | torch.Tensor:
    if variant == "rgb-standard":
        return torch.rand(1, 3, size, size, device=device)
    if compressor is None:
        raise RuntimeError("Compressor is required for compressed variants.")
    rgb = torch.rand(3, size, size)
    y_blocks, cbcr_blocks = compressor(rgb)
    return (
        y_blocks.unsqueeze(0).to(device=device, dtype=torch.float32),
        cbcr_blocks.unsqueeze(0).to(device=device, dtype=torch.float32),
    )


def _format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _resolve_presnet(model: torch.nn.Module) -> torch.nn.Module:
    backbone = model.backbone
    if hasattr(backbone, "backbone") and isinstance(backbone.backbone, torch.nn.Module):
        return backbone.backbone
    return backbone


def _collect_stage_shapes(model: torch.nn.Module, sample) -> List[tuple[int, ...]]:
    presnet = _resolve_presnet(model)
    targets = [1, 2, 3]
    captured: dict[int, tuple[int, ...]] = {}
    hooks = []

    for idx in targets:
        if idx >= len(presnet.res_layers):
            continue

        def _make_hook(stage_idx: int):
            def _hook(_module, _input, output):
                if isinstance(output, torch.Tensor):
                    captured[stage_idx] = tuple(output.shape)
                else:
                    captured[stage_idx] = tuple(output[0].shape)
            return _hook

        hooks.append(presnet.res_layers[idx].register_forward_hook(_make_hook(idx)))

    with torch.no_grad():
        _ = model.backbone(sample)

    for hook in hooks:
        hook.remove()

    shapes = [captured[idx] for idx in targets if idx in captured]
    return shapes


def _describe_inputs(sample) -> Tuple[str, str]:
    if isinstance(sample, torch.Tensor):
        dims = "x".join(str(dim) for dim in sample.shape[1:])
        return (dims, "n/a")
    if isinstance(sample, (list, tuple)) and len(sample) >= 2:
        y_blocks = sample[0]
        cbcr_blocks = sample[1]
        if isinstance(y_blocks, torch.Tensor):
            y_desc = "x".join(str(dim) for dim in y_blocks.shape[1:])
        else:
            y_desc = "?"
        if isinstance(cbcr_blocks, torch.Tensor):
            cbcr_desc = "x".join(str(dim) for dim in cbcr_blocks.shape[1:])
        else:
            cbcr_desc = "?"
        return (y_desc, cbcr_desc)
    return ("?", "?")


def _estimate_input_bytes(sample) -> float | None:
    def _per_sample_bytes(tensor: torch.Tensor) -> float:
        if tensor.shape[0] == 0:
            return 0.0
        return tensor.numel() / max(tensor.shape[0], 1) * tensor.element_size()

    if isinstance(sample, torch.Tensor):
        return _per_sample_bytes(sample)
    if isinstance(sample, (list, tuple)):
        total = 0.0
        has_tensor = False
        for item in sample:
            if isinstance(item, torch.Tensor):
                total += _per_sample_bytes(item)
                has_tensor = True
        return total if has_tensor else None
    return None


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    sizes = _ensure_multiple_of(args.sizes)
    coeff_windows = sorted(_ensure_multiple_of(args.coeff_windows, divisor=1))
    torch.manual_seed(args.seed)

    results = []
    for variant in args.variants:
        window_list: List[int | None]
        if variant == "rgb-standard":
            window_list = [None]
        else:
            window_list = coeff_windows

        for coeff_window in window_list:
            model = _build_model(variant, args.range_mode, coeff_window).to(device)
            model.eval()

            params_m = sum(p.numel() for p in model.parameters()) / 1e6

            compressor = None
            if coeff_window is not None:
                compressor = CompressToDCT(
                    coeff_window=coeff_window,
                    range_mode=args.range_mode,
                    dtype=torch.float32,
                    keep_original=False,
                )

            for size in sizes:
                sample = _make_input(
                    variant,
                    size,
                    device,
                    compressor,
                )
                gflops = None
                if FlopCountAnalysis is not None:
                    try:
                        flops = FlopCountAnalysis(model, (sample,)).total()
                        gflops = flops / 1e9
                    except Exception:  # pragma: no cover - best effort only
                        gflops = None
                peak_mem = None
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)

                repetitions = 10
                warmup = 3
                timings: List[float] = []
                with torch.no_grad():
                    for idx in range(warmup + repetitions):
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        start = time.perf_counter()
                        _ = model(sample)
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        elapsed = time.perf_counter() - start
                        if idx >= warmup:
                            timings.append(elapsed)
                avg_latency = float(sum(timings) / max(len(timings), 1))
                if device.type == "cuda":
                    peak_bytes = torch.cuda.max_memory_allocated(device)
                    peak_mem = peak_bytes / (1024 ** 2)
                out_desc = _collect_stage_shapes(model, sample)
                y_size, cbcr_size = _describe_inputs(sample)
                input_bytes = _estimate_input_bytes(sample)
                bs1 = bs64 = bs128 = None
                if input_bytes is not None:
                    bs1 = input_bytes / (1024 ** 2)
                    bs64 = input_bytes * 64 / (1024 ** 2)
                    bs128 = input_bytes * 128 / (1024 ** 2)
                param_bytes = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
                label = variant if coeff_window is None else f"{variant}@cw={coeff_window}"
                results.append(
                    (
                        label,
                        size,
                        params_m,
                        gflops,
                        out_desc,
                        peak_mem,
                        avg_latency,
                        y_size,
                        cbcr_size,
                        param_bytes,
                        bs1,
                        bs64,
                        bs128,
                    )
                )

    header = [
        "Model",
        "Input",
        "Params (M)",
        "GFLOPs",
        "Out Shapes",
        "Peak Mem (MB)",
        "Latency (ms)",
        "Y payload",
        "CbCr payload",
        "Model Mem (MB)",
        "Input@1 (MB)",
        "Input@64 (MB)",
        "Input@128 (MB)",
    ]
    rows = [header]
    for (
        model_name,
        size,
        params_m,
        gflops,
        shapes,
        peak_mem,
        latency,
        y_desc,
        cbcr_desc,
        model_mem,
        bs1,
        bs64,
        bs128,
    ) in results:
        out_desc = ", ".join(str(shape) for shape in shapes)
        rows.append(
            [
                model_name,
                f"{size}x{size}",
                f"{params_m:.2f}",
                _format_float(gflops),
                out_desc,
                _format_float(peak_mem) if peak_mem is not None else "n/a",
                f"{latency * 1000:.2f}",
                y_desc,
                cbcr_desc,
                _format_float(model_mem),
                _format_float(bs1),
                _format_float(bs64),
                _format_float(bs128),
            ]
        )

    col_widths = [max(len(row[col]) for row in rows) for col in range(len(header))]

    def _print_row(row: Sequence[str]) -> None:
        print(" | ".join(cell.ljust(col_widths[idx]) for idx, cell in enumerate(row)))

    _print_row(rows[0])
    print("-+-".join("-" * width for width in col_widths))
    for row in rows[1:]:
        _print_row(row)

    if FlopCountAnalysis is None:
        print("\nNote: Install fvcore (pip install fvcore) to enable FLOP reporting.")

    if args.csv_path is not None:
        csv_path = args.csv_path.expanduser()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            for row in rows:
                writer.writerow(row)
        print(f"Saved CSV results to {csv_path}")


if __name__ == "__main__":
    main()
