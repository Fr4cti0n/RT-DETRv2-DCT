#!/usr/bin/env python3
"""Backbone benchmarking utility.

Computes parameter counts and FLOPs for the standard RGB ResNet34 backbone and
all compressed-input variants at a set of spatial resolutions (default: 640x640,
matching the RT-DETR backbone input).
"""

from __future__ import annotations

import argparse
from typing import Iterable, List, Sequence

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
                torch.cuda.reset_peak_memory_stats(device)

            out_desc = _collect_stage_shapes(model, sample)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peak_bytes = torch.cuda.max_memory_allocated(device)
                peak_mem = peak_bytes / (1024 ** 2)
            label = variant if coeff_window is None else f"{variant}@cw={coeff_window}"
            results.append((label, size, params_m, gflops, out_desc, peak_mem))

    header = ["Model", "Input", "Params (M)", "GFLOPs", "Out Shapes", "Peak Mem (MB)"]
    rows = [header]
    for model_name, size, params_m, gflops, shapes, peak_mem in results:
        out_desc = ", ".join(str(shape) for shape in shapes)
        rows.append(
            [
                model_name,
                f"{size}x{size}",
                f"{params_m:.2f}",
                _format_float(gflops),
                out_desc,
                _format_float(peak_mem) if peak_mem is not None else "n/a",
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


if __name__ == "__main__":
    main()
