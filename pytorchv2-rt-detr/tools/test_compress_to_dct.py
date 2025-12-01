#!/usr/bin/env python3
"""Utility to inspect the output shape of the CompressToDCT transform."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

# Ensure project root is on sys.path so relative imports work when executed from tools/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.transforms.compress_reference_images import CompressToDCT  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Optional path to an image file.")
    parser.add_argument("--height", type=int, default=256, help="Height for the random image path.")
    parser.add_argument("--width", type=int, default=256, help="Width for the random image path.")
    parser.add_argument(
        "--components",
        nargs="+",
        default=("y", "cb", "cr"),
        choices=("y", "cb", "cr"),
        help="Component order to keep when stacking coefficients.",
    )
    parser.add_argument(
        "--coeff-window",
        type=int,
        default=4,
        choices=(1, 2, 4, 8),
        help="Square frequency window to retain within each 8x8 block.",
    )
    parser.add_argument(
        "--range-mode",
        type=str,
        default="studio",
        choices=("full", "studio"),
        help="Numeric range expected by the DCT compressor.",
    )
    parser.add_argument(
        "--input-range",
        type=str,
        default="auto",
        choices=("auto", "0-1", "-1-1", "0-255"),
        help="Range used when scaling floating tensors before compression.",
    )
    parser.add_argument(
        "--input-color",
        type=str,
        default="rgb",
        choices=("rgb", "bgr"),
        help="Declare whether the input tensor uses RGB or BGR ordering.",
    )
    parser.add_argument("--dtype", type=str, default="float32", help="Output tensor dtype.")
    parser.add_argument(
        "--output-structure",
        type=str,
        default="stacked",
        choices=("stacked", "dict", "tuple"),
        help="Controls how component tensors are returned (single tensor vs separated).",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="chw",
        choices=("chw", "hwc"),
        help="Layout for separated component tensors (ignored for stacked output).",
    )
    parser.add_argument(
        "--keep-full",
        action="store_true",
        help="Keep all 64 coefficients per block even when using a smaller coeff-window.",
    )
    parser.add_argument(
        "--metadata-key",
        type=str,
        default=None,
        help="Attach metadata under this key when using detection-style samples.",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for synthetic inputs.")
    parser.add_argument(
        "--uint8",
        action="store_true",
        help="Generate a random uint8 tensor instead of float when no image is provided.",
    )
    return parser.parse_args()


def load_input(args: argparse.Namespace) -> tuple[object, str]:
    if args.image is not None:
        if not args.image.exists():
            raise FileNotFoundError(f"Image path does not exist: {args.image}")
        image = Image.open(args.image).convert("RGB")
        description = f"PIL image -> RGB ({image.size[0]}x{image.size[1]})"
        return image, description

    torch.manual_seed(args.seed)
    shape = (3, args.height, args.width)
    if args.uint8:
        tensor = torch.randint(0, 256, shape, dtype=torch.uint8)
        description = f"Random uint8 tensor with shape {shape}"
    else:
        tensor = torch.rand(shape, dtype=torch.float32) * 2.0 - 1.0
        description = f"Random float tensor with shape {shape} (range [-1, 1])"
    return tensor, description


def main() -> None:
    args = parse_args()

    image, description = load_input(args)
    print(f"Loaded input: {description}")

    layout = args.layout
    if args.output_structure == "stacked" and layout != "chw":
        print("[info] Forcing layout to 'chw' because stacked output requires channel-first tensors.")
        layout = "chw"

    keep_full = True if args.keep_full else None

    transform = CompressToDCT(
        coeff_window=args.coeff_window,
        range_mode=args.range_mode,
        components=tuple(args.components),
        input_range=args.input_range,
        input_color=args.input_color,
        dtype=args.dtype,
        output_structure=args.output_structure,
        tensor_layout=layout,
        keep_full_coefficients=keep_full,
        metadata_key=args.metadata_key,
    )

    sample_target: dict[str, object] = {}
    output, target = transform(image, sample_target)

    print("\n=== CompressToDCT Result ===")
    print(f"Output structure    : {args.output_structure}")

    def print_tensor_stats(name: str, tensor: torch.Tensor) -> None:
        shape = tuple(tensor.shape)
        min_val = tensor.amin().item() if tensor.numel() else float("nan")
        max_val = tensor.amax().item() if tensor.numel() else float("nan")
        print(f"  {name}: shape={shape}, dtype={tensor.dtype}, range=[{min_val:.2f}, {max_val:.2f}]")

    if isinstance(output, torch.Tensor):
        print_tensor_stats("tensor", output)
    elif isinstance(output, dict):
        print("Component tensors:")
        for component, tensor in output.items():
            print_tensor_stats(component, tensor)
    elif isinstance(output, (list, tuple)):
        print("Component tensors:")
        for component, tensor in zip(args.components, output):
            print_tensor_stats(component, tensor)
    else:
        print(f"Unexpected output type: {type(output)!r}")

    if args.metadata_key is not None:
        metadata = target.get(args.metadata_key, {})
        print("\nMetadata:")
        if not metadata:
            print(f"  No metadata found under key '{args.metadata_key}'.")
        else:
            for key, value in metadata.items():
                print(f"  {key}: {value}")
    else:
        if isinstance(output, torch.Tensor):
            channel_per_component = output.shape[0] // len(args.components)
            print(
                f"Channels per component: {channel_per_component}"
                f" (components={list(args.components)})"
            )
        elif isinstance(output, dict):
            print("Returned components:")
            for component, tensor in output.items():
                if layout == "chw":
                    coeff_dim = tensor.shape[0]
                    spatial = tensor.shape[1:]
                else:
                    coeff_dim = tensor.shape[-1]
                    spatial = tensor.shape[:2]
                print(f"  {component}: coefficients={coeff_dim}, spatial={spatial}")
        elif isinstance(output, (list, tuple)):
            print("Returned components:")
            for component, tensor in zip(args.components, output):
                if layout == "chw":
                    coeff_dim = tensor.shape[0]
                    spatial = tensor.shape[1:]
                else:
                    coeff_dim = tensor.shape[-1]
                    spatial = tensor.shape[:2]
                print(f"  {component}: coefficients={coeff_dim}, spatial={spatial}")


if __name__ == "__main__":
    main()
