"""Utility script to inspect registered backbones.

Run with:
  python -m src.nn.backbone.inspect_backbone --backbone PResNet --height 640 --width 640 -u depth=34 pretrained=False
"""

import argparse
import ast
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import torch

# Ensure the repository root is on sys.path when executed as a module.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.core import GLOBAL_CONFIG  # noqa: E402
from src.nn import backbone as backbone_pkg  # noqa: F401,E402  # ensure registry is populated


def _parse_updates(pairs: Iterable[str]) -> Dict[str, Any]:
    updates: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Malformed override '{pair}'. Use key=value format.")
        key, raw_value = pair.split("=", 1)
        raw_value = raw_value.strip()
        if raw_value.lower() == "none":
            value: Any = None
        else:
            try:
                value = ast.literal_eval(raw_value)
            except (ValueError, SyntaxError):
                # Fallback to string if it cannot be parsed as a Python literal.
                value = raw_value
        updates[key.strip()] = value
    return updates


def _list_available() -> Dict[str, Dict[str, Any]]:
    backbones: Dict[str, Dict[str, Any]] = {}
    for name, cfg in GLOBAL_CONFIG.items():
        if not isinstance(cfg, dict):
            continue
        if "_pymodule" not in cfg:
            continue
        module_name = cfg["_pymodule"].__name__
        if not module_name.startswith("src.nn.backbone"):
            continue
        backbones[name] = cfg
    return backbones


def _build_backbone(name: str, overrides: Dict[str, Any]) -> torch.nn.Module:
    registry = _list_available()
    if name not in registry:
        available = ", ".join(sorted(registry))
        raise ValueError(f"Unknown backbone '{name}'. Available: {available}")

    cfg = registry[name]
    defaults = dict(cfg.get("_kwargs", {}))
    defaults.update(overrides)

    module_cls = getattr(cfg["_pymodule"], name)
    model = module_cls(**defaults)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect RT-DETRv2 backbone outputs.")
    parser.add_argument("--backbone", required=False, help="Registered backbone name (use --list to enumerate).")
    parser.add_argument("--height", type=int, default=640, help="Input image height.")
    parser.add_argument("--width", type=int, default=640, help="Input image width.")
    parser.add_argument("--batch-size", type=int, default=1, help="Input batch size.")
    parser.add_argument("--device", default="cpu", help="Device to run the forward pass (cpu or cuda).")
    parser.add_argument("--list", action="store_true", help="List available backbones and exit.")
    parser.add_argument("-u", "--update", nargs="*", default=[], help="Override constructor args as key=value.")

    args = parser.parse_args()

    available = _list_available()
    if args.list:
        print("Registered backbones:")
        for name, cfg in sorted(available.items()):
            kwargs = cfg.get("_kwargs", {})
            sig = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            print(f"  - {name}({sig})")
        return

    if not args.backbone:
        parser.error("--backbone is required unless --list is specified.")

    overrides = _parse_updates(args.update)
    model = _build_backbone(args.backbone, overrides)
    model.eval()

    device = torch.device(args.device)
    model.to(device)

    dummy_input = torch.randn(args.batch_size, 3, args.height, args.width, device=device)

    with torch.no_grad():
        outputs = model(dummy_input)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Backbone: {args.backbone}")
    print(f"Input tensor: batch={args.batch_size}, channels=3, height={args.height}, width={args.width}")
    print(f"Total parameters: {total_params:,} ({total_params / 1e6:.2f} M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params / 1e6:.2f} M)")

    if hasattr(model, "out_channels"):
        print(f"out_channels: {getattr(model, 'out_channels')}")
    if hasattr(model, "out_strides"):
        print(f"out_strides: {getattr(model, 'out_strides')}")

    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]

    for idx, feat in enumerate(outputs):
        shape = tuple(feat.shape)
        mem = feat.numel() * feat.element_size() / (1024 ** 2)
        print(f"Output[{idx}]: shape={shape}, approx_memory={mem:.2f} MB")


if __name__ == "__main__":
    main()
