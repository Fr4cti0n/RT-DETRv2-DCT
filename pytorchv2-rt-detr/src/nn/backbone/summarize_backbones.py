"""Generate a CSV summary for the PResNet and CSPDarkNet backbones."""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Union

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.core import GLOBAL_CONFIG  # noqa: E402
from src.nn import backbone as backbone_pkg  # noqa: F401,E402  # populate registry


TARGET_BACKBONES = {"PResNet", "CSPDarkNet"}


class ArgsNamespace:
    """Simple container for common arguments passed to helper functions."""

    def __init__(self, batch_size: int, height: int, width: int):
        self.batch_size = batch_size
        self.height = height
        self.width = width


def _list_backbones() -> Dict[str, Dict]:
    registered: Dict[str, Dict] = {}
    for name, cfg in GLOBAL_CONFIG.items():
        if not isinstance(cfg, dict):
            continue
        if "_pymodule" not in cfg:
            continue
        module_name = cfg["_pymodule"].__name__
        if module_name.startswith("src.nn.backbone"):
            registered[name] = cfg
    return registered


def _resolve_selection(available: Iterable[str], requested: Iterable[str]) -> List[str]:
    available_set = set(available)
    if not requested:
        return sorted(available_set)
    missing = [name for name in requested if name not in available_set]
    if missing:
        raise ValueError(f"Unknown backbone(s): {', '.join(missing)}")
    return sorted(set(requested))


def _format_shape(tensor: torch.Tensor) -> str:
    return "x".join(str(dim) for dim in tensor.shape)


def _format_input_shape(inp: Union[torch.Tensor, List[torch.Tensor]]) -> str:
    if isinstance(inp, (list, tuple)):
        return "[" + "; ".join(_format_shape(t) for t in inp) + "]"
    return _format_shape(inp)


BACKBONE_PROFILES: Dict[str, Dict] = {
    "PResNet": {
        "kwargs": {"depth": 34, "return_idx": [1, 2, 3], "pretrained": False},
    },
}


def _instantiate(name: str, cfg: Dict) -> torch.nn.Module:
    defaults = dict(cfg.get("_kwargs", {}))
    defaults.update(BACKBONE_PROFILES.get(name, {}).get("kwargs", {}))
    module_cls = getattr(cfg["_pymodule"], name)
    return module_cls(**defaults)


def _build_input(name: str, model: torch.nn.Module, args: ArgsNamespace, device: torch.device) -> Union[torch.Tensor, List[torch.Tensor]]:
    builder = BACKBONE_PROFILES.get(name, {}).get("input_builder")
    if builder is not None:
        return builder(model, args, device)
    return torch.randn(args.batch_size, 3, args.height, args.width, device=device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise RT-DETRv2 backbones into CSV.")
    parser.add_argument("--output", type=Path, default=Path("backbone_summary.csv"), help="Output CSV path.")
    parser.add_argument("--height", type=int, default=640, help="Input image height.")
    parser.add_argument("--width", type=int, default=640, help="Input image width.")
    parser.add_argument("--batch-size", type=int, default=1, help="Dummy batch size.")
    parser.add_argument("--device", default="cpu", help="Device to run the forward pass (cpu or cuda).")
    parser.add_argument("--include", nargs="*", default=[], help="Subset of backbone names to export.")

    args = parser.parse_args()

    available = {name: cfg for name, cfg in _list_backbones().items() if name in TARGET_BACKBONES}
    if not available:
        raise RuntimeError(f"None of the target backbones {sorted(TARGET_BACKBONES)} are registered.")
    selected_names = _resolve_selection(available.keys(), args.include)

    device = torch.device(args.device)
    cli_args = ArgsNamespace(args.batch_size, args.height, args.width)

    rows: List[Dict[str, Union[str, int, float]]] = []
    max_outputs = 0

    for name in selected_names:
        cfg = available[name]
        row: Dict[str, Union[str, int, float]] = {"backbone": name}

        try:
            model = _instantiate(name, cfg)
        except Exception as exc:  # pylint: disable=broad-except
            row["error"] = f"instantiate failed: {exc}"
            rows.append(row)
            continue

        model.eval().to(device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        row.update(
            {
                "total_params": total_params,
                "total_params_m": f"{total_params / 1e6:.2f}",
                "trainable_params": trainable_params,
                "trainable_params_m": f"{trainable_params / 1e6:.2f}",
            }
        )

        if hasattr(model, "out_channels"):
            row["out_channels"] = list(getattr(model, "out_channels"))
        if hasattr(model, "out_strides"):
            row["out_strides"] = list(getattr(model, "out_strides"))

        try:
            dummy_input = _build_input(name, model, cli_args, device)
        except Exception as exc:  # pylint: disable=broad-except
            row["error"] = f"input build failed: {exc}"
            rows.append(row)
            continue

        row["input_shape"] = _format_input_shape(dummy_input)

        try:
            with torch.no_grad():
                outputs = model(dummy_input)
        except Exception as exc:  # pylint: disable=broad-except
            row["error"] = f"forward failed: {exc}"
            rows.append(row)
            continue

        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        for idx, feat in enumerate(outputs):
            row[f"output_{idx}_shape"] = _format_shape(feat)
        max_outputs = max(max_outputs, len(outputs))

        rows.append(row)

    base_columns = [
        "backbone",
        "total_params",
        "total_params_m",
        "trainable_params",
        "trainable_params_m",
        "input_shape",
        "out_channels",
        "out_strides",
    ]
    extra_columns = [f"output_{idx}_shape" for idx in range(max_outputs)]
    header = base_columns + extra_columns + ["error"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in header})

    print(f"Wrote {len(rows)} backbone summaries to {args.output}")


if __name__ == "__main__":
    main()
