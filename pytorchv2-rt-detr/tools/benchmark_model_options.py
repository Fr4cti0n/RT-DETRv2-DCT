#!/usr/bin/env python3
"""Benchmark RT-DETR/RT-DETRv2 model options on synthetic inputs.

- Instantiates the model from a YAML config with optional per-profile overrides
  (e.g., num_queries, num_layers, channel_scale, coeff counts).
- Runs warmup + timed iterations on random tensors (RGB or compressed DCT) to
  report latency, throughput, peak VRAM, parameter count, FLOPs (if fvcore is
  installed), and optional GPU power draw (if NVML is available).

Usage example:
    python tools/benchmark_model_options.py \
        --config configs/rtdetrv2/lumafusion_coeffs/rtdetrv2_r34vd_lumafusion_coeffY16_Cb16_Cr16_120e_coco.yml \
        --checkpoint output/compressed_resnet34/luma-fusion_coeffY16_CbCr16_cs50_20260120-110950_epoch0002/checkpoint_last.pth \
        --batch-size 4 --height 640 --width 640 --iters 30 --warmup 5 \
        --profile base \
        --profile dec4:RTDETRTransformerv2.num_layers=4,RTDETRTransformerv2.num_queries=200 \
        --profile dec6:RTDETRTransformerv2.num_layers=6,RTDETRTransformerv2.num_queries=300

Notes:
- FLOP counting requires fvcore (pip install fvcore). If unavailable, FLOPs show
  as "n/a".
- GPU power draw is only reported on CUDA when NVIDIA NVML is available
  (pynvml import succeeds). Otherwise it is "n/a".
- Overrides are parsed with the repo's yaml_utils.parse_cli so values keep their
  types (ints/floats/bools).
"""

from __future__ import annotations

import argparse
import math
import re
import time
import copy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import itertools

import torch
import torch.profiler as torch_prof

try:
    from fvcore.nn import FlopCountAnalysis  # type: ignore
except Exception:  # pragma: no cover
    FlopCountAnalysis = None

try:
    import pynvml  # type: ignore
except Exception:  # pragma: no cover
    pynvml = None

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core import YAMLConfig, yaml_utils  # noqa: E402


def _nested_update(overrides: Dict[str, object]) -> Dict[str, object]:
    """Convert dotted keys to nested dicts (same as yaml_utils.parse_cli behavior)."""
    nested: Dict[str, object] = {}
    for key, val in overrides.items():
        nested = yaml_utils.merge_dict(nested, yaml_utils.dictify(key, val))
    return nested


def _parse_profile(spec: str) -> tuple[str, Dict[str, object]]:
    """Parse "name:key=val,key2=val2" into (name, overrides dict)."""
    if ":" not in spec:
        return spec.strip(), {}
    name, payload = spec.split(":", 1)
    updates = yaml_utils.parse_cli(payload.split(",")) if payload else {}
    return name.strip(), updates


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = None
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "model_state", "module"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                break
    if state is None and isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        state = ckpt
    if state is None:
        raise RuntimeError(f"Could not find model weights in checkpoint: {checkpoint_path}")
    missing = model.load_state_dict(state, strict=False)
    if missing.missing_keys:
        print(f"[warn] Missing keys when loading: {missing.missing_keys}")
    if missing.unexpected_keys:
        print(f"[warn] Unexpected keys when loading: {missing.unexpected_keys}")


def _detect_compressed(cfg: YAMLConfig) -> tuple[bool, int, int, int]:
    compressed_cfg = cfg.yaml_cfg.get("CompressedPResNet") if isinstance(cfg.yaml_cfg, dict) else None
    compressed = isinstance(compressed_cfg, dict) and compressed_cfg.get("compression_variant") is not None
    coeff_luma = int(compressed_cfg.get("coeff_count_luma", compressed_cfg.get("coeff_count", 64))) if compressed_cfg else 0
    coeff_cb = int(compressed_cfg.get("coeff_count_cb", compressed_cfg.get("coeff_count_chroma", coeff_luma))) if compressed_cfg else 0
    coeff_cr = int(compressed_cfg.get("coeff_count_cr", compressed_cfg.get("coeff_count_chroma", coeff_cb))) if compressed_cfg else 0
    return compressed, coeff_luma, coeff_cb, coeff_cr


def _make_dummy(
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    *,
    compressed: bool,
    coeff_luma: int,
    coeff_cb: int,
    coeff_cr: int,
) -> object:
    if compressed:
        h_blocks = max(1, height // 8)
        w_blocks = max(1, width // 8)
        h_chroma = max(1, h_blocks // 2)
        w_chroma = max(1, w_blocks // 2)
        y = torch.randn(batch_size, coeff_luma, h_blocks, w_blocks, device=device)
        cb = torch.randn(batch_size, coeff_cb, h_chroma, w_chroma, device=device) if coeff_cb > 0 else y.new_zeros((batch_size, 0, h_chroma, w_chroma))
        cr = torch.randn(batch_size, coeff_cr, h_chroma, w_chroma, device=device) if coeff_cr > 0 else y.new_zeros((batch_size, 0, h_chroma, w_chroma))
        return (y, (cb, cr))
    return torch.randn(batch_size, 3, height, width, device=device)


def _device_from_str(device_str: Optional[str]) -> torch.device:
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _nvml_prepare(device: torch.device):
    if pynvml is None or device.type != "cuda":
        return None
    try:
        pynvml.nvmlInit()
        idx = 0
        if device.index is not None:
            idx = int(device.index)
        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
        return handle
    except Exception:
        return None


def _nvml_power_w(handle) -> float:
    if handle is None:
        return float("nan")
    try:
        return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W
    except Exception:
        return float("nan")


def _measure(
    model: torch.nn.Module,
    dummy: object,
    device: torch.device,
    warmup: int,
    iters: int,
    warmup_seconds: float,
    measure_seconds: float,
    use_amp: bool,
    power_handle,
    power_sample_ms: float,
) -> tuple[float, float, float, float, float, float, float, float, float, int]:
    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    max_watt = -math.inf
    power_samples: List[float] = []
    sample_interval = max(0.0, power_sample_ms / 1000.0)
    last_sample_ts = -math.inf
    dtype_ctx = torch.cuda.amp.autocast if device.type == "cuda" else torch.cpu.amp.autocast
    autocast_ctx = dtype_ctx(enabled=use_amp)

    with torch.no_grad():
        # Warmup
        if warmup_seconds > 0:
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < warmup_seconds:
                with autocast_ctx:
                    _ = model(dummy)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
        else:
            for _ in range(max(0, warmup)):
                with autocast_ctx:
                    _ = model(dummy)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        start = time.perf_counter()
        iterations = 0
        if measure_seconds > 0:
            while time.perf_counter() - start < measure_seconds:
                with autocast_ctx:
                    _ = model(dummy)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                iterations += 1
                now = time.perf_counter()
                if now - last_sample_ts >= sample_interval:
                    watt = _nvml_power_w(power_handle)
                    if watt == watt:
                        max_watt = max(max_watt, watt)
                        power_samples.append(watt)
                        last_sample_ts = now
        else:
            while iterations < max(1, iters):
                with autocast_ctx:
                    _ = model(dummy)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                iterations += 1
                now = time.perf_counter()
                if now - last_sample_ts >= sample_interval:
                    watt = _nvml_power_w(power_handle)
                    if watt == watt:
                        max_watt = max(max_watt, watt)
                        power_samples.append(watt)
                        last_sample_ts = now
        elapsed = time.perf_counter() - start

    latency_ms = (elapsed / iterations) * 1000.0 if iterations > 0 else float("nan")
    throughput = (dummy[0].shape[0] if isinstance(dummy, (list, tuple)) and hasattr(dummy[0], "shape") else dummy.shape[0]) * iterations / elapsed if elapsed > 0 and iterations > 0 else float("nan")
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
    max_watt = max_watt if max_watt != -math.inf else float("nan")
    mean_watt = (sum(power_samples) / len(power_samples)) if power_samples else float("nan")
    p95_watt = float("nan")
    if power_samples:
        sorted_samples = sorted(power_samples)
        idx = min(len(sorted_samples) - 1, max(0, math.ceil(0.95 * len(sorted_samples)) - 1))
        p95_watt = sorted_samples[idx]
    energy_j = mean_watt * elapsed if mean_watt == mean_watt else float("nan")
    images = throughput * elapsed if throughput == throughput else float("nan")
    joules_per_img = energy_j / images if energy_j == energy_j and images and images == images else float("nan")
    return latency_ms, throughput, peak_mb, max_watt, mean_watt, p95_watt, energy_j, joules_per_img, elapsed, len(power_samples)


def _count_params(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000.0


def _is_compressed_sample(sample: object) -> bool:
    return isinstance(sample, (list, tuple)) and len(sample) >= 1 and isinstance(sample[0], torch.Tensor) and sample[0].dim() == 4


def _to_device_sample(sample: object, device: torch.device):
    if torch.is_tensor(sample):
        return sample.to(device=device)
    if isinstance(sample, (list, tuple)):
        return type(sample)(_to_device_sample(x, device) for x in sample)
    return sample


def _count_flops(model: torch.nn.Module, sample: object, *, allow_compressed: bool = False) -> float | float("nan"):
    if FlopCountAnalysis is None:
        print("[warn] fvcore not installed: pip install fvcore to enable FLOP counting")
        return float("nan")
    if _is_compressed_sample(sample) and not allow_compressed:
        print("[warn] FLOP counting skipped for compressed DCT inputs (fvcore/profiler lack support for nested tuples)")
        return float("nan")
    try:
        # fvcore expects the sample in the same structure the model forward consumes.
        flops = FlopCountAnalysis(model, (sample,)).total()
        return flops / 1e9
    except Exception as exc:
        print(f"[warn] FLOP counting failed: {exc}")
    # Fallback to torch.profiler with flops if fvcore failed
    try:
        activities = [torch_prof.ProfilerActivity.CPU]
        if torch.cuda.is_available() and any(p.is_cuda for p in model.parameters() if p is not None):
            activities.append(torch_prof.ProfilerActivity.CUDA)
        with torch.no_grad(), torch_prof.profile(activities=activities, with_flops=True) as prof:
            _ = model(sample)
        flops = prof.key_averages().total_average().flops
        if flops is None:
            raise RuntimeError("profiler returned None flops")
        return flops / 1e9
    except Exception as exc:
        print(f"[warn] torch.profiler FLOP counting failed: {exc}")
    # Final fallback: copy model to CPU and try fvcore
    try:
        model_cpu = copy.deepcopy(model).cpu()
        sample_cpu = _to_device_sample(sample, torch.device("cpu"))
        flops = FlopCountAnalysis(model_cpu, (sample_cpu,)).total()
        return flops / 1e9
    except Exception as exc:
        print(f"[warn] CPU fvcore FLOP counting failed: {exc}")
        return float("nan")


def _should_load_checkpoint(base_cfg: YAMLConfig, overrides: Dict[str, object]) -> bool:
    """Skip checkpoint if overrides change backbone shape-sensitive fields."""
    base_cmp = base_cfg.yaml_cfg.get("CompressedPResNet", {}) if isinstance(base_cfg.yaml_cfg, dict) else {}
    override_cmp = overrides.get("CompressedPResNet", {}) if isinstance(overrides, dict) else {}
    keys = [
        "channel_scale",
        "coeff_count_luma",
        "coeff_count_cb",
        "coeff_count_cr",
        "coeff_count_chroma",
        "depth",
    ]
    for key in keys:
        if key in override_cmp:
            if key not in base_cmp or base_cmp.get(key) != override_cmp.get(key):
                return False
    return True


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional checkpoint to load")
    parser.add_argument("--no-load", action="store_true", help="Skip loading checkpoints for all profiles")
    parser.add_argument("--device", type=str, default=None, help="Device string (cuda, cuda:0, cpu)")
    parser.add_argument("--height", type=int, default=640, help="Input height")
    parser.add_argument("--width", type=int, default=640, help="Input width")
    parser.add_argument("--batch-size", type=int, default=8, help="Synthetic batch size")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations (ignored if warmup-seconds > 0)")
    parser.add_argument("--iters", type=int, default=80, help="Timed iterations (ignored if measure-seconds > 0)")
    parser.add_argument("--warmup-seconds", type=float, default=30.0, help="Warmup duration in seconds (overrides warmup iters when >0)")
    parser.add_argument("--measure-seconds", type=float, default=60.0, help="Measurement duration in seconds (overrides iters when >0)")
    parser.add_argument("--power-sample-ms", type=float, default=200.0, help="Interval in milliseconds between power samples (e.g., 100-200 for 5-10 Hz)")
    parser.add_argument("--amp", action="store_true", help="Use autocast")
    parser.add_argument("--profile", action="append", default=["base"], help="Named override set: name:key=val,key2=val2")
    parser.add_argument("--layers", type=int, nargs="*", help="Decoder layers sweep (e.g., 4 3 2 1)")
    parser.add_argument("--queries", type=int, nargs="*", help="Num queries sweep (e.g., 300 200 100)")
    parser.add_argument("--channel-scales", type=float, nargs="*", help="Backbone channel_scale sweep (e.g., 0.5 0.6 0.7)")
    parser.add_argument(
        "--coeff-sets",
        type=str,
        nargs="*",
        help="Coeff overrides as Y,Cb,Cr (e.g., 64,64,64 16,16,16 64,0,0)",
    )
    parser.add_argument(
        "--approx-fused-flops",
        action="store_true",
        help="Approximate FLOPs by bypassing DCT fusion and feeding a fused conv1 output tensor (only for compressed backbones).",
    )
    parser.add_argument(
        "--rest-seconds",
        type=int,
        default=0,
        help="Seconds to rest between profiles (use >0 to let GPU cool; default 0).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    device = _device_from_str(args.device)
    power_handle = _nvml_prepare(device)
    base_cfg_for_ckpt = YAMLConfig(str(args.config))

    profiles = [_parse_profile(spec) for spec in args.profile]

    # Auto-generate grid profiles if sweeps are provided.
    coeff_sets: List[Tuple[int, int, int]] = []
    if args.coeff_sets:
        for spec in args.coeff_sets:
            parts = [p.strip() for p in spec.split(",") if p.strip()]
            if len(parts) != 3:
                raise SystemExit(f"Invalid coeff set '{spec}'. Expected Y,Cb,Cr.")
            y, cb, cr = map(int, parts)
            coeff_sets.append((y, cb, cr))

    if args.layers or args.queries or args.channel_scales or coeff_sets:
        layers = args.layers or [None]
        queries = args.queries or [None]
        channel_scales = args.channel_scales or [None]
        coeff_grid = coeff_sets or [(None, None, None)]
        for l, q, cs, (y, cb, cr) in itertools.product(layers, queries, channel_scales, coeff_grid):
            name_parts = []
            overrides: Dict[str, object] = {}
            if l is not None:
                name_parts.append(f"L{l}")
                overrides["RTDETRTransformerv2.num_layers"] = l
            if q is not None:
                name_parts.append(f"Q{q}")
                overrides["RTDETRTransformerv2.num_queries"] = q
            if cs is not None:
                name_parts.append(f"cs{cs}")
                overrides["CompressedPResNet.channel_scale"] = cs
            if y is not None:
                name_parts.append(f"Y{y}Cb{cb}Cr{cr}")
                overrides["CompressedPResNet.coeff_count_luma"] = y
                overrides["CompressedPResNet.coeff_count_cb"] = cb
                overrides["CompressedPResNet.coeff_count_cr"] = cr
            profile_name = "_".join(name_parts) if name_parts else "grid"
            profiles.append((profile_name, _nested_update(overrides)))
    rows: List[Dict[str, object]] = []

    for idx, (name, overrides) in enumerate(profiles):
        print(f"[info] Profile '{name}' overrides: {overrides if overrides else 'none'}")
        cfg = YAMLConfig(str(args.config), **overrides)
        model = cfg.model
        adapt = getattr(model, "adapt_to_backbone", None)
        if callable(adapt):
            adapt()
        if args.checkpoint and not args.no_load:
            if _should_load_checkpoint(base_cfg_for_ckpt, overrides):
                print(f"[info] Loading checkpoint: {args.checkpoint}")
                _load_checkpoint(model, args.checkpoint)
            else:
                print(f"[info] Skipping checkpoint load for profile '{name}' (shape-changing overrides)")
        else:
            if args.no_load:
                print("[info] --no-load set; skipping checkpoint for all profiles")
        model = model.to(device)

        compressed, coeff_luma, coeff_cb, coeff_cr = _detect_compressed(cfg)
        dummy = _make_dummy(
            batch_size=args.batch_size,
            height=args.height,
            width=args.width,
            device=device,
            compressed=compressed,
            coeff_luma=coeff_luma,
            coeff_cb=coeff_cb,
            coeff_cr=coeff_cr,
        )

        fused_dummy = None
        if args.approx_fused_flops and compressed:
            target_channels = getattr(model.backbone, "target_channels", None)
            if target_channels is None:
                print("[warn] approx-fused-flops requested but target_channels missing; skipping fused FLOPs")
            else:
                # After luma_down (stride 2) and chroma_proj (stride 1), fusion produces target_channels @ H/16, then upsample x4 to H/4
                h4 = max(1, args.height // 4)
                w4 = max(1, args.width // 4)
                fused_dummy = torch.randn(args.batch_size, int(target_channels), h4, w4, device=device)

        params_m = _count_params(model)
        flops_g = _count_flops(model, dummy)
        if fused_dummy is not None:
            try:
                if hasattr(model.backbone, "_forward_residual_stages"):
                    class _FusedRTDETR(torch.nn.Module):
                        def __init__(self, base_model):
                            super().__init__()
                            self.base = base_model
                        def forward(self, x):
                            feats = self.base.backbone._forward_residual_stages(x, skip_pool=True)
                            feats = self.base.encoder(feats)
                            return self.base.decoder(feats)
                    wrapper = _FusedRTDETR(model)
                    flops_g = _count_flops(wrapper, fused_dummy, allow_compressed=True)
                else:
                    print("[warn] approx-fused-flops requested but backbone has no _forward_residual_stages; skipping")
            except Exception as exc:
                print(f"[warn] approx-fused-flops failed: {exc}")
        latency_ms, throughput, peak_mb, max_watt, mean_watt, p95_watt, energy_j, joules_per_img, duration_s, samples_power = _measure(
            model,
            dummy,
            device,
            warmup=args.warmup,
            iters=args.iters,
            warmup_seconds=args.warmup_seconds,
            measure_seconds=args.measure_seconds,
            use_amp=args.amp,
            power_handle=power_handle,
            power_sample_ms=args.power_sample_ms,
        )

        print(
            "Profile {name}: params={params:.2f}M, FLOPs={flops}, latency={lat:.2f} ms, throughput={thr:.2f} img/s, "
            "peak_vram={vram:.1f} MB, max_power={power}".format(
                name=name,
                params=params_m,
                flops=f"{flops_g:.2f}G" if flops_g == flops_g else "n/a",
                lat=latency_ms,
                thr=throughput,
                vram=peak_mb,
                power=f"{max_watt:.1f} W" if max_watt == max_watt else "n/a",
            )
        )
        rows.append(
            {
                "profile": name,
                "params_m": params_m,
                "flops_g": flops_g,
                "latency_ms": latency_ms,
                "throughput": throughput,
                "peak_vram_mb": peak_mb,
                "max_power_w": max_watt,
                "mean_power_w": mean_watt,
                "p95_power_w": p95_watt,
                "energy_j": energy_j,
                "joules_per_img": joules_per_img,
                "duration_s": duration_s,
                "samples_power": samples_power,
            }
        )

        if args.rest_seconds and idx < len(profiles) - 1:
            print(f"[info] Resting {args.rest_seconds}s before next profile...")
            time.sleep(max(0, args.rest_seconds))

    if rows:
        headers = [
            "profile",
            "params_m",
            "flops_g",
            "latency_ms",
            "throughput",
            "peak_vram_mb",
            "max_power_w",
            "mean_power_w",
            "p95_power_w",
            "energy_j",
            "joules_per_img",
            "duration_s",
            "samples_power",
        ]

        def _fmt(val: object) -> str:
            if isinstance(val, float):
                if val != val:
                    return "n/a"
                return f"{val:.3f}" if abs(val) < 100 else f"{val:.1f}"
            return str(val)

        table = [[_fmt(row[h]) for h in headers] for row in rows]
        col_widths = [max(len(h), max(len(r[i]) for r in table)) for i, h in enumerate(headers)]

        def _print_row(cells: List[str]) -> None:
            print(" | ".join(cell.ljust(col_widths[idx]) for idx, cell in enumerate(cells)))

        _print_row(headers)
        print("-+-".join("-" * w for w in col_widths))
        for row in table:
            _print_row(row)


if __name__ == "__main__":
    main()
