"""Trimmed-input inference benchmark for compressed DCT backbones."""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T

from ...data.dataset.imagenet import ImageNetDataset
from ...data.dataset.subset import limit_total
from ..backbone.train_backbones import _move_to_device, build_resnet_transforms, normalise_logits
from ...misc.dct_coefficients import build_active_indices, resolve_coefficient_counts, validate_coeff_count


@dataclass
class BenchmarkResult:
    samples: int
    measured_batches: int
    batch_size: int
    coeff_channels: int
    coeff_channels_cb: int
    coeff_channels_cr: int
    luma_shape: tuple[int, int, int]
    chroma_shape_cb: tuple[int, int, int]
    chroma_shape_cr: tuple[int, int, int]
    input_mb_per_batch: float
    mean_latency_ms: float
    throughput_img_s: float
    peak_memory_mb: float | None

    loss: float | None = None
    acc1: float | None = None
    acc5: float | None = None

    def as_dict(self) -> dict[str, float | int | None | tuple[int, ...]]:
        return {
            "samples": self.samples,
            "measured_batches": self.measured_batches,
            "batch_size": self.batch_size,
            "coeff_channels": self.coeff_channels,
            "coeff_channels_cb": self.coeff_channels_cb,
            "coeff_channels_cr": self.coeff_channels_cr,
            "coeff_channels_chroma": self.coeff_channels_chroma,
            "luma_shape": self.luma_shape,
            "chroma_shape_cb": self.chroma_shape_cb,
            "chroma_shape_cr": self.chroma_shape_cr,
            "input_mb_per_batch": self.input_mb_per_batch,
            "mean_latency_ms": self.mean_latency_ms,
            "throughput_img_s": self.throughput_img_s,
            "peak_memory_mb": self.peak_memory_mb,
            "loss": self.loss,
            "acc1": self.acc1,
            "acc5": self.acc5,
        }

    @property
    def coeff_channels_chroma(self) -> int:
        return max(self.coeff_channels_cb, self.coeff_channels_cr)

    @property
    def chroma_shape(self) -> tuple[int, int, int, int]:
        cb_c, cb_h, cb_w = self.chroma_shape_cb
        cr_c, cr_h, cr_w = self.chroma_shape_cr
        return (
            2,
            self.coeff_channels_chroma,
            max(cb_h, cr_h),
            max(cb_w, cr_w),
        )


def _build_active_index(coeff_count: int) -> torch.Tensor | None:
    count = validate_coeff_count(int(coeff_count), name="coeff_count")
    if count <= 0 or count >= 64:
        return None
    indices = build_active_indices(count)
    return torch.tensor(indices, dtype=torch.long)


def _select_coefficients(
    plane: torch.Tensor,
    active_idx: torch.Tensor | None,
    expected: int,
) -> torch.Tensor:
    if expected <= 0:
        return plane.new_zeros((0, *plane.shape[1:]))

    trimmed = plane
    if plane.size(0) > expected:
        if active_idx is not None:
            max_idx = int(active_idx.max().item()) if active_idx.numel() > 0 else -1
            if max_idx < plane.size(0):
                idx = active_idx.to(device=plane.device)
                trimmed = torch.index_select(plane, 0, idx)
            else:
                trimmed = plane[:expected]
        else:
            trimmed = plane[:expected]

    if trimmed.size(0) < expected:
        pad_shape = (expected - trimmed.size(0), *trimmed.shape[1:])
        trimmed = torch.cat((trimmed, trimmed.new_zeros(pad_shape)), dim=0)

    return trimmed


def _expand_to_target(plane: torch.Tensor, target: int) -> torch.Tensor:
    if target <= 0:
        return plane.new_zeros((0, *plane.shape[1:]))
    if plane.size(0) < target:
        pad_shape = (target - plane.size(0), *plane.shape[1:])
        return torch.cat((plane, plane.new_zeros(pad_shape)), dim=0)
    if plane.size(0) > target:
        return plane[:target]
    return plane


def _prune_payload(
    payload: tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
    active_idx_luma: torch.Tensor | None,
    expected_luma: int,
    active_idx_cb: torch.Tensor | None,
    expected_cb: int,
    active_idx_cr: torch.Tensor | None,
    expected_cr: int,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    y_blocks, cbcr_blocks = payload

    y_trimmed = _select_coefficients(y_blocks, active_idx_luma, expected_luma)

    cb = cbcr_blocks[0]
    cr = cbcr_blocks[1]

    cb_trimmed = _select_coefficients(cb, active_idx_cb, expected_cb)
    cr_trimmed = _select_coefficients(cr, active_idx_cr, expected_cr)

    cb_final = _expand_to_target(cb_trimmed, expected_cb)
    cr_final = _expand_to_target(cr_trimmed, expected_cr)

    return y_trimmed, (cb_final, cr_final)


def _build_trimmed_collate_fn(
    coeff_count_luma: int,
    coeff_count_cb: int,
    coeff_count_cr: int,
):
    active_idx_luma = _build_active_index(coeff_count_luma)
    active_idx_cb = _build_active_index(coeff_count_cb)
    active_idx_cr = _build_active_index(coeff_count_cr)
    expected_luma = int(coeff_count_luma)
    expected_cb = int(coeff_count_cb)
    expected_cr = int(coeff_count_cr)

    def collate(
        batch: Iterable[tuple[tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]], int]]
    ):
        y_list = []
        cb_list = []
        cr_list = []
        targets = []
        for payload, target in batch:
            y_trimmed, (cb_trimmed, cr_trimmed) = _prune_payload(
                payload,
                active_idx_luma,
                expected_luma,
                active_idx_cb,
                expected_cb,
                active_idx_cr,
                expected_cr,
            )
            y_list.append(y_trimmed)
            cb_list.append(cb_trimmed)
            cr_list.append(cr_trimmed)
            targets.append(int(target))
        y_batch = torch.stack(y_list, dim=0)
        cb_batch = torch.stack(cb_list, dim=0)
        cr_batch = torch.stack(cr_list, dim=0)
        target_batch = torch.tensor(targets, dtype=torch.long)
        return (y_batch, (cb_batch, cr_batch)), target_batch

    return collate


def build_trimmed_eval_loader(
    val_dirs: Sequence[str],
    image_size: int,
    batch_size: int,
    workers: int,
    compression_cfg: dict[str, object],
    coeff_count_luma: int,
    coeff_count_cb: int,
    coeff_count_cr: int,
    max_samples: int | None,
    dct_normalizer: T.Transform | None,
    show_progress: bool,
    trim_coefficients: bool,
) -> DataLoader:
    compression = dict(compression_cfg)
    compression.setdefault("keep_original", False)
    chroma_count = coeff_count_cb if coeff_count_cb == coeff_count_cr else max(coeff_count_cb, coeff_count_cr)
    compression.setdefault("coeff_count", coeff_count_luma)
    compression.setdefault("coeff_count_luma", coeff_count_luma)
    compression.setdefault("coeff_count_chroma", chroma_count)
    compression.setdefault("coeff_count_cb", coeff_count_cb)
    compression.setdefault("coeff_count_cr", coeff_count_cr)
    _, val_tf = build_resnet_transforms(
        image_size,
        compression=compression,
        dct_normalizer_val=dct_normalizer,
        trim_coefficients=trim_coefficients,
    )
    dataset = ImageNetDataset(list(val_dirs), transforms=val_tf, show_progress=show_progress)
    if max_samples is not None and max_samples > 0:
        dataset = limit_total(dataset, max_samples)
    collate = _build_trimmed_collate_fn(coeff_count_luma, coeff_count_cb, coeff_count_cr)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, workers),
        pin_memory=True,
        drop_last=False,
        collate_fn=collate,
    )
    return loader


def _estimate_bytes(inputs) -> int:
    if isinstance(inputs, torch.Tensor):
        return inputs.element_size() * inputs.numel()
    if isinstance(inputs, (list, tuple)):
        return sum(_estimate_bytes(item) for item in inputs)
    if isinstance(inputs, dict):
        return sum(_estimate_bytes(value) for value in inputs.values())
    return 0


@contextlib.contextmanager
def _temporarily_patch_active_idx(
    backbone: nn.Module,
    coeff_count_luma: int,
    coeff_count_cb: int,
    coeff_count_cr: int,
):
    sentinel = object()
    patches: list[tuple[str, object]] = []
    chroma_target = max(int(coeff_count_cb), int(coeff_count_cr))
    for attr, count in (
        ("active_idx_luma", coeff_count_luma),
        ("active_idx_cb", coeff_count_cb),
        ("active_idx_cr", coeff_count_cr),
        ("active_idx_chroma", chroma_target),
    ):
        if not hasattr(backbone, attr):
            continue
        original = getattr(backbone, attr, sentinel)
        patches.append((attr, original))
        count_validated = validate_coeff_count(int(count), name="coeff_count")
        if count_validated <= 0 or count_validated >= 64:
            new_idx = None
        else:
            new_idx = torch.tensor(build_active_indices(count_validated), dtype=torch.long)
        setattr(backbone, attr, new_idx)
    try:
        yield
    finally:
        for attr, original in patches:
            if original is sentinel:
                delattr(backbone, attr)
            else:
                setattr(backbone, attr, original)


def run_trimmed_inference_benchmark(
    model: nn.Module,
    *,
    device: torch.device,
    coeff_window_luma: int | None,
    coeff_window_chroma: int | None,
    image_size: int,
    batch_size: int,
    val_dirs: Sequence[str],
    compression_cfg: dict[str, object],
    dct_normalizer: T.Transform | None,
    max_samples: int | None,
    workers: int,
    warmup_batches: int,
    measure_batches: int,
    show_progress: bool = False,
    collect_metrics: bool = False,
    trim_coefficients: bool = False,
) -> BenchmarkResult | None:
    if measure_batches <= 0:
        return None
    if batch_size <= 0:
        raise ValueError("Benchmark batch size must be > 0.")

    coeff_kwargs: dict[str, int] = {}
    for key in (
        "coeff_window",
        "coeff_count",
        "coeff_window_luma",
        "coeff_count_luma",
        "coeff_window_chroma",
        "coeff_count_chroma",
        "coeff_window_cb",
        "coeff_count_cb",
        "coeff_window_cr",
        "coeff_count_cr",
    ):
        value = compression_cfg.get(key)
        if value is not None:
            coeff_kwargs[key] = int(value)
    if coeff_window_luma is not None and "coeff_window_luma" not in coeff_kwargs:
        coeff_kwargs["coeff_window_luma"] = int(coeff_window_luma)
    if coeff_window_chroma is not None and "coeff_window_chroma" not in coeff_kwargs:
        coeff_kwargs["coeff_window_chroma"] = int(coeff_window_chroma)

    _, coeff_count_luma, coeff_count_cb, coeff_count_cr = resolve_coefficient_counts(**coeff_kwargs)
    coeff_count_luma = int(coeff_count_luma)
    coeff_count_cb = int(coeff_count_cb)
    coeff_count_cr = int(coeff_count_cr)
    coeff_count_chroma = coeff_count_cb if coeff_count_cb == coeff_count_cr else max(coeff_count_cb, coeff_count_cr)

    loader = build_trimmed_eval_loader(
        val_dirs,
        image_size,
        batch_size,
        workers,
        compression_cfg,
        coeff_count_luma,
        coeff_count_cb,
        coeff_count_cr,
        max_samples,
        dct_normalizer,
        show_progress,
        trim_coefficients,
    )
    if len(loader) == 0:
        return None

    total_batches = len(loader)
    warmup_batches = min(warmup_batches, max(0, total_batches - 1))
    measure_batches = min(measure_batches, total_batches - warmup_batches)
    if measure_batches <= 0:
        return None

    model.eval()
    backbone = getattr(model, "backbone", model)
    is_cuda = device.type == "cuda"

    total_samples = 0
    total_time = 0.0
    measured_batches = 0
    input_bytes = 0
    luma_shape: tuple[int, int, int] | None = None
    chroma_shape_cb: tuple[int, int, int] | None = None
    chroma_shape_cr: tuple[int, int, int] | None = None
    metric_samples = 0
    metric_loss = 0.0
    metric_correct1 = 0.0
    metric_correct5 = 0.0

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode(), _temporarily_patch_active_idx(
        backbone,
        coeff_count_luma,
        coeff_count_cb,
        coeff_count_cr,
    ):
        for batch_idx, (inputs, targets) in enumerate(loader):
            if batch_idx < warmup_batches:
                _ = model(_move_to_device(inputs, device))
                if is_cuda:
                    torch.cuda.synchronize(device)
                continue
            if measured_batches >= measure_batches:
                break
            inputs_device = _move_to_device(inputs, device)
            targets_device = targets.to(device=device) if collect_metrics else None
            if is_cuda:
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            outputs = model(inputs_device)
            if collect_metrics:
                outputs = normalise_logits(outputs)
            if is_cuda:
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            measured_batches += 1
            total_time += elapsed
            batch_inputs = inputs
            input_bytes += _estimate_bytes(batch_inputs)
            batch_size_actual = batch_inputs[0].size(0)
            total_samples += batch_size_actual
            if collect_metrics and targets_device is not None:
                metric_samples += targets_device.size(0)
                metric_loss += float(
                    F.cross_entropy(outputs, targets_device, reduction="sum").item()
                )
                maxk = min(5, outputs.size(1))
                _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
                pred = pred.t()
                correct = pred.eq(targets_device.view(1, -1).expand_as(pred))
                metric_correct1 += correct[:1].reshape(-1).float().sum().item()
                metric_correct5 += correct[:5].reshape(-1).float().sum().item()
            if luma_shape is None:
                y_tensor, chroma_tensors = batch_inputs
                cb_tensor, cr_tensor = chroma_tensors
                luma_shape = tuple(int(v) for v in y_tensor.shape[1:])
                chroma_shape_cb = tuple(int(v) for v in cb_tensor.shape[1:])
                chroma_shape_cr = tuple(int(v) for v in cr_tensor.shape[1:])

    if measured_batches == 0 or total_time <= 0.0:
        return None

    avg_latency_ms = (total_time / measured_batches) * 1000.0
    throughput = total_samples / total_time if total_time > 0 else 0.0
    mb_per_batch = (input_bytes / measured_batches) / (1024 ** 2)
    peak_memory = None
    if is_cuda:
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    mean_loss = metric_loss / metric_samples if metric_samples > 0 else None
    mean_acc1 = metric_correct1 / metric_samples if metric_samples > 0 else None
    mean_acc5 = metric_correct5 / metric_samples if metric_samples > 0 else None

    expected_luma = coeff_count_luma
    expected_cb = coeff_count_cb
    expected_cr = coeff_count_cr

    luma_channels = luma_shape[0] if luma_shape is not None else expected_luma
    cb_channels = chroma_shape_cb[0] if chroma_shape_cb is not None else expected_cb
    cr_channels = chroma_shape_cr[0] if chroma_shape_cr is not None else expected_cr

    result = BenchmarkResult(
        samples=total_samples,
        measured_batches=measured_batches,
        batch_size=batch_size,
        coeff_channels=luma_channels,
        coeff_channels_cb=cb_channels,
        coeff_channels_cr=cr_channels,
        luma_shape=luma_shape or (expected_luma, 0, 0),
        chroma_shape_cb=chroma_shape_cb or (expected_cb, 0, 0),
        chroma_shape_cr=chroma_shape_cr or (expected_cr, 0, 0),
        input_mb_per_batch=mb_per_batch,
        mean_latency_ms=avg_latency_ms,
        throughput_img_s=throughput,
        peak_memory_mb=peak_memory,
        loss=mean_loss,
        acc1=mean_acc1,
        acc5=mean_acc5,
    )
    return result
