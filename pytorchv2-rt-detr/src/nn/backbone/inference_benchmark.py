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


@dataclass
class BenchmarkResult:
    samples: int
    measured_batches: int
    batch_size: int
    coeff_channels: int
    luma_shape: tuple[int, int, int]
    chroma_shape: tuple[int, int, int, int]
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
            "luma_shape": self.luma_shape,
            "chroma_shape": self.chroma_shape,
            "input_mb_per_batch": self.input_mb_per_batch,
            "mean_latency_ms": self.mean_latency_ms,
            "throughput_img_s": self.throughput_img_s,
            "peak_memory_mb": self.peak_memory_mb,
            "loss": self.loss,
            "acc1": self.acc1,
            "acc5": self.acc5,
        }


def _build_active_index(coeff_window: int) -> torch.Tensor | None:
    if coeff_window >= 8:
        return None
    indices = [row + col * 8 for col in range(coeff_window) for row in range(coeff_window)]
    return torch.tensor(indices, dtype=torch.long)


def _prune_payload(
    payload: tuple[torch.Tensor, torch.Tensor],
    active_idx: torch.Tensor | None,
    expected_channels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if active_idx is None or expected_channels >= 64:
        return payload
    y_blocks, cbcr_blocks = payload
    if y_blocks.size(0) == expected_channels:
        return y_blocks, cbcr_blocks
    if y_blocks.size(0) <= 0 or cbcr_blocks.size(1) <= 0:
        return y_blocks, cbcr_blocks

    idx = active_idx
    if idx.numel() == 0:
        return y_blocks, cbcr_blocks

    valid_mask = idx < y_blocks.size(0)
    idx = idx[valid_mask]
    if idx.numel() == 0:
        return y_blocks, cbcr_blocks

    device = y_blocks.device
    mask = torch.zeros(y_blocks.size(0), dtype=y_blocks.dtype, device=device)
    mask[idx.to(device=device)] = 1.0
    mask_y = mask.view(-1, 1, 1)
    mask_cbcr = mask.view(1, -1, 1, 1)
    y_masked = y_blocks * mask_y
    cbcr_masked = cbcr_blocks * mask_cbcr
    return y_masked, cbcr_masked


def _build_trimmed_collate_fn(coeff_window: int):
    active_idx = _build_active_index(coeff_window)
    expected_channels = coeff_window * coeff_window

    def collate(batch: Iterable[tuple[tuple[torch.Tensor, torch.Tensor], int]]):
        y_list = []
        cbcr_list = []
        targets = []
        for payload, target in batch:
            y_trimmed, cbcr_trimmed = _prune_payload(payload, active_idx, expected_channels)
            y_list.append(y_trimmed)
            cbcr_list.append(cbcr_trimmed)
            targets.append(int(target))
        y_batch = torch.stack(y_list, dim=0)
        cbcr_batch = torch.stack(cbcr_list, dim=0)
        target_batch = torch.tensor(targets, dtype=torch.long)
        return (y_batch, cbcr_batch), target_batch

    return collate


def build_trimmed_eval_loader(
    val_dirs: Sequence[str],
    image_size: int,
    batch_size: int,
    workers: int,
    compression_cfg: dict[str, object],
    coeff_window: int,
    max_samples: int | None,
    dct_normalizer: T.Transform | None,
    show_progress: bool,
    trim_coefficients: bool,
) -> DataLoader:
    compression = dict(compression_cfg)
    compression.setdefault("keep_original", False)
    if not isinstance(compression.get("coeff_window", coeff_window), int):
        compression["coeff_window"] = coeff_window
    _, val_tf = build_resnet_transforms(
        image_size,
        compression=compression,
        dct_normalizer_val=dct_normalizer,
        trim_coefficients=trim_coefficients,
    )
    dataset = ImageNetDataset(list(val_dirs), transforms=val_tf, show_progress=show_progress)
    if max_samples is not None and max_samples > 0:
        dataset = limit_total(dataset, max_samples)
    collate = _build_trimmed_collate_fn(coeff_window)
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
def _temporarily_patch_active_idx(backbone: nn.Module, coeff_window: int):
    sentinel = object()
    if not hasattr(backbone, "active_idx"):
        yield
        return
    original = getattr(backbone, "active_idx", sentinel)
    if coeff_window >= 8:
        new_idx = None
    else:
        new_idx = torch.arange(coeff_window * coeff_window, dtype=torch.long)
    setattr(backbone, "active_idx", new_idx)
    try:
        yield
    finally:
        if original is sentinel:
            delattr(backbone, "active_idx")
        else:
            setattr(backbone, "active_idx", original)


def run_trimmed_inference_benchmark(
    model: nn.Module,
    *,
    device: torch.device,
    coeff_window: int,
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

    loader = build_trimmed_eval_loader(
        val_dirs,
        image_size,
        batch_size,
        workers,
        compression_cfg,
        coeff_window,
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
    chroma_shape: tuple[int, int, int, int] | None = None
    metric_samples = 0
    metric_loss = 0.0
    metric_correct1 = 0.0
    metric_correct5 = 0.0

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode(), _temporarily_patch_active_idx(backbone, coeff_window):
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
                y_tensor, cbcr_tensor = batch_inputs
                luma_shape = tuple(int(v) for v in y_tensor.shape[1:])
                chroma_shape = tuple(int(v) for v in cbcr_tensor.shape[1:])

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

    result = BenchmarkResult(
        samples=total_samples,
        measured_batches=measured_batches,
        batch_size=batch_size,
        coeff_channels=luma_shape[0] if luma_shape is not None else coeff_window * coeff_window,
        luma_shape=luma_shape or (coeff_window * coeff_window, 0, 0),
        chroma_shape=chroma_shape or (2, coeff_window * coeff_window, 0, 0),
        input_mb_per_batch=mb_per_batch,
        mean_latency_ms=avg_latency_ms,
        throughput_img_s=throughput,
        peak_memory_mb=peak_memory,
        loss=mean_loss,
        acc1=mean_acc1,
        acc5=mean_acc5,
    )
    return result
