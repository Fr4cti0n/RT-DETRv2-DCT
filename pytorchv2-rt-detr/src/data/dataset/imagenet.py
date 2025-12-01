"""ImageNet-1k dataset wrapper for classification training."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS, has_file_allowed_extension

from ...core import register

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional for progress view
    tqdm = None


torchvision.disable_beta_transforms_warning()


@register()
class ImageNetDataset(Dataset):
    """Load ImageNet-style folders (subfolders named by class id).

    Args:
        roots: One or more directories that follow the ImageFolder layout.
        transforms: Optional callable applied to the loaded PIL image.
    """

    __inject__ = ["transforms"]

    def __init__(
        self,
        roots: Sequence[str | Path],
        transforms=None,
        max_samples: Optional[int] = None,
        show_progress: bool = False,
    ) -> None:
        super().__init__()

        if isinstance(roots, (str, Path)):
            roots = [roots]

        if not roots:
            raise ValueError("Expected at least one root directory for ImageNetDataset")

        self.roots: List[Path] = [Path(root).expanduser().resolve() for root in roots]
        self.transforms = transforms
        self._max_samples = max_samples if max_samples is None or max_samples > 0 else None
        if show_progress and tqdm is None:
            print("[ImageNetDataset] tqdm not installed; progress bar disabled.")
        self._show_progress = show_progress and tqdm is not None

        # Build a unified class index so shards with partial class coverage stay consistent.
        class_names = set()
        for root in self.roots:
            if not root.exists():
                raise FileNotFoundError(f"ImageNet split directory not found: {root}")

            for entry in root.iterdir():
                if entry.is_dir():
                    class_names.add(entry.name)

        if not class_names:
            raise RuntimeError(
                "No class directories detected in the provided ImageNet roots."
            )

        self.class_to_idx = {name: idx for idx, name in enumerate(sorted(class_names))}
        self.samples: List[Tuple[str, int]] = []

        remaining = self._max_samples
        progress_bar = None
        if self._show_progress:
            total = self._max_samples
            progress_bar = tqdm(total=total, desc="Scanning ImageNet", unit="img")
        for root in self.roots:
            for class_dir in sorted(root.iterdir()):
                if not class_dir.is_dir():
                    continue
                class_name = class_dir.name
                target = self.class_to_idx[class_name]

                for candidate in sorted(class_dir.iterdir()):
                    if not candidate.is_file():
                        continue
                    if not has_file_allowed_extension(candidate.name, IMG_EXTENSIONS):
                        continue
                    self.samples.append((str(candidate), target))
                    if progress_bar is not None:
                        progress_bar.update(1)
                    if remaining is not None:
                        remaining -= 1
                        if remaining <= 0:
                            break
                if remaining is not None and remaining <= 0:
                    break
            if remaining is not None and remaining <= 0:
                break

        if not self.samples:
            raise RuntimeError(
                "No samples found when scanning ImageNet directories. Please check the dataset layout."
            )

        if progress_bar is not None:
            progress_bar.close()

        self.loader = default_loader
        self.targets = [target for _, target in self.samples]
        self._epoch = -1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, target = self.samples[index]
        image = self.loader(path)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    @property
    def epoch(self) -> int:
        return self._epoch

    def extra_repr(self) -> str:
        roots = "\n    ".join(str(root) for root in self.roots)
        return f"roots:\n    {roots}\nnum_classes: {len(self.class_to_idx or {})}"
