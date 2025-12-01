"""Dataset wrappers for limiting sample count per dataset or per class."""

from __future__ import annotations

import itertools
import random
from collections import defaultdict
from typing import Iterable, List, Sequence

from torch.utils.data import Dataset


class SubsetDataset(Dataset):
    """Wrap another dataset but expose only the desired indices."""

    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.dataset[self.indices[idx]]


def limit_per_class(dataset: Dataset, max_per_class: int, shuffle: bool = True) -> SubsetDataset:
    buckets = defaultdict(list)
    for idx, (_, target) in enumerate(dataset):
        buckets[target].append(idx)
    selected: List[int] = []
    for indices in buckets.values():
        if shuffle:
            random.shuffle(indices)
        selected.extend(indices[:max_per_class])
    if shuffle:
        random.shuffle(selected)
    return SubsetDataset(dataset, selected)


def limit_total(dataset: Dataset, max_items: int, shuffle: bool = True) -> SubsetDataset:
    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)
    return SubsetDataset(dataset, indices[:max_items])
