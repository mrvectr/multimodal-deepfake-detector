# data/samplers.py

import torch
from torch.utils.data import Sampler
from typing import Iterator, List


class DistributedEvalSampler(Sampler):
    """
    Sampler for evaluation that ensures every sample is seen exactly once
    across all processes in distributed training.
    Falls back to sequential sampling in single-GPU mode.
    """
    def __init__(self, dataset, num_replicas: int = 1, rank: int = 0):
        self.dataset       = dataset
        self.num_replicas  = num_replicas
        self.rank          = rank
        self.num_samples   = len(dataset) // num_replicas

    def __iter__(self) -> Iterator:
        indices = list(range(len(self.dataset)))
        indices = indices[self.rank::self.num_replicas]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples


class BalancedDatasetSampler(Sampler):
    """
    Samples evenly from multiple datasets combined into one.
    Used in full training when mixing LRS3 + VoxCeleb2 — prevents
    the larger dataset from dominating every batch.

    Args:
        dataset_sizes : list of sizes e.g. [100000, 200000]
        total_samples : total samples to draw per epoch
    """
    def __init__(self, dataset_sizes: List[int], total_samples: int):
        self.dataset_sizes  = dataset_sizes
        self.total_samples  = total_samples
        self.n_datasets     = len(dataset_sizes)
        self.offsets        = [0] + list(
            torch.cumsum(torch.tensor(dataset_sizes), dim=0).tolist()
        )

    def __iter__(self) -> Iterator:
        indices = []
        per_dataset = self.total_samples // self.n_datasets

        for i, size in enumerate(self.dataset_sizes):
            local_indices = torch.randperm(size)[:per_dataset].tolist()
            global_indices = [idx + self.offsets[i] for idx in local_indices]
            indices.extend(global_indices)

        # shuffle the combined list
        perm = torch.randperm(len(indices)).tolist()
        return iter([indices[i] for i in perm])

    def __len__(self) -> int:
        return self.total_samples