"""
Proportional Multi-Dataset Sampler for Distributed Training.

Sample from multiple datasets proportionally, using the smallest dataset as the baseline.
When the baseline dataset is exhausted, the epoch ends.
"""

import torch
from torch.utils.data import Sampler, ConcatDataset


class ProportionalMultiDatasetSampler(Sampler):
    """
    Sampler that samples from multiple datasets in a ConcatDataset proportionally.

    The sampling is based on a base dataset (usually the smallest one).
    When the base dataset is exhausted, the epoch ends.

    Args:
        combined_dataset: A ConcatDataset containing multiple datasets
        weights: List of weights, one per dataset. Higher weight = more samples from that dataset.
        base_dataset_idx: Index of the dataset to use as the baseline (default: 0, the first dataset)
        num_replicas: Number of processes participating in distributed training
        rank: Rank of the current process
        shuffle: If True, shuffle the indices
        seed: Random seed for shuffling

    Example:
        >>> # ConcatDataset with 3 datasets: [offline(1000), other(5000), openx(5000)]
        >>> sampler = ProportionalMultiDatasetSampler(
        ...     combined_dataset=combined_dataset,
        ...     weights=[1, 2, 2],  # sample 2x from other and openx
        ...     base_dataset_idx=0,  # offline is the baseline
        ... )
        >>> # Each epoch:
        >>> # - offline: all 1000 samples
        >>> # - other: 2000 samples (40% of dataset)
        >>> # - openx: 2000 samples (40% of dataset)
        >>> # Total: 5000 samples per epoch
    """

    def __init__(
        self,
        combined_dataset: ConcatDataset,
        weights: list,
        base_dataset_idx: int = 0,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
    ):
        if not isinstance(combined_dataset, ConcatDataset):
            raise ValueError(f"Expected ConcatDataset, got {type(combined_dataset)}")

        self.combined_dataset = combined_dataset
        self.weights = weights
        self.base_dataset_idx = base_dataset_idx
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Get the boundaries of each dataset
        self.cumulative_sizes = [0] + list(combined_dataset.cumulative_sizes)
        self.dataset_sizes = [
            end - start
            for start, end in zip(self.cumulative_sizes[:-1], self.cumulative_sizes[1:])
        ]

        if base_dataset_idx >= len(self.dataset_sizes):
            raise ValueError(f"base_dataset_idx {base_dataset_idx} out of range")

        if base_dataset_idx >= len(weights):
            raise ValueError(f"base_dataset_idx {base_dataset_idx} not in weights")

        # Size of the baseline dataset
        self.base_size = self.dataset_sizes[base_dataset_idx]
        base_weight = weights[base_dataset_idx]

        # Compute how many samples to draw from each dataset
        self.samples_per_dataset = []
        for i, size in enumerate(self.dataset_sizes):
            weight = weights[i] if i < len(weights) else 1.0
            # Ratio relative to the baseline
            ratio = weight / base_weight
            num_samples = int(self.base_size * ratio)

            # Ensure it doesn't exceed the dataset's actual size (sampling without replacement)
            if num_samples > size:
                print(f"Warning: Dataset {i} has {size} samples but wants to sample {num_samples}. Capping at {size}.")
                num_samples = size

            self.samples_per_dataset.append(num_samples)

        # Total number of samples
        self.total_samples = sum(self.samples_per_dataset)

        # Number of samples per rank (rounded down, to avoid an incomplete last batch)
        self.num_samples = self.total_samples // num_replicas

        print(f"ProportionalMultiDatasetSampler initialized:")
        print(f"  Base dataset idx: {base_dataset_idx}, size: {self.base_size}")
        for i, (size, num) in enumerate(zip(self.dataset_sizes, self.samples_per_dataset)):
            print(f"  Dataset {i}: {size} total -> {num} sampled ({num/size*100:.1f}%)")
        print(f"  Total samples per epoch: {self.total_samples}")
        print(f"  Samples per rank: {self.num_samples}")

    def set_epoch(self, epoch: int):
        """
        Set the epoch for this sampler.

        This ensures all replicas use a different random ordering for each epoch.

        Args:
            epoch: Epoch number
        """
        self.epoch = epoch

    def __iter__(self):
        # Pre-allocate the tensor to avoid dynamic resizing
        # Note: pin_memory=True is not used here, since indices don't need to be transferred to GPU
        # Using pinned memory would compete with the DataLoader for limited pinned memory resources
        all_indices = torch.zeros(self.total_samples, dtype=torch.long)

        # Generate sampling indices for each dataset (using tensor operations throughout)
        offset = 0
        for dataset_idx, num_samples in enumerate(self.samples_per_dataset):
            if num_samples == 0:
                continue

            start = self.cumulative_sizes[dataset_idx]
            dataset_len = self.dataset_sizes[dataset_idx]

            # Random sampling (with replacement), generating the tensor directly
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch + self.rank + dataset_idx)
            indices = torch.randint(0, dataset_len, (num_samples,), generator=g, dtype=torch.long)

            # Vectorized conversion to global indices
            all_indices[offset:offset + num_samples] = indices + start
            offset += num_samples

        # Key step: shuffle the global indices before sharding
        # This ensures each batch mixes samples from different datasets
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch + self.rank + 99999)  # different seed
            perm = torch.randperm(self.total_samples, generator=g)
            all_indices = all_indices[perm]

        # Distributed sharding: ensure each rank gets different samples
        # Use tensor slicing to take indices directly, avoiding tolist() then slicing
        indices = all_indices[self.rank::self.num_replicas]

        # Return an iterator over the numpy array, avoiding the overhead of tolist() creating Python int objects
        return iter(indices.numpy())

    def __len__(self):
        return self.num_samples


import torch
from torch.utils.data import Sampler, ConcatDataset
class BatchMixedProportionalSampler(Sampler):
    """
    Efficient proportional multi-dataset sampling, ensuring each batch mixes different datasets.

    Strategy: first split data by rank (ensuring no overlap), then mix batches within each rank (ensuring proportions).

    Args:
        combined_dataset: ConcatDataset
        weights: weight of each dataset; sampling is proportional to the baseline dataset
        base_dataset_idx: index of the baseline dataset (usually the smallest one)
        batch_size: number of samples per batch
        num_replicas: number of DDP processes
        rank: current rank
        shuffle: whether to shuffle
        seed: base random seed
    """
    def __init__(
        self,
        combined_dataset: ConcatDataset,
        weights: list,
        base_dataset_idx: int = 0,
        batch_size: int = 32,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0
    ):
        if not isinstance(combined_dataset, ConcatDataset):
            raise ValueError(f"Expected ConcatDataset, got {type(combined_dataset)}")

        self.combined_dataset = combined_dataset
        self.weights = weights
        self.base_dataset_idx = base_dataset_idx
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Dataset boundaries and sizes
        self.cumulative_sizes = [0] + list(combined_dataset.cumulative_sizes)
        self.dataset_sizes = [
            end - start
            for start, end in zip(self.cumulative_sizes[:-1], self.cumulative_sizes[1:])
        ]
        self.num_datasets = len(self.dataset_sizes)

        if base_dataset_idx >= self.num_datasets:
            raise ValueError(f"base_dataset_idx {base_dataset_idx} out of range")

        base_weight = weights[base_dataset_idx]
        self.base_size = self.dataset_sizes[base_dataset_idx]

        # Total global samples for each dataset
        self.samples_per_dataset = []
        for i, size in enumerate(self.dataset_sizes):
            weight = weights[i] if i < len(weights) else 1.0
            num_samples = int(self.base_size * (weight / base_weight))
            if num_samples > size:
                num_samples = size  # cap at dataset size
            self.samples_per_dataset.append(num_samples)

        # Number of samples allotted to the current rank for each dataset
        self.rank_samples_per_dataset = [n // num_replicas for n in self.samples_per_dataset]
        self.num_samples = sum(self.rank_samples_per_dataset)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        """
        Generate the indices for the current rank on each iteration.
        Logic:
          1. Global permutation -> contiguous split by rank (ensuring no overlap)
          2. Build batches proportionally within each rank (ensuring per-batch dataset proportions)
        """
        # Step 1: sample each dataset globally, then split contiguously by rank
        rank_dataset_indices = []
        for dataset_idx, global_num in enumerate(self.samples_per_dataset):
            dataset_len = self.dataset_sizes[dataset_idx]
            start = self.cumulative_sizes[dataset_idx]

            # All ranks use the same seed to generate the same permutation
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch + dataset_idx)

            perm = torch.randperm(dataset_len, generator=g)
            selected = perm[:global_num] + start

            # Contiguous split assigned to the current rank (ensuring no overlap across ranks)
            per_rank = global_num // self.num_replicas
            rank_start = self.rank * per_rank
            rank_end = rank_start + per_rank
            rank_dataset_indices.append(selected[rank_start:rank_end])

        # Step 2: build the proportional batch mix within the rank
        rank_total = sum(len(idx) for idx in rank_dataset_indices)

        # Compute how many samples each dataset contributes per batch, ensuring the sum == batch_size
        raw_portions = []
        for idx in rank_dataset_indices:
            raw_portions.append(max(1, int(self.batch_size * len(idx) / rank_total)))

        # Adjust the portion totals so they sum to batch_size
        diff = self.batch_size - sum(raw_portions)
        if diff > 0:
            # Add from the dataset with the largest weight
            sorted_ds = sorted(range(len(raw_portions)), key=lambda i: self.weights[i] if i < len(self.weights) else 1.0, reverse=True)
            for i in range(diff):
                raw_portions[sorted_ds[i % len(sorted_ds)]] += 1
        elif diff < 0:
            # Subtract from the dataset with the smallest weight
            sorted_ds = sorted(range(len(raw_portions)), key=lambda i: self.weights[i] if i < len(self.weights) else 1.0)
            for i in range(-diff):
                if raw_portions[sorted_ds[i % len(sorted_ds)]] > 1:
                    raw_portions[sorted_ds[i % len(sorted_ds)]] -= 1

        portions = raw_portions

        # Split each dataset into chunks according to its portion
        batch_slices = []
        for ds_idx, idx_tensor in enumerate(rank_dataset_indices):
            portion = portions[ds_idx]
            batches = [idx_tensor[i:i+portion] for i in range(0, len(idx_tensor), portion)]
            batch_slices.append(batches)

        # Merge chunks from different datasets batch-wise
        mixed_batches = []
        max_batches = max(len(b) for b in batch_slices)
        for batch_idx in range(max_batches):
            batch = []
            for ds_batches in batch_slices:
                if batch_idx < len(ds_batches):
                    batch.append(ds_batches[batch_idx])
            if batch:
                mixed_batches.append(torch.cat(batch))

        # Shuffle the order within each batch (using a different seed per batch)
        final_indices = []
        for batch_idx, batch in enumerate(mixed_batches):
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch + batch_idx + self.rank * 997)
                batch = batch[torch.randperm(len(batch), generator=g)]
            final_indices.append(batch)
        final_indices = torch.cat(final_indices)

        return iter(final_indices.numpy())

    def __len__(self):
        return self.num_samples

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import ConcatDataset

class SimpleMixedDistributedSampler(DistributedSampler):
    """
    DDP sampler for ConcatDataset, batch will naturally contain mixed datasets.
    No need for weights or base dataset.
    """

    def __init__(self, dataset: ConcatDataset, shuffle=True, seed=0, drop_last=False,
                 num_replicas=None, rank=None):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank,
                         shuffle=shuffle, drop_last=drop_last)
        self.seed = seed

    def __iter__(self):
        # 1. Build the global indices for the epoch
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.dataset), generator=g) if self.shuffle else torch.arange(len(self.dataset))

        # 2. DDP sharding
        indices = indices[self.rank:self.total_size:self.num_replicas]

        return iter(indices.numpy())

    def __len__(self):
        return self.num_samples