"""
Utility functions for handling dataset weights in distributed training scenarios.
"""

import torch
from torch.utils.data import WeightedRandomSampler
def expand_concat_dataset_weights(dataset, weights):
    """
    Expand per-dataset weights to per-sample weights for ConcatDataset.

    Args:
        dataset: A PyTorch Dataset (can be ConcatDataset or regular Dataset)
        weights: List of weights, one per dataset in ConcatDataset,
                 or a list of per-sample weights

    Returns:
        torch.Tensor: Per-sample weights with length equal to len(dataset)

    Examples:
        >>> # For ConcatDataset with 2 datasets
        >>> dataset = ConcatDataset([dataset1, dataset2])
        >>> weights = [1.0, 2.0]  # Per-dataset weights
        >>> sample_weights = expand_concat_dataset_weights(dataset, weights)
        >>> # sample_weights[i] = 1.0 if i from dataset1, 2.0 if from dataset2

        >>> # For regular dataset
        >>> dataset = TensorDataset(torch.randn(100, 10))
        >>> weights = [1.0, 2.0, 3.0]  # Will cycle through these
        >>> sample_weights = expand_concat_dataset_weights(dataset, weights)
    """
    # Check if this is a ConcatDataset
    if hasattr(dataset, 'datasets') and hasattr(dataset, 'cumulative_sizes'):
        # This is a ConcatDataset
        # Expand per-dataset weights to per-sample weights
        sample_weights = []
        cumulative_sizes = [0] + list(dataset.cumulative_sizes)

        for dataset_idx, (start, end) in enumerate(zip(cumulative_sizes[:-1], cumulative_sizes[1:])):
            dataset_len = end - start
            # Get weight for this dataset, default to 1.0 if not specified
            weight = weights[dataset_idx] if dataset_idx < len(weights) else 1.0
            sample_weights.extend([weight] * dataset_len)

        return torch.tensor(sample_weights, dtype=torch.float32)
    else:
        # Not a ConcatDataset
        # If weights length matches dataset length, use them directly
        if len(weights) == len(dataset):
            return torch.tensor(weights, dtype=torch.float32)
        # Otherwise, cycle through weights
        else:
            sample_weights = [weights[i % len(weights)] for i in range(len(dataset))]
            return torch.tensor(sample_weights, dtype=torch.float32)


def create_sampler_for_dataset(dataset, weights, num_replicas=1, rank=0, shuffle=True, replacement=True):
    """
    Create appropriate sampler for single or multi-GPU training.

    Args:
        dataset: PyTorch Dataset
        weights: Per-dataset weights (for ConcatDataset) or per-sample weights
        num_replicas: Number of processes participating in distributed training
        rank: Rank of the current process
        shuffle: Whether to shuffle the data
        replacement: Whether to sample with replacement

    Returns:
        DistributedSampler or WeightedRandomSampler based on num_replicas
    """
    from dataset.distributed_weighted_sampler import WeightedDistributedSampler

    # Expand weights to per-sample format
    sample_weights = expand_concat_dataset_weights(dataset, weights)

    if num_replicas > 1:
        # Multi-GPU: use custom WeightedDistributedSampler
        # Pass the expanded per-sample weights directly
        sampler = WeightedDistributedSampler(
            dataset,
            weights=sample_weights.tolist(),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle
        )
    else:
        # Single-GPU: use standard WeightedRandomSampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(dataset),
            replacement=replacement
        )

    return sampler
