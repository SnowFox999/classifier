# utils/sampler.py

import numpy as np
from torch.utils.data import WeightedRandomSampler


def make_sampler(labels):
    """
    labels: np.ndarray или list с class indices (0..C-1)
    """
    counts = np.bincount(labels)
    weights = 1.0 / counts
    sample_weights = weights[labels]

    return WeightedRandomSampler(
        sample_weights,
        len(sample_weights),
        replacement=True
    )
