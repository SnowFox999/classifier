import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

def make_sampler(labels):
    """
    labels: np.ndarray / list / pandas.Series с class indices (0..C-1)
    """
    labels = np.asarray(labels)

    n_classes = len(np.unique(labels))
    if n_classes == 1:
        # fallback: все веса = 1
        weights = torch.ones(len(labels))
        return WeightedRandomSampler(weights, len(weights), replacement=True)

    # считаем количество объектов каждого класса
    class_count = np.bincount(labels)

    # вес класса = N / count
    class_weights = len(labels) / class_count

    # вес каждого сэмпла
    sample_weights = class_weights[labels]

    sample_weights = torch.DoubleTensor(sample_weights)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
