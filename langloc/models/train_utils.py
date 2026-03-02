"""Training utilities: k-fold splitting and custom cross-entropy loss."""

import time
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
from tqdm import tqdm
import numpy as np
import random

from langloc.data_processing.scene_graph import SceneGraph


def k_fold(dataset, folds):
    """Splits a dataset into k folds using KFold (non-stratified).

    Args:
        dataset: The dataset to split (only its length is used).
        folds: Number of folds.

    Returns:
        Tuple of (train_indices, test_indices, val_indices), each a list
        of ``folds`` tensors of indices.
    """
    skf = KFold(folds, shuffle=True, random_state=12345)
    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset))):
        test_indices.append(torch.from_numpy(idx).to(torch.long))
    val_indices = [test_indices[i - 1] for i in range(folds)]
    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

    return train_indices, test_indices, val_indices


def cross_entropy(preds, targets, reduction='none', dim=-1):
    """Computes cross-entropy loss between predictions and soft targets.

    Args:
        preds: Prediction logits tensor.
        targets: Target distribution tensor (same shape as preds).
        reduction: ``'none'`` to return per-sample loss, ``'mean'`` for scalar.
        dim: Dimension along which to apply log-softmax.

    Returns:
        Loss tensor (per-sample if reduction is ``'none'``, scalar if ``'mean'``).
    """
    log_softmax = torch.nn.LogSoftmax(dim=dim)
    loss = (-targets * log_softmax(preds)).sum(1)
    assert all(loss >= 0), "Cross-entropy loss must be non-negative"
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


def k_fold_by_scene(dataset, folds: int):
    """Splits a dataset into k folds ensuring all graphs from the same scene
    stay in the same fold.

    Args:
        dataset: List of SceneGraph instances.
        folds: Number of folds.

    Returns:
        Iterator of (train_indices, val_indices) tuples.
    """
    scene_dataset = {}
    for i, graph in enumerate(dataset):
        if graph.scene_id not in scene_dataset:
            scene_dataset[graph.scene_id] = []
        scene_dataset[graph.scene_id].append(i)

    random.seed(0)
    scene_names = list(scene_dataset.keys())
    random.shuffle(scene_names)
    fold_size = len(scene_names) // folds
    train_indices, val_indices = [], []
    for i in range(folds):
        val_scene_names = scene_names[i * fold_size : (i + 1) * fold_size]
        val_indices.append([idx for scene_name in val_scene_names for idx in scene_dataset[scene_name]])
        train_indices.append([idx for scene_name in scene_names if scene_name not in val_scene_names for idx in scene_dataset[scene_name]])

    return zip(train_indices, val_indices)
