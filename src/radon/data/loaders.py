"""DataLoader construction.

Only an *evaluation* loader lives here, and that is deliberate. RADON's training
loop re-partitions the full training matrix into the inlier set ``L`` and the
outlier set ``S`` at every projection step, and rebuilds its own loaders from
that partition. A loader constructed before training would be discarded on the
first projection.

The original code did build one anyway (``create_dataloaders`` returned a
``train_loader`` alongside the test loader) and the notebook then passed the raw
``X_train`` tensor to ``fit`` instead, leaving that loader unused for the whole
run.
"""

from __future__ import annotations

import logging
from typing import cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset

from radon.data.base import AnomalyDataset

__all__ = ["build_eval_loader", "build_test_loader"]

logger = logging.getLogger(__name__)


def build_eval_loader(
    x: Tensor,
    y: Tensor,
    batch_size: int,
    device: torch.device | str | None = None,
) -> DataLoader[tuple[Tensor, Tensor]]:
    """Build a deterministic loader over ``(features, labels)`` pairs.

    Tensors are moved to ``device`` up front, so the whole split lives in device
    memory and no per-batch host-to-device copy happens. That is the right
    trade-off for datasets of this size (tens of MB) and is why the loader uses
    neither worker processes nor pinned memory: both are incompatible with data
    that already resides on the GPU.

    Args:
        x: Features of shape ``(N, D)``.
        y: Labels of shape ``(N,)``.
        batch_size: Samples per batch.
        device: Target device; ``None`` leaves the tensors where they are.

    Returns:
        A loader yielding ``(features, labels)`` batches in a fixed order.

    Raises:
        ValueError: If the tensors disagree in length or ``batch_size`` is not
            positive.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"build_eval_loader: x has {x.shape[0]} samples but y has {y.shape[0]}.")
    if batch_size <= 0:
        raise ValueError(f"build_eval_loader: batch_size must be positive, got {batch_size}.")

    if device is not None:
        x = x.to(device)
        y = y.to(device)

    # TensorDataset is not generic in torch's stubs, so its element type has to
    # be asserted here for the DataLoader's parameter to line up.
    dataset = cast("Dataset[tuple[Tensor, Tensor]]", TensorDataset(x, y))
    loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    logger.debug(
        "Evaluation loader: %d samples, %d batches of %d.",
        x.shape[0],
        len(loader),
        batch_size,
    )
    return loader


def build_test_loader(
    dataset: AnomalyDataset,
    batch_size: int,
    device: torch.device | str | None = None,
) -> DataLoader[tuple[Tensor, Tensor]]:
    """Build the evaluation loader for a dataset's test split.

    Args:
        dataset: The dataset to evaluate on.
        batch_size: Samples per batch.
        device: Target device; ``None`` leaves the tensors where they are.

    Returns:
        A loader over the test split.
    """
    return build_eval_loader(
        x=dataset.x_test, y=dataset.y_test, batch_size=batch_size, device=device
    )
