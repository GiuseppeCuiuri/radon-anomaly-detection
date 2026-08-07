"""Reproducibility helpers.

The original code seeded the *global* RNGs inside the preprocessing function and
then relied on that state for every subsequent ``torch.randperm`` call. That
couples dataset sampling to anything else that happens to draw from the global
generator, so inserting an unrelated random call anywhere upstream silently
changes which samples end up in the training set.

Here the global RNGs are still seeded (third-party code depends on them), but
components that need randomness receive an explicit
:class:`torch.Generator`, making their draws independent of unrelated code.
"""

from __future__ import annotations

import logging
import random

import numpy as np
import torch

__all__ = ["seed_worker", "set_seed"]

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = True) -> torch.Generator:
    """Seed every RNG in play and return a dedicated generator.

    Args:
        seed: Master seed, applied to :mod:`random`, NumPy and PyTorch.
        deterministic: Request deterministic cuDNN kernels and disable the
            autotuner. Reduces GPU throughput but makes runs repeatable. Has no
            effect when CUDA is unavailable.

    Returns:
        A CPU :class:`torch.Generator` seeded with ``seed``, to be passed to
        DataLoaders and sampling helpers.

    Note:
        This does not call :func:`torch.use_deterministic_algorithms`, which
        raises on several ops used here and would abort otherwise valid runs.
        Full bitwise determinism on GPU is therefore not guaranteed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic

    logger.debug("RNGs seeded with %d (deterministic=%s).", seed, deterministic)

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:  # noqa: ARG001 - fixed callback signature
    """Seed a DataLoader worker process.

    Pass as ``worker_init_fn`` when ``num_workers > 0``; without it each worker
    inherits the same NumPy seed and produces identical augmentations.

    Args:
        worker_id: Worker index supplied by the DataLoader. Unused: PyTorch has
            already given each worker a distinct base seed by the time this
            runs, so :func:`torch.initial_seed` is the value to propagate.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
