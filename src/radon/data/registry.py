"""Dataset builder selection.

Maps ``config.data.dataset`` to a concrete :class:`~radon.data.base.DatasetBuilder`,
so entry points never branch on the dataset name themselves.
"""

from __future__ import annotations

from radon.config import Config
from radon.data.base import AnomalyDataset, DatasetBuilder
from radon.data.mnist import MnistBuilder
from radon.data.nsl_kdd import NslKddBuilder
from radon.exceptions import ConfigError

__all__ = ["available_datasets", "build_dataset", "create_builder"]


def available_datasets() -> tuple[str, ...]:
    """Return the dataset names this package can build.

    Returns:
        The supported identifiers, sorted.
    """
    return ("mnist", "nsl_kdd")


def create_builder(cfg: Config) -> DatasetBuilder:
    """Instantiate the builder selected by ``cfg.data.dataset``.

    Args:
        cfg: The run configuration.

    Returns:
        A builder wired with the matching protocol block and the master seed.

    Raises:
        ConfigError: If the dataset name has no registered builder. Config
            validation already rejects unknown names, so reaching this means a
            builder was declared in the schema but never implemented.
    """
    name = cfg.data.dataset

    if name == "mnist":
        return MnistBuilder(data_dir=cfg.paths.data_dir, cfg=cfg.data.mnist, seed=cfg.seed)
    if name == "nsl_kdd":
        return NslKddBuilder(data_dir=cfg.paths.data_dir, cfg=cfg.data.nsl_kdd, seed=cfg.seed)

    raise ConfigError(
        f"data.dataset: no builder registered for '{name}'. "
        f"Available: {list(available_datasets())}."
    )


def build_dataset(cfg: Config) -> AnomalyDataset:
    """Build, preprocess and validate the dataset named in the config.

    Args:
        cfg: The run configuration.

    Returns:
        A validated :class:`~radon.data.base.AnomalyDataset`.

    Raises:
        ConfigError: If the dataset name is unknown.
        DatasetError: If loading or preprocessing fails.
    """
    return create_builder(cfg).build()
