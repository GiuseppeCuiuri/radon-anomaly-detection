"""MNIST one-vs-rest anomaly-detection protocol.

One digit is designated normal. The training set holds ``n_normal`` samples of
that digit plus ``n_anomaly`` contaminating samples drawn from
``anomaly_digits``. The test set holds inliers of the normal digit and outliers
drawn from *every* other digit.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import Tensor

from radon.config import MnistConfig
from radon.data.base import INLIER, OUTLIER, AnomalyDataset, DatasetBuilder, RawSplits
from radon.exceptions import DatasetError

__all__ = ["MnistBuilder"]

logger = logging.getLogger(__name__)


class MnistBuilder(DatasetBuilder):
    """Build the MNIST anomaly-detection protocol from torchvision's MNIST."""

    name = "mnist"

    def __init__(self, data_dir: Path, cfg: MnistConfig, seed: int) -> None:
        """Initialise the builder.

        Args:
            data_dir: Directory holding (or to receive) the ``MNIST`` folder.
            cfg: Protocol parameters.
            seed: Seed for sample selection and shuffling.
        """
        super().__init__(data_dir=data_dir, seed=seed)
        self.cfg = cfg

    def load(self) -> RawSplits:
        """Read MNIST and scale pixels into ``[0, 1]``.

        Images are read straight from the ``.data`` attribute and normalised
        here. No ``transform`` is passed to the torchvision dataset on purpose:
        transforms apply only when indexing a sample, so a ``ToTensor()``
        argument would be silently ignored by this access pattern - as it was in
        the original code, which passed one and then never used it.

        Returns:
            Raw splits with features of shape ``(N, 784)`` and digit labels.

        Raises:
            DatasetError: If MNIST is absent and cannot be downloaded.
        """
        # Imported lazily so that `import radon.data` does not pull in
        # torchvision (and its PIL dependency) for callers that never use MNIST.
        from torchvision import datasets

        try:
            train = datasets.MNIST(root=str(self.data_dir), train=True, download=self.cfg.download)
            test = datasets.MNIST(root=str(self.data_dir), train=False, download=self.cfg.download)
        except (RuntimeError, OSError) as exc:
            raise DatasetError(
                f"Cannot load MNIST from '{self.data_dir}': {exc}. "
                "Check the path, or set data.mnist.download=true to fetch it."
            ) from exc

        x_train = self._to_flat_float(train.data)
        x_test = self._to_flat_float(test.data)
        y_train = train.targets.clone().to(torch.int64)
        y_test = test.targets.clone().to(torch.int64)

        logger.info("MNIST loaded: train %s, test %s.", tuple(x_train.shape), tuple(x_test.shape))
        return RawSplits(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    @staticmethod
    def _to_flat_float(images: Tensor) -> Tensor:
        """Convert a uint8 image batch to flattened float32 in ``[0, 1]``.

        Args:
            images: Tensor of shape ``(N, H, W)`` with ``uint8`` pixels.

        Returns:
            Tensor of shape ``(N, H * W)``, ``float32``, scaled to ``[0, 1]``.
        """
        # `.float()` already allocates a new tensor, so the division can be
        # done in place: one full-size temporary saved per split.
        flat = images.reshape(images.shape[0], -1).float()
        return flat.div_(255.0)

    def preprocess(self, raw: RawSplits) -> AnomalyDataset:
        """Assemble the contaminated training set and the test set.

        Args:
            raw: Output of :meth:`load`.

        Returns:
            The dataset with binary labels.

        Raises:
            DatasetError: If the source has fewer samples than the protocol
                requires for the training split.
        """
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        x_train, y_train = self._build_train_split(raw, generator)
        x_test, y_test = self._build_test_split(raw, generator)

        return AnomalyDataset(
            name=self.name,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            feature_names=None,
        )

    def _build_train_split(
        self, raw: RawSplits, generator: torch.Generator
    ) -> tuple[Tensor, Tensor]:
        """Sample inliers and contaminating outliers, then shuffle.

        Args:
            raw: Raw splits.
            generator: RNG driving the sampling.

        Returns:
            A ``(features, labels)`` pair.

        Raises:
            DatasetError: If the requested counts exceed what MNIST provides.
                Silently returning fewer samples would change the experiment's
                contamination rate without any visible signal.
        """
        cfg = self.cfg

        normal_pool = (raw.y_train == cfg.normal_digit).nonzero(as_tuple=True)[0]
        self._require_enough(len(normal_pool), cfg.n_normal, f"digit {cfg.normal_digit}", "train")
        perm = torch.randperm(len(normal_pool), generator=generator)
        idx_normal = normal_pool[perm[: cfg.n_normal]]

        x_parts = [raw.x_train[idx_normal]]
        y_parts = [torch.full((len(idx_normal),), INLIER, dtype=torch.int64)]

        if cfg.n_anomaly > 0:
            wanted = torch.tensor(cfg.anomaly_digits, dtype=raw.y_train.dtype)
            anomaly_pool = torch.isin(raw.y_train, wanted).nonzero(as_tuple=True)[0]
            self._require_enough(
                len(anomaly_pool), cfg.n_anomaly, f"digits {list(cfg.anomaly_digits)}", "train"
            )
            perm = torch.randperm(len(anomaly_pool), generator=generator)
            idx_anomaly = anomaly_pool[perm[: cfg.n_anomaly]]

            x_parts.append(raw.x_train[idx_anomaly])
            y_parts.append(torch.full((len(idx_anomaly),), OUTLIER, dtype=torch.int64))

        x = torch.cat(x_parts, dim=0)
        y = torch.cat(y_parts, dim=0)

        shuffle = torch.randperm(x.shape[0], generator=generator)
        return x[shuffle], y[shuffle]

    def _build_test_split(
        self, raw: RawSplits, generator: torch.Generator
    ) -> tuple[Tensor, Tensor]:
        """Assemble the test split and shuffle it.

        Test outliers come from every digit other than ``normal_digit``, not
        just ``anomaly_digits``. A requested count larger than the pool is
        capped with a warning rather than raising: for the test split the count
        is a cap, and ``None`` explicitly means "keep everything".

        Args:
            raw: Raw splits.
            generator: RNG driving the sampling.

        Returns:
            A ``(features, labels)`` pair.
        """
        cfg = self.cfg

        normal_pool = (raw.y_test == cfg.normal_digit).nonzero(as_tuple=True)[0]
        idx_normal = self._take_at_most(normal_pool, cfg.n_test_normal, generator, "test inliers")

        outlier_pool = (raw.y_test != cfg.normal_digit).nonzero(as_tuple=True)[0]
        idx_outlier = self._take_at_most(
            outlier_pool, cfg.n_test_anomaly, generator, "test outliers"
        )

        x = torch.cat([raw.x_test[idx_normal], raw.x_test[idx_outlier]], dim=0)
        y = torch.cat(
            [
                torch.full((len(idx_normal),), INLIER, dtype=torch.int64),
                torch.full((len(idx_outlier),), OUTLIER, dtype=torch.int64),
            ],
            dim=0,
        )

        shuffle = torch.randperm(x.shape[0], generator=generator)
        return x[shuffle], y[shuffle]

    @staticmethod
    def _take_at_most(pool: Tensor, n: int | None, generator: torch.Generator, what: str) -> Tensor:
        """Randomly take ``n`` indices from ``pool``, or all of them.

        No random number is drawn when the whole pool is kept, which preserves
        the RNG consumption order of the original implementation and therefore
        the exact sample selection.

        Args:
            pool: Candidate indices.
            n: How many to keep; ``None`` or a value at least as large as the
                pool keeps everything.
            generator: RNG driving the permutation.
            what: Description used in the warning message.

        Returns:
            The selected indices.
        """
        if n is None or n >= len(pool):
            if n is not None and n > len(pool):
                logger.warning(
                    "Requested %d %s but only %d are available; keeping all of them.",
                    n,
                    what,
                    len(pool),
                )
            return pool
        perm = torch.randperm(len(pool), generator=generator)
        return pool[perm[:n]]

    @staticmethod
    def _require_enough(available: int, requested: int, source: str, split: str) -> None:
        """Raise when a split cannot be built with the requested sample count.

        Args:
            available: Samples present in the pool.
            requested: Samples the protocol asks for.
            source: Human-readable description of the pool.
            split: Split name, used in the message.

        Raises:
            DatasetError: If ``requested`` exceeds ``available``.
        """
        if requested > available:
            raise DatasetError(
                f"MNIST {split} split: {requested} sample(s) requested from {source} but only "
                f"{available} are available. Lower the count in the config, because proceeding "
                f"would silently change the contamination rate."
            )
