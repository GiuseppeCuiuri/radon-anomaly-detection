"""NSL-KDD anomaly-detection protocol.

The source is four pre-processed CSV files (``X_train``, ``Y_train``,
``X_test``, ``Y_test``) whose features are already scaled to ``[0, 1]`` and
whose categorical columns are already one-hot encoded. Labels are binary:
``0`` for normal traffic, ``1`` for an attack.

Training outliers are subsampled to reach a target contamination rate; the test
split is used as-is.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from radon.config import NslKddConfig
from radon.data.base import AnomalyDataset, DatasetBuilder, RawSplits
from radon.exceptions import DatasetError

__all__ = ["NslKddBuilder"]

logger = logging.getLogger(__name__)


class NslKddBuilder(DatasetBuilder):
    """Build the NSL-KDD protocol from pre-processed CSV splits."""

    name = "nsl_kdd"

    #: Sub-directory of ``data_dir`` holding the CSV files.
    SUBDIR = "NSL_KDD"
    #: Required files, in load order.
    FILES = ("X_train", "Y_train", "X_test", "Y_test")

    def __init__(self, data_dir: Path, cfg: NslKddConfig, seed: int) -> None:
        """Initialise the builder.

        Args:
            data_dir: Directory containing the ``NSL_KDD`` sub-directory.
            cfg: Protocol parameters.
            seed: Seed for outlier subsampling and shuffling.
        """
        super().__init__(data_dir=data_dir, seed=seed)
        self.cfg = cfg

    @property
    def split_dir(self) -> Path:
        """Directory holding the four CSV splits."""
        return self.data_dir / self.SUBDIR

    def load(self) -> RawSplits:
        """Read the four CSV splits and convert them to tensors.

        Returns:
            Raw splits with binary labels and aligned feature names.

        Raises:
            DatasetError: If a file is missing or unreadable, or if the train
                and test feature columns disagree.
        """
        paths = {name: self.split_dir / name for name in self.FILES}

        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            raise DatasetError(
                "NSL-KDD: missing file(s):\n  "
                + "\n  ".join(missing)
                + f"\nExpected all of {list(self.FILES)} inside '{self.split_dir}'."
            )

        frames = {name: self._read_csv(path) for name, path in paths.items()}

        x_train_df = self._drop_index_column(frames["X_train"])
        x_test_df = self._drop_index_column(frames["X_test"])
        self._require_aligned_columns(x_train_df, x_test_df)

        feature_names = tuple(str(c) for c in x_train_df.columns)

        raw = RawSplits(
            x_train=self._to_float_tensor(x_train_df),
            y_train=self._to_label_tensor(frames["Y_train"]),
            x_test=self._to_float_tensor(x_test_df),
            y_test=self._to_label_tensor(frames["Y_test"]),
            feature_names=feature_names,
        )

        logger.info(
            "NSL-KDD loaded: train %s, test %s, %d features.",
            tuple(raw.x_train.shape),
            tuple(raw.x_test.shape),
            len(feature_names),
        )
        return raw

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        """Read one CSV file into a DataFrame.

        Args:
            path: File to read.

        Returns:
            The parsed DataFrame.

        Raises:
            DatasetError: If the file cannot be read or parsed.
        """
        try:
            return pd.read_csv(path)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
            raise DatasetError(f"NSL-KDD: cannot parse '{path}': {exc}") from exc
        except OSError as exc:
            raise DatasetError(f"NSL-KDD: cannot read '{path}': {exc}") from exc

    def _drop_index_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop a leading unnamed index column left behind by ``to_csv``.

        Args:
            df: Feature frame.

        Returns:
            The frame without its stray index column, or unchanged.
        """
        if len(df.columns) == 0:
            return df
        first = str(df.columns[0])
        if first.startswith(self.cfg.index_column_prefix):
            logger.debug("Dropping stray index column '%s'.", first)
            return df.drop(columns=[df.columns[0]])
        return df

    @staticmethod
    def _require_aligned_columns(train: pd.DataFrame, test: pd.DataFrame) -> None:
        """Verify that train and test expose identical feature columns.

        The original code derived ``feature_names`` from the training frame
        alone and never compared it to the test frame. Any divergence in column
        order or membership silently misaligned every downstream feature
        attribution, since indices were mapped onto the wrong names.

        Args:
            train: Training feature frame.
            test: Test feature frame.

        Raises:
            DatasetError: If the column lists differ, naming the discrepancy.
        """
        train_cols = list(train.columns)
        test_cols = list(test.columns)
        if train_cols == test_cols:
            return

        only_train = [c for c in train_cols if c not in set(test_cols)]
        only_test = [c for c in test_cols if c not in set(train_cols)]
        details = []
        if only_train:
            details.append(f"only in train: {only_train[:10]}")
        if only_test:
            details.append(f"only in test: {only_test[:10]}")
        if not details:
            details.append("same columns but in a different order")

        raise DatasetError(
            "NSL-KDD: train and test feature columns do not match "
            f"({len(train_cols)} vs {len(test_cols)}). " + "; ".join(details)
        )

    @staticmethod
    def _to_float_tensor(df: pd.DataFrame) -> Tensor:
        """Convert a feature frame to a float32 tensor.

        Args:
            df: Feature frame.

        Returns:
            Tensor of shape ``(N, D)``.

        Raises:
            DatasetError: If the frame holds non-numeric data.
        """
        try:
            values = df.to_numpy(dtype=np.float32)
        except (ValueError, TypeError) as exc:
            raise DatasetError(
                f"NSL-KDD: feature columns must be numeric, conversion failed: {exc}"
            ) from exc
        return torch.from_numpy(values)

    @staticmethod
    def _to_label_tensor(df: pd.DataFrame) -> Tensor:
        """Convert a label frame to a 1-D int64 tensor.

        Uses ``reshape(-1)`` rather than ``squeeze()``: squeezing a
        single-row frame would collapse it to a 0-D scalar and break every
        downstream length check.

        Args:
            df: Label frame with a single column.

        Returns:
            Tensor of shape ``(N,)``.

        Raises:
            DatasetError: If the frame has more than one column or is not
                numeric.
        """
        if df.shape[1] != 1:
            raise DatasetError(
                f"NSL-KDD: expected a single label column, found {df.shape[1]}: "
                f"{list(df.columns)[:10]}."
            )
        try:
            values = df.to_numpy(dtype=np.int64)
        except (ValueError, TypeError) as exc:
            raise DatasetError(f"NSL-KDD: labels must be numeric: {exc}") from exc
        return torch.from_numpy(values).reshape(-1)

    def preprocess(self, raw: RawSplits) -> AnomalyDataset:
        """Subsample training outliers to hit the target contamination.

        All inliers are kept. The number of outliers follows from
        ``n_out = c * n_in / (1 - c)``, which is the count that makes outliers a
        fraction ``c`` of the resulting set. The test split is left untouched
        apart from the optional clamp into ``[0, 1]``.

        Args:
            raw: Output of :meth:`load`.

        Returns:
            The dataset at the requested contamination rate.

        Raises:
            DatasetError: If the training split contains no inliers.
        """
        x_test = raw.x_test
        if self.cfg.clamp_test_to_unit_range:
            below = int((x_test < 0.0).sum().item())
            above = int((x_test > 1.0).sum().item())
            if below or above:
                logger.info(
                    "Clamping %d test value(s) below 0 and %d above 1 into [0, 1].",
                    below,
                    above,
                )
                x_test = x_test.clamp(0.0, 1.0)

        x_train, y_train = self._subsample_outliers(raw)

        return AnomalyDataset(
            name=self.name,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=raw.y_test,
            feature_names=raw.feature_names,
        )

    def _subsample_outliers(self, raw: RawSplits) -> tuple[Tensor, Tensor]:
        """Select all inliers plus enough outliers to reach the target rate.

        Args:
            raw: Raw splits.

        Returns:
            A shuffled ``(features, labels)`` pair.

        Raises:
            DatasetError: If there are no inliers to build the split around.
        """
        target = self.cfg.target_contamination
        labels = raw.y_train.numpy()

        idx_inlier = np.flatnonzero(labels == 0)
        idx_outlier = np.flatnonzero(labels == 1)

        if idx_inlier.size == 0:
            raise DatasetError(
                "NSL-KDD: the training split contains no inliers (label 0); "
                "cannot build a contaminated training set."
            )

        n_wanted = int((target * idx_inlier.size) / (1.0 - target)) if target > 0.0 else 0
        if n_wanted > idx_outlier.size:
            logger.warning(
                "Target contamination %.4f needs %d outliers but only %d are available; "
                "using all of them (effective rate %.4f).",
                target,
                n_wanted,
                idx_outlier.size,
                idx_outlier.size / (idx_inlier.size + idx_outlier.size),
            )
            n_wanted = idx_outlier.size

        rng = np.random.default_rng(self.seed)
        chosen = rng.choice(idx_outlier, size=n_wanted, replace=False)
        selected = np.concatenate([idx_inlier, chosen])
        rng.shuffle(selected)

        index = torch.from_numpy(selected).long()
        x = raw.x_train[index]
        y = raw.y_train[index]

        logger.info(
            "NSL-KDD training split: %d inliers + %d outliers = %d samples "
            "(contamination %.4f, target %.4f).",
            idx_inlier.size,
            n_wanted,
            x.shape[0],
            float(y.float().mean().item()),
            target,
        )
        return x, y
