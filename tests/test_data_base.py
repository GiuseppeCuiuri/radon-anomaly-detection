"""Tests for AnomalyDataset invariants and label statistics."""

from __future__ import annotations

import logging

import pytest

torch = pytest.importorskip("torch")

from radon.data.base import AnomalyDataset, label_statistics  # noqa: E402
from radon.exceptions import DatasetError  # noqa: E402


def _dataset(**overrides) -> AnomalyDataset:
    """Build a minimal valid dataset, with fields replaceable by keyword."""
    defaults = {
        "name": "toy",
        "x_train": torch.rand(10, 4, dtype=torch.float32),
        "y_train": torch.tensor([0] * 8 + [1] * 2, dtype=torch.int64),
        "x_test": torch.rand(6, 4, dtype=torch.float32),
        "y_test": torch.tensor([0] * 4 + [1] * 2, dtype=torch.int64),
        "feature_names": None,
    }
    return AnomalyDataset(**{**defaults, **overrides})


class TestLabelStatistics:
    def test_counts_and_contamination(self) -> None:
        stats = label_statistics(torch.tensor([0, 0, 0, 1], dtype=torch.int64))
        assert (stats.n_inliers, stats.n_outliers, stats.n_total) == (3, 1, 4)
        assert stats.contamination == pytest.approx(0.25)

    def test_empty_tensor_is_rejected(self) -> None:
        with pytest.raises(DatasetError, match="empty"):
            label_statistics(torch.empty(0, dtype=torch.int64))


class TestValidate:
    def test_valid_dataset_passes(self) -> None:
        _dataset().validate()

    def test_properties(self) -> None:
        ds = _dataset()
        assert (ds.n_features, ds.n_train, ds.n_test) == (4, 10, 6)
        assert ds.train_stats.contamination == pytest.approx(0.2)

    def test_rejects_three_dimensional_features(self) -> None:
        with pytest.raises(DatasetError, match="must be 2-D"):
            _dataset(x_train=torch.rand(10, 2, 2, dtype=torch.float32)).validate()

    def test_rejects_length_mismatch(self) -> None:
        with pytest.raises(DatasetError, match="10 samples but y_train has 9"):
            _dataset(y_train=torch.zeros(9, dtype=torch.int64)).validate()

    def test_rejects_non_binary_labels(self) -> None:
        y = torch.tensor([0] * 8 + [1, 7], dtype=torch.int64)
        with pytest.raises(DatasetError, match=r"only \{0, 1\}"):
            _dataset(y_train=y).validate()

    def test_rejects_wrong_feature_dtype(self) -> None:
        with pytest.raises(DatasetError, match="must be float32"):
            _dataset(x_train=torch.rand(10, 4, dtype=torch.float64)).validate()

    def test_rejects_wrong_label_dtype(self) -> None:
        with pytest.raises(DatasetError, match="must be int64"):
            _dataset(y_train=torch.zeros(10, dtype=torch.int32)).validate()

    def test_rejects_nan_features(self) -> None:
        x = torch.rand(10, 4, dtype=torch.float32)
        x[3, 1] = float("nan")
        with pytest.raises(DatasetError, match="non-finite"):
            _dataset(x_train=x).validate()

    def test_rejects_feature_count_mismatch_between_splits(self) -> None:
        with pytest.raises(DatasetError, match="feature-count mismatch"):
            _dataset(x_test=torch.rand(6, 5, dtype=torch.float32)).validate()

    def test_rejects_misaligned_feature_names(self) -> None:
        with pytest.raises(DatasetError, match="feature name"):
            _dataset(feature_names=("a", "b")).validate()

    def test_rejects_empty_split(self) -> None:
        with pytest.raises(DatasetError, match="empty"):
            _dataset(
                x_train=torch.empty(0, 4, dtype=torch.float32),
                y_train=torch.empty(0, dtype=torch.int64),
            ).validate()

    def test_out_of_range_features_warn_but_pass(self, caplog) -> None:
        """A sigmoid decoder cannot reproduce values outside [0, 1]."""
        x = torch.rand(10, 4, dtype=torch.float32) * 5.0
        with caplog.at_level(logging.WARNING, logger="radon.data.base"):
            _dataset(x_train=x).validate()
        assert "outside [0, 1]" in caplog.text


class TestTo:
    def test_to_returns_a_copy(self) -> None:
        ds = _dataset()
        moved = ds.to("cpu")
        assert moved is not ds
        assert torch.equal(moved.x_train, ds.x_train)
