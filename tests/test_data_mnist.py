"""Tests for the MNIST anomaly-detection protocol.

Preprocessing is exercised against synthetic raw splits, so the suite never
downloads MNIST. A single opt-in test (``--runslow``) checks the real loader.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from radon.config import MnistConfig  # noqa: E402
from radon.data.mnist import MnistBuilder  # noqa: E402
from radon.exceptions import DatasetError  # noqa: E402


@pytest.fixture
def cfg() -> MnistConfig:
    """A protocol small enough for the synthetic fixture (10 samples/digit)."""
    return MnistConfig(
        normal_digit=4,
        n_normal=8,
        n_anomaly=3,
        anomaly_digits=(0, 2, 3),
        n_test_normal=None,
        n_test_anomaly=5,
        download=False,
    )


def _builder(cfg: MnistConfig, seed: int = 42) -> MnistBuilder:
    return MnistBuilder(data_dir="unused", cfg=cfg, seed=seed)


class TestTrainSplit:
    def test_composition_matches_the_config(self, cfg: MnistConfig, raw_mnist_splits) -> None:
        ds = _builder(cfg).preprocess(raw_mnist_splits)
        assert ds.n_train == cfg.n_normal + cfg.n_anomaly
        assert ds.train_stats.n_inliers == cfg.n_normal
        assert ds.train_stats.n_outliers == cfg.n_anomaly

    def test_inliers_come_from_the_normal_digit_only(
        self, cfg: MnistConfig, raw_mnist_splits
    ) -> None:
        """The fixture encodes each digit as a constant pixel value d/10."""
        ds = _builder(cfg).preprocess(raw_mnist_splits)
        inliers = ds.x_train[ds.y_train == 0]
        assert torch.allclose(inliers, torch.full_like(inliers, cfg.normal_digit / 10.0))

    def test_train_outliers_come_only_from_anomaly_digits(
        self, cfg: MnistConfig, raw_mnist_splits
    ) -> None:
        ds = _builder(cfg).preprocess(raw_mnist_splits)
        outliers = ds.x_train[ds.y_train == 1]
        observed = {round(v * 10) for v in outliers[:, 0].tolist()}
        assert observed <= set(cfg.anomaly_digits)

    def test_uncontaminated_training_set(self, cfg: MnistConfig, raw_mnist_splits) -> None:
        ds = _builder(replace(cfg, n_anomaly=0)).preprocess(raw_mnist_splits)
        assert ds.train_stats.n_outliers == 0
        assert ds.train_stats.contamination == 0.0

    def test_requesting_more_inliers_than_available_raises(
        self, cfg: MnistConfig, raw_mnist_splits
    ) -> None:
        """Silently shrinking the split would change the contamination rate."""
        with pytest.raises(DatasetError, match="only 10 are available"):
            _builder(replace(cfg, n_normal=999)).preprocess(raw_mnist_splits)

    def test_requesting_more_train_outliers_than_available_raises(
        self, cfg: MnistConfig, raw_mnist_splits
    ) -> None:
        with pytest.raises(DatasetError, match="30 are available"):
            _builder(replace(cfg, n_anomaly=999)).preprocess(raw_mnist_splits)


class TestTestSplit:
    def test_null_keeps_every_inlier(self, cfg: MnistConfig, raw_mnist_splits) -> None:
        ds = _builder(cfg).preprocess(raw_mnist_splits)
        assert ds.test_stats.n_inliers == 4  # per_digit_test in the fixture

    def test_outlier_cap_is_applied(self, cfg: MnistConfig, raw_mnist_splits) -> None:
        ds = _builder(cfg).preprocess(raw_mnist_splits)
        assert ds.test_stats.n_outliers == cfg.n_test_anomaly

    def test_test_outliers_span_every_other_digit(self, cfg: MnistConfig, raw_mnist_splits) -> None:
        """Unlike training, test outliers are not restricted to anomaly_digits."""
        ds = _builder(replace(cfg, n_test_anomaly=None)).preprocess(raw_mnist_splits)
        outliers = ds.x_test[ds.y_test == 1]
        observed = {round(v * 10) for v in outliers[:, 0].tolist()}
        assert observed == set(range(10)) - {cfg.normal_digit}

    def test_oversized_cap_keeps_all_and_warns(
        self, cfg: MnistConfig, raw_mnist_splits, caplog
    ) -> None:
        ds = _builder(replace(cfg, n_test_anomaly=10_000)).preprocess(raw_mnist_splits)
        assert ds.test_stats.n_outliers == 36  # 9 digits x 4 samples
        assert "keeping all of them" in caplog.text


class TestReproducibility:
    def test_same_seed_gives_identical_splits(self, cfg: MnistConfig, raw_mnist_splits) -> None:
        first = _builder(cfg, seed=7).preprocess(raw_mnist_splits)
        second = _builder(cfg, seed=7).preprocess(raw_mnist_splits)
        assert torch.equal(first.x_train, second.x_train)
        assert torch.equal(first.y_train, second.y_train)

    def test_different_seeds_give_different_splits(
        self, cfg: MnistConfig, raw_mnist_splits
    ) -> None:
        first = _builder(cfg, seed=7).preprocess(raw_mnist_splits)
        second = _builder(cfg, seed=8).preprocess(raw_mnist_splits)
        assert not torch.equal(first.y_train, second.y_train)

    def test_explicit_generator_matches_the_global_rng(self) -> None:
        """Guards the switch from global seeding to an explicit generator.

        The original code called ``torch.manual_seed(seed)`` and then relied on
        global ``torch.randperm``. Preprocessing now uses a private generator;
        this asserts the two produce the same permutations, so sample selection
        is unchanged by the refactor.
        """
        torch.manual_seed(42)
        expected = [torch.randperm(n) for n in (5000, 3000, 128)]

        generator = torch.Generator()
        generator.manual_seed(42)
        actual = [torch.randperm(n, generator=generator) for n in (5000, 3000, 128)]

        assert all(torch.equal(a, b) for a, b in zip(expected, actual, strict=True))


class TestOutput:
    def test_result_passes_validation(self, cfg: MnistConfig, raw_mnist_splits) -> None:
        _builder(cfg).preprocess(raw_mnist_splits).validate()

    def test_features_are_flat_float32(self, cfg: MnistConfig, raw_mnist_splits) -> None:
        ds = _builder(cfg).preprocess(raw_mnist_splits)
        assert ds.x_train.ndim == 2
        assert ds.x_train.dtype == torch.float32

    def test_pixels_are_scaled_to_unit_range(self) -> None:
        images = torch.randint(0, 256, (4, 28, 28), dtype=torch.uint8)
        flat = MnistBuilder._to_flat_float(images)
        assert flat.shape == (4, 784)
        assert flat.dtype == torch.float32
        assert float(flat.min()) >= 0.0
        assert float(flat.max()) <= 1.0


@pytest.mark.slow
def test_real_mnist_loads(tmp_path) -> None:
    """Opt-in check against the real dataset: `pytest -m slow`."""
    pytest.importorskip("torchvision")
    builder = MnistBuilder(data_dir=tmp_path, cfg=MnistConfig(download=True), seed=42)
    dataset = builder.build()
    assert dataset.n_features == 784
    assert dataset.n_train == 4859 + 265
