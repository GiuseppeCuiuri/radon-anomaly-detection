"""Tests for evaluation DataLoader construction."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from radon.data.base import AnomalyDataset  # noqa: E402
from radon.data.loaders import build_eval_loader, build_test_loader  # noqa: E402


@pytest.fixture
def dataset() -> AnomalyDataset:
    return AnomalyDataset(
        name="toy",
        x_train=torch.rand(20, 3, dtype=torch.float32),
        y_train=torch.zeros(20, dtype=torch.int64),
        x_test=torch.rand(10, 3, dtype=torch.float32),
        y_test=torch.tensor([0] * 6 + [1] * 4, dtype=torch.int64),
    )


def test_batches_cover_every_sample(dataset: AnomalyDataset) -> None:
    loader = build_test_loader(dataset, batch_size=4)
    assert len(loader) == 3  # 4 + 4 + 2, drop_last is off
    assert sum(x.shape[0] for x, _ in loader) == 10


def test_order_is_deterministic(dataset: AnomalyDataset) -> None:
    """Evaluation must not shuffle, or scores and labels would decorrelate."""
    loader = build_test_loader(dataset, batch_size=4)
    first = torch.cat([y for _, y in loader])
    second = torch.cat([y for _, y in loader])
    assert torch.equal(first, dataset.y_test)
    assert torch.equal(first, second)


def test_length_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="10 samples but y has 9"):
        build_eval_loader(torch.rand(10, 3), torch.zeros(9, dtype=torch.int64), batch_size=4)


def test_non_positive_batch_size_is_rejected(dataset: AnomalyDataset) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        build_test_loader(dataset, batch_size=0)


def test_device_placement(dataset: AnomalyDataset) -> None:
    loader = build_test_loader(dataset, batch_size=4, device="cpu")
    x, _ = next(iter(loader))
    assert x.device.type == "cpu"
