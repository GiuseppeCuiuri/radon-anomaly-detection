"""Shared pytest fixtures.

Torch-dependent modules are imported lazily inside fixtures so that the
configuration test suite runs in an environment without a deep-learning stack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml


@pytest.fixture
def write_yaml(tmp_path: Path):
    """Return a helper that writes a mapping to a temporary YAML file.

    Returns:
        A callable ``(name, content) -> Path``.
    """

    def _write(name: str, content: dict[str, Any]) -> Path:
        path = tmp_path / name
        path.write_text(yaml.safe_dump(content), encoding="utf-8")
        return path

    return _write


@pytest.fixture
def empty_base(tmp_path: Path) -> Path:
    """An empty base config layer, so tests start from dataclass defaults."""
    path = tmp_path / "empty_base.yaml"
    path.write_text("", encoding="utf-8")
    return path


@pytest.fixture
def raw_mnist_splits():
    """Build a small synthetic stand-in for MNIST's raw splits.

    Ten samples per digit in train and four per digit in test, with pixel values
    keyed to the digit so that a sample's provenance can be asserted after
    shuffling. Avoids downloading 50 MB to test index arithmetic.

    Returns:
        A :class:`~radon.data.base.RawSplits` instance.
    """
    torch = pytest.importorskip("torch")
    from radon.data.base import RawSplits

    n_features = 12
    per_digit_train, per_digit_test = 10, 4

    def _make(per_digit: int):
        xs, ys = [], []
        for digit in range(10):
            block = torch.full((per_digit, n_features), digit / 10.0, dtype=torch.float32)
            xs.append(block)
            ys.append(torch.full((per_digit,), digit, dtype=torch.int64))
        return torch.cat(xs), torch.cat(ys)

    x_train, y_train = _make(per_digit_train)
    x_test, y_test = _make(per_digit_test)
    return RawSplits(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
