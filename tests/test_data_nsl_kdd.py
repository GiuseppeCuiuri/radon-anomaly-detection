"""Tests for the NSL-KDD loader and contamination targeting."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from radon.config import NslKddConfig  # noqa: E402
from radon.data.nsl_kdd import NslKddBuilder  # noqa: E402
from radon.exceptions import DatasetError  # noqa: E402

FEATURES = ["duration", "src_bytes", "protocol_type_tcp", "protocol_type_udp"]


@pytest.fixture
def split_dir(tmp_path: Path) -> Path:
    """Write a synthetic NSL-KDD split: 100 inliers and 50 outliers in train."""
    directory = tmp_path / "NSL_KDD"
    directory.mkdir()
    rng = np.random.default_rng(0)

    def _write(name: str, frame: pd.DataFrame) -> None:
        frame.to_csv(directory / name, index=False)

    _write("X_train", pd.DataFrame(rng.random((150, len(FEATURES))), columns=FEATURES))
    _write("Y_train", pd.DataFrame({"label": [0] * 100 + [1] * 50}))
    _write("X_test", pd.DataFrame(rng.random((40, len(FEATURES))), columns=FEATURES))
    _write("Y_test", pd.DataFrame({"label": [0] * 25 + [1] * 15}))
    return tmp_path


def _builder(data_dir: Path, cfg: NslKddConfig | None = None, seed: int = 42) -> NslKddBuilder:
    return NslKddBuilder(data_dir=data_dir, cfg=cfg or NslKddConfig(), seed=seed)


class TestLoad:
    def test_loads_shapes_dtypes_and_names(self, split_dir: Path) -> None:
        raw = _builder(split_dir).load()
        assert raw.x_train.shape == (150, 4)
        assert raw.x_train.dtype == torch.float32
        assert raw.y_train.dtype == torch.int64
        assert raw.y_train.ndim == 1
        assert raw.feature_names == tuple(FEATURES)

    def test_missing_files_are_listed_together(self, tmp_path: Path) -> None:
        (tmp_path / "NSL_KDD").mkdir()
        with pytest.raises(DatasetError) as exc:
            _builder(tmp_path).load()
        message = str(exc.value)
        assert all(name in message for name in ("X_train", "Y_train", "X_test", "Y_test"))

    def test_stray_index_column_is_dropped(self, split_dir: Path) -> None:
        path = split_dir / "NSL_KDD" / "X_train"
        frame = pd.read_csv(path)
        frame.insert(0, "Unnamed: 0", range(len(frame)))
        frame.to_csv(path, index=False)

        raw = _builder(split_dir).load()
        assert raw.feature_names == tuple(FEATURES)
        assert raw.x_train.shape[1] == 4

    def test_column_mismatch_between_splits_raises(self, split_dir: Path) -> None:
        """The original code took feature names from train and never checked test."""
        path = split_dir / "NSL_KDD" / "X_test"
        frame = pd.read_csv(path)
        frame = frame.rename(columns={"src_bytes": "src_octets"})
        frame.to_csv(path, index=False)

        with pytest.raises(DatasetError, match="do not match"):
            _builder(split_dir).load()

    def test_column_reordering_is_caught(self, split_dir: Path) -> None:
        path = split_dir / "NSL_KDD" / "X_test"
        frame = pd.read_csv(path)
        frame[list(reversed(FEATURES))].to_csv(path, index=False)

        with pytest.raises(DatasetError, match="different order"):
            _builder(split_dir).load()

    def test_multi_column_label_frame_raises(self, split_dir: Path) -> None:
        path = split_dir / "NSL_KDD" / "Y_train"
        pd.DataFrame({"label": [0] * 150, "extra": [1] * 150}).to_csv(path, index=False)

        with pytest.raises(DatasetError, match="single label column"):
            _builder(split_dir).load()

    def test_single_row_labels_stay_one_dimensional(self, tmp_path: Path) -> None:
        """`squeeze()` would collapse a one-row frame to a 0-D scalar."""
        directory = tmp_path / "NSL_KDD"
        directory.mkdir()
        one_row = pd.DataFrame([[0.1, 0.2, 1.0, 0.0]], columns=FEATURES)
        for name in ("X_train", "X_test"):
            one_row.to_csv(directory / name, index=False)
        for name in ("Y_train", "Y_test"):
            pd.DataFrame({"label": [0]}).to_csv(directory / name, index=False)

        raw = _builder(tmp_path).load()
        assert raw.y_train.shape == (1,)


class TestContamination:
    @pytest.mark.parametrize("target", [0.05, 0.10, 0.20])
    def test_effective_rate_hits_the_target(self, split_dir: Path, target: float) -> None:
        builder = _builder(split_dir, NslKddConfig(target_contamination=target))
        dataset = builder.build()
        # n_out = round-down of c*n_in/(1-c), so the realised rate lands just at
        # or below the target, never above it.
        assert dataset.train_stats.contamination <= target + 1e-9
        assert dataset.train_stats.contamination > target - 0.02

    def test_all_inliers_are_kept(self, split_dir: Path) -> None:
        dataset = _builder(split_dir).build()
        assert dataset.train_stats.n_inliers == 100

    def test_zero_contamination_drops_every_outlier(self, split_dir: Path) -> None:
        builder = _builder(split_dir, NslKddConfig(target_contamination=0.0))
        dataset = builder.build()
        assert dataset.train_stats.n_outliers == 0

    def test_unreachable_target_caps_and_warns(self, split_dir: Path, caplog) -> None:
        builder = _builder(split_dir, NslKddConfig(target_contamination=0.9))
        dataset = builder.build()
        assert dataset.train_stats.n_outliers == 50
        assert "using all of them" in caplog.text

    def test_test_split_is_untouched(self, split_dir: Path) -> None:
        dataset = _builder(split_dir).build()
        assert dataset.n_test == 40
        assert dataset.test_stats.n_outliers == 15

    def test_same_seed_gives_identical_selection(self, split_dir: Path) -> None:
        first = _builder(split_dir, seed=3).build()
        second = _builder(split_dir, seed=3).build()
        assert torch.equal(first.x_train, second.x_train)


class TestClamping:
    def test_out_of_range_test_values_are_clamped(self, split_dir: Path) -> None:
        path = split_dir / "NSL_KDD" / "X_test"
        frame = pd.read_csv(path)
        frame.iloc[0, 0] = 7.5
        frame.iloc[1, 0] = -2.0
        frame.to_csv(path, index=False)

        dataset = _builder(split_dir).build()
        assert float(dataset.x_test.max()) <= 1.0
        assert float(dataset.x_test.min()) >= 0.0

    def test_clamping_can_be_disabled(self, split_dir: Path) -> None:
        path = split_dir / "NSL_KDD" / "X_test"
        frame = pd.read_csv(path)
        frame.iloc[0, 0] = 7.5
        frame.to_csv(path, index=False)

        cfg = NslKddConfig(clamp_test_to_unit_range=False)
        dataset = _builder(split_dir, cfg).build()
        assert float(dataset.x_test.max()) == pytest.approx(7.5)


class TestOutput:
    def test_result_passes_validation(self, split_dir: Path) -> None:
        _builder(split_dir).build().validate()

    def test_feature_names_align_with_columns(self, split_dir: Path) -> None:
        dataset = _builder(split_dir).build()
        assert dataset.feature_names is not None
        assert len(dataset.feature_names) == dataset.n_features

    def test_no_inliers_raises(self, split_dir: Path) -> None:
        path = split_dir / "NSL_KDD" / "Y_train"
        pd.DataFrame({"label": [1] * 150}).to_csv(path, index=False)

        with pytest.raises(DatasetError, match="no inliers"):
            _builder(split_dir).build()


class TestConfigDefaults:
    def test_replace_keeps_other_fields(self) -> None:
        cfg = replace(NslKddConfig(), target_contamination=0.01)
        assert cfg.target_contamination == 0.01
        assert cfg.clamp_test_to_unit_range is True
