"""Tests for the command-line entry points."""

from __future__ import annotations

import importlib
import logging

import pytest

torch = pytest.importorskip("torch")

from radon.cli import explain as explain_cli  # noqa: E402
from radon.cli import train as train_cli  # noqa: E402
from radon.data import AnomalyDataset  # noqa: E402
from radon.explain import profile_from_dict  # noqa: E402


class TestLoggerNames:
    """Guards a bug that made CLI log records vanish.

    These modules run as ``python -m radon.cli.X``, so ``__name__`` is
    ``"__main__"``. A logger by that name is not a child of the configured
    ``radon`` logger, which has ``propagate = False`` and owns the handlers, so
    every record emitted through it was silently dropped.
    """

    @pytest.mark.parametrize("module_name", ["train", "explain", "common"])
    def test_cli_loggers_sit_under_the_radon_logger(self, module_name: str) -> None:
        module = importlib.import_module(f"radon.cli.{module_name}")
        assert module.logger.name.startswith("radon."), (
            f"radon.cli.{module_name} uses logger '{module.logger.name}', which will not "
            "reach the handlers attached to the 'radon' logger."
        )

    def test_records_reach_the_radon_logger(self, caplog) -> None:
        with caplog.at_level(logging.INFO, logger="radon"):
            train_cli.logger.info("marker")
        assert "marker" in caplog.text


class TestParsers:
    def test_train_parser_accepts_the_documented_flags(self) -> None:
        args = train_cli.build_parser().parse_args(
            ["--config", "configs/mnist.yaml", "--set", "training.epochs=5", "--repetitions", "3"]
        )
        assert args.overrides == ["training.epochs=5"]
        assert args.repetitions == 3
        assert args.no_plots is False

    def test_repeated_set_flags_accumulate(self) -> None:
        args = train_cli.build_parser().parse_args(["--set", "seed=1", "--set", "training.lr=0.01"])
        assert args.overrides == ["seed=1", "training.lr=0.01"]

    def test_explain_parser_has_its_own_options(self) -> None:
        args = explain_cli.build_parser().parse_args(["--samples", "5"])
        assert args.samples == 5
        assert args.checkpoint is None

    def test_defaults(self) -> None:
        args = train_cli.build_parser().parse_args([])
        assert args.config is None
        assert args.overrides == []
        assert args.repetitions == 1


class TestErrorHandling:
    def test_bad_override_exits_nonzero(self, tmp_path, monkeypatch) -> None:
        """A malformed override is a ConfigError, which main() must report."""
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit) as exit_info:
            raise SystemExit(train_cli.main(["--set", "not-an-assignment"]))
        assert exit_info.value.code == 1

    def test_unknown_config_key_exits_nonzero(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        assert train_cli.main(["--set", "training.epocs=5"]) == 1


class TestCheckpointProfile:
    """The inference service explains records from the profile stored at training time."""

    @staticmethod
    def _dataset(feature_names: tuple[str, ...] | None) -> AnomalyDataset:
        x = torch.rand(20, 3, generator=torch.Generator().manual_seed(0))
        y = torch.zeros(20, dtype=torch.long)
        y[:2] = 1
        return AnomalyDataset(
            name="toy", x_train=x, y_train=y, x_test=x, y_test=y, feature_names=feature_names
        )

    def test_tabular_dataset_stores_the_inlier_profile(self) -> None:
        names = ("duration", "protocol_type_tcp", "protocol_type_udp")
        extra = train_cli.inlier_profile_extra(self._dataset(names))

        profile = profile_from_dict(extra["inlier_profile"])
        assert profile.feature_names == names
        assert profile.onehot_groups == {"protocol_type": (1, 2)}

    def test_image_dataset_stores_nothing(self) -> None:
        assert train_cli.inlier_profile_extra(self._dataset(None)) == {}
