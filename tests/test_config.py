"""Tests for layered configuration loading and coercion.

These run without PyTorch installed, which is why :mod:`radon.config` must not
import it.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from radon.config import PROJECT_ROOT, Config, MnistConfig, _project_root, load_config
from radon.exceptions import ConfigError


class TestDefaults:
    def test_empty_base_yields_dataclass_defaults(self, empty_base: Path) -> None:
        cfg = load_config(base_path=empty_base)
        assert cfg.seed == 42
        assert cfg.data.dataset == "mnist"
        assert cfg.data.mnist.normal_digit == 4
        assert cfg.compat.any_enabled is False

    def test_shipped_configs_load(self) -> None:
        """The committed YAML files must match the current schema."""
        for name in ("mnist", "nsl_kdd"):
            cfg = load_config(PROJECT_ROOT / "configs" / f"{name}.yaml")
            assert cfg.data.dataset == name

    def test_relative_paths_resolve_against_project_root(self, empty_base: Path) -> None:
        cfg = load_config(base_path=empty_base)
        assert cfg.paths.data_dir == (PROJECT_ROOT / "data").resolve()
        assert cfg.paths.checkpoint_dir == cfg.paths.output_dir / "checkpoints"

    def test_project_root_is_the_checkout_in_a_source_tree(self) -> None:
        assert (PROJECT_ROOT / "pyproject.toml").is_file()
        assert (PROJECT_ROOT / "src" / "radon" / "config.py").is_file()

    def test_installed_package_resolves_against_the_working_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Installed with pip, config.py sits in site-packages, two levels below
        # a directory of the virtual environment that holds no pyproject.toml.
        installed = tmp_path / "venv" / "lib" / "site-packages" / "radon" / "config.py"
        installed.parent.mkdir(parents=True)
        installed.touch()
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)

        assert _project_root(installed) == work

    def test_absolute_paths_are_left_alone(self, empty_base: Path, tmp_path: Path) -> None:
        cfg = load_config(base_path=empty_base, overrides=[f"paths.data_dir={tmp_path.as_posix()}"])
        assert cfg.paths.data_dir == tmp_path


class TestLayering:
    def test_dataset_file_overrides_base(self, write_yaml, empty_base: Path) -> None:
        override = write_yaml("over.yaml", {"seed": 7, "data": {"batch_size": 64}})
        cfg = load_config(override, base_path=empty_base)
        assert cfg.seed == 7
        assert cfg.data.batch_size == 64
        # Untouched sibling keys survive the merge.
        assert cfg.data.dataset == "mnist"

    def test_cli_override_wins_over_file(self, write_yaml, empty_base: Path) -> None:
        override = write_yaml("over.yaml", {"seed": 7})
        cfg = load_config(override, overrides=["seed=99"], base_path=empty_base)
        assert cfg.seed == 99

    def test_nested_override_preserves_siblings(self, empty_base: Path) -> None:
        cfg = load_config(base_path=empty_base, overrides=["data.mnist.normal_digit=7"])
        assert cfg.data.mnist.normal_digit == 7
        assert cfg.data.mnist.n_normal == MnistConfig.n_normal

    def test_list_override_replaces_rather_than_extends(self, empty_base: Path) -> None:
        cfg = load_config(base_path=empty_base, overrides=["data.mnist.anomaly_digits=[1,5]"])
        assert cfg.data.mnist.anomaly_digits == (1, 5)


class TestOverrideParsing:
    @pytest.mark.parametrize(
        ("override", "expected"),
        [
            ("seed=1", 1),
            ("seed=0", 0),
        ],
    )
    def test_values_parse_as_yaml_scalars(
        self, empty_base: Path, override: str, expected: int
    ) -> None:
        assert load_config(base_path=empty_base, overrides=[override]).seed == expected

    def test_boolean_override(self, empty_base: Path) -> None:
        cfg = load_config(base_path=empty_base, overrides=["compat.legacy_auprc_trapezoid=true"])
        assert cfg.compat.legacy_auprc_trapezoid is True
        assert cfg.compat.enabled_flags() == ("legacy_auprc_trapezoid",)

    def test_null_override(self, empty_base: Path) -> None:
        cfg = load_config(base_path=empty_base, overrides=["data.mnist.n_test_anomaly=null"])
        assert cfg.data.mnist.n_test_anomaly is None

    @pytest.mark.parametrize("bad", ["seed", "=5", "data..seed=1"])
    def test_malformed_override_is_rejected(self, empty_base: Path, bad: str) -> None:
        with pytest.raises(ConfigError, match=r"[Mm]alformed"):
            load_config(base_path=empty_base, overrides=[bad])


class TestStrictness:
    def test_unknown_root_key_is_rejected(self, write_yaml, empty_base: Path) -> None:
        bad = write_yaml("bad.yaml", {"sed": 42})
        with pytest.raises(ConfigError, match="Unknown key"):
            load_config(bad, base_path=empty_base)

    def test_unknown_nested_key_names_its_location(self, write_yaml, empty_base: Path) -> None:
        bad = write_yaml("bad.yaml", {"data": {"mnist": {"normal_digits": 4}}})
        with pytest.raises(ConfigError, match=r"data\.mnist"):
            load_config(bad, base_path=empty_base)

    def test_wrong_type_is_rejected(self, empty_base: Path) -> None:
        with pytest.raises(ConfigError, match="expected an integer"):
            load_config(base_path=empty_base, overrides=["seed=notanumber"])

    def test_bool_is_not_accepted_as_int(self, empty_base: Path) -> None:
        with pytest.raises(ConfigError, match="expected an integer"):
            load_config(base_path=empty_base, overrides=["seed=true"])

    def test_missing_file_is_reported_clearly(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "nope.yaml")

    def test_invalid_yaml_is_reported_clearly(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("key: [unclosed\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="Invalid YAML"):
            load_config(bad)

    def test_non_mapping_top_level_is_rejected(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="must be a mapping"):
            load_config(bad)


class TestSemanticValidation:
    def test_unknown_dataset_is_rejected(self, empty_base: Path) -> None:
        with pytest.raises(ConfigError, match="not supported"):
            load_config(base_path=empty_base, overrides=["data.dataset=cifar"])

    def test_dataset_name_is_normalised(self, empty_base: Path) -> None:
        cfg = load_config(base_path=empty_base, overrides=["data.dataset=' MNIST '"])
        assert cfg.data.dataset == "mnist"

    def test_normal_digit_cannot_be_an_anomaly_digit(self, empty_base: Path) -> None:
        with pytest.raises(ConfigError, match="disjoint"):
            load_config(base_path=empty_base, overrides=["data.mnist.normal_digit=0"])

    def test_contamination_must_be_below_one(self, empty_base: Path) -> None:
        with pytest.raises(ConfigError, match="must lie in"):
            load_config(base_path=empty_base, overrides=["data.nsl_kdd.target_contamination=1.0"])

    def test_invalid_log_level_is_rejected(self, empty_base: Path) -> None:
        with pytest.raises(ConfigError, match="not a valid level"):
            load_config(base_path=empty_base, overrides=["logging.level=CHATTY"])

    def test_non_positive_batch_size_is_rejected(self, empty_base: Path) -> None:
        with pytest.raises(ConfigError, match="must be positive"):
            load_config(base_path=empty_base, overrides=["data.batch_size=0"])


class TestSerialisation:
    def test_round_trips_through_yaml(self, empty_base: Path, tmp_path: Path) -> None:
        cfg = load_config(base_path=empty_base, overrides=["seed=123"])
        dumped = cfg.dump_yaml(tmp_path / "effective.yaml")
        reloaded = load_config(dumped, base_path=empty_base)
        assert reloaded == cfg

    def test_to_dict_is_yaml_native(self, empty_base: Path) -> None:
        as_dict = load_config(base_path=empty_base).to_dict()
        assert isinstance(as_dict["paths"]["data_dir"], str)
        assert isinstance(as_dict["data"]["mnist"]["anomaly_digits"], list)

    def test_config_is_immutable(self, empty_base: Path) -> None:
        cfg: Config = load_config(base_path=empty_base)
        with pytest.raises(FrozenInstanceError):
            cfg.seed = 1  # type: ignore[misc]
