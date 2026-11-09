"""Typed, layered configuration for RADON.

Configuration is assembled in three layers, each overriding the previous one:

1. ``configs/default.yaml`` - the base values committed to the repository.
2. An optional dataset-specific file, e.g. ``configs/mnist.yaml``.
3. Command-line overrides in dotted ``key.path=value`` form.

The merged mapping is then coerced into the :class:`Config` dataclass tree.
Unknown keys are a hard error rather than a silent no-op, because a typo in a
hyper-parameter name is one of the easiest ways to invalidate an experiment.

This module must not import :mod:`torch`, so that configuration handling stays
testable in an environment without a deep-learning stack.

Example:
    >>> cfg = load_config("configs/mnist.yaml", overrides=["data.batch_size=512"])
    >>> cfg.data.batch_size
    512
"""

from __future__ import annotations

import dataclasses
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, Union, cast, get_args, get_origin, get_type_hints

import yaml

from radon.exceptions import ConfigError

__all__ = [
    "PROJECT_ROOT",
    "CompatConfig",
    "Config",
    "ConsistencyConfig",
    "CurriculumConfig",
    "DataConfig",
    "EarlyStoppingConfig",
    "LoggingConfig",
    "MnistConfig",
    "ModelConfig",
    "NslKddConfig",
    "PathsConfig",
    "RampConfig",
    "SmoothingConfig",
    "TrainingConfig",
    "load_config",
]

T = TypeVar("T")


def _project_root(config_file: Path = Path(__file__)) -> Path:
    """Directory that relative paths and ``configs/default.yaml`` resolve against.

    Args:
        config_file: Location of this module; a parameter only so tests can
            place it elsewhere.

    Returns:
        The repository root when running from a source checkout, recognised by
        ``pyproject.toml`` next to ``src/``, so runs behave the same from any
        working directory. Otherwise, as for a package installed with pip, the
        current working directory: the directory above the package would be
        inside the virtual environment.
    """
    checkout = config_file.resolve().parents[2]
    return checkout if (checkout / "pyproject.toml").is_file() else Path.cwd()


#: Root for relative paths: the repository in a checkout, else the working directory.
PROJECT_ROOT: Path = _project_root()

#: Base configuration layer, always loaded first when it exists.
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "default.yaml"

_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
_DATASETS = frozenset({"mnist", "nsl_kdd"})


# --------------------------------------------------------------------------- #
# Configuration schema
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LoggingConfig:
    """Logging behaviour for a run.

    Attributes:
        level: Root level for the ``radon`` logger. Case-insensitive.
        file: Log file name, created under ``PathsConfig.log_dir``. ``None``
            disables file logging and keeps console output only.
        format: :mod:`logging` format string.
        date_format: :mod:`logging` date format string.
    """

    level: str = "INFO"
    file: str | None = "radon.log"
    format: str = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
    date_format: str = "%H:%M:%S"

    def __post_init__(self) -> None:
        if self.level.upper() not in _LOG_LEVELS:
            raise ConfigError(
                f"logging.level: '{self.level}' is not a valid level. "
                f"Expected one of {sorted(_LOG_LEVELS)}."
            )


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem locations for a run.

    Relative paths are resolved against :data:`PROJECT_ROOT`: in a source
    checkout a run behaves identically regardless of the working directory it
    was launched from; an installed package uses the working directory.

    Attributes:
        data_dir: Root directory holding ``MNIST/`` and ``NSL_KDD/``.
        output_dir: Root directory for every artefact a run produces.
    """

    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")

    def __post_init__(self) -> None:
        # Frozen dataclasses forbid plain assignment; ``object.__setattr__`` is
        # the documented way to normalise fields inside ``__post_init__``.
        for name in ("data_dir", "output_dir"):
            value: Path = getattr(self, name)
            if not value.is_absolute():
                object.__setattr__(self, name, (PROJECT_ROOT / value).resolve())

    @property
    def checkpoint_dir(self) -> Path:
        """Directory for trained model weights."""
        return self.output_dir / "checkpoints"

    @property
    def log_dir(self) -> Path:
        """Directory for log files."""
        return self.output_dir / "logs"

    @property
    def figure_dir(self) -> Path:
        """Directory for generated figures."""
        return self.output_dir / "figures"

    @property
    def report_dir(self) -> Path:
        """Directory for CSV dumps and metric reports."""
        return self.output_dir / "reports"


@dataclass(frozen=True)
class MnistConfig:
    """One-vs-rest anomaly detection protocol on MNIST.

    One digit is designated normal; the training set is contaminated with a
    small number of samples drawn from ``anomaly_digits``.

    Note:
        Training anomalies are drawn only from ``anomaly_digits``, while test
        anomalies are drawn from *every* digit other than ``normal_digit``.
        This makes the test set a generalisation check rather than an in-
        distribution one. The behaviour is inherited from the original code and
        is preserved deliberately.

    Attributes:
        normal_digit: The digit treated as the inlier class.
        n_normal: Number of normal samples in the training set.
        n_anomaly: Number of contaminating anomalies in the training set.
        anomaly_digits: Digits eligible as training contamination.
        n_test_normal: Test inliers to keep; ``None`` keeps all of them.
        n_test_anomaly: Test outliers to keep; ``None`` keeps all of them.
        image_shape: Original image geometry, used only to reshape flattened
            vectors back for visualisation.
        download: Whether torchvision may download MNIST if it is absent.
    """

    normal_digit: int = 4
    n_normal: int = 4859
    n_anomaly: int = 265
    anomaly_digits: tuple[int, ...] = (0, 2, 3)
    n_test_normal: int | None = None
    n_test_anomaly: int | None = 265
    image_shape: tuple[int, int] = (28, 28)
    download: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.normal_digit <= 9:
            raise ConfigError(
                f"data.mnist.normal_digit: expected a digit in 0..9, got {self.normal_digit}."
            )
        if self.normal_digit in self.anomaly_digits:
            raise ConfigError(
                f"data.mnist: normal_digit ({self.normal_digit}) also appears in "
                f"anomaly_digits ({list(self.anomaly_digits)}); the two must be disjoint."
            )
        if self.n_normal <= 0:
            raise ConfigError(f"data.mnist.n_normal: must be positive, got {self.n_normal}.")
        if self.n_anomaly < 0:
            raise ConfigError(f"data.mnist.n_anomaly: must be non-negative, got {self.n_anomaly}.")
        if self.n_anomaly > 0 and not self.anomaly_digits:
            raise ConfigError("data.mnist: n_anomaly > 0 requires a non-empty anomaly_digits list.")
        for name in ("n_test_normal", "n_test_anomaly"):
            value: int | None = getattr(self, name)
            if value is not None and value <= 0:
                raise ConfigError(
                    f"data.mnist.{name}: must be positive or null (meaning 'keep all'), "
                    f"got {value}."
                )


@dataclass(frozen=True)
class NslKddConfig:
    """NSL-KDD protocol, built from pre-processed CSV splits on disk.

    Attributes:
        target_contamination: Desired fraction of outliers in the training set.
            Outliers are randomly subsampled to hit this ratio; inliers are all
            kept.
        clamp_test_to_unit_range: Clamp test features into ``[0, 1]``. The
            autoencoder's decoder ends in a sigmoid, so out-of-range inputs can
            never be reconstructed and inflate the anomaly score artificially.
        index_column_prefix: Column-name prefix identifying a stray pandas index
            column to drop on load.
    """

    target_contamination: float = 0.05
    clamp_test_to_unit_range: bool = True
    index_column_prefix: str = "Unnamed"

    def __post_init__(self) -> None:
        if not 0.0 <= self.target_contamination < 1.0:
            raise ConfigError(
                "data.nsl_kdd.target_contamination: must lie in [0, 1), got "
                f"{self.target_contamination}."
            )


@dataclass(frozen=True)
class DataConfig:
    """Dataset selection and the per-dataset protocol blocks.

    Attributes:
        dataset: Which builder to use; one of ``mnist`` or ``nsl_kdd``.
        batch_size: Batch size for evaluation loaders.
        mnist: MNIST protocol, used when ``dataset == "mnist"``.
        nsl_kdd: NSL-KDD protocol, used when ``dataset == "nsl_kdd"``.
    """

    dataset: str = "mnist"
    batch_size: int = 256
    mnist: MnistConfig = field(default_factory=MnistConfig)
    nsl_kdd: NslKddConfig = field(default_factory=NslKddConfig)

    def __post_init__(self) -> None:
        normalised = self.dataset.strip().lower()
        if normalised not in _DATASETS:
            raise ConfigError(
                f"data.dataset: '{self.dataset}' is not supported. "
                f"Expected one of {sorted(_DATASETS)}."
            )
        object.__setattr__(self, "dataset", normalised)
        if self.batch_size <= 0:
            raise ConfigError(f"data.batch_size: must be positive, got {self.batch_size}.")


@dataclass(frozen=True)
class ModelConfig:
    """Architecture of the RADON autoencoder.

    The network is a symmetric fully-connected autoencoder
    ``input_dim -> hidden_dim -> latent_dim -> hidden_dim -> input_dim``, with
    ReLU after every layer except the output, which uses a sigmoid. Only the
    two inner widths are configurable; ``input_dim`` follows from the dataset.

    Note:
        The original code picked these with an inline rule - ``(256, 128)`` when
        ``input_dim > 200``, otherwise ``(64, 32)`` - buried in a notebook cell.
        They are stated explicitly per dataset instead, so a run's capacity is
        visible in its config rather than implied by a threshold.

    Attributes:
        hidden_dim: Width of the first encoder and last decoder layer (``dim1``
            in the original code).
        latent_dim: Bottleneck width (``dim2`` in the original code).
    """

    hidden_dim: int = 256
    latent_dim: int = 128

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ConfigError(f"model.hidden_dim: must be positive, got {self.hidden_dim}.")
        if self.latent_dim <= 0:
            raise ConfigError(f"model.latent_dim: must be positive, got {self.latent_dim}.")
        if self.latent_dim >= self.hidden_dim:
            raise ConfigError(
                f"model: latent_dim ({self.latent_dim}) must be smaller than hidden_dim "
                f"({self.hidden_dim}); without a bottleneck the encoder has nothing to compress."
            )


@dataclass(frozen=True)
class CurriculumConfig:
    """Gradual admission of outliers into the negative-learning objective.

    At each projection the candidate outlier set ``S`` is trimmed to its
    ``fraction`` lowest-scoring members, where ``fraction`` ramps linearly from
    ``start`` to ``end`` across the negative-learning phase. Early on the model
    therefore only learns from the most confident outliers.

    Attributes:
        enabled: Whether to trim ``S`` at all.
        start: Fraction admitted at the first negative-learning epoch.
        end: Fraction admitted at the last epoch.
        min_s: Skip trimming while ``S`` holds fewer than this many samples;
            quantiles over a tiny set are too noisy to act on.
    """

    enabled: bool = True
    start: float = 0.10
    end: float = 1.00
    min_s: int = 200

    def __post_init__(self) -> None:
        for name in ("start", "end"):
            value: float = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ConfigError(f"training.curriculum.{name}: must lie in [0, 1], got {value}.")
        if self.min_s < 0:
            raise ConfigError(f"training.curriculum.min_s: must be non-negative, got {self.min_s}.")


@dataclass(frozen=True)
class RampConfig:
    """Linear warm-in of the negative-learning loss weight.

    Attributes:
        enabled: Whether to ramp. When disabled the weight applies at full
            strength from the first negative-learning epoch.
        epochs: Number of epochs over which the factor climbs from 0 to 1.
    """

    enabled: bool = True
    epochs: int = 50

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ConfigError(f"training.ramp.epochs: must be at least 1, got {self.epochs}.")


@dataclass(frozen=True)
class ConsistencyConfig:
    """Optional consistency penalty on perturbed outliers.

    Adds ``weight * MSE(f(s), f(s + noise))`` so that reconstructions of
    outliers do not swing wildly under small input perturbations.

    Attributes:
        weight: Loss weight. Zero disables the term entirely, which is what
            every configuration in the original notebook used.
        sigma: Standard deviation of the Gaussian noise, applied to numeric
            features only; binary features are left intact.
    """

    weight: float = 0.0
    sigma: float = 0.02

    def __post_init__(self) -> None:
        if self.weight < 0.0:
            raise ConfigError(
                f"training.consistency.weight: must be non-negative, got {self.weight}."
            )
        if self.sigma < 0.0:
            raise ConfigError(
                f"training.consistency.sigma: must be non-negative, got {self.sigma}."
            )


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Stop when the monitored metric stops improving across projections.

    Attributes:
        enabled: Whether to monitor at all. The best weights are restored at the
            end of training regardless of whether the patience ran out.
        patience: Consecutive non-improving projections tolerated.
        min_delta: Improvement below this is treated as noise.
        metric: Which inlier loss to monitor.

            - ``frozen_l0``: the inlier set captured when negative learning
              begins, held fixed thereafter. Comparable across projections.
            - ``current_l``: the inlier set of the current projection, which is
              rebuilt every time. This is what the original code used, and it is
              not comparable over time: an ``L`` that shrinks towards the
              easiest samples lowers the loss without the model improving.
            - ``full_x``: the whole training matrix. Comparable, but negative
              learning deliberately pushes outliers away from their own
              reconstruction, so this metric worsens by design.
    """

    enabled: bool = True
    patience: int = 3
    min_delta: float = 1e-6
    metric: str = "frozen_l0"

    def __post_init__(self) -> None:
        valid = {"frozen_l0", "current_l", "full_x"}
        if self.metric not in valid:
            raise ConfigError(
                f"training.early_stopping.metric: '{self.metric}' is not supported. "
                f"Expected one of {sorted(valid)}."
            )
        if self.patience < 1:
            raise ConfigError(
                f"training.early_stopping.patience: must be at least 1, got {self.patience}."
            )
        if self.min_delta < 0.0:
            raise ConfigError(
                f"training.early_stopping.min_delta: must be non-negative, got {self.min_delta}."
            )


@dataclass(frozen=True)
class SmoothingConfig:
    """Exponential-moving-average factors used by the projection step.

    Every value here was a magic number inside the original projection routine.
    A factor of ``0.9`` keeps 90% of the running estimate at each update.

    Attributes:
        threshold_s: Smoothing of the outlier threshold across projections.
        cardinality: Smoothing of ``|S|``, which feeds the loss weight.
        lambda_card: Smoothing of the cardinality ratio ``|L| / |S|``.
        lambda_eff: Smoothing of the resulting loss weight.
    """

    threshold_s: float = 0.90
    cardinality: float = 0.90
    lambda_card: float = 0.90
    lambda_eff: float = 0.90

    def __post_init__(self) -> None:
        for f in fields(self):
            value: float = getattr(self, f.name)
            if not 0.0 <= value < 1.0:
                raise ConfigError(f"training.smoothing.{f.name}: must lie in [0, 1), got {value}.")


@dataclass(frozen=True)
class TrainingConfig:
    """The RADON training schedule.

    Training runs in two phases. During *warm-up* the model simply learns to
    reconstruct, while periodic projections split the data into an inlier set
    ``L`` and an outlier set ``S``. From ``warmup_epochs`` onward *negative
    learning* takes over: inliers are still reconstructed, but outliers are
    pushed towards their feature-wise inverse.

    Attributes:
        epochs: Total number of epochs.
        batch_size: Samples per optimisation step.
        lr: Adam learning rate.
        warmup_epochs: Epochs before negative learning begins.
        bin_estimator: Histogram estimator for the projection step.
        projection_every_warmup: Projection interval during warm-up. A
            projection is always forced at the final warm-up epoch.
        projection_every_nl: Projection interval during negative learning.
        lambda_s: Base weight of the negative-learning loss.
        lambda_card_mode: How the ``|L| / |S|`` ratio scales that weight;
            ``linear``, ``sqrt`` or ``log``.
        curriculum: Gradual admission of outliers.
        ramp: Warm-in of the loss weight.
        consistency: Optional consistency penalty.
        early_stopping: Stopping rule.
        smoothing: EMA factors for the projection step.
    """

    epochs: int = 200
    batch_size: int = 256
    lr: float = 1e-3
    warmup_epochs: int = 50
    bin_estimator: str = "knuth"
    projection_every_warmup: int = 25
    projection_every_nl: int = 50
    lambda_s: float = 1.0
    lambda_card_mode: str = "sqrt"
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    ramp: RampConfig = field(default_factory=RampConfig)
    consistency: ConsistencyConfig = field(default_factory=ConsistencyConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ConfigError(f"training.epochs: must be at least 1, got {self.epochs}.")
        if self.batch_size < 1:
            raise ConfigError(f"training.batch_size: must be at least 1, got {self.batch_size}.")
        if self.lr <= 0.0:
            raise ConfigError(f"training.lr: must be positive, got {self.lr}.")
        if self.warmup_epochs < 0:
            raise ConfigError(
                f"training.warmup_epochs: must be non-negative, got {self.warmup_epochs}."
            )
        if self.warmup_epochs > self.epochs:
            raise ConfigError(
                f"training.warmup_epochs ({self.warmup_epochs}) exceeds epochs "
                f"({self.epochs}); negative learning would never start."
            )
        for name in ("projection_every_warmup", "projection_every_nl"):
            value: int = getattr(self, name)
            if value < 1:
                raise ConfigError(f"training.{name}: must be at least 1, got {value}.")
        if self.lambda_s < 0.0:
            raise ConfigError(f"training.lambda_s: must be non-negative, got {self.lambda_s}.")
        valid_modes = {"linear", "sqrt", "log"}
        if self.lambda_card_mode not in valid_modes:
            raise ConfigError(
                f"training.lambda_card_mode: '{self.lambda_card_mode}' is not supported. "
                f"Expected one of {sorted(valid_modes)}."
            )


@dataclass(frozen=True)
class CompatConfig:
    """Opt-in switches that restore pre-refactor behaviour.

    Each flag corresponds to a bug fixed during the refactor whose correction
    changes published numbers. Setting all of them to ``True`` reproduces the
    original results bit-for-bit, which doubles as a numerical-equivalence test
    for the refactor itself.

    Attributes:
        legacy_warmup_no_shuffle: Keep ``shuffle=False`` in the warm-up loader,
            freezing batch order and the L/S pairing across epochs.
        legacy_auprc_trapezoid: Compute AUPRC as ``auc(recall, precision)``
            instead of ``average_precision_score``. Trapezoidal interpolation of
            a precision-recall curve is optimistically biased.
        legacy_early_stop_metric: Score early stopping on the inlier loss over
            the *current* L, which is rebuilt at every projection and therefore
            not comparable across projections.
    """

    legacy_warmup_no_shuffle: bool = False
    legacy_auprc_trapezoid: bool = False
    legacy_early_stop_metric: bool = False

    @property
    def any_enabled(self) -> bool:
        """Whether at least one legacy behaviour is active."""
        return any(getattr(self, f.name) for f in fields(self))

    def enabled_flags(self) -> tuple[str, ...]:
        """Names of the currently enabled legacy flags."""
        return tuple(f.name for f in fields(self) if getattr(self, f.name))


@dataclass(frozen=True)
class Config:
    """Root configuration object for a RADON run.

    Attributes:
        seed: Master seed for Python, NumPy and PyTorch.
        device: ``auto``, ``cpu``, ``cuda`` or an explicit ``cuda:N``.
        deterministic: Request deterministic cuDNN kernels. Costs throughput but
            makes GPU runs reproducible.
        paths: Filesystem locations.
        logging: Logging behaviour.
        data: Dataset selection and protocol.
        model: Autoencoder architecture.
        training: Training schedule.
        compat: Legacy-behaviour switches.
    """

    seed: int = 42
    device: str = "auto"
    deterministic: bool = True
    paths: PathsConfig = field(default_factory=PathsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    compat: CompatConfig = field(default_factory=CompatConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, YAML-serialisable mapping of the whole config.

        Paths are rendered as strings and tuples as lists, so the result can be
        dumped next to a run's artefacts to document exactly what was executed.

        Returns:
            A nested dictionary mirroring the dataclass tree.
        """
        return cast("dict[str, Any]", _to_plain(dataclasses.asdict(self)))

    def dump_yaml(self, path: Path) -> Path:
        """Write the effective configuration to ``path``.

        Args:
            path: Destination file. Parent directories are created as needed.

        Returns:
            The path that was written.

        Raises:
            ConfigError: If the file cannot be written.
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(self.to_dict(), handle, sort_keys=False, allow_unicode=True)
        except OSError as exc:
            raise ConfigError(f"Cannot write the effective config to '{path}': {exc}") from exc
        return path


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_config(
    config_path: str | Path | None = None,
    overrides: Sequence[str] = (),
    base_path: str | Path | None = None,
) -> Config:
    """Build a :class:`Config` from the layered YAML files plus CLI overrides.

    Args:
        config_path: Dataset-specific YAML applied on top of the base layer.
            ``None`` uses the base layer alone.
        overrides: Dotted ``key.path=value`` strings, applied last. Values are
            parsed as YAML, so ``data.batch_size=512`` yields an int and
            ``compat.legacy_auprc_trapezoid=true`` yields a bool.
        base_path: Base layer. Defaults to :data:`DEFAULT_CONFIG_PATH`; a
            missing default file is tolerated and dataclass defaults are used.

    Returns:
        The fully merged and validated configuration.

    Raises:
        ConfigError: If a file is missing or malformed, an override is not in
            ``key=value`` form, a key is unknown, or a value has the wrong type.
    """
    resolved_base = Path(base_path) if base_path is not None else DEFAULT_CONFIG_PATH
    merged: dict[str, Any] = {}
    # An explicitly requested base must exist; the built-in default may not, in
    # which case the dataclass defaults stand in for it.
    if base_path is not None or resolved_base.exists():
        merged = _read_yaml(resolved_base)

    if config_path is not None:
        merged = _deep_merge(merged, _read_yaml(Path(config_path)))

    for override in overrides:
        merged = _deep_merge(merged, _override_to_mapping(override))

    return _build(Config, merged)


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping from ``path``.

    Args:
        path: File to read.

    Returns:
        The parsed mapping; an empty file yields an empty dict.

    Raises:
        ConfigError: If the file is missing, unreadable, invalid YAML, or its
            top level is not a mapping.
    """
    try:
        with path.open("r", encoding="utf-8") as handle:
            content = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration file not found: '{path}'.") from exc
    except OSError as exc:
        raise ConfigError(f"Cannot read the configuration file '{path}': {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in '{path}': {exc}") from exc

    if content is None:
        return {}
    if not isinstance(content, Mapping):
        raise ConfigError(
            f"'{path}': the top level must be a mapping, found {type(content).__name__}."
        )
    return dict(content)


def _override_to_mapping(override: str) -> dict[str, Any]:
    """Expand a dotted ``key.path=value`` override into a nested mapping.

    Args:
        override: For example ``data.mnist.normal_digit=7``.

    Returns:
        A nested dict such as ``{"data": {"mnist": {"normal_digit": 7}}}``.

    Raises:
        ConfigError: If the override is not in ``key=value`` form or the key is
            empty.
    """
    key, separator, raw_value = override.partition("=")
    if not separator:
        raise ConfigError(
            f"Malformed override '{override}': expected the form key.path=value, "
            "e.g. data.batch_size=512."
        )
    key = key.strip()
    if not key:
        raise ConfigError(f"Malformed override '{override}': the key is empty.")

    parts = key.split(".")
    if any(not part for part in parts):
        raise ConfigError(f"Malformed override '{override}': empty segment in the key '{key}'.")

    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Override '{override}': cannot parse the value: {exc}") from exc

    nested: dict[str, Any] = {}
    cursor = nested
    for part in parts[:-1]:
        child: dict[str, Any] = {}
        cursor[part] = child
        cursor = child
    cursor[parts[-1]] = value
    return nested


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating either.

    Nested mappings are merged key by key; every other type is replaced
    wholesale, so a list in an override replaces the base list rather than
    extending it.

    Args:
        base: The lower-priority mapping.
        override: The higher-priority mapping.

    Returns:
        A new merged dictionary.
    """
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


# --------------------------------------------------------------------------- #
# Dataclass coercion
# --------------------------------------------------------------------------- #


def _build(cls: type[T], data: Mapping[str, Any], prefix: str = "") -> T:
    """Instantiate the dataclass ``cls`` from a plain mapping.

    Args:
        cls: Target dataclass type.
        data: Mapping of field name to raw value.
        prefix: Dotted path prefix used to build readable error messages.

    Returns:
        An instance of ``cls``.

    Raises:
        ConfigError: On unknown keys, missing required fields, or type errors.
    """
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]

    unknown = sorted(set(data) - known)
    if unknown:
        location = prefix.rstrip(".") or "the root of the config"
        raise ConfigError(f"Unknown key(s) {unknown} in {location}. Valid keys: {sorted(known)}.")

    kwargs: dict[str, Any] = {}
    for f in fields(cls):  # type: ignore[arg-type]
        if f.name in data:
            kwargs[f.name] = _coerce(hints[f.name], data[f.name], f"{prefix}{f.name}")

    try:
        return cls(**kwargs)
    except ConfigError:
        raise
    except TypeError as exc:
        location = prefix.rstrip(".") or "the root of the config"
        raise ConfigError(f"Cannot build {location}: {exc}") from exc


def _coerce(annotation: Any, value: Any, path: str) -> Any:
    """Convert a raw YAML value to the type declared by a dataclass field.

    Args:
        annotation: The declared type.
        value: The raw value coming from YAML.
        path: Dotted path to the field, used in error messages.

    Returns:
        The converted value.

    Raises:
        ConfigError: If the value cannot be represented as ``annotation``.
    """
    origin = get_origin(annotation)

    # Optional[X] and X | Y
    if origin is Union or origin is getattr(types, "UnionType", None):
        options = get_args(annotation)
        if value is None:
            if type(None) in options:
                return None
            raise ConfigError(f"'{path}': null is not allowed here.")
        concrete = [opt for opt in options if opt is not type(None)]
        for option in concrete:
            try:
                return _coerce(option, value, path)
            except ConfigError:
                continue
        names = ", ".join(getattr(opt, "__name__", str(opt)) for opt in concrete)
        raise ConfigError(f"'{path}': {value!r} does not match any of ({names}).")

    # `is_dataclass` also accepts dataclass *instances*; only a class can be
    # instantiated from a mapping, hence the extra `isinstance(..., type)`.
    if is_dataclass(annotation) and isinstance(annotation, type):
        if not isinstance(value, Mapping):
            raise ConfigError(f"'{path}': expected a mapping, found {type(value).__name__}.")
        return _build(annotation, value, f"{path}.")

    if annotation is Path:
        if not isinstance(value, (str, Path)):
            raise ConfigError(f"'{path}': expected a path string, found {type(value).__name__}.")
        return Path(value)

    if origin is list:
        return [
            _coerce(get_args(annotation)[0], item, f"{path}[{i}]")
            for i, item in enumerate(_as_sequence(value, path))
        ]

    if origin is tuple:
        args = get_args(annotation)
        items = _as_sequence(value, path)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(args[0], item, f"{path}[{i}]") for i, item in enumerate(items))
        if len(args) != len(items):
            raise ConfigError(f"'{path}': expected {len(args)} element(s), found {len(items)}.")
        return tuple(
            _coerce(arg, item, f"{path}[{i}]")
            for i, (arg, item) in enumerate(zip(args, items, strict=True))
        )

    if annotation is bool:
        if not isinstance(value, bool):
            raise ConfigError(
                f"'{path}': expected a boolean (true/false), found {type(value).__name__}."
            )
        return value

    if annotation is int:
        # bool is a subclass of int; accepting it here would silently turn
        # `true` into 1 for a numeric field.
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"'{path}': expected an integer, found {type(value).__name__}.")
        return value

    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"'{path}': expected a number, found {type(value).__name__}.")
        return float(value)

    if annotation is str:
        if not isinstance(value, str):
            raise ConfigError(f"'{path}': expected a string, found {type(value).__name__}.")
        return value

    if annotation is Any:
        return value

    raise ConfigError(
        f"'{path}': unsupported type annotation {annotation!r}. "
        "Extend radon.config._coerce to handle it."
    )


def _as_sequence(value: Any, path: str) -> list[Any]:
    """Return ``value`` as a list, rejecting strings and mappings.

    Args:
        value: The candidate sequence.
        path: Dotted path used in the error message.

    Returns:
        The value as a list.

    Raises:
        ConfigError: If ``value`` is not a list or tuple.
    """
    if isinstance(value, (list, tuple)):
        return list(value)
    raise ConfigError(f"'{path}': expected a list, found {type(value).__name__}.")


def _to_plain(value: Any) -> Any:
    """Recursively convert Paths to strings and tuples to lists for YAML dumping.

    Args:
        value: Any nested structure of dicts, lists, tuples, Paths and scalars.

    Returns:
        An equivalent structure containing only YAML-native types.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    return value
