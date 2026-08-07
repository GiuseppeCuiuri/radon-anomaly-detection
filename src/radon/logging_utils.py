"""Logging setup for RADON.

Every module obtains its logger with ``logging.getLogger(__name__)``, which
yields names like ``radon.data.mnist``. This module configures the ``radon``
logger once, at process start, from a :class:`~radon.config.LoggingConfig`.

Library code never calls :func:`print`: console output is a presentation
concern owned by the entry point, and structured results are returned as data.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from radon.config import LoggingConfig, PathsConfig

__all__ = ["setup_logging"]

#: Third-party loggers that emit DEBUG noise we never want to see.
_NOISY_LIBRARIES = ("matplotlib", "matplotlib.font_manager", "PIL", "fsspec", "urllib3")


def setup_logging(
    cfg: LoggingConfig | None = None,
    paths: PathsConfig | None = None,
    *,
    force: bool = True,
) -> logging.Logger:
    """Configure the ``radon`` logger and return it.

    Console output goes to stdout so that it interleaves correctly with normal
    program output when a run is piped to a file. When ``cfg.file`` is set, a
    second handler mirrors everything to ``paths.log_dir / cfg.file``.

    Args:
        cfg: Logging settings. Defaults to :class:`LoggingConfig` defaults.
        paths: Provides ``log_dir``; required only when ``cfg.file`` is set.
        force: Remove handlers already attached to the ``radon`` logger. Keeps
            repeated calls idempotent, which matters in notebooks where a cell
            may be re-run.

    Returns:
        The configured ``radon`` logger.

    Raises:
        ValueError: If ``cfg.file`` is set but ``paths`` is ``None``.
    """
    cfg = cfg or LoggingConfig()
    logger = logging.getLogger("radon")
    logger.setLevel(cfg.level.upper())

    # The radon logger owns its handlers; it must not also bubble up to the
    # root logger, or every record would be emitted twice under pytest.
    logger.propagate = False

    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(fmt=cfg.format, datefmt=cfg.date_format)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if cfg.file:
        if paths is None:
            raise ValueError(
                "setup_logging: logging.file is set but no PathsConfig was provided, "
                "so the log directory is unknown."
            )
        log_path = _prepare_log_file(paths.log_dir, cfg.file, logger)
        if log_path is not None:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    for name in _NOISY_LIBRARIES:
        logging.getLogger(name).setLevel(logging.WARNING)

    return logger


def _prepare_log_file(log_dir: Path, filename: str, logger: logging.Logger) -> Path | None:
    """Create the log directory and return the log file path.

    A failure here must not abort a training run, so the error is reported on
    the console handler that is already attached and file logging is skipped.

    Args:
        log_dir: Directory that should contain the log file.
        filename: Log file name.
        logger: Logger used to report a failure.

    Returns:
        The log file path, or ``None`` if the directory could not be created.
    """
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning(
            "Cannot create the log directory '%s' (%s); logging to console only.", log_dir, exc
        )
        return None
    return log_dir / filename
