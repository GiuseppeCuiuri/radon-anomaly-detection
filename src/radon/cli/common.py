"""Shared command-line plumbing."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from radon.config import Config, load_config
from radon.device import resolve_device
from radon.logging_utils import setup_logging
from radon.seeding import set_seed

__all__ = ["add_common_arguments", "bootstrap", "checkpoint_path"]

# Explicit name, not __name__: these modules run as `python -m radon.cli.X`, where
# __name__ is "__main__". A logger by that name does not sit under the configured
# "radon" logger, so every record from it would be silently dropped.
logger = logging.getLogger("radon.cli.common")


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach the arguments every RADON command accepts.

    Args:
        parser: The parser to extend.

    Returns:
        The same parser, for chaining.
    """
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Dataset config layered on top of configs/default.yaml.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override any config value, e.g. --set training.epochs=50. "
            "Repeatable. Values are parsed as YAML."
        ),
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Independent runs, for mean and standard deviation of the metrics.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip figure generation. Metrics and CSVs are still written.",
    )
    return parser


def bootstrap(args: argparse.Namespace) -> tuple[Config, torch.device]:
    """Load the config, set up logging, seed the RNGs and pick a device.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The configuration and the resolved device.
    """
    cfg = load_config(args.config, overrides=args.overrides)

    setup_logging(cfg.logging, cfg.paths)
    logger.info("RADON starting.")
    if args.config:
        logger.info("Config: %s", args.config)
    if args.overrides:
        logger.info("Overrides: %s", ", ".join(args.overrides))

    set_seed(cfg.seed, deterministic=cfg.deterministic)
    device = resolve_device(cfg.device)

    cfg.dump_yaml(cfg.paths.report_dir / "effective_config.yaml")
    return cfg, device


def checkpoint_path(cfg: Config, repetition: int) -> Path:
    """Where the weights for one repetition live.

    Args:
        cfg: The run configuration.
        repetition: Zero-based repetition index.

    Returns:
        The checkpoint path.
    """
    return cfg.paths.checkpoint_dir / f"radon_{cfg.data.dataset}_rep{repetition}.pth"
