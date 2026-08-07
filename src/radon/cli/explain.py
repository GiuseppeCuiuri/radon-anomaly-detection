"""``python -m radon.cli.explain`` - explain a trained model's decisions.

For NSL-KDD this produces feature attributions; for MNIST it produces
reconstruction and error-map images.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

from radon.cli.common import add_common_arguments, bootstrap, checkpoint_path
from radon.config import Config
from radon.data import AnomalyDataset, build_dataset, build_test_loader
from radon.exceptions import RadonError
from radon.models import RadonAutoencoder, load_checkpoint

# Explicit name, not __name__: these modules run as `python -m radon.cli.X`, where
# __name__ is "__main__". A logger by that name does not sit under the configured
# "radon" logger, so every record from it would be silently dropped.
logger = logging.getLogger("radon.cli.explain")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for this command."""
    parser = argparse.ArgumentParser(
        prog="radon-explain",
        description="Explain the decisions of a trained RADON model.",
        epilog=(
            "Examples:\n"
            "  python -m radon.cli.explain --config configs/nsl_kdd.yaml\n"
            "  python -m radon.cli.explain --config configs/mnist.yaml --samples 5"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint to explain. Defaults to repetition 0 of this config.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Samples to explain per class.",
    )
    return parser


def explain_tabular(
    cfg: Config,
    dataset: AnomalyDataset,
    model: RadonAutoencoder,
    device: torch.device,
    n_samples: int,
    plots: bool,
) -> None:
    """Attribute NSL-KDD anomalies to features and groups."""
    from radon.explain import build_inlier_profile, explain_samples

    if dataset.feature_names is None:
        raise RadonError(
            f"Dataset '{dataset.name}' has no feature names, so tabular attribution "
            "cannot label its columns."
        )

    profile = build_inlier_profile(dataset.x_train, dataset.y_train, dataset.feature_names)
    loader = build_test_loader(dataset, batch_size=cfg.data.batch_size, device=device)
    explanations = explain_samples(
        model, loader, profile, device, n_anomalies=n_samples, n_normals=n_samples
    )

    for explanation in explanations:
        logger.info("\n%s", explanation.describe())

    if not plots:
        return

    from radon.viz import save_figure
    from radon.viz.explain import plot_categorical_attribution, plot_numeric_attribution

    for index, explanation in enumerate(explanations):
        stem = explanation.tag.replace(" ", "_")
        if explanation.categorical:
            save_figure(
                plot_categorical_attribution(explanation),
                cfg.paths.figure_dir / f"explain_{index}_{stem}_categorical.png",
            )
        if explanation.numeric_causal:
            save_figure(
                plot_numeric_attribution(explanation),
                cfg.paths.figure_dir / f"explain_{index}_{stem}_numeric.png",
            )
    logger.info("Attribution figures written to '%s'.", cfg.paths.figure_dir)


def explain_images(
    cfg: Config,
    dataset: AnomalyDataset,
    model: RadonAutoencoder,
    device: torch.device,
    n_samples: int,
    plots: bool,
) -> None:
    """Show MNIST reconstructions and their error maps."""
    from radon.viz.reconstruction import collect_reconstructions, plot_reconstructions

    loader = build_test_loader(dataset, batch_size=cfg.data.batch_size, device=device)
    inliers, outliers = collect_reconstructions(
        model, loader, device, cfg.data.mnist.image_shape, n_per_class=n_samples
    )

    for name, samples in (("inliers", inliers), ("outliers", outliers)):
        if samples:
            mean_score = sum(s.score for s in samples) / len(samples)
            logger.info("%s: %d sample(s), mean score %.4f", name, len(samples), mean_score)

    if not plots:
        return

    from radon.viz import save_figure

    for name, samples in (("inliers", inliers), ("outliers", outliers)):
        if samples:
            save_figure(
                plot_reconstructions(samples, title=name.capitalize()),
                cfg.paths.figure_dir / f"reconstructions_{name}.png",
            )
    logger.info("Reconstruction figures written to '%s'.", cfg.paths.figure_dir)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)

    try:
        cfg, device = bootstrap(args)
        dataset = build_dataset(cfg)

        path = args.checkpoint or checkpoint_path(cfg, 0)
        model = load_checkpoint(path, device=device).model

        handler = explain_tabular if dataset.feature_names else explain_images
        handler(cfg, dataset, model, device, args.samples, not args.no_plots)

    except RadonError as exc:
        logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logger.warning("Interrupted.")
        return 130

    return 0


if __name__ == "__main__":
    sys.exit(main())
