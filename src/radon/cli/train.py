"""``python -m radon.cli.train`` - train RADON and evaluate it.

Replaces the notebook that used to drive every experiment, where the dataset
choice, the hyper-parameters, the repetition loop and three absolute Windows
paths all lived in top-level cells.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

import torch

from radon.cli.common import add_common_arguments, bootstrap, checkpoint_path
from radon.config import Config
from radon.data import AnomalyDataset, build_dataset, build_test_loader
from radon.evaluation import EvaluationResult, aggregate_metrics, evaluate_split, score_split
from radon.exceptions import RadonError
from radon.explain import build_inlier_profile, profile_to_dict
from radon.models import build_model, save_checkpoint
from radon.training import RadonTrainer, TrainingHistory

# Explicit name, not __name__: these modules run as `python -m radon.cli.X`, where
# __name__ is "__main__". A logger by that name does not sit under the configured
# "radon" logger, so every record from it would be silently dropped.
logger = logging.getLogger("radon.cli.train")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for this command."""
    parser = argparse.ArgumentParser(
        prog="radon-train",
        description="Train RADON on MNIST or NSL-KDD and report AUROC and AUPRC.",
        epilog=(
            "Examples:\n"
            "  python -m radon.cli.train --config configs/mnist.yaml\n"
            "  python -m radon.cli.train --config configs/nsl_kdd.yaml --repetitions 5\n"
            "  python -m radon.cli.train --config configs/mnist.yaml "
            "--set training.epochs=50 --set seed=7"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    return add_common_arguments(parser)


def run_repetition(
    cfg: Config,
    dataset: AnomalyDataset,
    device: torch.device,
    repetition: int,
    total: int,
) -> tuple[TrainingHistory, EvaluationResult]:
    """Train and evaluate one model.

    Args:
        cfg: The run configuration.
        dataset: The prepared dataset.
        device: Device to train on.
        repetition: Zero-based repetition index.
        total: Total number of repetitions, for the log line.

    Returns:
        The training history and the evaluation of the resulting model.
    """
    logger.info("=" * 70)
    logger.info("Repetition %d of %d", repetition + 1, total)
    logger.info("=" * 70)

    model = build_model(cfg.model, input_dim=dataset.n_features).to(device)
    trainer = RadonTrainer(model=model, cfg=cfg.training, device=device, compat=cfg.compat)
    history = trainer.train(dataset.x_train, feature_names=dataset.feature_names)

    _, threshold_outlier = history.final_thresholds()

    loader = build_test_loader(dataset, batch_size=cfg.data.batch_size, device=device)
    scored = score_split(model, loader, device, threshold_cosine=threshold_outlier)
    result = evaluate_split(scored, legacy_auprc=cfg.compat.legacy_auprc_trapezoid)

    save_checkpoint(
        model,
        checkpoint_path(cfg, repetition),
        extra={
            "threshold_inlier": history.final_thresholds()[0],
            "threshold_outlier": threshold_outlier,
            "best_epoch": history.best_epoch or -1,
            "dataset": dataset.name,
            **inlier_profile_extra(dataset),
        },
    )
    history.write_csv(cfg.paths.report_dir, f"{dataset.name}_rep{repetition}")

    return history, result


def inlier_profile_extra(dataset: AnomalyDataset) -> dict[str, Any]:
    """Checkpoint payload that lets the inference service explain records.

    Feature attribution needs the inlier profile of the training data. Storing
    it in the checkpoint keeps the service independent of the dataset files.

    Args:
        dataset: The dataset the model was trained on.

    Returns:
        ``{"inlier_profile": ...}`` for a tabular dataset, an empty dict for an
        image dataset, which has no per-feature attribution.
    """
    if dataset.feature_names is None:
        return {}
    profile = build_inlier_profile(dataset.x_train, dataset.y_train, dataset.feature_names)
    return {"inlier_profile": profile_to_dict(profile)}


def write_figures(
    cfg: Config,
    dataset: AnomalyDataset,
    scored_result: EvaluationResult,
    repetition: int,
) -> None:
    """Render and save the standard figure set for one repetition."""
    from radon.viz import save_figure
    from radon.viz.evaluation import plot_precision_recall, plot_roc

    prefix = f"{dataset.name}_rep{repetition}"
    save_figure(
        plot_roc(scored_result, title=f"ROC - {dataset.name} (rep {repetition + 1})"),
        cfg.paths.figure_dir / f"{prefix}_roc.png",
    )
    save_figure(
        plot_precision_recall(
            scored_result, title=f"Precision-Recall - {dataset.name} (rep {repetition + 1})"
        ),
        cfg.paths.figure_dir / f"{prefix}_pr.png",
    )
    logger.info("Figures written to '%s'.", cfg.paths.figure_dir)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 on success, 1 on an expected RADON failure.
    """
    args = build_parser().parse_args(argv)

    try:
        cfg, device = bootstrap(args)
        dataset = build_dataset(cfg)

        results: list[EvaluationResult] = []
        for repetition in range(args.repetitions):
            _history, result = run_repetition(cfg, dataset, device, repetition, args.repetitions)
            results.append(result)

            if not args.no_plots:
                write_figures(cfg, dataset, result, repetition)

        if len(results) > 1:
            aggregate_metrics(results)

        logger.info("Done. Artefacts in '%s'.", cfg.paths.output_dir)

    except RadonError as exc:
        logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logger.warning("Interrupted.")
        return 130

    return 0


if __name__ == "__main__":
    sys.exit(main())
