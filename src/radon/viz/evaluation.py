"""Evaluation plots: score separation, ROC and precision-recall curves."""

from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from radon.evaluation.metrics import EvaluationResult
from radon.evaluation.scoring import ScoredSplit

__all__ = ["plot_precision_recall", "plot_roc", "plot_score_distribution"]

_INLIER_COLOUR = "#4c72b0"
_OUTLIER_COLOUR = "#c44e52"


def _figure(ax: Axes | None, size: tuple[float, float]) -> tuple[Figure, Axes]:
    """Return the axes to draw on and the figure that owns it."""
    import matplotlib.pyplot as plt

    if ax is None:
        return plt.subplots(figsize=size)

    parent = ax.get_figure()
    if not isinstance(parent, Figure):
        raise ValueError("The supplied axes must belong to a top-level figure.")
    return parent, ax


def plot_score_distribution(
    scored: ScoredSplit,
    title: str = "Anomaly score by class",
    inlier_label: str = "Inliers",
    outlier_label: str = "Outliers",
    ax: Axes | None = None,
) -> Figure:
    """Box-plot the anomaly score of inliers against outliers.

    The clearest single picture of whether the model separates the classes at
    all, independently of where any threshold is placed.

    Args:
        scored: The scored split.
        title: Plot title.
        inlier_label: Label for the inlier box.
        outlier_label: Label for the outlier box.
        ax: Axes to draw on; a new figure is created when omitted.

    Returns:
        The figure.
    """
    fig, ax = _figure(ax, (7, 6))

    inliers = scored.scores[scored.labels == 0]
    outliers = scored.scores[scored.labels == 1]

    boxes = ax.boxplot(
        [inliers, outliers],
        positions=[0, 1],
        patch_artist=True,
        widths=0.6,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    for patch, colour in zip(boxes["boxes"], (_INLIER_COLOUR, _OUTLIER_COLOUR), strict=True):
        patch.set_facecolor(colour)
        patch.set_alpha(0.7)

    if scored.threshold is not None:
        ax.axhline(
            scored.threshold,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=f"Decision threshold = {scored.threshold:.4f}",
        )
        ax.legend(loc="upper left", fontsize=9)

    ax.set_xticks([0, 1])
    ax.set_xticklabels([inlier_label, outlier_label])
    ax.set_ylabel("Anomaly score  (1 - cosine)")
    ax.set_title(title)
    ax.set_xlabel(f"inlier mean {inliers.mean():.3f}   |   outlier mean {outliers.mean():.3f}")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return fig


def plot_roc(result: EvaluationResult, title: str = "ROC", ax: Axes | None = None) -> Figure:
    """Plot the ROC curve with its Youden-optimal operating point.

    Args:
        result: Metrics for one split.
        title: Plot title.
        ax: Axes to draw on; a new figure is created when omitted.

    Returns:
        The figure.
    """
    fig, ax = _figure(ax, (7, 6))
    roc = result.roc

    ax.plot(roc.fpr, roc.tpr, color="#dd8452", linewidth=2, label=f"AUC = {roc.auc:.5f}")
    ax.plot([0, 1], [0, 1], color="#4c72b0", linewidth=1.5, linestyle="--", label="Chance")
    ax.scatter(
        roc.fpr[roc.best_index],
        roc.tpr[roc.best_index],
        marker="o",
        color="black",
        s=80,
        zorder=5,
        label=f"Youden J at score {roc.best_threshold:.4f}",
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def plot_precision_recall(
    result: EvaluationResult, title: str = "Precision-Recall", ax: Axes | None = None
) -> Figure:
    """Plot the precision-recall curve against its chance baseline.

    The baseline is drawn explicitly because AUPRC has no fixed chance level:
    without it the number cannot be judged.

    Args:
        result: Metrics for one split.
        title: Plot title.
        ax: Axes to draw on; a new figure is created when omitted.

    Returns:
        The figure.
    """
    fig, ax = _figure(ax, (7, 6))
    pr = result.pr

    ax.plot(
        pr.recall,
        pr.precision,
        color="#4c72b0",
        linewidth=2,
        label=f"AP = {pr.average_precision:.5f}",
    )
    ax.axhline(
        pr.baseline,
        linestyle="--",
        color="gray",
        linewidth=1.5,
        label=f"Baseline (positive rate) = {pr.baseline:.4f}",
    )
    ax.scatter(
        pr.recall[pr.best_index],
        pr.precision[pr.best_index],
        marker="o",
        color="#c44e52",
        s=80,
        zorder=5,
        label=f"Best F1 = {pr.best_f1:.4f} at score {pr.best_threshold:.4f}",
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig
