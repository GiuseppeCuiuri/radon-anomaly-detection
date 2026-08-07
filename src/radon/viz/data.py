"""Dataset composition plots."""

from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from radon.data.base import LabelStats

__all__ = ["plot_label_distribution"]

_INLIER_COLOUR = "#4c72b0"
_OUTLIER_COLOUR = "#c44e52"


def plot_label_distribution(
    stats: LabelStats,
    title: str,
    inlier_label: str = "Inliers",
    outlier_label: str = "Outliers",
    ax: Axes | None = None,
) -> Figure:
    """Draw the inlier/outlier composition of a split as a bar chart.

    Takes pre-computed :class:`~radon.data.base.LabelStats` rather than a raw
    label tensor, so the counts can be asserted in a test without ever touching
    matplotlib.

    Args:
        stats: Counts to plot, from
            :func:`~radon.data.base.label_statistics`.
        title: Plot title.
        inlier_label: X-axis label for the inlier bar.
        outlier_label: X-axis label for the outlier bar.
        ax: Axes to draw on. A new figure is created when omitted.

    Returns:
        The figure containing the plot.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        # `get_figure` may return a SubFigure, which has no `savefig`; callers
        # expect a real Figure back.
        parent = ax.get_figure()
        if not isinstance(parent, Figure):
            raise ValueError(
                "plot_label_distribution: the supplied axes belongs to a SubFigure "
                "(or to nothing at all); pass axes from a top-level figure."
            )
        fig = parent

    counts = [stats.n_inliers, stats.n_outliers]
    bars = ax.bar([0, 1], counts, color=[_INLIER_COLOUR, _OUTLIER_COLOUR], width=0.6, alpha=0.85)

    for bar, count in zip(bars, counts, strict=True):
        ax.annotate(
            f"{count:,}",
            xy=(bar.get_x() + bar.get_width() / 2, count),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels([inlier_label, outlier_label])
    ax.set_ylabel("Samples")
    ax.set_title(title)
    ax.set_xlabel(f"Contamination: {stats.contamination * 100:.2f}%")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return fig
