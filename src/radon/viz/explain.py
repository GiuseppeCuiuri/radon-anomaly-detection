"""Attribution plots."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from radon.explain.nsl_kdd import SampleExplanation

__all__ = ["plot_categorical_attribution", "plot_numeric_attribution"]

_CULPRIT = "#c44e52"
_INNOCENT = "#55a868"


def plot_categorical_attribution(explanation: SampleExplanation) -> Figure:
    """Bar-plot the score change from normalising each categorical group.

    Bars below zero mark groups that were driving the anomaly: resetting them
    to the inlier mode made the sample look more normal.

    Args:
        explanation: The sample's attribution.

    Returns:
        The figure.

    Raises:
        ValueError: If the sample has no categorical attributions.
    """
    import matplotlib.pyplot as plt

    if not explanation.categorical:
        raise ValueError(
            "plot_categorical_attribution: no categorical attribution survived filtering."
        )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [a.group for a in explanation.categorical]
    deltas = [a.delta_score for a in explanation.categorical]
    colours = [_CULPRIT if d < 0 else _INNOCENT for d in deltas]

    ax.bar(np.arange(len(labels)), deltas, color=colours, alpha=0.85)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Change in anomaly score")
    ax.set_title(
        f"{explanation.tag} (score {explanation.score:.4f}) - categorical attribution\n"
        "negative = this group was driving the anomaly",
        fontsize=11,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return fig


def plot_numeric_attribution(explanation: SampleExplanation) -> Figure:
    """Plot observed value, reconstruction and inlier range per feature.

    The shaded band is the normal interquantile range; the line on the right
    axis is the score change from clipping the feature into that band.

    Args:
        explanation: The sample's attribution.

    Returns:
        The figure.

    Raises:
        ValueError: If the sample has no numeric attributions.
    """
    import matplotlib.pyplot as plt

    attributions = explanation.numeric_causal
    if not attributions:
        raise ValueError("plot_numeric_attribution: no numeric attribution survived filtering.")

    fig, ax = plt.subplots(figsize=(12, 6))
    positions = np.arange(len(attributions))
    width = 0.25

    values = np.array([a.value for a in attributions])
    reconstructions = np.array([a.reconstruction for a in attributions])
    lows = np.array([a.quantile_low for a in attributions])
    highs = np.array([a.quantile_high for a in attributions])
    deltas = np.array([a.delta_score for a in attributions])

    for index in positions:
        ax.fill_between(
            [index - 0.4, index + 0.4],
            lows[index],
            highs[index],
            alpha=0.15,
            color=_INNOCENT,
            label="Inlier range" if index == 0 else None,
        )

    ax.bar(positions - width, values, width=width, label="Observed", color="#4c72b0")
    ax.bar(positions, reconstructions, width=width, label="Reconstructed", color="#dd8452")
    ax.bar(
        positions + width,
        np.clip(values, lows, highs),
        width=width,
        label="Baseline (clipped)",
        color=_INNOCENT,
    )

    ax.set_xticks(positions)
    ax.set_xticklabels([a.name for a in attributions], rotation=45, ha="right")
    ax.set_ylabel("Feature value")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    twin = ax.twinx()
    twin.plot(positions, deltas, marker="o", linestyle="--", linewidth=2, color="#8172b3")
    twin.axhline(0.0, color="gray", linestyle=":", linewidth=1)
    twin.set_ylabel("Change in anomaly score", color="#8172b3")

    ax.set_title(
        f"{explanation.tag} (score {explanation.score:.4f}) - numeric attribution",
        fontsize=12,
    )
    fig.tight_layout()
    return fig
