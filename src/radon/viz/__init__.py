"""Plotting helpers.

Every function here returns a :class:`~matplotlib.figure.Figure` and never calls
``plt.show()`` or ``plt.savefig()`` itself. Displaying and saving are decisions
for the caller: a notebook shows, a CLI run saves, and a test does neither. The
original code interleaved computation, printing and ``plt.show()`` inside single
functions, which made the numbers untestable and the pipeline unusable in a
headless environment.
"""

from radon.viz.data import plot_label_distribution
from radon.viz.io import save_figure

__all__ = ["plot_label_distribution", "save_figure"]
