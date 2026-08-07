"""Figure persistence."""

from __future__ import annotations

import logging
from pathlib import Path

from matplotlib.figure import Figure

__all__ = ["save_figure"]

logger = logging.getLogger(__name__)


def save_figure(fig: Figure, path: Path, dpi: int = 150, close: bool = True) -> Path:
    """Write a figure to disk, creating parent directories as needed.

    Args:
        fig: The figure to write.
        path: Destination file; the suffix selects the format.
        dpi: Output resolution.
        close: Close the figure afterwards. Leave enabled in loops, or
            matplotlib accumulates every figure in memory and warns once twenty
            are open.

    Returns:
        The path that was written.

    Raises:
        OSError: If the directory cannot be created or the file cannot be
            written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    logger.debug("Figure saved to '%s'.", path)
    if close:
        import matplotlib.pyplot as plt

        plt.close(fig)
    return path
