"""Histogram thresholding for the projection step.

RADON decides which samples are inliers by histogramming their cosine
reconstruction scores and cutting the distribution at a threshold found with the
*triangle* (Rosin) method: draw the chord from the histogram's peak to its last
non-empty bin, and take the bin lying furthest from that chord.

Two estimators choose the bin count: any estimator NumPy accepts
(``"fd"``, ``"sturges"``, ...) or Knuth's Bayesian rule, which is the default.

The mathematics is unchanged from the reference implementation. What changed is
that degenerate inputs now raise instead of silently returning a threshold, and
that the search loops are vectorised.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln

from radon.exceptions import ThresholdError

__all__ = [
    "KnuthCost",
    "UnimodalThreshold",
    "histogram_bin_edges",
    "knuth_num_bins",
    "unimodal_threshold",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UnimodalThreshold:
    """Result of the triangle thresholding method.

    Attributes:
        index: Index of the selected bin.
        value: Centre of the selected bin - the threshold itself.
        degenerate: True when the selected bin is the last non-empty one, which
            means the histogram had no descending tail for the method to cut.
            The threshold then sits at the very edge of the distribution and the
            partition it induces is almost empty on one side.
    """

    index: int
    value: float
    degenerate: bool = False


def unimodal_threshold(
    hist: NDArray[np.integer] | NDArray[np.floating],
    bin_centers: NDArray[np.floating],
) -> UnimodalThreshold:
    """Find the triangle-method threshold of a score histogram.

    The chord runs from the peak ``(argmax, max)`` to ``(last_non_empty + 1, 0)``.
    Every bin from the peak up to that point is tested, and the one with the
    greatest perpendicular distance from the chord wins. Ties keep the leftmost
    bin, matching the original strict ``>`` comparison.

    Args:
        hist: Bin counts. Must be ordered so that the tail to be cut lies to the
            right of the peak; callers reverse the array when cutting the other
            side.
        bin_centers: Centre of each bin, same length as ``hist``.

    Returns:
        The chosen bin index and its centre.

    Raises:
        ThresholdError: If the arrays are empty, differ in length, hold negative
            counts, or the histogram contains no samples at all. The original
            returned a threshold regardless: an all-zero histogram yielded the
            first bin centre and a single empty bin fell through to an
            unguarded ``-1`` index, silently returning the last bin.
    """
    hist = np.asarray(hist)
    bin_centers = np.asarray(bin_centers, dtype=np.float64)

    if hist.ndim != 1 or bin_centers.ndim != 1:
        raise ThresholdError(
            f"unimodal_threshold: both arrays must be 1-D, got shapes "
            f"{hist.shape} and {bin_centers.shape}."
        )
    if hist.size == 0:
        raise ThresholdError("unimodal_threshold: the histogram is empty.")
    if hist.size != bin_centers.size:
        raise ThresholdError(
            f"unimodal_threshold: {hist.size} bin(s) but {bin_centers.size} bin centre(s); "
            "the two must be aligned."
        )
    if np.any(hist < 0):
        raise ThresholdError("unimodal_threshold: the histogram contains negative counts.")

    non_empty = np.flatnonzero(hist)
    if non_empty.size == 0:
        raise ThresholdError(
            "unimodal_threshold: the histogram contains no samples, so no threshold can be "
            "estimated. This usually means the scores fell outside the binning range."
        )

    peak_index = int(np.argmax(hist))
    last_non_empty = int(non_empty[-1])
    chord_end = last_non_empty + 1

    # Chord from the peak to the foot of the distribution. `chord_end` is a
    # coordinate, not an index, so it may legitimately sit one past the array.
    chord = np.array([peak_index - chord_end, float(hist[peak_index])], dtype=np.float64)
    chord_length = float(np.hypot(chord[0], chord[1]))
    if chord_length == 0.0:
        raise ThresholdError(
            "unimodal_threshold: the peak coincides with the foot of the distribution, "
            "leaving no chord to measure distances against."
        )

    # Perpendicular distance of every candidate bin from the chord, via the
    # magnitude of the 2-D cross product. Vectorised over the whole range; the
    # original evaluated one bin per Python iteration.
    candidates = np.arange(peak_index, chord_end)
    offset_x = candidates - chord_end
    offset_y = hist[candidates].astype(np.float64)
    cross = chord[0] * offset_y - offset_x * chord[1]
    distances = np.abs(cross) / chord_length

    # argmax returns the first maximum, matching the original `d > max_dist`.
    best = int(candidates[int(np.argmax(distances))])

    degenerate = best == last_non_empty
    if degenerate:
        logger.warning(
            "Triangle thresholding selected the last non-empty bin (index %d of %d). The "
            "histogram rises monotonically instead of showing a tail, so the threshold sits "
            "at the edge of the score distribution and one side of the resulting partition "
            "will be nearly empty.",
            best,
            hist.size,
        )

    return UnimodalThreshold(index=best, value=float(bin_centers[best]), degenerate=degenerate)


class KnuthCost:
    """Knuth's Bayesian cost function for the number of histogram bins.

    Maximising this over ``M`` yields the bin count best supported by the data.
    The data is sorted once at construction so that each evaluation can count
    bin occupancies with a binary search instead of a full pass.

    Attributes:
        data: The sorted sample.
    """

    def __init__(self, data: NDArray[np.floating]) -> None:
        """Sort the sample and cache the terms that do not depend on ``M``.

        Args:
            data: One-dimensional sample of scores.

        Raises:
            ThresholdError: If the data is not 1-D, is empty, or is constant.
        """
        # The input dtype is preserved on purpose: bin edges are a linspace over
        # the data range, so promoting float32 scores to float64 would move
        # every edge by about float32 epsilon and shift the resulting threshold.
        values = np.asarray(data).copy()
        if not np.issubdtype(values.dtype, np.floating):
            values = values.astype(np.float64)
        if values.ndim != 1:
            raise ThresholdError(f"KnuthCost: data must be 1-D, got shape {values.shape}.")
        if values.size == 0:
            raise ThresholdError("KnuthCost: the sample is empty.")
        if not np.isfinite(values).all():
            raise ThresholdError("KnuthCost: the sample contains NaN or infinite values.")

        values.sort()
        if values[0] == values[-1]:
            raise ThresholdError(
                "KnuthCost: every score is identical, so no binning is meaningful. "
                "This usually means the model collapsed to a constant output."
            )

        self.data = values
        self.n = values.size
        self._gammaln_half = float(gammaln(0.5))

    def bin_counts(self, n_bins: int) -> NDArray[np.int64]:
        """Occupancy of each bin under an equal-width binning.

        Equivalent to ``np.histogram(data, bins=n_bins)[0]`` but O(M log N)
        rather than O(N) per call, which matters because the estimator sweeps a
        few hundred candidate values of ``M``.

        Args:
            n_bins: Number of equal-width bins.

        Returns:
            Counts of length ``n_bins``.
        """
        edges = np.linspace(self.data[0], self.data[-1], n_bins + 1)
        # Interior edges only: everything below the first edge and above the
        # last belongs to the outer bins by construction, and this reproduces
        # NumPy's rule that the rightmost bin is closed on both sides.
        interior = np.searchsorted(self.data, edges[1:-1], side="left")
        boundaries = np.concatenate(([0], interior, [self.n]))
        return np.diff(boundaries).astype(np.int64)

    def evaluate(self, n_bins: int) -> float:
        """Knuth's cost for a given number of bins.

        Args:
            n_bins: Candidate bin count.

        Returns:
            The cost; lower is better. Non-positive ``n_bins`` yields infinity.
        """
        if n_bins <= 0:
            return float(np.inf)

        counts = self.bin_counts(n_bins)
        return -float(
            self.n * np.log(n_bins)
            + gammaln(0.5 * n_bins)
            - n_bins * self._gammaln_half
            - gammaln(self.n + 0.5 * n_bins)
            + np.sum(gammaln(counts + 0.5))
        )


def knuth_num_bins(
    data: NDArray[np.floating],
    n_min: int = 10,
    n_max: int = 250,
    step: int = 1,
) -> int:
    """Choose a bin count with Knuth's rule.

    Args:
        data: One-dimensional sample of scores.
        n_min: Smallest candidate bin count.
        n_max: Largest candidate bin count, inclusive.
        step: Stride across the candidate range.

    Returns:
        The bin count minimising Knuth's cost.

    Raises:
        ThresholdError: If the sample is unusable or the candidate range is
            empty.
    """
    if n_min < 1:
        raise ThresholdError(f"knuth_num_bins: n_min must be at least 1, got {n_min}.")
    if n_max < n_min:
        raise ThresholdError(
            f"knuth_num_bins: n_max ({n_max}) must not be smaller than n_min ({n_min})."
        )
    if step < 1:
        raise ThresholdError(f"knuth_num_bins: step must be at least 1, got {step}.")

    cost = KnuthCost(data)
    candidates = np.arange(n_min, n_max + 1, step)
    costs = np.array([cost.evaluate(int(m)) for m in candidates], dtype=np.float64)

    best = int(candidates[int(np.argmin(costs))])
    logger.debug("Knuth estimator: %d bins over %d candidates.", best, candidates.size)
    return best


def histogram_bin_edges(
    scores: NDArray[np.floating],
    estimator: str = "knuth",
    n_min: int = 10,
    n_max: int = 250,
) -> NDArray[np.floating]:
    """Compute histogram bin edges for a set of scores.

    Unifies the two estimator paths. The original returned a bin *count* for
    Knuth and an *edge array* for NumPy estimators, leaving every caller to
    handle both shapes; edges are returned in either case here.

    Args:
        scores: One-dimensional sample of scores.
        estimator: ``"knuth"`` or any estimator accepted by
            :func:`numpy.histogram_bin_edges` (``"fd"``, ``"sturges"``, ...).
        n_min: Smallest candidate bin count, Knuth only.
        n_max: Largest candidate bin count, Knuth only.

    Returns:
        Monotonically increasing bin edges of length ``n_bins + 1``.

    Raises:
        ThresholdError: If the sample is unusable or the estimator is unknown.
    """
    # The input dtype is preserved: see the note in KnuthCost.__init__.
    values = np.asarray(scores)
    if values.ndim != 1:
        raise ThresholdError(f"histogram_bin_edges: scores must be 1-D, got shape {values.shape}.")
    if values.size == 0:
        raise ThresholdError("histogram_bin_edges: the score array is empty.")

    if estimator == "knuth":
        n_bins = knuth_num_bins(values, n_min=n_min, n_max=n_max)
        return np.linspace(values.min(), values.max(), n_bins + 1)

    try:
        return np.asarray(np.histogram_bin_edges(values, bins=estimator))
    except (ValueError, TypeError) as exc:
        raise ThresholdError(
            f"histogram_bin_edges: '{estimator}' is not a usable estimator. Expected "
            f"'knuth' or a NumPy estimator name such as 'fd', 'sturges' or 'auto'. ({exc})"
        ) from exc
