"""Tests for histogram thresholding.

Includes a differential suite against the original ``unimodal`` and
``knuth_bin_width``: the refactor must not move a single threshold on any
well-formed histogram.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from radon.exceptions import ThresholdError
from radon.training.thresholding import (
    KnuthCost,
    histogram_bin_edges,
    knuth_num_bins,
    unimodal_threshold,
)


def reference_unimodal(hist, bin_centers):
    """The original implementation, verbatim, as a differential oracle."""
    pos_peak = np.argmax(hist)
    peak_max = hist[pos_peak]
    p1 = np.array([pos_peak, peak_max])

    nz = np.where(hist != 0)[0]
    last_zeroBin = nz[-1] + 1 if len(nz) > 0 else len(hist) - 1

    p2 = np.array([last_zeroBin, 0])
    best_idx = -1
    max_dist = -1
    for x0 in range(pos_peak, last_zeroBin):
        y0 = hist[x0]
        a = p1 - p2
        b = [x0, y0] - p2
        cross_ab = a[0] * b[1] - b[0] * a[1]
        d = np.linalg.norm(cross_ab) / np.linalg.norm(a)
        if d > max_dist:
            best_idx = x0
            max_dist = d
    return best_idx, bin_centers[best_idx]


def centers(n: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n)


class TestUnimodalCorrectness:
    def test_textbook_histogram(self) -> None:
        hist = np.array([1, 5, 20, 9, 4, 2, 1, 0, 0, 0])
        result = unimodal_threshold(hist, centers(hist.size))
        assert result.index == 4
        assert result.value == pytest.approx(4 / 9)
        assert result.degenerate is False

    def test_peak_at_first_bin(self) -> None:
        hist = np.array([20, 4, 3, 2, 1])
        assert unimodal_threshold(hist, centers(hist.size)).index == 1

    def test_single_populated_bin(self) -> None:
        hist = np.array([0, 0, 7, 0, 0])
        assert unimodal_threshold(hist, centers(hist.size)).index == 2

    def test_ties_keep_the_leftmost_bin(self) -> None:
        """Matches the original strict `d > max_dist` comparison."""
        hist = np.array([10, 5, 5, 5, 0])
        result = unimodal_threshold(hist, centers(hist.size))
        assert result.index == reference_unimodal(hist, centers(hist.size))[0]


class TestUnimodalDegenerate:
    def test_monotone_histogram_is_flagged(self, caplog) -> None:
        """A rising histogram has no tail to cut; the cut lands at the edge."""
        hist = np.array([1, 2, 3, 4, 20])
        with caplog.at_level(logging.WARNING, logger="radon.training.thresholding"):
            result = unimodal_threshold(hist, centers(hist.size))
        assert result.degenerate is True
        assert result.index == hist.size - 1
        assert "nearly empty" in caplog.text

    def test_all_zero_histogram_raises(self) -> None:
        """The original returned the first bin centre with no signal at all."""
        with pytest.raises(ThresholdError, match="no samples"):
            unimodal_threshold(np.zeros(10, dtype=int), centers(10))

    def test_single_empty_bin_raises(self) -> None:
        """The original fell through to index -1 and returned the last bin."""
        with pytest.raises(ThresholdError, match="no samples"):
            unimodal_threshold(np.zeros(1, dtype=int), centers(1))

    def test_empty_histogram_raises(self) -> None:
        with pytest.raises(ThresholdError, match="empty"):
            unimodal_threshold(np.array([], dtype=int), np.array([]))

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ThresholdError, match="aligned"):
            unimodal_threshold(np.array([1, 2, 3]), centers(5))

    def test_negative_counts_raise(self) -> None:
        with pytest.raises(ThresholdError, match="negative"):
            unimodal_threshold(np.array([1, -2, 3]), centers(3))

    def test_two_dimensional_input_raises(self) -> None:
        with pytest.raises(ThresholdError, match="1-D"):
            unimodal_threshold(np.ones((2, 3)), centers(3))


class TestUnimodalDifferential:
    """The refactor must reproduce the original on every well-formed input."""

    @pytest.mark.parametrize("seed", range(40))
    def test_matches_reference_on_random_histograms(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        size = int(rng.integers(3, 60))
        hist = rng.integers(0, 200, size=size)
        if hist.sum() == 0:
            pytest.skip("all-zero histogram is rejected by design")

        bins = centers(size)
        expected_index, expected_value = reference_unimodal(hist, bins)
        result = unimodal_threshold(hist, bins)

        assert result.index == expected_index
        assert result.value == pytest.approx(expected_value)

    @pytest.mark.parametrize("seed", range(20))
    def test_matches_reference_on_realistic_score_histograms(self, seed: int) -> None:
        """Skewed, heavy-tailed shapes like real cosine-score distributions."""
        rng = np.random.default_rng(seed)
        scores = np.concatenate([rng.beta(8, 2, 4000), rng.beta(2, 8, int(rng.integers(50, 400)))])
        edges = histogram_bin_edges(scores, estimator="fd")
        hist, _ = np.histogram(scores, bins=edges)
        bins = (edges[1:] + edges[:-1]) / 2

        expected_index, expected_value = reference_unimodal(hist, bins)
        result = unimodal_threshold(hist, bins)

        assert result.index == expected_index
        assert result.value == pytest.approx(expected_value)

    def test_matches_reference_on_reversed_histograms(self) -> None:
        """The projection cuts the other side by reversing the arrays."""
        rng = np.random.default_rng(7)
        scores = rng.beta(8, 2, 3000)
        edges = histogram_bin_edges(scores, estimator="sturges")
        hist, _ = np.histogram(scores, bins=edges)
        bins = (edges[1:] + edges[:-1]) / 2

        reversed_hist = hist[::-1]
        reversed_bins = bins[::-1]
        expected_index, expected_value = reference_unimodal(reversed_hist, reversed_bins)
        result = unimodal_threshold(reversed_hist, reversed_bins)

        assert result.index == expected_index
        assert result.value == pytest.approx(expected_value)


class TestKnuthCost:
    def test_bin_counts_match_numpy_histogram(self) -> None:
        """The searchsorted counting must reproduce np.histogram exactly."""
        rng = np.random.default_rng(0)
        data = rng.normal(0.5, 0.2, 5000)
        cost = KnuthCost(data)
        sorted_data = np.sort(data)

        for n_bins in (2, 3, 17, 64, 199, 250):
            edges = np.linspace(sorted_data[0], sorted_data[-1], n_bins + 1)
            expected, _ = np.histogram(sorted_data, bins=edges)
            assert np.array_equal(cost.bin_counts(n_bins), expected)

    def test_bin_counts_are_exhaustive(self) -> None:
        rng = np.random.default_rng(1)
        cost = KnuthCost(rng.random(500))
        assert int(cost.bin_counts(23).sum()) == 500

    def test_non_positive_bins_cost_infinity(self) -> None:
        cost = KnuthCost(np.random.default_rng(2).random(100))
        assert cost.evaluate(0) == float("inf")
        assert cost.evaluate(-5) == float("inf")

    def test_constant_data_raises(self) -> None:
        with pytest.raises(ThresholdError, match="identical"):
            KnuthCost(np.full(100, 0.42))

    def test_empty_data_raises(self) -> None:
        with pytest.raises(ThresholdError, match="empty"):
            KnuthCost(np.array([]))

    def test_non_finite_data_raises(self) -> None:
        with pytest.raises(ThresholdError, match="NaN or infinite"):
            KnuthCost(np.array([0.1, np.nan, 0.3]))

    def test_input_is_not_mutated(self) -> None:
        data = np.array([0.9, 0.1, 0.5])
        KnuthCost(data)
        assert np.array_equal(data, [0.9, 0.1, 0.5])


class TestKnuthDifferential:
    @staticmethod
    def reference_knuth(data: np.ndarray, n_min: int, n_max: int, step: int) -> int:
        """The original estimator, using np.histogram on each candidate."""
        from scipy.special import gammaln

        values = np.array(data, copy=True)
        values.sort()
        n = values.size

        def evaluate(m: int) -> float:
            if m <= 0:
                return np.inf
            edges = np.linspace(values[0], values[-1], int(m) + 1)
            nk, _ = np.histogram(values, edges)
            return -(
                n * np.log(m)
                + gammaln(0.5 * m)
                - m * gammaln(0.5)
                - gammaln(n + 0.5 * m)
                + np.sum(gammaln(nk + 0.5))
            )

        candidates = range(n_min, n_max + 1, step)
        return candidates[int(np.argmin([evaluate(m) for m in candidates]))]

    @pytest.mark.parametrize("seed", range(8))
    def test_matches_reference_estimator(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        data = np.concatenate([rng.beta(8, 2, 3000), rng.beta(2, 8, 200)])
        assert knuth_num_bins(data, 10, 250, 1) == self.reference_knuth(data, 10, 250, 1)

    def test_matches_reference_on_narrow_range(self) -> None:
        data = np.random.default_rng(3).normal(0.0, 1.0, 2000)
        assert knuth_num_bins(data, 5, 40, 5) == self.reference_knuth(data, 5, 40, 5)


class TestKnuthValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"n_min": 0}, "n_min must be at least 1"),
            ({"n_min": 50, "n_max": 10}, "must not be smaller"),
            ({"step": 0}, "step must be at least 1"),
        ],
    )
    def test_invalid_ranges_are_rejected(self, kwargs: dict, match: str) -> None:
        data = np.random.default_rng(0).random(200)
        with pytest.raises(ThresholdError, match=match):
            knuth_num_bins(data, **kwargs)


class TestHistogramBinEdges:
    def test_knuth_returns_edges_not_a_count(self) -> None:
        """The original returned an int here and an array for NumPy estimators."""
        data = np.random.default_rng(0).beta(8, 2, 2000)
        edges = histogram_bin_edges(data, estimator="knuth")
        assert edges.ndim == 1
        assert edges.size == knuth_num_bins(data, 10, 250) + 1
        assert np.all(np.diff(edges) > 0)

    def test_knuth_edges_reproduce_the_bin_count_path(self) -> None:
        """np.histogram(data, bins=M) and bins=linspace(min,max,M+1) agree."""
        data = np.random.default_rng(1).beta(8, 2, 2000)
        n_bins = knuth_num_bins(data, 10, 250)
        by_count, _ = np.histogram(data, bins=n_bins)
        by_edges, _ = np.histogram(data, bins=histogram_bin_edges(data, "knuth"))
        assert np.array_equal(by_count, by_edges)

    @pytest.mark.parametrize("estimator", ["fd", "sturges", "auto", "scott"])
    def test_numpy_estimators(self, estimator: str) -> None:
        data = np.random.default_rng(2).normal(0.5, 0.1, 3000)
        edges = histogram_bin_edges(data, estimator=estimator)
        assert np.array_equal(edges, np.histogram_bin_edges(data, bins=estimator))

    def test_unknown_estimator_raises(self) -> None:
        data = np.random.default_rng(3).random(500)
        with pytest.raises(ThresholdError, match="not a usable estimator"):
            histogram_bin_edges(data, estimator="telepathy")

    def test_empty_scores_raise(self) -> None:
        with pytest.raises(ThresholdError, match="empty"):
            histogram_bin_edges(np.array([]), estimator="knuth")
