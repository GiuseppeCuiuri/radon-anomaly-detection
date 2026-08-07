"""Tests for anomaly scoring and evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

torch = pytest.importorskip("torch")

from radon.data.loaders import build_eval_loader  # noqa: E402
from radon.evaluation import (  # noqa: E402
    ScoredSplit,
    aggregate_metrics,
    evaluate_split,
    score_split,
)
from radon.exceptions import RadonError  # noqa: E402
from radon.models import RadonAutoencoder  # noqa: E402

DEVICE = torch.device("cpu")


def make_split(n_inliers: int = 80, n_outliers: int = 20, seed: int = 0) -> ScoredSplit:
    """A separable score distribution: inliers low, outliers high."""
    rng = np.random.default_rng(seed)
    scores = np.concatenate([rng.uniform(0.0, 0.4, n_inliers), rng.uniform(0.6, 1.0, n_outliers)])
    labels = np.concatenate([np.zeros(n_inliers), np.ones(n_outliers)]).astype(np.int64)
    return ScoredSplit(scores=scores, labels=labels)


class TestScoreSplit:
    def test_scores_every_sample_once(self) -> None:
        torch.manual_seed(0)
        model = RadonAutoencoder(8, 6, 3)
        x = torch.rand(37, 8)
        y = torch.randint(0, 2, (37,))
        loader = build_eval_loader(x, y, batch_size=8)

        scored = score_split(model, loader, DEVICE)
        assert scored.n_samples == 37
        assert np.array_equal(scored.labels, y.numpy())

    def test_score_is_one_minus_cosine(self) -> None:
        torch.manual_seed(0)
        model = RadonAutoencoder(8, 6, 3).eval()
        x = torch.rand(10, 8)
        loader = build_eval_loader(x, torch.zeros(10, dtype=torch.int64), batch_size=10)

        with torch.no_grad():
            expected = 1.0 - torch.nn.CosineSimilarity(dim=1, eps=1e-5)(model(x), x)

        scored = score_split(model, loader, DEVICE)
        assert np.allclose(scored.scores, expected.numpy(), atol=1e-6)

    def test_threshold_produces_predictions(self) -> None:
        torch.manual_seed(0)
        model = RadonAutoencoder(8, 6, 3)
        loader = build_eval_loader(
            torch.rand(20, 8), torch.zeros(20, dtype=torch.int64), batch_size=8
        )

        scored = score_split(model, loader, DEVICE, threshold_cosine=0.5)
        assert scored.predictions is not None
        assert scored.threshold == pytest.approx(0.5)
        assert np.array_equal(scored.predictions, (scored.scores > 0.5).astype(np.int64))

    def test_confusion_without_predictions_raises(self) -> None:
        with pytest.raises(ValueError, match="no predictions"):
            make_split().confusion()


class TestMetrics:
    def test_perfect_separation_scores_one(self) -> None:
        result = evaluate_split(make_split())
        assert result.auroc == pytest.approx(1.0)
        assert result.auprc == pytest.approx(1.0)

    def test_random_scores_land_near_chance(self) -> None:
        rng = np.random.default_rng(3)
        scored = ScoredSplit(
            scores=rng.random(2000),
            labels=rng.integers(0, 2, 2000).astype(np.int64),
        )
        result = evaluate_split(scored)
        assert 0.45 < result.auroc < 0.55

    def test_baseline_is_the_positive_rate(self) -> None:
        result = evaluate_split(make_split(n_inliers=90, n_outliers=10))
        assert result.pr.baseline == pytest.approx(0.10)

    def test_auprc_uses_average_precision_by_default(self) -> None:
        """The trapezoid estimator is computed too, but is not the headline."""
        scored = make_split(seed=5)
        result = evaluate_split(scored)
        expected = average_precision_score(scored.labels, scored.scores)
        assert result.auprc == pytest.approx(expected)
        assert result.pr.average_precision == pytest.approx(expected)

    def test_legacy_flag_selects_the_trapezoid_estimator(self) -> None:
        rng = np.random.default_rng(11)
        scored = ScoredSplit(
            scores=rng.random(500),
            labels=rng.integers(0, 2, 500).astype(np.int64),
        )
        modern = evaluate_split(scored, legacy_auprc=False)
        legacy = evaluate_split(scored, legacy_auprc=True)

        assert legacy.auprc == pytest.approx(legacy.pr.trapezoid_auc)
        assert modern.auprc == pytest.approx(modern.pr.average_precision)
        # Both estimators are always available regardless of the flag.
        assert legacy.pr.average_precision == pytest.approx(modern.pr.average_precision)

    def test_the_two_estimators_disagree(self) -> None:
        """Why the default changed: the estimators are not interchangeable.

        Trapezoidal interpolation of a precision-recall curve has no error
        bound, because precision is not monotone in recall. The two values
        therefore differ, and which is larger is not fixed — only average
        precision has a defined meaning.
        """
        rng = np.random.default_rng(7)
        scores = rng.random(3000)
        labels = (rng.random(3000) < 0.05).astype(np.int64)
        result = evaluate_split(ScoredSplit(scores=scores, labels=labels))
        assert result.pr.trapezoid_auc != pytest.approx(result.pr.average_precision, abs=1e-6)

    def test_single_class_split_raises(self) -> None:
        scored = ScoredSplit(scores=np.linspace(0, 1, 10), labels=np.zeros(10, dtype=np.int64))
        with pytest.raises(RadonError, match="only class"):
            evaluate_split(scored)

    def test_threshold_metrics_are_reported(self) -> None:
        scored = make_split()
        scored = ScoredSplit(
            scores=scored.scores,
            labels=scored.labels,
            predictions=(scored.scores > 0.5).astype(np.int64),
            threshold=0.5,
        )
        result = evaluate_split(scored)
        assert result.f1 == pytest.approx(1.0)
        assert result.confusion == {"tp": 20, "fp": 0, "tn": 80, "fn": 0}

    def test_describe_mentions_the_baseline(self) -> None:
        assert "baseline" in evaluate_split(make_split()).describe()


class TestAggregate:
    def test_mean_and_spread(self) -> None:
        results = [evaluate_split(make_split(seed=s)) for s in range(4)]
        aggregate = aggregate_metrics(results)
        assert aggregate.n_repetitions == 4
        assert aggregate.auroc_mean == pytest.approx(1.0)
        assert aggregate.auroc_std == pytest.approx(0.0)
        assert len(aggregate.auroc_values) == 4

    def test_empty_raises(self) -> None:
        with pytest.raises(RadonError, match="no results"):
            aggregate_metrics([])
