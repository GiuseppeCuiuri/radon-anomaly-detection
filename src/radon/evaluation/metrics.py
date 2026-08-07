"""Ranking and classification metrics.

Two notes on AUPRC, because the original got it wrong and the difference is not
cosmetic:

1. The area under a precision-recall curve must not be computed with the
   trapezoid rule. Precision is not monotone in recall and moves in jumps, so
   linear interpolation between operating points credits the model with
   performance it never achieved. ``average_precision_score`` sums
   ``(R_n - R_{n-1}) * P_n`` instead, which is the standard estimator.
2. AUPRC has no fixed chance level: the baseline is the positive rate. An AUPRC
   of 0.57 is worthless on a split that is 57% anomalies and excellent on one
   that is 5%. The baseline is therefore reported alongside the value.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from radon.evaluation.scoring import ScoredSplit
from radon.exceptions import RadonError

__all__ = [
    "AggregateMetrics",
    "EvaluationResult",
    "PrecisionRecallCurve",
    "RocCurve",
    "aggregate_metrics",
    "evaluate_split",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RocCurve:
    """A receiver operating characteristic curve.

    Attributes:
        fpr: False positive rate at each operating point.
        tpr: True positive rate at each operating point.
        thresholds: Score threshold at each operating point.
        auc: Area under the curve.
        best_index: Operating point maximising Youden's J (``tpr - fpr``).
    """

    fpr: NDArray[np.float64]
    tpr: NDArray[np.float64]
    thresholds: NDArray[np.float64]
    auc: float
    best_index: int

    @property
    def best_threshold(self) -> float:
        """Score threshold at the Youden-optimal operating point."""
        return float(self.thresholds[self.best_index])


@dataclass(frozen=True)
class PrecisionRecallCurve:
    """A precision-recall curve.

    Attributes:
        precision: Precision at each operating point.
        recall: Recall at each operating point.
        thresholds: Score threshold at each operating point; one shorter than
            the other arrays, as scikit-learn returns it.
        average_precision: The recommended AUPRC estimator.
        trapezoid_auc: ``auc(recall, precision)``, the original's estimator,
            reported only for comparison with previously published numbers.
        baseline: Positive rate, the chance level for this split.
        best_index: Operating point maximising F1.
    """

    precision: NDArray[np.float64]
    recall: NDArray[np.float64]
    thresholds: NDArray[np.float64]
    average_precision: float
    trapezoid_auc: float
    baseline: float
    best_index: int

    @property
    def best_threshold(self) -> float:
        """Score threshold at the F1-optimal operating point."""
        index = min(self.best_index, self.thresholds.size - 1)
        return float(self.thresholds[index])

    @property
    def best_f1(self) -> float:
        """The maximum F1 attainable on this curve."""
        precision = self.precision[self.best_index]
        recall = self.recall[self.best_index]
        return float(2 * precision * recall / (precision + recall + 1e-12))


@dataclass(frozen=True)
class EvaluationResult:
    """Every metric for one scored split.

    Attributes:
        roc: The ROC curve and its area.
        pr: The precision-recall curve and its areas.
        auprc: The AUPRC actually reported, selected by the legacy flag.
        precision: Precision at the applied threshold, if any.
        recall: Recall at the applied threshold, if any.
        f1: F1 at the applied threshold, if any.
        confusion: Confusion counts at the applied threshold, if any.
    """

    roc: RocCurve
    pr: PrecisionRecallCurve
    auprc: float
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    confusion: dict[str, int] | None = None

    @property
    def auroc(self) -> float:
        """Area under the ROC curve."""
        return self.roc.auc

    def describe(self) -> str:
        """A multi-line summary suitable for logging."""
        lines = [
            f"AUROC = {self.auroc:.5f}",
            f"AUPRC = {self.auprc:.5f} (baseline {self.pr.baseline:.4f}, "
            f"lift {self.auprc / max(self.pr.baseline, 1e-12):.2f}x)",
        ]
        if self.f1 is not None:
            lines.append(
                f"At the applied threshold: precision={self.precision:.4f} "
                f"recall={self.recall:.4f} f1={self.f1:.4f} {self.confusion}"
            )
        lines.append(f"Best attainable F1 on the curve = {self.pr.best_f1:.4f}")
        return "\n".join(lines)


def evaluate_split(scored: ScoredSplit, legacy_auprc: bool = False) -> EvaluationResult:
    """Compute every metric for a scored split.

    Args:
        scored: Scores, labels and optional decisions.
        legacy_auprc: Report ``auc(recall, precision)`` as the AUPRC instead of
            average precision, reproducing the original's optimistic estimator.
            Both are computed regardless; this only selects which is headline.

    Returns:
        The metrics.

    Raises:
        RadonError: If the split contains only one class, which leaves AUROC
            and AUPRC undefined.
    """
    labels = scored.labels
    scores = scored.scores

    present = np.unique(labels)
    if present.size < 2:
        raise RadonError(
            f"evaluate_split: the split contains only class {present.tolist()}, so "
            "AUROC and AUPRC are undefined. Check the test-set construction."
        )

    fpr, tpr, roc_thresholds = roc_curve(labels, scores, pos_label=1)
    roc = RocCurve(
        fpr=fpr,
        tpr=tpr,
        thresholds=roc_thresholds,
        auc=float(roc_auc_score(labels, scores)),
        best_index=int(np.argmax(tpr - fpr)),
    )

    precision, recall, pr_thresholds = precision_recall_curve(labels, scores, pos_label=1)
    f1_curve = 2 * precision * recall / (precision + recall + 1e-12)
    pr = PrecisionRecallCurve(
        precision=precision,
        recall=recall,
        thresholds=pr_thresholds,
        average_precision=float(average_precision_score(labels, scores)),
        trapezoid_auc=float(auc(recall, precision)),
        baseline=float(labels.mean()),
        best_index=int(np.argmax(f1_curve)),
    )

    auprc = pr.trapezoid_auc if legacy_auprc else pr.average_precision
    if legacy_auprc:
        logger.warning(
            "Reporting AUPRC as auc(recall, precision) = %.5f for compatibility; "
            "average_precision_score gives %.5f. The trapezoid estimator is "
            "optimistically biased.",
            pr.trapezoid_auc,
            pr.average_precision,
        )

    precision_at, recall_at, f1_at, confusion = _threshold_metrics(scored)
    result = EvaluationResult(
        roc=roc,
        pr=pr,
        auprc=auprc,
        precision=precision_at,
        recall=recall_at,
        f1=f1_at,
        confusion=confusion,
    )
    logger.info("Evaluation:\n%s", result.describe())
    return result


def _threshold_metrics(
    scored: ScoredSplit,
) -> tuple[float | None, float | None, float | None, dict[str, int] | None]:
    """Precision, recall and F1 at the applied threshold, if there is one.

    Args:
        scored: The scored split.

    Returns:
        Precision, recall, F1 and the confusion counts; all ``None`` when no
        threshold was applied.
    """
    if scored.predictions is None:
        return None, None, None, None

    counts = scored.confusion()
    true_positive = counts["tp"]
    precision = true_positive / max(true_positive + counts["fp"], 1)
    recall = true_positive / max(true_positive + counts["fn"], 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return precision, recall, f1, counts


@dataclass(frozen=True)
class AggregateMetrics:
    """Mean and spread of a metric across independent repetitions.

    Attributes:
        auroc_mean: Mean AUROC.
        auroc_std: Population standard deviation of AUROC.
        auprc_mean: Mean AUPRC.
        auprc_std: Population standard deviation of AUPRC.
        baseline: Positive rate, identical across repetitions.
        n_repetitions: How many runs were aggregated.
        auroc_values: The individual AUROC values.
        auprc_values: The individual AUPRC values.
    """

    auroc_mean: float
    auroc_std: float
    auprc_mean: float
    auprc_std: float
    baseline: float
    n_repetitions: int
    auroc_values: tuple[float, ...]
    auprc_values: tuple[float, ...]

    def describe(self) -> str:
        """A two-line summary suitable for a results table."""
        return (
            f"AUROC = {self.auroc_mean:.5f} +/- {self.auroc_std:.5f}  "
            f"(n={self.n_repetitions}, values {[f'{v:.5f}' for v in self.auroc_values]})\n"
            f"AUPRC = {self.auprc_mean:.5f} +/- {self.auprc_std:.5f}  "
            f"(baseline {self.baseline:.4f})"
        )


def aggregate_metrics(results: list[EvaluationResult]) -> AggregateMetrics:
    """Summarise metrics across repetitions.

    Args:
        results: One result per repetition.

    Returns:
        The aggregate.

    Raises:
        RadonError: If ``results`` is empty.
    """
    if not results:
        raise RadonError("aggregate_metrics: no results to aggregate.")

    auroc = np.array([r.auroc for r in results], dtype=np.float64)
    auprc = np.array([r.auprc for r in results], dtype=np.float64)

    aggregate = AggregateMetrics(
        auroc_mean=float(auroc.mean()),
        auroc_std=float(auroc.std()),
        auprc_mean=float(auprc.mean()),
        auprc_std=float(auprc.std()),
        baseline=results[0].pr.baseline,
        n_repetitions=len(results),
        auroc_values=tuple(float(v) for v in auroc),
        auprc_values=tuple(float(v) for v in auprc),
    )
    logger.info("Aggregate over %d repetition(s):\n%s", len(results), aggregate.describe())
    return aggregate
