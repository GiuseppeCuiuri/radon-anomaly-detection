"""Feature attribution for NSL-KDD anomalies.

Answers "why did this connection score as anomalous?" by ablation: replace one
feature (or one categorical group) with a value typical of normal traffic,
re-score, and record how much the anomaly score moved. A feature whose
normalisation *lowers* the score was carrying the anomaly.

Two attributions are produced per sample:

- **categorical**: each one-hot group is reset to the category most common
  among inliers;
- **numeric**: each numeric feature is clipped into the inlier interquantile
  range, one feature at a time.

Everything here returns data. The original interleaved computation with two
dozen ``print`` statements and three ``plt.show()`` calls, which made the
attribution impossible to test, export or aggregate.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.utils.data import DataLoader

from radon.exceptions import RadonError
from radon.features.tabular import ONEHOT_PREFIXES
from radon.models.autoencoder import RadonAutoencoder

__all__ = [
    "CategoricalAttribution",
    "InlierProfile",
    "NumericAttribution",
    "SampleExplanation",
    "build_inlier_profile",
    "explain_record",
    "explain_samples",
    "profile_from_dict",
    "profile_to_dict",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InlierProfile:
    """What normal traffic looks like, used as the ablation baseline.

    Attributes:
        quantile_low: Lower quantile of each feature over inliers.
        quantile_high: Upper quantile of each feature over inliers.
        onehot_groups: Column indices of each categorical group, by name.
        group_modes: The most frequent column within each group.
        numeric_indices: Columns not belonging to any categorical group.
        feature_names: Column names.
    """

    quantile_low: Tensor
    quantile_high: Tensor
    onehot_groups: dict[str, tuple[int, ...]]
    group_modes: dict[str, int]
    numeric_indices: tuple[int, ...]
    feature_names: tuple[str, ...]

    def short_name(self, index: int) -> str:
        """Feature name with its group prefix stripped, for display."""
        name = self.feature_names[index]
        return name.split("_", 1)[-1] if "_" in name else name


@dataclass(frozen=True)
class NumericAttribution:
    """Effect of clipping one numeric feature into the inlier range.

    Attributes:
        index: Column index.
        name: Column name.
        value: The observed value.
        reconstruction: What the model reconstructed for it.
        quantile_low: Lower end of the inlier range.
        quantile_high: Upper end of the inlier range.
        outside_range: Whether the observed value falls outside that range.
        delta_score: Change in anomaly score when the feature is normalised.
            Negative means this feature was driving the anomaly.
        error_positive: ``|value - reconstruction|``.
        error_negative: ``|(1 - value) - reconstruction|``; smaller than
            ``error_positive`` means the model reconstructed the *inverted*
            target, which is negative learning working as intended.
    """

    index: int
    name: str
    value: float
    reconstruction: float
    quantile_low: float
    quantile_high: float
    outside_range: bool
    delta_score: float
    error_positive: float
    error_negative: float

    @property
    def prefers_inverted_target(self) -> bool:
        """Whether the reconstruction is closer to ``1 - x`` than to ``x``."""
        return self.error_negative < self.error_positive


@dataclass(frozen=True)
class CategoricalAttribution:
    """Effect of resetting one categorical group to its inlier mode.

    Attributes:
        group: Group name, e.g. ``service``.
        observed: The active categories and their values, most active first.
        baseline: The category the group was reset to.
        delta_score: Change in anomaly score. Negative means this group was
            driving the anomaly.
        mismatch: Whether the observed top category differs from the baseline.
    """

    group: str
    observed: tuple[tuple[str, float], ...]
    baseline: str
    delta_score: float
    mismatch: bool


@dataclass(frozen=True)
class SampleExplanation:
    """The full attribution for one sample.

    Attributes:
        tag: How the sample was selected, e.g. ``top anomaly``.
        score: Its anomaly score.
        label: Ground truth, 1 for an outlier; ``None`` when it is unknown, as
            for a record sent to the inference service.
        categorical: Categorical attributions, most incriminating first.
        numeric_causal: Numeric attributions ranked by ``delta_score``.
        numeric_violations: Numeric features ranked by reconstruction error.
    """

    tag: str
    score: float
    label: int | None
    categorical: tuple[CategoricalAttribution, ...]
    numeric_causal: tuple[NumericAttribution, ...]
    numeric_violations: tuple[NumericAttribution, ...]

    def describe(self, top_k: int = 3) -> str:
        """A short human-readable account of why this sample scored as it did."""
        lines = [f"{self.tag}: score={self.score:.4f}, label={self.label}"]

        for attribution in self.categorical[:top_k]:
            observed = (
                ", ".join(
                    f"{name}={value:.2f}" for name, value in attribution.observed if value > 0.01
                )
                or "none active"
            )
            flag = "MISMATCH" if attribution.mismatch else "matches"
            lines.append(
                f"  [{attribution.group}] {observed} -> {attribution.baseline} "
                f"({flag}), delta={attribution.delta_score:+.4f}"
            )

        for numeric in self.numeric_causal[:top_k]:
            position = "outside" if numeric.outside_range else "inside"
            lines.append(
                f"  [{numeric.name}] x={numeric.value:.3f} "
                f"clip=[{numeric.quantile_low:.3f},{numeric.quantile_high:.3f}] "
                f"({position}), delta={numeric.delta_score:+.4f}"
            )

        inverted = sum(1 for a in self.numeric_violations if a.prefers_inverted_target)
        lines.append(
            f"  negative learning: {inverted}/{len(self.numeric_violations)} features "
            "reconstructed closer to the inverted target"
        )
        return "\n".join(lines)


def build_inlier_profile(
    x_train: Tensor,
    y_train: Tensor,
    feature_names: Sequence[str],
    quantile_low: float = 0.05,
    quantile_high: float = 0.95,
) -> InlierProfile:
    """Summarise normal traffic to serve as the ablation baseline.

    Args:
        x_train: Training features.
        y_train: Training labels; only inliers contribute.
        feature_names: Column names.
        quantile_low: Lower quantile of the normal range.
        quantile_high: Upper quantile of the normal range.

    Returns:
        The profile.

    Raises:
        RadonError: If the names do not match the data width, or there are no
            inliers to profile.
    """
    if len(feature_names) != x_train.shape[1]:
        raise RadonError(
            f"build_inlier_profile: {len(feature_names)} feature name(s) but "
            f"{x_train.shape[1]} column(s)."
        )

    labels = y_train.reshape(-1)
    inliers = x_train[labels == 0].float()
    if inliers.shape[0] == 0:
        raise RadonError("build_inlier_profile: the training split contains no inliers.")

    low = torch.quantile(inliers, quantile_low, dim=0)
    high = torch.quantile(inliers, quantile_high, dim=0)

    groups: dict[str, list[int]] = {}
    for index, name in enumerate(feature_names):
        for prefix in ONEHOT_PREFIXES:
            if name.startswith(prefix):
                groups.setdefault(prefix.rstrip("_"), []).append(index)
                break

    onehot_groups = {name: tuple(indices) for name, indices in groups.items()}
    grouped = {index for indices in onehot_groups.values() for index in indices}
    numeric_indices = tuple(i for i in range(len(feature_names)) if i not in grouped)

    group_modes = {}
    for name, indices in onehot_groups.items():
        frequencies = inliers[:, list(indices)].mean(dim=0)
        group_modes[name] = indices[int(torch.argmax(frequencies).item())]

    logger.info(
        "Inlier profile: %d numeric feature(s), %d categorical group(s), from %d inlier(s).",
        len(numeric_indices),
        len(onehot_groups),
        inliers.shape[0],
    )
    return InlierProfile(
        quantile_low=low,
        quantile_high=high,
        onehot_groups=onehot_groups,
        group_modes=group_modes,
        numeric_indices=numeric_indices,
        feature_names=tuple(feature_names),
    )


def profile_to_dict(profile: InlierProfile) -> dict[str, Any]:
    """Plain-data form of a profile, for a checkpoint's ``extra`` slot.

    Only tensors, strings, integers, lists and dicts, which
    ``torch.load(weights_only=True)`` accepts.

    Args:
        profile: The profile to store.

    Returns:
        A dictionary that :func:`profile_from_dict` turns back into the profile.
    """
    return {
        "quantile_low": profile.quantile_low.detach().cpu(),
        "quantile_high": profile.quantile_high.detach().cpu(),
        "onehot_groups": {name: list(indices) for name, indices in profile.onehot_groups.items()},
        "group_modes": dict(profile.group_modes),
        "numeric_indices": list(profile.numeric_indices),
        "feature_names": list(profile.feature_names),
    }


def profile_from_dict(data: Mapping[str, Any]) -> InlierProfile:
    """Rebuild a profile stored by :func:`profile_to_dict`.

    Args:
        data: The stored dictionary.

    Returns:
        The profile, with its quantiles on the CPU.

    Raises:
        RadonError: If a field is missing or malformed, or the quantiles do not
            have one value per feature.
    """
    try:
        profile = InlierProfile(
            quantile_low=torch.as_tensor(data["quantile_low"], dtype=torch.float32),
            quantile_high=torch.as_tensor(data["quantile_high"], dtype=torch.float32),
            onehot_groups={
                str(name): tuple(int(i) for i in indices)
                for name, indices in data["onehot_groups"].items()
            },
            group_modes={str(name): int(index) for name, index in data["group_modes"].items()},
            numeric_indices=tuple(int(i) for i in data["numeric_indices"]),
            feature_names=tuple(str(name) for name in data["feature_names"]),
        )
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise RadonError(f"Malformed inlier profile: {exc!r}") from exc

    width = len(profile.feature_names)
    for name, quantiles in (("low", profile.quantile_low), ("high", profile.quantile_high)):
        if tuple(quantiles.shape) != (width,):
            raise RadonError(
                f"Malformed inlier profile: {width} feature name(s) but quantile_{name} "
                f"has shape {tuple(quantiles.shape)}."
            )
    return profile


@torch.no_grad()
def explain_record(
    model: RadonAutoencoder,
    record: Tensor,
    profile: InlierProfile,
    device: torch.device,
    min_abs_delta: float = 1e-3,
    min_clip_delta: float = 0.02,
    top_k: int = 8,
) -> SampleExplanation:
    """Explain one record whose ground truth is unknown.

    The same two ablations as :func:`explain_samples`, for a single record
    instead of the extremes of a split.

    Args:
        model: The trained model.
        record: Feature vector of shape ``(input_dim,)`` or ``(1, input_dim)``.
        profile: The inlier baseline.
        device: Device to run on.
        min_abs_delta: Ignore attributions smaller than this.
        min_clip_delta: Ignore numeric features whose clipped value barely
            differs from the observed one.
        top_k: How many attributions to keep per ranking.

    Returns:
        The explanation, with ``label`` set to ``None``.

    Raises:
        RadonError: If the record width does not match the profile.
    """
    features = record.reshape(1, -1).to(device).float()
    if features.shape[1] != len(profile.feature_names):
        raise RadonError(
            f"explain_record: the record has {features.shape[1]} feature(s), the profile "
            f"{len(profile.feature_names)}."
        )

    model.eval()
    model.to(device)
    criterion = nn.CosineSimilarity(dim=1, eps=1e-5)
    reconstruction = model(features)
    score = float((1.0 - criterion(reconstruction, features)).item())

    return _explain_one(
        model=model,
        criterion=criterion,
        tag="record",
        score=score,
        label=None,
        features=features,
        reconstruction=reconstruction,
        profile=profile,
        low=profile.quantile_low.to(device),
        high=profile.quantile_high.to(device),
        min_abs_delta=min_abs_delta,
        min_clip_delta=min_clip_delta,
        top_k=top_k,
    )


@torch.no_grad()
def explain_samples(
    model: RadonAutoencoder,
    loader: DataLoader[tuple[Tensor, Tensor]],
    profile: InlierProfile,
    device: torch.device,
    n_anomalies: int = 1,
    n_normals: int = 2,
    include_borderline: bool = True,
    pool_size: int = 400,
    min_abs_delta: float = 1e-3,
    min_clip_delta: float = 0.02,
    top_k: int = 8,
) -> list[SampleExplanation]:
    """Explain the most anomalous and most normal samples in a split.

    Args:
        model: The trained model.
        loader: Yields ``(features, labels)`` batches.
        profile: The inlier baseline.
        device: Device to run on.
        n_anomalies: How many top-scoring outliers to explain.
        n_normals: How many lowest-scoring inliers to explain.
        include_borderline: Also explain the highest-scoring inlier, which is
            the most informative false-positive candidate.
        pool_size: Stop scanning once this many of each class have been seen.
        min_abs_delta: Ignore attributions smaller than this.
        min_clip_delta: Ignore numeric features whose clipped value barely
            differs from the observed one.
        top_k: How many attributions to keep per ranking.

    Returns:
        One explanation per selected sample.

    Raises:
        RadonError: If either class is absent from the scanned pool.
    """
    model.eval()
    model.to(device)
    criterion = nn.CosineSimilarity(dim=1, eps=1e-5)

    anomalies, normals = _collect_pools(model, loader, criterion, device, pool_size)
    if not anomalies:
        raise RadonError("explain_samples: no outliers found in the scanned pool.")
    if not normals:
        raise RadonError("explain_samples: no inliers found in the scanned pool.")

    anomalies.sort(key=lambda item: item[0], reverse=True)
    normals.sort(key=lambda item: item[0])

    selected: list[tuple[str, tuple[float, Tensor, Tensor, int]]] = []
    selected += [("top anomaly", item) for item in anomalies[:n_anomalies]]
    selected += [("clear normal", item) for item in normals[:n_normals]]
    if include_borderline and len(normals) > 10:
        selected.append(("borderline normal", normals[-1]))

    low = profile.quantile_low.to(device)
    high = profile.quantile_high.to(device)

    explanations = []
    for tag, (score, features, reconstruction, label) in selected:
        explanations.append(
            _explain_one(
                model=model,
                criterion=criterion,
                tag=tag,
                score=score,
                label=label,
                features=features.to(device),
                reconstruction=reconstruction.to(device),
                profile=profile,
                low=low,
                high=high,
                min_abs_delta=min_abs_delta,
                min_clip_delta=min_clip_delta,
                top_k=top_k,
            )
        )

    logger.info("Explained %d sample(s).", len(explanations))
    return explanations


def _collect_pools(
    model: RadonAutoencoder,
    loader: DataLoader[tuple[Tensor, Tensor]],
    criterion: nn.CosineSimilarity,
    device: torch.device,
    pool_size: int,
) -> tuple[list[tuple[float, Tensor, Tensor, int]], list[tuple[float, Tensor, Tensor, int]]]:
    """Scan the loader until enough of each class has been scored.

    Reconstructions are kept alongside the scores so the attribution never has
    to re-run the model on the same sample. The original scored each sample
    once here and then twice more per explanation.

    Args:
        model: The trained model.
        loader: Yields ``(features, labels)`` batches.
        criterion: Cosine similarity over dimension 1.
        device: Device to run on.
        pool_size: Target size for each pool.

    Returns:
        The outlier and inlier pools as ``(score, features, reconstruction,
        label)`` tuples.
    """
    anomalies: list[tuple[float, Tensor, Tensor, int]] = []
    normals: list[tuple[float, Tensor, Tensor, int]] = []

    for features, labels in loader:
        features = features.to(device).float()
        reconstruction = model(features)
        scores = 1.0 - criterion(reconstruction, features)

        score_values = scores.detach().cpu().numpy()
        label_values = labels.reshape(-1).cpu().numpy()

        for index in range(features.shape[0]):
            label = int(label_values[index])
            target = anomalies if label == 1 else normals
            if len(target) >= pool_size:
                continue
            target.append(
                (
                    float(score_values[index]),
                    features[index : index + 1].detach(),
                    reconstruction[index : index + 1].detach(),
                    label,
                )
            )

        if len(anomalies) >= pool_size and len(normals) >= pool_size:
            break

    return anomalies, normals


def _explain_one(
    model: RadonAutoencoder,
    criterion: nn.CosineSimilarity,
    tag: str,
    score: float,
    label: int | None,
    features: Tensor,
    reconstruction: Tensor,
    profile: InlierProfile,
    low: Tensor,
    high: Tensor,
    min_abs_delta: float,
    min_clip_delta: float,
    top_k: int,
) -> SampleExplanation:
    """Run both ablations for one sample."""
    categorical = _categorical_ablation(model, criterion, features, score, profile, min_abs_delta)
    numeric = _numeric_ablation(
        model, criterion, features, reconstruction, score, profile, low, high
    )

    causal = [
        attribution
        for attribution in numeric
        if _is_causally_relevant(attribution, min_abs_delta, min_clip_delta)
    ]
    causal.sort(key=lambda a: a.delta_score)
    violations = sorted(numeric, key=lambda a: a.error_positive, reverse=True)

    return SampleExplanation(
        tag=tag,
        score=score,
        label=label,
        categorical=tuple(categorical[:top_k]),
        numeric_causal=tuple(causal[:top_k]),
        numeric_violations=tuple(violations[:top_k]),
    )


def _categorical_ablation(
    model: RadonAutoencoder,
    criterion: nn.CosineSimilarity,
    features: Tensor,
    score: float,
    profile: InlierProfile,
    min_abs_delta: float,
) -> list[CategoricalAttribution]:
    """Reset each categorical group to its inlier mode and re-score."""
    attributions = []
    values = features[0].detach().cpu().numpy()

    for group, indices in profile.onehot_groups.items():
        perturbed = features.clone()
        perturbed[0, list(indices)] = 0.0
        perturbed[0, profile.group_modes[group]] = 1.0

        delta = float((1.0 - criterion(model(perturbed), perturbed)).item()) - score
        if abs(delta) < min_abs_delta:
            continue

        order = np.argsort(values[list(indices)])[::-1][:2]
        observed = tuple((profile.short_name(indices[i]), float(values[indices[i]])) for i in order)
        baseline = profile.short_name(profile.group_modes[group])
        top = observed[0][0] if observed else "none active"

        attributions.append(
            CategoricalAttribution(
                group=group,
                observed=observed,
                baseline=baseline,
                delta_score=delta,
                mismatch=top != baseline,
            )
        )

    attributions.sort(key=lambda a: a.delta_score)
    return attributions


def _numeric_ablation(
    model: RadonAutoencoder,
    criterion: nn.CosineSimilarity,
    features: Tensor,
    reconstruction: Tensor,
    score: float,
    profile: InlierProfile,
    low: Tensor,
    high: Tensor,
) -> list[NumericAttribution]:
    """Clip each numeric feature into the inlier range and re-score.

    All perturbations are evaluated in a single batched forward pass, one row
    per feature.
    """
    indices = list(profile.numeric_indices)
    if not indices:
        return []

    device = features.device
    clipped = torch.clamp(features, low, high)

    columns = torch.tensor(indices, device=device, dtype=torch.long)
    perturbed = features.repeat(len(indices), 1)
    perturbed[torch.arange(len(indices), device=device), columns] = clipped[0, columns]

    deltas = (1.0 - criterion(model(perturbed), perturbed)).detach().cpu().numpy() - score

    values = features[0].detach().cpu().numpy()
    reconstructed = reconstruction[0].detach().cpu().numpy()
    low_np = low.detach().cpu().numpy()
    high_np = high.detach().cpu().numpy()

    error_positive = np.abs(values - reconstructed)
    error_negative = np.abs((1.0 - values) - reconstructed)

    return [
        NumericAttribution(
            index=column,
            name=profile.feature_names[column],
            value=float(values[column]),
            reconstruction=float(reconstructed[column]),
            quantile_low=float(low_np[column]),
            quantile_high=float(high_np[column]),
            outside_range=bool(values[column] < low_np[column] or values[column] > high_np[column]),
            delta_score=float(deltas[position]),
            error_positive=float(error_positive[column]),
            error_negative=float(error_negative[column]),
        )
        for position, column in enumerate(indices)
    ]


def _is_causally_relevant(
    attribution: NumericAttribution, min_abs_delta: float, min_clip_delta: float
) -> bool:
    """Whether a numeric attribution is worth reporting.

    Discards features whose inlier range is degenerate, whose observed value is
    already effectively inside it, or whose ablation moved the score negligibly.
    """
    if (attribution.quantile_high - attribution.quantile_low) < 1e-3:
        return False

    clipped = min(max(attribution.value, attribution.quantile_low), attribution.quantile_high)
    if not attribution.outside_range and abs(attribution.value - clipped) < min_clip_delta:
        return False

    return abs(attribution.delta_score) >= min_abs_delta


def attribution_table(explanation: SampleExplanation) -> NDArray[np.object_]:
    """Render a sample's numeric attributions as a table.

    Args:
        explanation: The explanation to render.

    Returns:
        A 2-D object array whose first row is the header.
    """
    header = ["feature", "x", "x_hat", "q_low", "q_high", "outside", "delta", "prefers"]
    rows = [
        [
            a.name,
            f"{a.value:.3f}",
            f"{a.reconstruction:.3f}",
            f"{a.quantile_low:.3f}",
            f"{a.quantile_high:.3f}",
            "yes" if a.outside_range else "no",
            f"{a.delta_score:+.4f}",
            "inverted" if a.prefers_inverted_target else "identity",
        ]
        for a in explanation.numeric_causal
    ]
    return np.array([header, *rows], dtype=object)
