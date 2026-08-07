"""Feature-type analysis for tabular data.

Negative learning pushes an outlier towards its feature-wise inverse. For a
continuous feature that inverse is ``1 - x``, but for a binary indicator - a
one-hot category, say - flipping the bit would ask the model to reconstruct a
different category, which is not what "unlike itself" should mean. The masks
built here let the inverter treat the two kinds of column differently.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = ["FeatureMasks", "analyze_features", "make_feature_inverter"]

logger = logging.getLogger(__name__)

#: Prefixes identifying one-hot groups in the NSL-KDD feature names.
ONEHOT_PREFIXES = ("protocol_type_", "service_", "flag_")


@dataclass(frozen=True)
class FeatureMasks:
    """Column classification for one dataset.

    Attributes:
        binary: True where a column only ever takes the values 0 or 1.
        numeric: Complement of ``binary``; the columns the inverter flips.
        onehot_groups: Index groups that behave as mutually exclusive
            categories. Empty when none could be identified.
    """

    binary: Tensor
    numeric: Tensor
    onehot_groups: tuple[tuple[int, ...], ...] = ()

    @property
    def n_features(self) -> int:
        """Total number of columns."""
        return int(self.binary.numel())

    @property
    def n_binary(self) -> int:
        """Number of binary columns."""
        return int(self.binary.sum().item())

    @property
    def n_numeric(self) -> int:
        """Number of numeric columns."""
        return int(self.numeric.sum().item())

    def describe(self) -> str:
        """A one-line summary for logging."""
        return (
            f"{self.n_features} columns: {self.n_numeric} numeric, {self.n_binary} binary, "
            f"{len(self.onehot_groups)} one-hot group(s)"
        )


def analyze_features(
    x: Tensor,
    feature_names: Sequence[str] | None = None,
    max_samples: int = 5000,
    tolerance: float = 1e-6,
) -> FeatureMasks:
    """Classify each column of ``x`` as binary or numeric.

    One-hot groups are read from ``feature_names`` when they are available, and
    inferred from column co-occurrence otherwise: within a one-hot group exactly
    one column is hot per row, so members never co-occur.

    Args:
        x: Feature matrix of shape ``(N, D)``.
        feature_names: Column names, used to recognise known one-hot prefixes.
        max_samples: Rows to inspect. The classification is stable well before
            the full dataset is read.
        tolerance: Absolute tolerance when testing a value against 0 or 1.

    Returns:
        The masks describing each column.
    """
    with torch.no_grad():
        sample = x[: min(x.shape[0], max_samples)].detach()

        is_zero = (sample - 0.0).abs() <= tolerance
        is_one = (sample - 1.0).abs() <= tolerance
        binary = (is_zero | is_one).all(dim=0)
        numeric = ~binary

        if feature_names is not None:
            groups = _groups_from_names(feature_names)
            source = "names"
        else:
            groups = _groups_from_cooccurrence(sample, binary)
            source = "co-occurrence"

    masks = FeatureMasks(binary=binary, numeric=numeric, onehot_groups=groups)
    logger.info("Feature analysis (%s): %s.", source, masks.describe())
    return masks


def _groups_from_names(names: Sequence[str]) -> tuple[tuple[int, ...], ...]:
    """Group column indices by known one-hot prefixes.

    Args:
        names: Column names.

    Returns:
        One tuple of indices per group with at least two members.
    """
    grouped: defaultdict[str, list[int]] = defaultdict(list)
    for index, name in enumerate(names):
        for prefix in ONEHOT_PREFIXES:
            if name.startswith(prefix):
                grouped[prefix].append(index)
                break

    return tuple(tuple(sorted(indices)) for indices in grouped.values() if len(indices) >= 2)


def _groups_from_cooccurrence(
    sample: Tensor,
    binary: Tensor,
    cooccurrence_tolerance: float = 1e-4,
    min_positive_rate: float = 1e-4,
    min_group_size: int = 3,
    min_onehot_fraction: float = 0.90,
    sum_tolerance: float = 1e-3,
) -> tuple[tuple[int, ...], ...]:
    """Infer one-hot groups from binary columns that never fire together.

    Builds the graph of columns whose co-occurrence rate is negligible, takes
    its connected components, and keeps those whose rows sum to 0 or 1 often
    enough to look like a genuine categorical encoding.

    Args:
        sample: Feature matrix subsample.
        binary: Mask of binary columns.
        cooccurrence_tolerance: Co-occurrence rate below which two columns are
            considered mutually exclusive.
        min_positive_rate: Columns firing less often than this are ignored.
        min_group_size: Smallest component accepted as a group.
        min_onehot_fraction: Fraction of rows that must sum to 0 or 1.
        sum_tolerance: Absolute tolerance on those row sums.

    Returns:
        One tuple of column indices per inferred group.
    """
    n_features = sample.shape[1]
    candidates = torch.where(binary)[0]
    if candidates.numel() == 0:
        return ()

    columns = sample[:, candidates].float()
    frequent = columns.mean(dim=0) >= min_positive_rate
    candidates = candidates[frequent]
    columns = columns[:, frequent]
    n_rows, n_candidates = columns.shape
    if n_candidates == 0:
        return ()

    cooccurrence = (columns.T @ columns) / max(float(n_rows), 1.0)
    cooccurrence.fill_diagonal_(0.0)
    adjacency = cooccurrence <= cooccurrence_tolerance

    groups: list[tuple[int, ...]] = []
    for component in _connected_components(adjacency):
        if len(component) < min_group_size:
            continue

        member_columns = torch.tensor(component, device=sample.device, dtype=torch.long)
        row_sums = columns[:, member_columns].sum(dim=1)
        is_empty_row = (row_sums - 0.0).abs() <= sum_tolerance
        is_hot_row = (row_sums - 1.0).abs() <= sum_tolerance

        looks_onehot = float((is_empty_row | is_hot_row).float().mean().item())
        ever_hot = float(is_hot_row.float().mean().item())
        if looks_onehot >= min_onehot_fraction and ever_hot >= 0.05:
            globals_ = sorted({int(candidates[c].item()) for c in component})
            groups.append(tuple(globals_))

    if groups:
        logger.debug(
            "Inferred %d one-hot group(s) covering %d of %d columns.",
            len(groups),
            sum(len(g) for g in groups),
            n_features,
        )
    return tuple(groups)


def _connected_components(adjacency: Tensor) -> list[list[int]]:
    """Enumerate connected components of a boolean adjacency matrix.

    Args:
        adjacency: Square boolean matrix.

    Returns:
        One list of node indices per component.
    """
    size = adjacency.shape[0]
    visited = [False] * size
    components: list[list[int]] = []

    for start in range(size):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbour in torch.where(adjacency[node])[0].tolist():
                if not visited[neighbour]:
                    visited[neighbour] = True
                    stack.append(neighbour)
        components.append(component)

    return components


def make_feature_inverter(numeric_mask: Tensor) -> Callable[[Tensor], Tensor]:
    """Build the negative-learning target function.

    The returned callable maps a batch to the target that negative learning
    pushes outliers towards: numeric features become ``1 - x`` while binary
    features keep their value, so a one-hot category is not silently turned into
    a different one.

    Args:
        numeric_mask: Boolean mask over columns, from :func:`analyze_features`.

    Returns:
        A function mapping ``(N, D)`` to ``(N, D)``.
    """

    def invert(batch: Tensor) -> Tensor:
        inverted = batch.clone()
        inverted[:, numeric_mask] = 1.0 - batch[:, numeric_mask]
        return inverted

    return invert
