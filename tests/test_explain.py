"""Tests for NSL-KDD feature attribution."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from radon.data.loaders import build_eval_loader  # noqa: E402
from radon.exceptions import RadonError  # noqa: E402
from radon.explain import build_inlier_profile, explain_samples  # noqa: E402
from radon.models import RadonAutoencoder  # noqa: E402

DEVICE = torch.device("cpu")

FEATURE_NAMES = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "protocol_type_tcp",
    "protocol_type_udp",
    "protocol_type_icmp",
    "flag_SF",
    "flag_S0",
]


@pytest.fixture
def training_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Inliers use tcp/SF with low numeric values; outliers use icmp/S0."""
    generator = torch.Generator().manual_seed(0)

    inliers = torch.zeros(200, 8)
    inliers[:, :3] = 0.1 + 0.05 * torch.rand(200, 3, generator=generator)
    inliers[:, 3] = 1.0  # protocol_type_tcp
    inliers[:, 6] = 1.0  # flag_SF

    outliers = torch.zeros(20, 8)
    outliers[:, :3] = 0.85 + 0.1 * torch.rand(20, 3, generator=generator)
    outliers[:, 5] = 1.0  # protocol_type_icmp
    outliers[:, 7] = 1.0  # flag_S0

    x = torch.cat([inliers, outliers]).clamp(0.0, 1.0)
    y = torch.cat([torch.zeros(200), torch.ones(20)]).long()
    return x, y


@pytest.fixture
def model() -> RadonAutoencoder:
    torch.manual_seed(0)
    return RadonAutoencoder(input_dim=8, hidden_dim=6, latent_dim=3)


class TestInlierProfile:
    def test_identifies_onehot_groups(self, training_data) -> None:
        x, y = training_data
        profile = build_inlier_profile(x, y, FEATURE_NAMES)
        assert set(profile.onehot_groups) == {"protocol_type", "flag"}
        assert profile.onehot_groups["protocol_type"] == (3, 4, 5)

    def test_numeric_indices_exclude_categoricals(self, training_data) -> None:
        x, y = training_data
        profile = build_inlier_profile(x, y, FEATURE_NAMES)
        assert profile.numeric_indices == (0, 1, 2)

    def test_group_mode_is_the_inlier_majority(self, training_data) -> None:
        """Inliers are all tcp/SF, so those must be the baselines."""
        x, y = training_data
        profile = build_inlier_profile(x, y, FEATURE_NAMES)
        assert profile.group_modes["protocol_type"] == 3  # tcp
        assert profile.group_modes["flag"] == 6  # SF

    def test_quantiles_come_from_inliers_only(self, training_data) -> None:
        x, y = training_data
        profile = build_inlier_profile(x, y, FEATURE_NAMES)
        # Outlier numeric values sit near 0.9 and must not widen the range.
        assert float(profile.quantile_high[0]) < 0.3

    def test_name_length_mismatch_raises(self, training_data) -> None:
        x, y = training_data
        with pytest.raises(RadonError, match="feature name"):
            build_inlier_profile(x, y, FEATURE_NAMES[:4])

    def test_no_inliers_raises(self, training_data) -> None:
        x, _ = training_data
        with pytest.raises(RadonError, match="no inliers"):
            build_inlier_profile(x, torch.ones(x.shape[0]).long(), FEATURE_NAMES)


class TestExplainSamples:
    def test_produces_one_explanation_per_selection(self, training_data, model) -> None:
        x, y = training_data
        profile = build_inlier_profile(x, y, FEATURE_NAMES)
        loader = build_eval_loader(x, y, batch_size=32)

        explanations = explain_samples(
            model,
            loader,
            profile,
            DEVICE,
            n_anomalies=1,
            n_normals=2,
            include_borderline=False,
        )
        assert len(explanations) == 3
        assert explanations[0].tag == "top anomaly"

    def test_selection_picks_the_extremes_within_each_class(self, training_data, model) -> None:
        """The anomaly is the highest-scoring outlier, the normal the lowest inlier.

        Note this says nothing about the two relative to each other: on an
        untrained model the classes are not separated, so the "top anomaly" can
        legitimately score below the "clear normal".
        """
        from radon.evaluation import score_split

        x, y = training_data
        profile = build_inlier_profile(x, y, FEATURE_NAMES)
        loader = build_eval_loader(x, y, batch_size=32)

        scored = score_split(model, loader, DEVICE)
        expected_anomaly = scored.scores[scored.labels == 1].max()
        expected_normal = scored.scores[scored.labels == 0].min()

        anomaly, normal = explain_samples(
            model,
            loader,
            profile,
            DEVICE,
            n_anomalies=1,
            n_normals=1,
            include_borderline=False,
        )
        assert anomaly.score == pytest.approx(expected_anomaly, abs=1e-6)
        assert normal.score == pytest.approx(expected_normal, abs=1e-6)

    def test_attributions_carry_ranges_and_deltas(self, training_data, model) -> None:
        x, y = training_data
        profile = build_inlier_profile(x, y, FEATURE_NAMES)
        loader = build_eval_loader(x, y, batch_size=32)

        explanation = explain_samples(
            model,
            loader,
            profile,
            DEVICE,
            n_anomalies=1,
            n_normals=1,
            include_borderline=False,
            min_abs_delta=0.0,
            min_clip_delta=0.0,
        )[0]

        assert explanation.numeric_violations
        for attribution in explanation.numeric_violations:
            assert attribution.quantile_low <= attribution.quantile_high
            assert attribution.name in FEATURE_NAMES

    def test_outlier_numeric_features_flagged_out_of_range(self, training_data, model) -> None:
        """The synthetic outliers sit far above the inlier quantile range."""
        x, y = training_data
        profile = build_inlier_profile(x, y, FEATURE_NAMES)
        loader = build_eval_loader(x, y, batch_size=32)

        anomaly = explain_samples(
            model,
            loader,
            profile,
            DEVICE,
            n_anomalies=1,
            n_normals=1,
            include_borderline=False,
            min_abs_delta=0.0,
            min_clip_delta=0.0,
        )[0]

        assert any(a.outside_range for a in anomaly.numeric_violations)

    def test_describe_is_readable(self, training_data, model) -> None:
        x, y = training_data
        profile = build_inlier_profile(x, y, FEATURE_NAMES)
        loader = build_eval_loader(x, y, batch_size=32)

        text = explain_samples(
            model,
            loader,
            profile,
            DEVICE,
            n_anomalies=1,
            n_normals=1,
            include_borderline=False,
        )[0].describe()
        assert "score=" in text
        assert "negative learning" in text

    def test_missing_class_raises(self, training_data, model) -> None:
        x, _ = training_data
        profile = build_inlier_profile(x, torch.zeros(x.shape[0]).long(), FEATURE_NAMES)
        loader = build_eval_loader(x, torch.zeros(x.shape[0]).long(), batch_size=32)

        with pytest.raises(RadonError, match="no outliers"):
            explain_samples(model, loader, profile, DEVICE)
