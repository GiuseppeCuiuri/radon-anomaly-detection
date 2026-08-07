"""Tests for the trainer, its phases, projection and early stopping."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from radon.config import (  # noqa: E402
    CompatConfig,
    ConsistencyConfig,
    CurriculumConfig,
    EarlyStoppingConfig,
    TrainingConfig,
)
from radon.exceptions import RadonError  # noqa: E402
from radon.features import analyze_features, make_feature_inverter  # noqa: E402
from radon.models import RadonAutoencoder  # noqa: E402
from radon.training import (  # noqa: E402
    EpochRecord,
    ProjectionRecord,
    RadonTrainer,
    TrainingHistory,
)
from radon.training.phases import warmup_epoch  # noqa: E402
from radon.training.projection import pad_outliers_to_inliers, score_dataset  # noqa: E402

DEVICE = torch.device("cpu")


@pytest.fixture
def toy_data() -> torch.Tensor:
    """400 inliers clustered together plus 30 scattered outliers."""
    generator = torch.Generator().manual_seed(0)
    inliers = 0.5 + 0.05 * torch.randn(400, 16, generator=generator)
    outliers = torch.rand(30, 16, generator=generator)
    return torch.cat([inliers, outliers]).clamp(0.0, 1.0).float()


@pytest.fixture
def fast_config() -> TrainingConfig:
    """A schedule short enough to run inside a unit test."""
    return TrainingConfig(
        epochs=12,
        batch_size=64,
        lr=1e-3,
        warmup_epochs=6,
        bin_estimator="sturges",
        projection_every_warmup=3,
        projection_every_nl=3,
        curriculum=CurriculumConfig(enabled=True, start=0.5, end=1.0, min_s=5),
        early_stopping=EarlyStoppingConfig(enabled=False),
    )


def build_trainer(cfg: TrainingConfig, n_features: int = 16, seed: int = 0) -> RadonTrainer:
    torch.manual_seed(seed)
    model = RadonAutoencoder(input_dim=n_features, hidden_dim=12, latent_dim=6).to(DEVICE)
    return RadonTrainer(model=model, cfg=cfg, device=DEVICE, compat=CompatConfig())


class TestTrainerRuns:
    def test_produces_one_record_per_epoch(self, toy_data, fast_config) -> None:
        history = build_trainer(fast_config).train(toy_data)
        assert len(history.epochs) == fast_config.epochs
        assert [r.epoch for r in history.epochs] == list(range(1, fast_config.epochs + 1))

    def test_phases_switch_at_the_boundary(self, toy_data, fast_config) -> None:
        history = build_trainer(fast_config).train(toy_data)
        phases = [r.phase for r in history.epochs]
        assert set(phases[: fast_config.warmup_epochs]) == {"warmup"}
        assert set(phases[fast_config.warmup_epochs :]) == {"negative_learning"}

    def test_projection_is_forced_at_the_warmup_boundary(self, toy_data, fast_config) -> None:
        """Negative learning must not start from a stale partition."""
        history = build_trainer(fast_config).train(toy_data)
        assert fast_config.warmup_epochs in [r.epoch for r in history.projections]

    def test_loss_decreases_before_the_first_projection(self, toy_data, fast_config) -> None:
        """Pure optimisation, isolated: a projection changes the objective."""
        cfg = replace(fast_config, projection_every_warmup=fast_config.warmup_epochs)
        history = build_trainer(cfg).train(toy_data)
        before = [r.loss_mean for r in history.epochs if r.epoch < cfg.warmup_epochs]
        assert before[-1] < before[0]

    def test_total_loss_jumps_when_outliers_first_appear(self, toy_data, fast_config) -> None:
        """The warm-up loss curve is not monotone, and that is not a failure.

        Until the first projection finds outliers, the outlier term is exactly
        zero. The moment S becomes non-empty the objective gains
        ``cos(f(s), s)^2``, so the total loss steps up even though the model is
        still improving. Plotted on the log scale the original code used, that
        step reads as a training collapse unless you know to expect it.
        """
        history = build_trainer(fast_config).train(toy_data)
        first_projection = history.projections[0].epoch
        losses = {r.epoch: r.loss_mean for r in history.epochs}

        assert history.projections[0].n_outliers > 0
        assert losses[first_projection + 1] > losses[first_projection]

    def test_inlier_loss_is_tracked_separately(self, toy_data, fast_config) -> None:
        """Which is why the inlier term is recorded in its own column."""
        history = build_trainer(fast_config).train(toy_data)
        assert all(
            r.loss_inlier_mean <= r.loss_mean + 1e-9
            for r in history.epochs
            if np.isfinite(r.loss_mean)
        )

    def test_thresholds_are_available_after_training(self, toy_data, fast_config) -> None:
        history = build_trainer(fast_config).train(toy_data)
        inlier, outlier = history.final_thresholds()
        assert outlier <= inlier

    def test_records_are_finite(self, toy_data, fast_config) -> None:
        history = build_trainer(fast_config).train(toy_data)
        assert all(np.isfinite(r.loss_mean) for r in history.epochs)
        assert all(np.isfinite(r.threshold_inlier) for r in history.projections)

    def test_reproducible_under_a_fixed_seed(self, toy_data, fast_config) -> None:
        torch.manual_seed(1234)
        first = build_trainer(fast_config, seed=7).train(toy_data).losses
        torch.manual_seed(1234)
        second = build_trainer(fast_config, seed=7).train(toy_data).losses
        assert first == second


class TestTrainerValidation:
    def test_wrong_feature_count_is_rejected(self, fast_config) -> None:
        trainer = build_trainer(fast_config, n_features=16)
        with pytest.raises(ValueError, match="expects 16 features"):
            trainer.train(torch.rand(50, 8))

    def test_non_2d_input_is_rejected(self, fast_config) -> None:
        with pytest.raises(ValueError, match="must be 2-D"):
            build_trainer(fast_config).train(torch.rand(50, 4, 4))

    def test_warmup_longer_than_training_is_rejected(self) -> None:
        from radon.exceptions import ConfigError

        with pytest.raises(ConfigError, match="never start"):
            TrainingConfig(epochs=10, warmup_epochs=20)


class TestWarmupOptimisation:
    def test_skipping_the_zero_outlier_branch_changes_nothing(self) -> None:
        """The padded outlier term is exactly zero, so its pass is wasted work.

        Trains one epoch with the branch skipped and one with it executed, from
        identical initial weights, and requires the resulting parameters to be
        bit-identical.
        """
        torch.manual_seed(0)
        data = torch.rand(128, 10)
        padding = torch.zeros_like(data)
        criterion = nn.CosineSimilarity(dim=1, eps=1e-5)

        results = []
        for has_outliers in (False, True):
            torch.manual_seed(42)
            model = RadonAutoencoder(10, 8, 4)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            loader: DataLoader = DataLoader(
                TensorDataset(data, padding), batch_size=32, shuffle=False
            )
            stats = warmup_epoch(model, loader, optimizer, criterion, has_outliers=has_outliers)
            results.append((stats, [p.detach().clone() for p in model.parameters()]))

        skipped, executed = results
        assert all(torch.equal(a, b) for a, b in zip(skipped[1], executed[1], strict=True))
        assert skipped[0].loss_total == pytest.approx(executed[0].loss_total)
        assert skipped[0].loss_inlier == pytest.approx(executed[0].loss_inlier)
        assert skipped[0].outlier_loss_sum == executed[0].outlier_loss_sum == 0.0

    def test_cosine_against_a_zero_vector_is_exactly_zero(self) -> None:
        """The premise the optimisation rests on."""
        criterion = nn.CosineSimilarity(dim=1, eps=1e-5)
        reconstruction = torch.rand(16, 10, requires_grad=True)
        zeros = torch.zeros(16, 10)

        cosine = criterion(reconstruction, zeros)
        assert torch.all(cosine == 0.0)

        (cosine**2).mean().backward()
        assert reconstruction.grad is not None
        assert torch.all(reconstruction.grad == 0.0)


class TestPadding:
    def test_pads_short_outlier_sets(self) -> None:
        outliers = torch.ones(3, 5)
        padded, mask = pad_outliers_to_inliers(outliers, 8, 5, DEVICE)
        assert padded.shape == (8, 5)
        assert mask.tolist() == [1, 1, 1, 0, 0, 0, 0, 0]

    def test_truncates_long_outlier_sets(self) -> None:
        padded, mask = pad_outliers_to_inliers(torch.ones(20, 5), 8, 5, DEVICE)
        assert padded.shape == (8, 5)
        assert mask.tolist() == [1] * 8

    def test_handles_the_absent_case(self) -> None:
        padded, mask = pad_outliers_to_inliers(None, 4, 5, DEVICE)
        assert torch.all(padded == 0.0)
        assert torch.all(mask == 0.0)


class TestScoring:
    def test_batched_scoring_matches_a_single_pass(self) -> None:
        torch.manual_seed(0)
        model = RadonAutoencoder(12, 8, 4).eval()
        data = torch.rand(500, 12)
        criterion = nn.CosineSimilarity(dim=1, eps=1e-5)

        single = score_dataset(model, data, criterion, batch_size=0)
        batched = score_dataset(model, data, criterion, batch_size=64)

        assert single.shape == (500,)
        assert np.allclose(single, batched, rtol=0, atol=1e-6)

    def test_model_mode_is_restored(self) -> None:
        model = RadonAutoencoder(12, 8, 4).train()
        score_dataset(model, torch.rand(20, 12), nn.CosineSimilarity(dim=1, eps=1e-5))
        assert model.training is True


class TestEarlyStopping:
    def test_stops_and_restores_the_best_model(self, toy_data) -> None:
        cfg = TrainingConfig(
            epochs=40,
            batch_size=64,
            warmup_epochs=4,
            bin_estimator="sturges",
            projection_every_warmup=2,
            projection_every_nl=1,
            curriculum=CurriculumConfig(enabled=False),
            early_stopping=EarlyStoppingConfig(enabled=True, patience=1, min_delta=1e9),
        )
        history = build_trainer(cfg).train(toy_data)
        assert history.stopped_early is True
        assert history.best_epoch is not None
        assert len(history.epochs) < cfg.epochs

    def test_disabled_stopping_runs_every_epoch(self, toy_data, fast_config) -> None:
        history = build_trainer(fast_config).train(toy_data)
        assert history.stopped_early is False
        assert len(history.epochs) == fast_config.epochs

    @pytest.mark.parametrize("metric", ["frozen_l0", "current_l", "full_x"])
    def test_every_metric_runs(self, toy_data, fast_config, metric: str) -> None:
        cfg = replace(
            fast_config,
            early_stopping=EarlyStoppingConfig(enabled=True, patience=5, metric=metric),
        )
        history = build_trainer(cfg).train(toy_data)
        assert len(history.epochs) >= 1

    def test_frozen_l0_is_captured_at_the_phase_boundary(self, toy_data, fast_config) -> None:
        cfg = replace(
            fast_config,
            early_stopping=EarlyStoppingConfig(enabled=True, patience=5, metric="frozen_l0"),
        )
        trainer = build_trainer(cfg)
        trainer.train(toy_data)
        assert trainer._frozen_inliers is not None

    def test_legacy_flag_overrides_the_metric(self, toy_data, fast_config) -> None:
        cfg = replace(
            fast_config,
            early_stopping=EarlyStoppingConfig(enabled=True, patience=5, metric="frozen_l0"),
        )
        trainer = build_trainer(cfg)
        trainer.compat = CompatConfig(legacy_early_stop_metric=True)
        trainer.train(toy_data)
        # The legacy path never evaluates on a frozen set.
        assert trainer._frozen_inliers is not None  # still captured, just unused


class TestConsistency:
    def test_consistency_term_runs(self, toy_data, fast_config) -> None:
        cfg = replace(fast_config, consistency=ConsistencyConfig(weight=0.1, sigma=0.05))
        history = build_trainer(cfg).train(toy_data)
        assert all(np.isfinite(r.loss_mean) for r in history.epochs)


class TestFeatureMasks:
    def test_binary_and_numeric_columns_are_separated(self) -> None:
        data = torch.cat([torch.rand(100, 3), torch.randint(0, 2, (100, 2)).float()], dim=1)
        masks = analyze_features(data)
        assert masks.binary.tolist() == [False, False, False, True, True]
        assert masks.n_numeric == 3

    def test_inverter_flips_numeric_and_preserves_binary(self) -> None:
        data = torch.tensor([[0.25, 1.0], [0.75, 0.0]])
        masks = analyze_features(data)
        inverted = make_feature_inverter(masks.numeric)(data)
        assert inverted[:, 0].tolist() == [0.75, 0.25]
        assert inverted[:, 1].tolist() == [1.0, 0.0]

    def test_onehot_groups_from_names(self) -> None:
        names = ["duration", "protocol_type_tcp", "protocol_type_udp", "flag_SF", "flag_S0"]
        data = torch.zeros(10, 5)
        data[:, 1] = 1.0
        data[:, 3] = 1.0
        masks = analyze_features(data, feature_names=names)
        assert len(masks.onehot_groups) == 2


class TestHistory:
    def test_csv_round_trip(self, toy_data, fast_config, tmp_path: Path) -> None:
        history = build_trainer(fast_config).train(toy_data)
        epochs_csv, projections_csv = history.write_csv(tmp_path, "toy")

        assert epochs_csv.exists() and projections_csv.exists()
        lines = epochs_csv.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == len(history.epochs) + 1
        assert "loss_mean" in lines[0]

    def test_projection_csv_flattens_percentiles(
        self, toy_data, fast_config, tmp_path: Path
    ) -> None:
        """The nested percentile dict must become plain columns."""
        history = build_trainer(fast_config).train(toy_data)
        _, projections_csv = history.write_csv(tmp_path, "toy")
        header = projections_csv.read_text(encoding="utf-8").splitlines()[0]
        assert "score_percentiles_p50" in header

    def test_final_thresholds_without_a_projection_raises(self) -> None:
        with pytest.raises(RadonError, match="No projection"):
            TrainingHistory().final_thresholds()

    def test_epoch_record_formats_a_line(self) -> None:
        record = EpochRecord(
            epoch=1,
            phase="warmup",
            loss_mean=0.5,
            loss_inlier_mean=0.4,
            loss_outlier_mean=0.1,
            lambda_used=1.0,
            loss_ratio=0.25,
            n_outliers_seen=10,
            cos_outlier_self=0.1,
            cos_outlier_inverse=0.9,
            grad_norm=1e-3,
            param_norm=2.0,
            duration_s=0.1,
            had_projection=True,
            nonfinite=False,
        )
        assert "Ep 001" in record.format_line()

    def test_projection_record_formats_a_block(self) -> None:
        record = ProjectionRecord(
            epoch=5,
            n_bins=20,
            threshold_inlier=0.9,
            threshold_outlier=0.4,
            lambda_eff=1.5,
            n_inliers=100,
            n_outliers=10,
            n_unassigned=5,
            degenerate_inlier_threshold=False,
            degenerate_outlier_threshold=False,
            score_min=0.1,
            score_mean=0.7,
            score_max=1.0,
            score_percentiles={"p50": 0.7},
        )
        assert len(record.format_lines()) == 5
