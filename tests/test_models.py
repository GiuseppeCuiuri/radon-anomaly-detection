"""Tests for the RADON autoencoder and its checkpoints."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from radon.config import ModelConfig  # noqa: E402
from radon.exceptions import CheckpointError, ConfigError  # noqa: E402
from radon.models import (  # noqa: E402
    RadonAutoencoder,
    build_model,
    load_checkpoint,
    save_checkpoint,
)


@pytest.fixture
def model() -> RadonAutoencoder:
    torch.manual_seed(0)
    return RadonAutoencoder(input_dim=12, hidden_dim=8, latent_dim=4)


class TestArchitecture:
    def test_layer_shapes_match_the_reference(self, model: RadonAutoencoder) -> None:
        assert model.fc1.weight.shape == (8, 12)
        assert model.fc2.weight.shape == (4, 8)
        assert model.fc3.weight.shape == (8, 4)
        assert model.fc4.weight.shape == (12, 8)

    def test_forward_preserves_shape(self, model: RadonAutoencoder) -> None:
        out = model(torch.rand(5, 12))
        assert out.shape == (5, 12)

    def test_output_lies_in_the_unit_interval(self, model: RadonAutoencoder) -> None:
        """The sigmoid output is why inputs must be scaled to [0, 1]."""
        with torch.no_grad():
            out = model(torch.rand(64, 12) * 10 - 5)
        assert float(out.min()) > 0.0
        assert float(out.max()) < 1.0

    def test_encode_then_decode_equals_forward(self, model: RadonAutoencoder) -> None:
        x = torch.rand(5, 12)
        assert torch.equal(model.decode(model.encode(x)), model(x))

    def test_latent_shape(self, model: RadonAutoencoder) -> None:
        assert model.encode(torch.rand(7, 12)).shape == (7, 4)

    def test_encoder_output_is_non_negative(self, model: RadonAutoencoder) -> None:
        """The bottleneck sits behind a ReLU, exactly as in the original."""
        with torch.no_grad():
            latent = model.encode(torch.rand(32, 12))
        assert float(latent.min()) >= 0.0

    def test_parameter_count(self, model: RadonAutoencoder) -> None:
        # (12*8+8) + (8*4+4) + (4*8+8) + (8*12+12) = 104 + 36 + 40 + 108
        assert model.num_parameters() == 288

    def test_gradients_flow(self, model: RadonAutoencoder) -> None:
        model(torch.rand(4, 12)).sum().backward()
        assert all(p.grad is not None for p in model.parameters())


class TestInputValidation:
    def test_wrong_feature_count_raises(self, model: RadonAutoencoder) -> None:
        with pytest.raises(ValueError, match="expected 12 features, got 7"):
            model(torch.rand(5, 7))

    def test_non_2d_input_raises(self, model: RadonAutoencoder) -> None:
        """The original reshaped at each call site, hiding shape mistakes."""
        with pytest.raises(ValueError, match="2-D batch"):
            model(torch.rand(5, 1, 12))

    @pytest.mark.parametrize("dims", [(0, 8, 4), (12, 0, 4), (12, 8, 0), (-1, 8, 4)])
    def test_non_positive_dimensions_raise(self, dims: tuple[int, int, int]) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            RadonAutoencoder(*dims)


class TestBuildModel:
    def test_uses_the_configured_widths(self) -> None:
        built = build_model(ModelConfig(hidden_dim=64, latent_dim=32), input_dim=122)
        assert (built.input_dim, built.hidden_dim, built.latent_dim) == (122, 64, 32)

    def test_warns_when_the_bottleneck_does_not_compress(self, caplog) -> None:
        """latent_dim >= input_dim lets the network learn the identity map."""
        with caplog.at_level(logging.WARNING, logger="radon.models.autoencoder"):
            build_model(ModelConfig(hidden_dim=64, latent_dim=32), input_dim=16)
        assert "identity map" in caplog.text

    def test_latent_not_smaller_than_hidden_is_rejected_at_config_time(self) -> None:
        with pytest.raises(ConfigError, match="bottleneck"):
            ModelConfig(hidden_dim=32, latent_dim=32)


class TestCheckpoint:
    def test_round_trip_preserves_weights(self, model: RadonAutoencoder, tmp_path: Path) -> None:
        save_checkpoint(model, tmp_path / "m.pth")
        restored = load_checkpoint(tmp_path / "m.pth").model

        x = torch.rand(6, 12)
        assert torch.equal(model(x), restored(x))

    def test_architecture_is_recovered_without_being_supplied(
        self, model: RadonAutoencoder, tmp_path: Path
    ) -> None:
        """A bare state_dict could not be reloaded without knowing the widths."""
        save_checkpoint(model, tmp_path / "m.pth")
        restored = load_checkpoint(tmp_path / "m.pth").model
        assert restored.architecture == {"input_dim": 12, "hidden_dim": 8, "latent_dim": 4}

    def test_extra_payload_round_trips(self, model: RadonAutoencoder, tmp_path: Path) -> None:
        extra = {"threshold_l": 0.83, "threshold_s": 0.41, "epoch": 57}
        save_checkpoint(model, tmp_path / "m.pth", extra=extra)
        assert load_checkpoint(tmp_path / "m.pth").extra == extra

    def test_parent_directories_are_created(self, model: RadonAutoencoder, tmp_path: Path) -> None:
        """The original torch.save call assumed its directory already existed."""
        target = tmp_path / "deep" / "nested" / "m.pth"
        save_checkpoint(model, target)
        assert target.exists()

    def test_version_is_recorded(self, model: RadonAutoencoder, tmp_path: Path) -> None:
        save_checkpoint(model, tmp_path / "m.pth")
        assert load_checkpoint(tmp_path / "m.pth").radon_version == "0.1.0"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(CheckpointError, match="not found"):
            load_checkpoint(tmp_path / "absent.pth")

    def test_bare_state_dict_is_reported_helpfully(
        self, model: RadonAutoencoder, tmp_path: Path
    ) -> None:
        """Checkpoints produced by the original code land here."""
        path = tmp_path / "legacy.pth"
        torch.save(model.state_dict(), path)
        with pytest.raises(CheckpointError, match="missing the field"):
            load_checkpoint(path)

    def test_unknown_format_version_raises(self, model: RadonAutoencoder, tmp_path: Path) -> None:
        path = tmp_path / "future.pth"
        torch.save(
            {
                "format": 99,
                "architecture": model.architecture,
                "state_dict": model.state_dict(),
            },
            path,
        )
        with pytest.raises(CheckpointError, match="format 99"):
            load_checkpoint(path)

    def test_mismatched_weights_are_reported(self, model: RadonAutoencoder, tmp_path: Path) -> None:
        path = tmp_path / "bad.pth"
        torch.save(
            {
                "format": 1,
                "architecture": {"input_dim": 99, "hidden_dim": 8, "latent_dim": 4},
                "state_dict": model.state_dict(),
            },
            path,
        )
        with pytest.raises(CheckpointError, match="do not match"):
            load_checkpoint(path)

    def test_loads_onto_the_requested_device(self, model: RadonAutoencoder, tmp_path: Path) -> None:
        save_checkpoint(model, tmp_path / "m.pth")
        restored = load_checkpoint(tmp_path / "m.pth", device="cpu").model
        assert next(restored.parameters()).device.type == "cpu"
