"""Tests for the HTTP inference service.

Every test serves a tiny checkpoint written into ``tmp_path``: no network, no
dataset downloads.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from radon.evaluation.scoring import anomaly_scores  # noqa: E402
from radon.exceptions import CheckpointError  # noqa: E402
from radon.explain import build_inlier_profile, profile_to_dict  # noqa: E402
from radon.models import RadonAutoencoder, save_checkpoint  # noqa: E402
from radon.service.app import ServiceConfigError, Settings, create_app  # noqa: E402

FEATURE_NAMES = (
    "duration",
    "src_bytes",
    "protocol_type_tcp",
    "protocol_type_udp",
    "flag_SF",
    "flag_REJ",
)
WIDTH = len(FEATURE_NAMES)
NORMAL = [0.1, 0.2, 1.0, 0.0, 1.0, 0.0]
UNUSUAL = [0.9, 0.95, 0.0, 1.0, 0.0, 1.0]


@pytest.fixture
def model() -> RadonAutoencoder:
    torch.manual_seed(0)
    return RadonAutoencoder(input_dim=WIDTH, hidden_dim=4, latent_dim=2)


@pytest.fixture
def tabular_checkpoint(tmp_path: Path, model: RadonAutoencoder) -> Path:
    generator = torch.Generator().manual_seed(0)
    x = torch.tensor([NORMAL] * 40) + 0.05 * torch.rand(40, WIDTH, generator=generator)
    x = x.clamp(0.0, 1.0)
    y = torch.zeros(40, dtype=torch.long)
    profile = build_inlier_profile(x, y, FEATURE_NAMES)
    return save_checkpoint(
        model,
        tmp_path / "tabular.pth",
        extra={"dataset": "nsl_kdd", "inlier_profile": profile_to_dict(profile)},
    )


@pytest.fixture
def image_checkpoint(tmp_path: Path, model: RadonAutoencoder) -> Path:
    return save_checkpoint(model, tmp_path / "image.pth", extra={"dataset": "mnist"})


def client_for(checkpoint: Path, **settings: object) -> TestClient:
    return TestClient(create_app(Settings(checkpoint=checkpoint, **settings)))  # type: ignore[arg-type]


class TestSettings:
    def test_checkpoint_is_required(self) -> None:
        with pytest.raises(ServiceConfigError, match="RADON_CHECKPOINT"):
            Settings.from_env({})

    def test_reads_every_variable(self) -> None:
        settings = Settings.from_env(
            {
                "RADON_CHECKPOINT": "model.pth",
                "RADON_MAX_BATCH_SIZE": "8",
                "RADON_SCORE_THRESHOLD": "0.3",
                "RADON_DEVICE": "cpu",
            }
        )
        assert settings == Settings(
            checkpoint=Path("model.pth"), max_batch_size=8, score_threshold=0.3, device="cpu"
        )

    def test_defaults(self) -> None:
        settings = Settings.from_env({"RADON_CHECKPOINT": "model.pth"})
        assert settings.max_batch_size == 1024
        assert settings.score_threshold is None
        assert settings.device == "cpu"

    @pytest.mark.parametrize("value", ["zero", "0", "-3"])
    def test_bad_batch_size_raises(self, value: str) -> None:
        with pytest.raises(ServiceConfigError, match="RADON_MAX_BATCH_SIZE"):
            Settings.from_env({"RADON_CHECKPOINT": "model.pth", "RADON_MAX_BATCH_SIZE": value})

    def test_bad_threshold_raises(self) -> None:
        with pytest.raises(ServiceConfigError, match="RADON_SCORE_THRESHOLD"):
            Settings.from_env({"RADON_CHECKPOINT": "m.pth", "RADON_SCORE_THRESHOLD": "high"})


class TestStartup:
    def test_missing_checkpoint_fails_clearly(self, tmp_path: Path) -> None:
        with pytest.raises(CheckpointError, match="not found"):
            create_app(Settings(checkpoint=tmp_path / "missing.pth"))

    def test_unknown_device_fails_clearly(self, tabular_checkpoint: Path) -> None:
        with pytest.raises(ServiceConfigError, match="RADON_DEVICE"):
            create_app(Settings(checkpoint=tabular_checkpoint, device="quantum"))


class TestHealth:
    def test_describes_the_served_model(self, tabular_checkpoint: Path) -> None:
        body = client_for(tabular_checkpoint).get("/health").json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["architecture"] == {"input_dim": WIDTH, "hidden_dim": 4, "latent_dim": 2}
        assert body["device"] == "cpu"
        assert body["trained_on"] == "nsl_kdd"
        assert body["explain_available"] is True
        assert body["score_threshold"] is None

    def test_image_checkpoint_cannot_explain(self, image_checkpoint: Path) -> None:
        body = client_for(image_checkpoint).get("/health").json()
        assert body["explain_available"] is False


class TestScore:
    def test_scores_match_the_package(self, tabular_checkpoint: Path, model) -> None:
        client = client_for(tabular_checkpoint)
        response = client.post("/score", json={"records": [NORMAL, UNUSUAL]})

        assert response.status_code == 200
        body = response.json()
        with torch.no_grad():
            expected = anomaly_scores(model, torch.tensor([NORMAL, UNUSUAL])).tolist()
        assert body["scores"] == pytest.approx(expected, abs=1e-6)

    def test_no_label_without_a_threshold(self, tabular_checkpoint: Path) -> None:
        body = client_for(tabular_checkpoint).post("/score", json={"records": [NORMAL]}).json()
        assert body["threshold"] is None
        assert body["above_threshold"] is None

    def test_configured_threshold_is_labelled_as_such(self, tabular_checkpoint: Path) -> None:
        client = client_for(tabular_checkpoint, score_threshold=0.0)
        body = client.post("/score", json={"records": [NORMAL, UNUSUAL]}).json()

        assert body["threshold"]["value"] == 0.0
        assert "not a calibrated decision threshold" in body["threshold"]["note"]
        assert body["above_threshold"] == [s > 0.0 for s in body["scores"]]

    def test_wrong_width_is_rejected(self, tabular_checkpoint: Path) -> None:
        response = client_for(tabular_checkpoint).post(
            "/score", json={"records": [NORMAL, NORMAL[:-1]]}
        )
        assert response.status_code == 422
        assert response.json()["detail"] == "Record 1 has 5 feature(s); the model expects 6."

    @pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
    def test_non_finite_values_are_rejected(self, tabular_checkpoint: Path, token: str) -> None:
        # Python's JSON parser accepts these tokens, so they reach validation as floats.
        body = f'{{"records": [[{token}, 0.2, 1.0, 0.0, 1.0, 0.0]]}}'
        response = client_for(tabular_checkpoint).post(
            "/score", content=body, headers={"content-type": "application/json"}
        )
        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["body", "records", 0, 0]

    @pytest.mark.parametrize("value", [-0.01, 1.01])
    def test_out_of_range_values_are_rejected(self, tabular_checkpoint: Path, value: float) -> None:
        record = [value, *NORMAL[1:]]
        response = client_for(tabular_checkpoint).post("/score", json={"records": [record]})
        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["body", "records", 0, 0]

    def test_empty_batch_is_rejected(self, tabular_checkpoint: Path) -> None:
        response = client_for(tabular_checkpoint).post("/score", json={"records": []})
        assert response.status_code == 422

    def test_batch_above_the_limit_is_rejected(self, tabular_checkpoint: Path) -> None:
        client = client_for(tabular_checkpoint, max_batch_size=2)
        response = client.post("/score", json={"records": [NORMAL] * 3})
        assert response.status_code == 413
        assert "maximum batch size is 2" in response.json()["detail"]


class TestExplain:
    def test_attribution_for_one_record(self, tabular_checkpoint: Path) -> None:
        client = client_for(tabular_checkpoint)
        response = client.post("/explain", json={"record": UNUSUAL})

        assert response.status_code == 200
        body = response.json()
        score = client.post("/score", json={"records": [UNUSUAL]}).json()["scores"][0]
        assert body["score"] == pytest.approx(score, abs=1e-6)
        assert body["above_threshold"] is None
        for item in body["categorical"]:
            assert set(item) == {"group", "observed", "baseline", "delta_score", "mismatch"}
        for item in body["numeric"]:
            assert item["name"] in FEATURE_NAMES

    def test_image_checkpoint_returns_409(self, image_checkpoint: Path) -> None:
        response = client_for(image_checkpoint).post("/explain", json={"record": NORMAL})
        assert response.status_code == 409
        assert "trained on 'mnist'" in response.json()["detail"]

    def test_checkpoint_without_profile_returns_409(self, tmp_path: Path, model) -> None:
        path = save_checkpoint(model, tmp_path / "old.pth", extra={"dataset": "nsl_kdd"})
        response = client_for(path).post("/explain", json={"record": NORMAL})
        assert response.status_code == 409
        assert "retrain" in response.json()["detail"]

    def test_wrong_width_is_rejected(self, tabular_checkpoint: Path) -> None:
        response = client_for(tabular_checkpoint).post("/explain", json={"record": NORMAL[:3]})
        assert response.status_code == 422
        assert response.json()["detail"] == "The record has 3 feature(s); the model expects 6."


class TestLogging:
    def test_requests_are_logged_without_feature_values(
        self, tabular_checkpoint: Path, caplog
    ) -> None:
        marker = 0.123456789
        client = client_for(tabular_checkpoint)
        with caplog.at_level(logging.INFO, logger="radon"):
            client.post("/score", json={"records": [[marker, *NORMAL[1:]]]})
            client.post("/explain", json={"record": [marker, *NORMAL[1:]]})

        assert "POST /score -> 200" in caplog.text
        assert "POST /explain -> 200" in caplog.text
        assert str(marker) not in caplog.text
        assert "0.1234" not in caplog.text
