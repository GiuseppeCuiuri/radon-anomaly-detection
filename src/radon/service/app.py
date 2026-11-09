"""HTTP inference service for one trained RADON checkpoint.

Start it with::

    RADON_CHECKPOINT=outputs/checkpoints/radon_nsl_kdd_rep0.pth \\
        uvicorn radon.service.app:create_app --factory

Endpoints: ``GET /health``, ``POST /score`` and ``POST /explain``. Records are
feature vectors preprocessed like the training data: every value in ``[0, 1]``,
categorical columns one-hot encoded. The service does not preprocess raw data,
and it never logs feature values.

Configuration comes from the environment:

- ``RADON_CHECKPOINT`` (required): checkpoint written by ``radon-train``.
- ``RADON_MAX_BATCH_SIZE`` (default 1024): most records accepted by ``/score``.
- ``RADON_SCORE_THRESHOLD`` (optional): an operator-chosen anomaly-score
  threshold. Without it the service returns scores only.
- ``RADON_DEVICE`` (default ``cpu``): torch device to run on.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import torch
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from radon import __version__
from radon.config import LoggingConfig
from radon.evaluation.scoring import anomaly_scores
from radon.exceptions import RadonError
from radon.explain.nsl_kdd import explain_record, profile_from_dict
from radon.logging_utils import setup_logging
from radon.models.checkpoint import load_checkpoint

__all__ = ["ServiceConfigError", "Settings", "create_app"]

logger = logging.getLogger(__name__)

#: Attached to every response that compares scores with a threshold.
THRESHOLD_NOTE = (
    "Set by the operator through RADON_SCORE_THRESHOLD. It is not a calibrated decision "
    "threshold: above_threshold only says that a score exceeds it."
)

#: One feature value: finite and inside the [0, 1] range the model was trained on.
FeatureValue = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]


class ServiceConfigError(RadonError):
    """The service configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    """Service configuration.

    Attributes:
        checkpoint: Checkpoint file to serve.
        max_batch_size: Most records accepted in one ``/score`` request.
        score_threshold: Optional anomaly-score threshold set by the operator.
        device: Torch device to run on.
    """

    checkpoint: Path
    max_batch_size: int = 1024
    score_threshold: float | None = None
    device: str = "cpu"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        """Read the configuration from environment variables.

        Args:
            environ: Variables to read; defaults to ``os.environ``.

        Returns:
            The settings.

        Raises:
            ServiceConfigError: If ``RADON_CHECKPOINT`` is missing or a value
                does not parse.
        """
        env = os.environ if environ is None else environ

        checkpoint = env.get("RADON_CHECKPOINT", "").strip()
        if not checkpoint:
            raise ServiceConfigError(
                "RADON_CHECKPOINT is not set. Point it at a checkpoint written by radon-train, "
                "e.g. outputs/checkpoints/radon_nsl_kdd_rep0.pth."
            )

        raw_batch = env.get("RADON_MAX_BATCH_SIZE", "1024").strip()
        try:
            max_batch_size = int(raw_batch)
        except ValueError as exc:
            raise ServiceConfigError(
                f"RADON_MAX_BATCH_SIZE must be an integer, got {raw_batch!r}."
            ) from exc
        if max_batch_size < 1:
            raise ServiceConfigError(
                f"RADON_MAX_BATCH_SIZE must be at least 1, got {max_batch_size}."
            )

        score_threshold: float | None = None
        raw_threshold = env.get("RADON_SCORE_THRESHOLD", "").strip()
        if raw_threshold:
            try:
                score_threshold = float(raw_threshold)
            except ValueError as exc:
                raise ServiceConfigError(
                    f"RADON_SCORE_THRESHOLD must be a number, got {raw_threshold!r}."
                ) from exc

        device = env.get("RADON_DEVICE", "cpu").strip() or "cpu"
        return cls(
            checkpoint=Path(checkpoint),
            max_batch_size=max_batch_size,
            score_threshold=score_threshold,
            device=device,
        )


# --------------------------------------------------------------------------- #
# Request and response bodies
# --------------------------------------------------------------------------- #


class ScoreRequest(BaseModel):
    """A batch of records to score."""

    records: list[list[FeatureValue]] = Field(min_length=1)


class ExplainRequest(BaseModel):
    """One record to explain."""

    record: list[FeatureValue] = Field(min_length=1)


class ThresholdInfo(BaseModel):
    """The operator-chosen threshold, labelled as such."""

    value: float
    note: str = THRESHOLD_NOTE


class ScoreResponse(BaseModel):
    """One anomaly score per record, ``1 - cos(f(x), x)``: higher is more anomalous."""

    scores: list[float]
    threshold: ThresholdInfo | None = None
    above_threshold: list[bool] | None = None


class CategoricalAttributionOut(BaseModel):
    """Effect of resetting one categorical group to its most common inlier value."""

    model_config = ConfigDict(from_attributes=True)

    group: str
    observed: list[tuple[str, float]]
    baseline: str
    delta_score: float
    mismatch: bool


class NumericAttributionOut(BaseModel):
    """Effect of clipping one numeric feature into the inlier range."""

    model_config = ConfigDict(from_attributes=True)

    index: int
    name: str
    value: float
    reconstruction: float
    quantile_low: float
    quantile_high: float
    outside_range: bool
    delta_score: float
    prefers_inverted_target: bool


class ExplainResponse(BaseModel):
    """Per-feature attribution for one record.

    A negative ``delta_score`` means that normalising the feature lowered the
    anomaly score, so the feature was driving the anomaly.
    """

    score: float
    threshold: ThresholdInfo | None = None
    above_threshold: bool | None = None
    categorical: list[CategoricalAttributionOut]
    numeric: list[NumericAttributionOut]


class HealthResponse(BaseModel):
    """What is being served."""

    status: str
    model_loaded: bool
    architecture: dict[str, int]
    checkpoint_radon_version: str
    trained_on: str | None
    device: str
    max_batch_size: int
    score_threshold: float | None
    explain_available: bool


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #


def create_app(settings: Settings | None = None) -> FastAPI:
    """Load the checkpoint and build the application.

    Args:
        settings: Configuration. When omitted, as with
            ``uvicorn radon.service.app:create_app --factory``, the process is a
            standalone server: console logging is set up and the settings are
            read from the environment.

    Returns:
        The FastAPI application.

    Raises:
        ServiceConfigError: If the configuration is invalid.
        CheckpointError: If the checkpoint cannot be loaded.
    """
    if settings is None:
        # Nothing else configures the radon logger under uvicorn; without this the
        # service's own log lines would be dropped.
        setup_logging(LoggingConfig(file=None))
        settings = Settings.from_env()

    try:
        device = torch.device(settings.device)
    except RuntimeError as exc:
        raise ServiceConfigError(f"RADON_DEVICE: unknown device {settings.device!r}.") from exc

    loaded = load_checkpoint(settings.checkpoint, device=device)
    model = loaded.model.eval()
    input_dim = model.input_dim
    raw_dataset = loaded.extra.get("dataset")
    trained_on = str(raw_dataset) if raw_dataset is not None else None

    raw_profile = loaded.extra.get("inlier_profile")
    profile = profile_from_dict(raw_profile) if raw_profile is not None else None
    if profile is not None and len(profile.feature_names) != input_dim:
        raise ServiceConfigError(
            f"'{settings.checkpoint}': the inlier profile has {len(profile.feature_names)} "
            f"feature(s) but the model expects {input_dim}."
        )

    logger.info(
        "Serving '%s' (%s) on %s; explain %s.",
        settings.checkpoint,
        model.extra_repr(),
        device,
        "available" if profile is not None else "unavailable",
    )

    app = FastAPI(title="RADON inference service", version=__version__)

    @app.middleware("http")
    async def log_requests(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Method, path, status and duration only: the body holds feature values.
        start = time.perf_counter()
        response = await call_next(request)
        logger.info(
            "%s %s -> %d in %.1f ms",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - start) * 1000.0,
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default handler echoes every offending input back. A NaN cannot be
        # written as JSON, so that turned a 422 into a 500; the location and the
        # message are enough to fix the request.
        detail = [
            {"loc": list(error["loc"]), "msg": error["msg"], "type": error["type"]}
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": detail})

    def threshold_info() -> ThresholdInfo | None:
        if settings.score_threshold is None:
            return None
        return ThresholdInfo(value=settings.score_threshold)

    def check_width(record: list[float], position: str) -> None:
        if len(record) != input_dim:
            raise HTTPException(
                status_code=422,
                detail=f"{position} has {len(record)} feature(s); the model expects {input_dim}.",
            )

    @app.get("/health")
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_loaded=True,
            architecture=model.architecture,
            checkpoint_radon_version=loaded.radon_version,
            trained_on=trained_on,
            device=str(device),
            max_batch_size=settings.max_batch_size,
            score_threshold=settings.score_threshold,
            explain_available=profile is not None,
        )

    @app.post("/score")
    def score(request: ScoreRequest) -> ScoreResponse:
        n_records = len(request.records)
        if n_records > settings.max_batch_size:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"{n_records} records sent; the maximum batch size is "
                    f"{settings.max_batch_size} (RADON_MAX_BATCH_SIZE)."
                ),
            )
        for index, record in enumerate(request.records):
            check_width(record, f"Record {index}")

        with torch.inference_mode():
            features = torch.tensor(request.records, dtype=torch.float32, device=device)
            scores: list[float] = anomaly_scores(model, features).cpu().tolist()

        logger.info("Scored %d record(s).", n_records)
        threshold = threshold_info()
        return ScoreResponse(
            scores=scores,
            threshold=threshold,
            above_threshold=None if threshold is None else [s > threshold.value for s in scores],
        )

    @app.post("/explain")
    def explain(request: ExplainRequest) -> ExplainResponse:
        if profile is None:
            if trained_on == "mnist":
                reason = "this checkpoint was trained on 'mnist'"
            else:
                reason = (
                    "this checkpoint carries no inlier profile; retrain it with this version "
                    "of radon-train"
                )
            raise HTTPException(
                status_code=409,
                detail=f"Feature attribution is only available for tabular checkpoints: {reason}.",
            )
        check_width(request.record, "The record")

        record = torch.tensor(request.record, dtype=torch.float32)
        explanation = explain_record(model, record, profile, device)

        threshold = threshold_info()
        return ExplainResponse(
            score=explanation.score,
            threshold=threshold,
            above_threshold=None if threshold is None else explanation.score > threshold.value,
            categorical=[
                CategoricalAttributionOut.model_validate(item) for item in explanation.categorical
            ],
            numeric=[
                NumericAttributionOut.model_validate(item) for item in explanation.numeric_causal
            ],
        )

    return app
