# CPU images for RADON.
#
# Training and evaluation (the default target):
#   docker build -t radon:cpu .
#   docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/outputs:/app/outputs" \
#       radon:cpu --config configs/mnist.yaml
#
# Inference service:
#   docker build --target service -t radon:service .
#   docker run --rm -p 8000:8000 -v "$(pwd)/outputs:/app/outputs" \
#       -e RADON_CHECKPOINT=outputs/checkpoints/radon_nsl_kdd_rep0.pth radon:service

# --------------------------------------------------------------------------- #
# Builder: a virtual environment with CPU PyTorch and the package.
# --------------------------------------------------------------------------- #
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# The largest layer, kept separate so that code changes do not reinstall it.
RUN pip install --upgrade pip setuptools wheel \
 && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[service]"

# --------------------------------------------------------------------------- #
# Runtime base: the virtual environment and the configs, nothing else. The
# installed package resolves configs/, data/ and outputs/ against the working
# directory, /app.
# --------------------------------------------------------------------------- #
FROM python:3.12-slim AS runtime

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN useradd --create-home --uid 1000 radon

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY configs ./configs

# Mount points for the datasets and for everything a run produces.
RUN mkdir data outputs && chown radon:radon data outputs

USER radon

# --------------------------------------------------------------------------- #
# Inference service: docker build --target service
# --------------------------------------------------------------------------- #
FROM runtime AS service

EXPOSE 8000
ENTRYPOINT ["uvicorn", "radon.service.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

# --------------------------------------------------------------------------- #
# Training and evaluation: the default target, so it stays last.
# --------------------------------------------------------------------------- #
FROM runtime AS train

ENTRYPOINT ["python", "-m", "radon.cli.train"]
CMD ["--help"]
