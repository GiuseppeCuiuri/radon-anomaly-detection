# CPU image for training and evaluating RADON.
#
#   docker build -t radon:cpu .
#   docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/outputs:/app/outputs" \
#       radon:cpu --config configs/mnist.yaml

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
COPY pyproject.toml README.md ./
COPY src ./src
# Editable on purpose: radon resolves configs/, data/ and outputs/ against the
# directory above src/ (radon.config.PROJECT_ROOT), so the package has to stay in /app/src.
RUN pip install -e .

# --------------------------------------------------------------------------- #
# Runtime: the virtual environment, the source and the configs, nothing else.
# --------------------------------------------------------------------------- #
FROM python:3.12-slim

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN useradd --create-home --uid 1000 radon

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/src ./src
COPY configs ./configs

# Mount points for the datasets and for everything a run produces.
RUN mkdir data outputs && chown radon:radon data outputs

USER radon

ENTRYPOINT ["python", "-m", "radon.cli.train"]
CMD ["--help"]
