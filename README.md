# RADON — Robust Autoencoder for Unsupervised Anomaly Detection

[![CI](https://github.com/GiuseppeCuiuri/radon-anomaly-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/GiuseppeCuiuri/radon-anomaly-detection/actions/workflows/ci.yml)

An autoencoder trained to reconstruct inliers while a *negative learning* objective pushes
suspected outliers towards their feature-wise inverse, so anomalies end up badly reconstructed
on purpose. Periodic **projection** steps re-partition the training set by thresholding the
distribution of cosine reconstruction scores.

Two protocols are implemented:

- **MNIST** — one digit is normal, the training set is lightly contaminated with other digits.
- **NSL-KDD** — network intrusion detection, with outliers subsampled to a target contamination
  rate, plus feature attribution for the anomalies it flags.

Also: [CI](.github/workflows/ci.yml) on Python 3.10–3.12 · [Docker images](#docker) ·
an [HTTP inference service](#inference-service), documented in [docs/SERVICE.md](docs/SERVICE.md) ·
[serving benchmarks](#serving-performance).

---

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

The pip upgrade is not optional. Editable installs from a `pyproject.toml` need
PEP 660 support, which arrived in pip 21.3; Python 3.10.0 still ships 21.2.3, and
that version fails with a misleading `File "setup.py" or "setup.cfg" not found`.

```bash
python -m radon.cli.train --config configs/mnist.yaml
```

MNIST downloads itself. NSL-KDD expects four pre-processed CSV files in `data/NSL_KDD/`
(`X_train`, `Y_train`, `X_test`, `Y_test`), features already scaled to `[0, 1]` with
categorical columns one-hot encoded.

Everything a run produces lands under `outputs/`: checkpoints, per-epoch and per-projection
CSVs, figures, logs, and the effective configuration that generated them.

## Commands

```bash
# Train and evaluate, five independent repetitions
python -m radon.cli.train --config configs/nsl_kdd.yaml --repetitions 5

# Override anything from the command line
python -m radon.cli.train --config configs/mnist.yaml --set training.epochs=50 --set seed=7

# Explain what the trained model reacted to
python -m radon.cli.explain --config configs/nsl_kdd.yaml
```

`notebooks/01_radon_demo.ipynb` walks the same pipeline interactively. It contains no logic of
its own — it calls the package.

## Docker

A CPU image with the package and the configs. Datasets and results stay on the host, mounted
into `/app/data` and `/app/outputs`:

```bash
docker build -t radon:cpu .

# One epoch on MNIST: downloads into ./data, writes everything to ./outputs
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/outputs:/app/outputs" \
    radon:cpu --config configs/mnist.yaml --set training.epochs=1 --set training.warmup_epochs=1

# The explain command
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/outputs:/app/outputs" \
    --entrypoint python radon:cpu -m radon.cli.explain --config configs/mnist.yaml
```

In PowerShell, write `${PWD}` instead of `$(pwd)` and keep each command on one line. Without
arguments the image prints the `radon-train` help. The container runs as an unprivileged user
(uid 1000), so on Linux the mounted folders must be writable by that user.

## Inference service

An optional HTTP service that serves one trained checkpoint. [docs/SERVICE.md](docs/SERVICE.md)
walks through training a small model, the requests and their responses, and the limits.

```bash
pip install -e ".[service]"
RADON_CHECKPOINT=outputs/checkpoints/radon_mnist_rep0.pth \
    uvicorn radon.service.app:create_app --factory --port 8000
```

| Endpoint | Body | Returns |
| --- | --- | --- |
| `GET /health` | | the served architecture, the device, whether `/explain` is available |
| `POST /score` | `{"records": [[...], ...]}` | one anomaly score per record, `1 - cos(f(x), x)`: higher is more anomalous |
| `POST /explain` | `{"record": [...]}` | the per-feature attribution of one record (tabular checkpoints only) |

Records must be preprocessed like the training data: `input_dim` values, each in `[0, 1]`.
Wrong lengths, NaN or out-of-range values return 422, and batches above `RADON_MAX_BATCH_SIZE`
(default 1024) return 413. The service returns scores, not anomaly/normal labels, because the
projection threshold is a poor decision threshold (see [Known caveats](#known-caveats)). If
`RADON_SCORE_THRESHOLD` is set, responses also say which scores exceed it and label it as an
operator choice. `RADON_DEVICE` selects the torch device (default `cpu`). Requests are logged
with method, path, status and duration, never with feature values.

`/explain` compares a record with the inlier profile of the training data, which `radon-train`
stores in checkpoints of tabular datasets. MNIST checkpoints, and tabular ones written by an
older version, answer 409.

In Docker:

```bash
docker build --target service -t radon:service .
docker run --rm -p 8000:8000 -v "$(pwd)/outputs:/app/outputs" \
    -e RADON_CHECKPOINT=outputs/checkpoints/radon_mnist_rep0.pth radon:service
```

## Serving performance

`POST /score` on the MNIST architecture (784 → 256 → 128), 200 timed requests per batch size
after 20 warm-up requests. Intel Core i7-10750H, 6 torch threads, 15.8 GB RAM, no GPU; Windows 10,
Python 3.12.7, torch 2.15.0+cpu; commit `9254efc`:

| Batch | p50 ms | p95 ms | p99 ms | Records/s | Peak RSS MB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6.55 [6.48, 6.60] | 7.27 [7.04, 7.52] | 8.07 [7.45, 8.45] | 152 [151, 153] | 436 |
| 8 | 9.12 [9.08, 9.16] | 9.66 [9.55, 9.73] | 9.89 [9.68, 10.20] | 879 [875, 883] | 576 |
| 32 | 17.01 [16.95, 17.07] | 17.67 [17.52, 17.87] | 18.40 [17.79, 97.94] | 1869 [1849, 1881] | 728 |
| 128 | 50.53 [50.14, 50.70] | 56.94 [56.68, 57.24] | 58.12 [57.16, 58.67] | 2496 [2477, 2513] | 1169 |

Brackets are bootstrap 95% confidence intervals. The wide p99 interval at batch 32 comes from a
single slow request. Peak RSS is the process peak after each batch size, so it accumulates.
Raw data, table and figure are in [`benchmarks/results/`](benchmarks/results/). To reproduce:

```bash
python benchmarks/serving.py                  # about 30 s on the machine above
python benchmarks/serving.py --report-only    # rebuild the table and figure from the CSV
```

These numbers come from one laptop, with synthetic requests sent in-process through FastAPI's
test client, so they include JSON parsing, validation and inference but no HTTP server, no
network and no concurrent clients.

## Configuration

Three layers, each overriding the previous: `configs/default.yaml`, then a dataset file, then
`--set key.path=value`. Values parse as YAML, so types come out right:

```bash
--set training.epochs=50                    # int
--set data.mnist.anomaly_digits=[1,5,7]     # list
--set data.mnist.n_test_anomaly=null        # None, meaning "keep all"
--set compat.legacy_auprc_trapezoid=true    # bool
```

Unknown keys are a **hard error**. A typo in a hyper-parameter name silently invalidates an
experiment, so it fails at load time instead:

```
ConfigError: Unknown key(s) ['normal_digits'] in data.mnist.
Valid keys: ['anomaly_digits', 'download', 'image_shape', 'n_anomaly', ...]
```

## Reproducing the original results

This code is a refactor of a working implementation, preserved in [`legacy/`](legacy/). The
refactor is verified by differential testing against it: on both real datasets the data
pipeline, the thresholds, the per-epoch losses and the final weights come out **bit-identical**.

Four defects were fixed whose correction changes published numbers. Three are switchable:

| Flag | Original behaviour | Why it was wrong |
|---|---|---|
| `compat.legacy_warmup_no_shuffle` | Warm-up loader used `shuffle=False` | Batch order and the `L`/`S` pairing were frozen for every epoch |
| `compat.legacy_auprc_trapezoid` | `auc(recall, precision)` | Precision is not monotone in recall, so trapezoidal interpolation credits performance never achieved; `average_precision_score` is the standard estimator |
| `compat.legacy_early_stop_metric` | Inlier loss over the *current* `L` | `L` is rebuilt at every projection, so the metric is not comparable across projections — a shrinking `L` lowers the loss without the model improving |

```bash
python -m radon.cli.train --config configs/mnist.yaml \
  --set compat.legacy_warmup_no_shuffle=true \
  --set compat.legacy_auprc_trapezoid=true \
  --set compat.legacy_early_stop_metric=true
```

The fourth has no flag, because the original behaviour is simply incorrect: **the restored
decision threshold did not belong to the restored model.** `best_proj_snapshot` was written
once, at the first negative-learning projection, and never refreshed when a later projection
improved. The weights advanced; the thresholds did not. On MNIST that meant classifying with
epoch-16 thresholds using epoch-30 weights. It happens to be harmless whenever the best model
*is* the first negative-learning projection, which is why NSL-KDD was unaffected.

## Known caveats

Preserved deliberately — changing them would alter the experimental protocol rather than fix a
defect:

- **The projection threshold is a poor decision threshold.** It is tuned to select *confident*
  outliers for the training loss, so it is deliberately conservative. Reused as a classification
  rule it buys precision at a heavy cost in recall: on a short MNIST run, precision 1.00 and
  recall 0.18 where the curve's best F1 was 0.84. AUROC and AUPRC are threshold-free and
  unaffected; precision, recall and F1 are not.
- **MNIST train and test anomalies differ.** Training contamination comes only from
  `anomaly_digits` (`[0, 2, 3]`); test outliers come from *every* other digit. The test split
  measures generalisation to unseen anomaly types.
- **The MNIST test split is not balanced.** With `n_test_normal: null` and
  `n_test_anomaly: 265`, digit 4 gives roughly 79% / 21%. AUPRC has no fixed chance level, so
  the baseline is reported next to it — an AUPRC of 0.57 is worthless on a 57%-positive split.
- **Triangle thresholding degenerates on rising histograms.** When the score distribution has
  no descending tail the cut lands on the extreme bin and one side of the partition is nearly
  empty. The condition is detected and logged rather than silently accepted.

## Project layout

```
configs/            layered YAML configuration
src/radon/
  config.py         typed config tree, strict loading, CLI overrides
  logging_utils.py  logger setup — library code never prints
  seeding.py        RNG seeding with explicit generators
  device.py         device resolution
  data/             loading and anomaly-detection protocols
  models/           autoencoder, checkpoints that carry their own architecture
  features/         binary vs numeric column analysis
  training/         thresholding, projection, phases, early stopping, trainer, records
  evaluation/       anomaly scoring and metrics
  explain/          feature attribution by ablation
  viz/              plotting; returns figures, never shows or saves them
  cli/              entry points
legacy/             the original implementation, used as a differential test oracle
tests/              pytest suite
```

Four conventions worth knowing before reading the code:

- **Computation, logging and plotting are separate.** Functions return data; `viz/` turns data
  into figures; the entry point decides whether to show or save. Numbers stay testable and the
  pipeline runs headless.
- **Telemetry is structured, never parsed.** The trainer appends typed records. The original
  printed formatted lines, captured stdout, and re-parsed them with ten regular expressions to
  rebuild CSVs — which silently produced empty files whenever `verbose` was left at its default.
- **Validation raises, it does not assert.** Bare `assert` statements vanish under `python -O`,
  which is exactly the flag someone reaches for when training feels slow.
- **Checkpoints are self-describing.** They store the architecture alongside the weights, so a
  file can be reloaded without independently remembering three dimensions.

## Testing

```bash
pytest
pytest -m slow          # also downloads MNIST and exercises the real loader
ruff check src tests && mypy
```

The suite includes differential tests that import `legacy/` and require the refactored code to
reproduce it exactly. Delete `legacy/` and those tests skip; the rest still run.

## References

Najari, N. *RADON: Robust Autoencoder for Unsupervised Anomaly Detection.* The untouched
reference release is in [`legacy/reference/`](legacy/reference/); background papers are in
[`docs/`](docs/).

## License

MIT.
