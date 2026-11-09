# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-09

First release: the RADON reference implementation refactored into an installable, tested
package, with CI, Docker images, an inference service and serving benchmarks.

### Added

- **Package and command line.** The `radon` package with strict layered YAML configuration,
  the `radon-train` and `radon-explain` commands, and checkpoints that store their own
  architecture. Differential tests check the refactor against the reference implementation in
  `legacy/`; three compatibility flags reproduce the original behaviour where a defect was
  fixed (see the README).
- **CI.** GitHub Actions runs ruff, mypy and the fast tests on Python 3.10, 3.11 and 3.12 for
  pushes to `master` and pull requests, and the slow tests weekly or on demand.
- **Docker.** A multi-stage CPU image that runs as an unprivileged user, with `data/` and
  `outputs/` as mount points, in two targets: training (the default) and `service`.
- **Inference service.** The optional `service` extra: a FastAPI app that serves one
  checkpoint through `GET /health`, `POST /score` and `POST /explain`. It validates input
  (length, NaN, range, batch size), returns scores without labels unless an operator sets
  `RADON_SCORE_THRESHOLD`, and logs requests without feature values. Documented in
  [docs/SERVICE.md](docs/SERVICE.md).
- **Attribution for single records.** `explain_record`, and the inlier profile stored by
  `radon-train` in checkpoints of tabular datasets, so `/explain` needs no dataset files.
- **Serving benchmark.** `benchmarks/serving.py` measures `/score` latency (p50, p95, p99) and
  throughput for several batch sizes with bootstrap 95% confidence intervals; results are in
  `benchmarks/results/` and the README.
- **License.** The MIT license file, declared as an SPDX expression in `pyproject.toml`.

### Fixed

- Four defects of the reference implementation that changed published numbers: a frozen
  warm-up batch order, a trapezoidal AUPRC, an early-stopping metric not comparable across
  projections, and a restored decision threshold that did not belong to the restored model.
- An installed package resolved `data/` and `outputs/` inside the virtual environment; it now
  uses the working directory. A source checkout still uses the repository root.
- Category names in attributions: `protocol_type_udp` is shown as `udp`, not `type_udp`.
- mypy errors that appeared with numpy 2.5 and pandas-stubs 3.

[0.1.0]: https://github.com/GiuseppeCuiuri/radon-anomaly-detection/releases/tag/v0.1.0
