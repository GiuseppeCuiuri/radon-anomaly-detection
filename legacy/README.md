# Legacy implementation

The original RADON code, kept for one concrete reason: the test suite uses it as
a **differential oracle**. Several tests re-implement or import these functions
and assert that the refactored code reproduces them exactly, which is how the
refactor demonstrates that it did not move any published number.

Nothing here is imported by `src/radon/`. Delete this directory and the
differential tests will skip; the rest of the suite is unaffected.

## Contents

| File | Role |
|---|---|
| `RADON_AE_SAD.py` | The working implementation: model, `fit`, projection, thresholding |
| `Preprocessing.py` | Dataset loading and the MNIST / NSL-KDD protocols |
| `Training.py` | `fit` wrappers plus the stdout-regex CSV dump |
| `Testing.py` | Evaluation, metrics and NSL-KDD explainability |
| `RADON.py` | Najari's reference implementation, as vendored into the project |
| `notebook/RADON.ipynb` | The notebook that drove every experiment |
| `reference/` | Najari's untouched original release |

## Mapping to the refactored package

| Legacy | Refactored |
|---|---|
| `Preprocessing.get_data`, `preprocess_dataset` | `radon.data.mnist`, `radon.data.nsl_kdd` |
| `Preprocessing.create_dataloaders` | `radon.data.loaders` |
| `RADON_AE_SAD.RADON` | `radon.models.autoencoder` |
| `RADON_AE_SAD.fit` | `radon.training.trainer` |
| `RADON_AE_SAD.unimodal`, `knuth_bin_width` | `radon.training.thresholding` |
| `RADON_AE_SAD._perform_projection` | `radon.training.projection` |
| `RADON_AE_SAD.TabularFeatureAnalyzer` | `radon.features.tabular` |
| `Training.parse_fit_stdout`, `write_csv` | `radon.training.records` (regex removed) |
| `Testing.evaluate_and_plot`, `plot_aucroc`, `plot_auprc` | `radon.evaluation`, `radon.viz.evaluation` |
| `Testing.explain_nslkdd` | `radon.explain.nsl_kdd`, `radon.viz.explain` |
| `RADON.ipynb` | `radon.cli`, `notebooks/01_radon_demo.ipynb` |
