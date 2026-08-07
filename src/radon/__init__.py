"""RADON: Robust Autoencoder for Unsupervised Anomaly Detection.

This package deliberately performs no eager submodule imports: ``radon.config``
must stay importable without PyTorch installed so that configuration handling
can be tested in a minimal environment.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
