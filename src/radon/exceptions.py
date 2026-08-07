"""Exception hierarchy for the RADON package.

Every error raised deliberately by this package derives from :class:`RadonError`,
so callers can distinguish expected failures (bad config, missing data) from
genuine bugs.
"""


class RadonError(Exception):
    """Base class for all errors raised by the ``radon`` package."""


class ConfigError(RadonError):
    """Raised when a configuration file or override is invalid."""


class DatasetError(RadonError):
    """Raised when a dataset cannot be loaded or fails its integrity checks."""


class ThresholdError(RadonError):
    """Raised when a threshold cannot be estimated from a score histogram."""


class CheckpointError(RadonError):
    """Raised when a model checkpoint cannot be written or restored."""
