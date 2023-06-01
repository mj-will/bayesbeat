"""bayesbeat

Bayesian inference for two decaying beating signals.
"""

import logging
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    pass

logging.getLogger(__name__).addHandler(logging.NullHandler())
