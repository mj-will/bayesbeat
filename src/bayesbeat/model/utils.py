"""Utilities for the model classes"""

import copy
import logging
from typing import Callable, Optional
from warnings import warn

import numpy as np

from .base import BaseModel

logger = logging.getLogger(__name__)


try:
    from numba import jit
except ImportError:
    warn("Could not import numba", RuntimeWarning)

    # Based on https://stackoverflow.com/a/73275170
    def jit(f=None, *args, **kwargs):
        def decorator(func):
            return func

        if callable(f):
            return f
        else:
            return decorator


def get_model_class(name: str) -> Callable:
    from . import _MODELS

    ModelClass = _MODELS.get(name.lower())
    if ModelClass is None:
        raise ValueError(
            f"Unknown model name: {name.lower()}, "
            f"choose from: {_MODELS.keys()}"
        )
    return ModelClass


def get_model(
    name: str,
    *,
    x_data: np.ndarray,
    y_data: np.ndarray,
    model_config: Optional[dict] = None,
    **kwargs,
) -> BaseModel:
    """Get an instance of model from a name, x and y data."""

    ModelClass = get_model_class(name)

    if model_config is None:
        model_config = {}
    config = copy.deepcopy(model_config)
    config.update(kwargs)
    logger.debug(f"Creating instance of {ModelClass} with config: {config}")

    model = ModelClass(
        x_data=x_data,
        y_data=y_data,
        **config,
    )
    return model
