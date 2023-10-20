"""Utilities for the model classes"""
import copy
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def get_model(
    name: str,
    *,
    x_data: np.ndarray,
    y_data: np.ndarray,
    model_config: Optional[dict] = None,
    **kwargs
):
    """Get an instance of model from a name, x and y data."""

    from . import _MODELS
    if model_config is None:
        model_config = {}

    config = copy.deepcopy(model_config)
    config.update(kwargs)

    ModelClass = _MODELS.get(name.lower())
    if ModelClass is None:
        raise ValueError(
            f"Unknown model name: {name.lower()}, "
            f"choose from: {_MODELS.keys()}"
        )
    logger.debug(f"Creating instance of {ModelClass} with config: {config}")
    model = ModelClass(
        x_data=x_data,
        y_data=y_data,
        **config,
    )
    return model
