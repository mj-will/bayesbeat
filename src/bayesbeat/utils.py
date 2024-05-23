"""General utilities"""

import ast
import logging
import os
import sys
import time
from typing import Any

from .model.base import BaseModel


def configure_logger(
    output=None,
    label="bayesbeat",
    log_level="INFO",
):
    """
    Configure the logger.

    Base of the logger in nessai.


    Parameters
    ----------
    output : str, optional
        Path of to output directory.
    label : str, optional
        Label for this instance of the logger.
    log_level : {'ERROR', 'WARNING', 'INFO', 'DEBUG'}, optional
        Level of logging passed to logger.

    Returns
    -------
    :obj:`logging.Logger`
        Instance of the Logger class.
    """
    from . import __version__ as version

    if type(log_level) is str:
        try:
            level = getattr(logging, log_level.upper())
        except AttributeError:
            raise ValueError("log_level {} not understood".format(log_level))
    else:
        level = int(log_level)

    logger = logging.getLogger("bayesbeat")
    logger.setLevel(level)

    if (
        any([type(h) == logging.StreamHandler for h in logger.handlers])
        is False
    ):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s bayesbeat %(levelname)-8s: %(message)s",
                datefmt="%m-%d %H:%M",
            )
        )
        stream_handler.setLevel(level)
        logger.addHandler(stream_handler)

    if any([type(h) == logging.FileHandler for h in logger.handlers]) is False:
        if label:
            if output:
                if not os.path.exists(output):
                    os.makedirs(output, exist_ok=True)
            else:
                output = "."
            log_file = os.path.join(output, f"{label}.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s %(levelname)-8s: %(message)s", datefmt="%H:%M"
                )
            )

            file_handler.setLevel(level)
            logger.addHandler(file_handler)

    for handler in logger.handlers:
        handler.setLevel(level)

    logger.info(f"Running bayesbeat version {version}")

    return logger


def try_literal_eval(value: Any, /) -> Any:
    """Try to call literal eval return value if an error is raised"""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def time_likelihood(model: BaseModel, n: int = 100) -> float:
    """Time the likelihood"""
    x = model.new_point(n)
    # Call once since likelihood may use JIT
    _ = model.log_likelihood(x[0])
    start = time.perf_counter()
    for xx in x:
        _ = model.log_likelihood(xx)
    end = time.perf_counter()
    return (end - start) / n


def read_hdf5_to_dict(file_path):
    import h5py

    data_dict = {}
    with h5py.File(file_path, "r") as f:

        def visitor_func(name, obj):
            if isinstance(obj, h5py.Dataset):
                data_dict[name] = obj[()]

        f.visititems(visitor_func)

    return data_dict
