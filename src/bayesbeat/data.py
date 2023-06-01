import logging
from typing import Optional, Tuple

import hdf5storage
import numpy as np


logger = logging.getLogger(__name__)


def get_n_entries(filename: str) -> int:
    """Get the number of entries in a datafile"""
    data = hdf5storage.loadmat(filename)
    return len(data["ring_times"].T)


def get_data(
    filename: str,
    index: int,
    maximum_amplitude: Optional[float] = None,
    rescale_amplitude: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Get a specific piece of data from a file.

    Parameters
    ----------
    filename : str
        Filename including the complete path
    index : int
        Index in the data to analyse. Start at zero.
    max_amp : Optional[float], optional
        Truncate the data at a maximum amplitude, by default None and the data
        is truncated.
    rescale_amplitude : bool, optional
        If true, the data is rescaled to the maximum amplitude is one, by
        default True.

    Returns
    -------
    times : numpy.ndarray
        Array of times
    amplitudes : numpy.ndarray
        Array of amplitudes
    freq : float
        Frequency of the data

    Raises
    ------
    ValueError
        If an index is not specified.
    """
    if index is None:
        raise ValueError("Must specify index")

    matdata = hdf5storage.loadmat(filename)
    times = matdata["ring_times"].T
    amplitudes = matdata["ring_amps"].T
    freqs = matdata["freq"]

    times = times[index]
    amplitudes = amplitudes[index]

    keep = ~np.isnan(times)
    times = times[keep]
    amplitudes = amplitudes[keep]

    if maximum_amplitude:
        logger.info(f"Initial maximum amplitude: {amplitudes.max()}")
        start = np.flatnonzero(amplitudes > maximum_amplitude)[-1]
        times, amplitudes = times[start:], amplitudes[start:]
        times = times - times[0]

    if rescale_amplitude:
        amplitudes = amplitudes / amplitudes.max()
    return times, amplitudes, freqs[index]
