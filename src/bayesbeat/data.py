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
    use_bryan_model: bool = True,
    sample_rate: int = 250e3,
    samples_per_measurement: int = 50e3,
    reduce_factor: int = 1,
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

    if use_bryan_model:
        red_sample_rate = int(
            reduce_factor * sample_rate
        )  # Data acquisition rate of hardware (Hz)
        red_samples_per_measurement = int(
            reduce_factor * samples_per_measurement
        )  # Number of samples per measurment (n.a)
        measurement_duration = (
            red_samples_per_measurement / red_sample_rate
        )  # Length of each individual measurement (s)

        times_full = np.zeros((len(times), red_samples_per_measurement))
        for p, t1 in enumerate(times):
            # measurement time recorded at end of processing
            times_full[p] = np.linspace(
                t1 - measurement_duration, t1, red_samples_per_measurement
            )

        return times_full, amplitudes, freqs[index]

    else:
        return times, amplitudes, freqs[index]
