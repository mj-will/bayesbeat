import copy
import inspect
import logging
import os
from typing import Callable, Optional, Tuple

from nessai.livepoint import dict_to_live_points
import numpy as np

from .model.utils import get_model_class
from .utils import read_hdf5_to_dict


logger = logging.getLogger(__name__)


def get_n_entries(filename: str) -> int:
    """Get the number of entries in a datafile"""
    import hdf5storage

    try:
        data = hdf5storage.loadmat(filename)
    except ValueError:
        data = read_hdf5_to_dict(filename)

    return len(data["ring_times"].T)


def get_data(
    filename: str,
    index: int,
    maximum_amplitude: Optional[float] = None,
    rescale_amplitude: bool = False,
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
    import hdf5storage

    if index is None:
        raise ValueError("Must specify index")

    if not filename:
        raise ValueError("Must specific a data file!")

    if not os.path.exists(filename):
        raise RuntimeError("Data file does not exist!")

    try:
        matdata = hdf5storage.loadmat(filename)
    except ValueError:
        matdata = read_hdf5_to_dict(filename)

    times = matdata["ring_times"].T
    amplitudes = matdata["ring_amps"].T
    freqs = matdata["freq"]

    times = times[index]
    amplitudes = amplitudes[index]

    keep = ~np.isnan(times)
    times = times[keep]
    amplitudes = amplitudes[keep]

    if "ring_amps_inj" in matdata:
        signal = matdata["ring_amps_inj"].T[index]
        signal = signal[keep]
    else:
        signal = None

    if maximum_amplitude:
        logger.info(f"Initial maximum amplitude: {amplitudes.max()}")
        start = np.flatnonzero(amplitudes > maximum_amplitude)[-1]
        times, amplitudes = times[start:], amplitudes[start:]
        times = times - times[0]

    if rescale_amplitude:
        amplitudes = amplitudes / amplitudes.max()

    return times, amplitudes, freqs[index], signal


def simulate_data_from_model(
    model,
    parameters: np.ndarray,
    gaussian_noise: bool = True,
    noise_scale: Optional[float] = None,
):
    if gaussian_noise:
        y_signal = model.signal_model(parameters)
        y_data = y_signal + noise_scale * np.random.randn(len(y_signal))
    else:
        try:
            y_data = model.signal_model_with_noise(
                parameters, noise_scale=noise_scale
            )
            y_signal = model.signal_model_with_noise(
                parameters, noise_scale=0.0
            )
        except NotImplementedError:
            raise RuntimeError("model only supports Gaussian noise")
    return y_data, y_signal


def simulate_data(
    model_name: str,
    sample_rate: float,
    duration: float,
    sigma_noise: float,
    maximum_amplitude: Optional[float] = None,
    rescale_amplitude: Optional[float] = None,
    gaussian_noise: bool = True,
    zero_noise: bool = False,
    **kwargs,
):
    """

    Parameters
    ----------
    sample_rate
        Sample rate in Hz
    duration
        Duration in seconds
    sigma_noise
        Standard deviation of the Gaussian noise
    """
    ModelClass = get_model_class(model_name)

    sig = inspect.signature(ModelClass)
    allowed_kwargs = sig.parameters.keys()

    if zero_noise:
        sigma_noise = 0.0

    parameters = copy.deepcopy(kwargs)
    model_kwargs = {}
    for k in kwargs:
        if k in allowed_kwargs:
            model_kwargs[k] = parameters.pop(k)
    times = np.linspace(0, duration, int(sample_rate * duration))
    model = ModelClass(x_data=times, y_data=None, **model_kwargs)

    if isinstance(parameters, dict):
        parameters = dict_to_live_points(
            parameters, non_sampling_parameters=False
        )

    logger.info(
        f"Simulating signal with {ModelClass} model and parameters {parameters}"
    )

    y_data, y_signal = simulate_data_from_model(
        model,
        parameters,
        gaussian_noise=gaussian_noise,
        noise_scale=sigma_noise,
    )

    if maximum_amplitude:
        logger.info(f"Initial maximum amplitude: {y_data.max()}")
        start = np.flatnonzero(y_data > maximum_amplitude)[-1]
        times, y_data = times[start:], y_data[start:]
        times = times - times[0]

    if rescale_amplitude:
        y_data = y_data / y_data.max()

    return times, y_data, y_signal
