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


def get_frequency(filename: str, index: int) -> float:
    """Get the frequency of a specific index in a data file"""
    import hdf5storage

    try:
        data = hdf5storage.loadmat(filename)
    except ValueError:
        data = read_hdf5_to_dict(filename)
    freq = data["freq"][index]
    return float(freq)


def get_data(
    filename: str,
    index: int,
    minimum_amplitude: Optional[float] = None,
    maximum_amplitude: Optional[float] = None,
    t_end: Optional[float] = None,
    rescale_amplitude: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Get a specific piece of data from a file.

    Parameters
    ----------
    filename : str
        Filename including the complete path
    index : int
        Index in the data to analyse. Start at zero.
    minimum_amplitude : Optional[float]
        Truncate the data once a minimum amplitude has been reached, by
        default no truncation is applied.
    maximum_amplitude : Optional[float]
        Truncate the data at a maximum amplitude, by default None and the data
        is not truncated.
    t_end : Optional[float]
        Truncate the data at a specific time, by default None and the data is
        not truncated.
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
        raise RuntimeError(f"Data file {filename} does not exist!")

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

    if minimum_amplitude:
        logger.info(f"Truncating after minimum amplitude: {minimum_amplitude}")
        end = np.argmax(amplitudes < minimum_amplitude)
        logger.info(f"Discarding data after  {times[end]:.2f} s")
        times, amplitudes = times[:end], amplitudes[:end]
        if signal is not None:
            signal = signal[:end]

    if t_end:
        logger.info(f"Truncating after {t_end} s")
        end = np.argmax(times > t_end)
        logger.info(f"Discarding data after  {times[end]:.2f} s")
        times, amplitudes = times[:end], amplitudes[:end]
        if signal is not None:
            signal = signal[:end]

    if rescale_amplitude:
        amplitudes = amplitudes / amplitudes.max()

    return times, amplitudes, freqs[index], signal


def simulate_data_from_model(
    model,
    parameters: np.ndarray,
    gaussian_noise: bool = True,
    noise_scale: Optional[float] = None,
    zero_noise: bool = False,
    **kwargs,
):
    if gaussian_noise or zero_noise:
        if zero_noise:
            noise_scale = 0.0
        y_signal = model.signal_model(parameters)
        y_data = y_signal + noise_scale * np.random.randn(len(y_signal))
    else:
        logger.info(f"Simulating non-Gaussian noise with {kwargs}")
        try:
            y_data = model.signal_model_with_noise(parameters, **kwargs)
            y_signal = model.signal_model(parameters)
        except NotImplementedError:
            raise RuntimeError("model only supports Gaussian noise")
    return y_data, y_signal


def simulate_data(
    model_name: str,
    sample_rate: float,
    duration: float,
    minimum_amplitude: Optional[float] = None,
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

    parameters = copy.deepcopy(kwargs)
    model_kwargs = {}
    for k in kwargs:
        if k in allowed_kwargs:
            model_kwargs[k] = parameters.pop(k)
    times = np.linspace(0, duration, int(sample_rate * duration))
    model = ModelClass(x_data=times, y_data=None, **model_kwargs)

    signal_parameters = {}
    noise_parameters = {}
    for k, v in parameters.items():
        if "noise" in k:
            noise_parameters[k] = v
        else:
            signal_parameters[k] = v

    signal_parameters = dict_to_live_points(
        signal_parameters, non_sampling_parameters=False
    )

    logger.info(
        f"Simulating signal with {ModelClass} model and parameters {signal_parameters}"
    )

    y_data, y_signal = simulate_data_from_model(
        model,
        signal_parameters,
        gaussian_noise=gaussian_noise,
        zero_noise=zero_noise,
        **noise_parameters,
    )

    if maximum_amplitude:
        logger.info(f"Initial maximum amplitude: {y_data.max()}")
        start = np.flatnonzero(y_data > maximum_amplitude)[-1]
        times, y_data = times[start:], y_data[start:]
        times = times - times[0]

    if minimum_amplitude:
        logger.info(f"Truncating after minimum amplitude: {minimum_amplitude}")
        end = np.argmax(amplitudes < minimum_amplitude)
        logger.info(f"Discarding data after  {times[end]}s")
        times, amplitudes = times[:end], amplitudes[:end]

    if rescale_amplitude:
        y_data = y_data / y_data.max()

    return times, y_data, y_signal
