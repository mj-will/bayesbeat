import numpy as np
from scipy.signal import butter, filtfilt


def highpass_filter(data, cutoff, fs, order=4):
    """Apply a high-pass filter to the data.

    Parameters
    ----------
    data : np.ndarray
        Data to be filtered.
    cutoff : float
        Cutoff frequency.
    fs : float
        Sampling frequency.
    order : int
        Order of the filter.
    """
    return _filter(data, cutoff, fs, "high", order)


def lowpass_filter(data, cutoff, fs, order=4):
    """Apply a low-pass filter to the data.

    Parameters
    ----------
    data : np.ndarray
        Data to be filtered.
    cutoff : float
        Cutoff frequency.
    fs : float
        Sampling frequency.
    order : int
        Order of the filter.
    """
    return _filter(data, cutoff, fs, "low", order)


def _filter(data, cutoff, fs, btype, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    y = filtfilt(b, a, data)
    return y


def filter_and_rfft(
    x_data,
    y_data,
    f_lower=None,
    f_higher=None,
    sampling_rate=None,
    filter_order=4,
):
    if sampling_rate is None:
        sampling_rate = 1 / (x_data[1] - x_data[0])

    if f_higher:
        y_data = lowpass_filter(
            y_data, f_higher, sampling_rate, order=filter_order
        )
    if f_lower:
        y_data = highpass_filter(
            y_data, f_lower, sampling_rate, order=filter_order
        )

    x_freq = np.fft.rfftfreq(len(y_data), d=1 / sampling_rate)
    y_freq = np.fft.rfft(y_data)
    return x_freq, y_freq
