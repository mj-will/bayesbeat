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
