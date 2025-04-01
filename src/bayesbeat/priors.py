from scipy.signal import find_peaks
import numpy as np
from typing import Optional

from .filter import filter_and_rfft


def estimate_frequency(
    x_data: np.ndarray,
    y_data: np.ndarray,
    sampling_rate: Optional[float] = None,
    f_lower: float = 1e-2,
    f_higher: float = None,
    filter_order: int = 4,
):

    x_freq, y_freq = filter_and_rfft(
        x_data,
        y_data,
        f_lower=f_lower,
        f_higher=f_higher,
        sampling_rate=sampling_rate,
        filter_order=filter_order,
    )

    if f_lower:
        # Require peaks be above value at f_lower
        idx = np.argmin(abs(x_freq - f_lower))
        y_min = np.abs(y_freq[idx])
    else:
        y_min = None

    peaks = find_peaks(np.abs(y_freq), height=y_min)[0]
    if not len(peaks):
        return None, None
    freq_vals = x_freq[peaks]
    power_vals = np.abs(y_freq[peaks])
    max_idx = np.argmax(power_vals)
    f_peak = freq_vals[max_idx]
    power_peak = power_vals[max_idx]
    return f_peak, power_peak


def estimate_domega_prior(
    x_data, y_data, sampling_rate, minimum_width=0.1 * np.pi, **kwargs
):
    f_peak, power_peak = estimate_frequency(
        x_data, y_data, sampling_rate, **kwargs
    )
    if f_peak is None:
        priors_range = [0, 0.5]
    else:
        domega_peak = 2 * np.pi * f_peak
        if domega_peak < minimum_width:
            priors_range = [0, minimum_width]
        else:
            delta = minimum_width / 2
            priors_range = [max(0, domega_peak - delta), domega_peak + delta]
    return priors_range


def estimate_initial_priors(
    x_data: np.ndarray,
    y_data: np.ndarray,
    sampling_rate: Optional[float] = None,
    parameters: list[str] = None,
    minimum_domega_width: float = 0.1 * np.pi,
):
    if parameters is None:
        raise ValueError("No parameters specified for estimating priors.")
    parameters = parameters.copy()
    priors = {}
    if "domega" in parameters:
        priors["domega"] = estimate_domega_prior(
            x_data, y_data, sampling_rate, minimum_width=minimum_domega_width
        )
        parameters.remove("domega")
    if parameters:
        raise ValueError(f"Unknown parameters: {parameters}")
    return priors
