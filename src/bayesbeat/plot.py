"""Plotting functions"""

from typing import Optional, Union

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np


def plot_data(
    x_data: np.ndarray,
    y_data: np.ndarray,
    signal: Optional[np.ndarray] = None,
    filename: Optional[str] = None,
) -> Union[Figure, None]:

    fig = plt.figure()
    plt.plot(x_data, y_data, ls="-")
    if signal is not None:
        plt.plot(x_data, signal, label="Signal")

    plt.xlabel("Time [s]")
    # TODO: what are the units?
    plt.ylabel("Amplitude")

    plt.tight_layout()

    if filename is not None:
        fig.savefig(filename)
    else:
        return fig


def plot_fit(
    x_data: np.ndarray,
    y_data: np.ndarray,
    y_fit: np.ndarray,
    sigma_amp_noise: float = 0.0,
    sigma_constant_noise: float = 0.0,
    filename: Optional[str] = None,
) -> Union[Figure, None]:
    """Plot the fit and residuals"""
    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].scatter(x_data, y_data, label="Data", s=2.0, color="grey")
    axs[0].plot(x_data, y_fit, label="Fit", color="C0", ls="-")
    axs[0].set_ylabel("Amplitude")
    axs[0].legend()

    # Fallback to just data - fit
    if sigma_amp_noise == 0 and sigma_constant_noise == 0:
        sigma_constant_noise = 1

    res = (y_data - y_fit) / np.sqrt(
        (y_fit * sigma_amp_noise) ** 2 + sigma_constant_noise**2,
    )

    axs[1].scatter(x_data, res, label="Data - fit", color="C0", s=2.0)
    axs[1].legend()
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Residuals")
    plt.tight_layout()

    if filename is not None:
        fig.savefig(filename)
    else:
        return fig
