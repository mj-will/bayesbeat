import logging
import math

import numpy as np
import torch

from .base import BaseModel

logger = logging.getLogger(__name__)


class GaussianBeamModel(BaseModel):
    """Model of a double decaying sinusoid including the gaussian beam shape
    and gap between sensors.
    
    """

    def __init__(
        self,
        x_data,
        y_data,
        *,
        photodiode_gap: float,
        photodiode_size: float,
        sample_rate: float,
        samples_per_measurement: int,
        allow_positive_beat: bool = True,
        allow_negative_beat: bool = False,
        rescale: bool = False,
        use_ratio: bool = False,
        reduce_factor: float = 0.1,
        device: str = "cpu",
    ):
        # define param names as list
        self.rescale = rescale
        self.use_ratio = use_ratio
        self.allow_positive_beat = allow_positive_beat
        self.allow_negative_beat = allow_negative_beat
        self.vectorised_likelihood = False

        if self.rescale:
            names = ["A1"]
            bounds = {
                "A1": (0.5, 1),
            }
        elif self.use_ratio:
            names = ["A1", "A_ratio"]
            bounds = {"A1": (1e-7, 5e-4), "A_ratio": (0.1, 1)}
        else:
            names = ["A1", "A2"]
            bounds = {"A1": (1e-7, 5e-4), "A2": (1e-7, 5e-4)}

        names += [
            "dw",
            "dp",
            "decay1",
            "decay2",
            # "x_offset",
            "beam_radius",
            "sigma",
        ]

        bounds["dw"] = [
            -1.0 if self.allow_negative_beat else 0.0,
            1.0 if self.allow_positive_beat else 0.0,
        ]

        if bounds["dw"][0] == bounds["dw"][1]:
            raise RuntimeError("Invalid beat prior")
        
        bounds["dp"] = [0, 2 * np.pi]

        other_bounds = {
            # "f1": tuple(np.array([5e2, 40e3]) * reduce_factor),
            # "ph1": (0, 2 * np.pi),
            "decay1": (10, 8000),
            "decay2": (10, 8000),
            # "x_offset": (-0.005, 0.005),  # offset in m
            "beam_radius": (0.0005, 0.005),  # beam radius in m
            "sigma": (0, 50),
        }

        self.device = device
        self.f1 = 2000
        self.ph1 = 0.0

        if self.device == "cpu":
            logger.warning(
                "Running likelihood with CPU, sampling may be very slow!"
            )

        self.PD_gap = photodiode_gap
        self.PD_size = photodiode_size
        self.x_offset = 0.0
        self.names = names
        self.bounds = bounds | other_bounds

        self.sample_rate = sample_rate
        self.samples_per_measurement = samples_per_measurement
        self.reduce_factor = reduce_factor

        x_data, y_data = self.prep_data(
            x_data,
            y_data,
            self.sample_rate,
            self.samples_per_measurement,
            self.reduce_factor,
            self.device,
        )
        super().__init__(x_data, y_data)

    def prep_data(
        self,
        x,
        y,
        sample_rate: float,
        samples_per_measurement: int,
        reduce_factor: float,
        device: str,
    ):
        red_sample_rate = int(
            reduce_factor * sample_rate
        )  # Data acquisition rate of hardware (Hz)
        red_samples_per_measurement = int(
            reduce_factor * samples_per_measurement
        )  # Number of samples per measurement (n.a)
        measurement_duration = (
            red_samples_per_measurement / red_sample_rate
        )  # Length of each individual measurement (s)

        times_full = np.zeros((len(x), red_samples_per_measurement))
        for p, t1 in enumerate(x):
            # measurement time recorded at end of processing
            times_full[p] = np.linspace(
                t1 - measurement_duration,
                t1, red_samples_per_measurement,
                # endpoint=False,
            )
        times_full = (
            torch.from_numpy(times_full).to(torch.get_default_dtype()).to(device)
        )
        y = torch.from_numpy(y).to(device)
        return times_full, y

    def log_prior(self, x):
        """
        Returns log of prior given a live point assuming uniform
        priors on each parameter.
        """
        log_p = np.log(self.in_bounds(x), dtype="float")
        for n in self.names:
            log_p -= np.log(self.bounds[n][1] - self.bounds[n][0])
        return log_p

    def signal_model(self, x):
        """Return the signal for a given point"""
        A1 = x["A1"].item()
        if self.rescale:
            A2 = 1 - A1
        elif self.use_ratio:
            A2 = x["A_ratio"].item() * A1
        else:
            A2 = x["A2"]
        
        f2 = x["f1"] + x["df"]
        ph2 = np.mod(self.ph1 + x["dp"], 2 * np.pi)

        with torch.inference_mode():
            return (
                signal_model(
                    self.x_data,
                    self.PD_gap,
                    self.PD_size,
                    A1,
                    x["f1"],
                    self.ph1,
                    x["decay1"],
                    A2,
                    f2,
                    ph2,
                    x["decay2"],
                    self.x_offset,#x["x_offset"],
                    x["beam_radius"],
                )
            ).cpu().numpy()

    def log_likelihood(self, x):
        A1 = x["A1"].item()
        if self.rescale:
            A2 = 1 - A1
        elif self.use_ratio:
            A2 = x["A_ratio"].item() * A1
        else:
            A2 = x["A2"]
        f2 = self.f1 - (x["dw"] / np.pi)
        ph2 = np.mod(self.ph1 + x["dp"], 2 * np.pi)
        with torch.inference_mode():
            logl = (
                log_likelihood(
                    self.x_data,
                    self.y_data,
                    self.n_samples,
                    self.PD_gap,
                    self.PD_size,
                    A1,
                    self.f1,
                    self.ph1,
                    x["decay1"],
                    A2,
                    f2,
                    ph2,
                    x["decay2"],
                    self.x_offset,# x["x_offset"],
                    x["beam_radius"],
                    x["sigma"],
                )
            ).cpu().numpy()
        return logl


@torch.jit.script
def gaussian_cdf(val: float, loc: torch.Tensor, scale: float) -> torch.Tensor:
    """Gaussian CDF implemented in torch"""
    return 0.5 * torch.special.erfc((loc - val) / (math.sqrt(2) * scale))


@torch.jit.script
def decaying_sine(
    t_vec: torch.Tensor,
    Amp_0: float,
    freq_0: float,
    phase: float,
    tau: float,
) -> torch.Tensor:
    A_0 = Amp_0 * torch.exp(-1 * t_vec / tau)
    D_0 = A_0 * torch.sin(2.0 * np.pi * freq_0 * t_vec + phase)
    return D_0


@torch.jit.script
def int_from_disp(
    d: torch.Tensor, gap: float, edges: float, omega: float
) -> torch.Tensor:
    """
    Compute the signal in both halveds of the photodiode with a given offset of d
    Args
    -------
    d:
        offset of the beam
    gap: float
        size of the gap between photodiodes
    edges: float

    omega: float
        beam size

    returns
    ---------
    PD_left:
    PD_right:
    PD_sum:
    PD_diff:
    """
    PD_left = gaussian_cdf(edges, loc=-1 * d, scale=omega / 2) - gaussian_cdf(
        gap, loc=-1 * d, scale=omega / 2
    )
    PD_right = gaussian_cdf(edges, loc=1 * d, scale=omega / 2) - gaussian_cdf(
        gap, loc=1 * d, scale=omega / 2
    )
    return PD_right - PD_left


@torch.jit.script
def signal_model(
    x_data: torch.Tensor,
    PD_gap: float,
    PD_size: float,
    Amp1: float,
    f1: float,
    ph1: float,
    decay1: float,
    Amp2: float,
    f2: float,
    ph2: float,
    decay2: float,
    x_offset: float,
    beam_radius: float,
) -> torch.Tensor:
    """get the model for the peaks"""
    disp = (
        decaying_sine(x_data, Amp1, f1, ph1, decay1)
        + decaying_sine(x_data, Amp2, f2, ph2, decay2)
        + x_offset
    )
    Diff = int_from_disp(disp, PD_gap, PD_size, beam_radius)
    Difft = torch.fft.rfft(Diff, dim=1)
    peaks_total = torch.max(torch.abs(Difft), dim=1)[0]
    return peaks_total


@torch.jit.script
def log_likelihood(
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    n_samples: int,
    PD_gap: float,
    PD_size: float,
    Amp1: float,
    f1: float,
    ph1: float,
    decay1: float,
    Amp2: float,
    f2: float,
    ph2: float,
    decay2: float,
    x_offset: float,
    beam_radius: float,
    sigma: float,
) -> torch.Tensor:
    norm_const = -0.5 * n_samples * math.log(2 * math.pi * sigma**2)
    y_signal = signal_model(
        x_data,
        PD_gap,
        PD_size,
        Amp1,
        f1,
        ph1,
        decay1,
        Amp2,
        f2,
        ph2,
        decay2,
        x_offset,
        beam_radius,
    )
    return norm_const + torch.sum(
        -0.5 * ((y_data - y_signal) ** 2 / (sigma**2)),
        dtype=y_data.dtype,
    )
