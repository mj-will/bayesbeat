import logging
import inspect
import math
from typing import Optional

from nessai.livepoint import live_points_to_dict
import numpy as np
import torch

from .base import BaseModel

logger = logging.getLogger(__name__)


class GaussianBeamModel(BaseModel):
    """Model of a double decaying sinusoid including the gaussian beam shape
    and gap between sensors.

    """
    _model_parameters = None

    def __init__(
        self,
        x_data,
        y_data,
        *,
        photodiode_gap: float,
        photodiode_size: float,
        pre_fft_sample_rate: float,
        samples_per_measurement: int,
        allow_positive_beat: bool = True,
        allow_negative_beat: bool = False,
        beam_radius: Optional[float] = None,
        x_offset: Optional[float] = None,
        sigma_noise: Optional[float] = None,
        prior_bounds: Optional[dict] = None,
        decay_constraint: bool = False,
        amplitude_constraint: bool = False,
        rescale: bool = False,
        use_ratio: bool = False,
        reduce_factor: float = 0.1,
        device: str = "cpu",
        **kwargs,
    ):
        # define param names as list
        self.rescale = rescale
        self.use_ratio = use_ratio
        self.allow_positive_beat = allow_positive_beat
        self.allow_negative_beat = allow_negative_beat
        self.vectorised_likelihood = False

        self.constant_parameters = dict(
            f1=2000.0,
            phi_1=0.0,
            photodiode_gap=photodiode_gap,
            photodiode_size=photodiode_size,
        )

        if self.rescale:
            bounds = {
                "a_1": [0.5, 1],
            }
        elif self.use_ratio:
            bounds = {"a_1": [1e-7, 5e-4], "a_ratio": [0.1, 1]}
        else:
            bounds = {"a_1": [1e-7, 5e-4], "a_2": [1e-7, 5e-4]}

        bounds.update({
            "tau_1": [10, 8000],
            "tau_2": [10, 8000],
            "dphi": [0, 2 * np.pi],
        })

        bounds["domega"] = [
            -1.0 if self.allow_negative_beat else 0.0,
            1.0 if self.allow_positive_beat else 0.0,
        ]

        if bounds["domega"][0] == bounds["domega"][1]:
            raise RuntimeError("Invalid beat prior")

        if beam_radius is None:
            bounds["beam_radius"] = [1e-3, 1e-2]
        else:
            self.constant_parameters["beam_radius"] = beam_radius

        if x_offset is None:
            bounds["x_offset"] = [-1e-3, 1e-3]
        else:
            self.constant_parameters["x_offset"] = x_offset

        if sigma_noise is None:
            bounds["sigma_noise"] = [1e-5, 1.0]
        else:
            self.constant_parameters["sigma_noise"] = sigma_noise

        if prior_bounds is not None:
            bounds.update(prior_bounds)

        for k, v in kwargs.items():
            if k in self.model_parameters:
                bounds.pop(k)
                self.constant_parameters[k] = v

        self.names = list(bounds.keys())
        self.bounds = bounds

        self.sample_rate = pre_fft_sample_rate
        self.samples_per_measurement = samples_per_measurement
        self.reduce_factor = reduce_factor
        self.decay_constraint = decay_constraint
        self.amplitude_constraint = amplitude_constraint

        self.device = device
        if self.device == "cpu":
            logger.warning(
                "Running likelihood with CPU, sampling may be very slow!"
            )

        x_data, y_data = self.prep_data(
            x_data,
            y_data,
            self.sample_rate,
            self.samples_per_measurement,
            self.reduce_factor,
            self.device,
        )
        super().__init__(x_data, y_data)

    @property
    def model_parameters(self) -> list[str]:
        if self._model_parameters is None:
            params = set(inspect.signature(signal_model).parameters.keys())
            self._model_parameters = params - {"x_data"}
        return self._model_parameters

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
                t1,
                red_samples_per_measurement,
                # endpoint=False,
            )
        times_full = (
            torch.from_numpy(times_full)
            .to(torch.float64)
            .to(device)
        )
        if y is not None:
            y = torch.from_numpy(y).to(device)
        return times_full, y

    def evaluate_constraints(self, x):
        """Evaluate any prior constraints"""
        out = np.ones(x.size, dtype=bool)
        if self.decay_constraint:
            out &= (x["tau_1"] > x["tau_2"])
        if self.amplitude_constraint:
            out &= (x["a_1"] > x["a_2"])
        return out

    def log_prior(self, x):
        """
        Returns log of prior given a live point assuming uniform
        priors on each parameter.
        """
        with np.errstate(divide="ignore"):
            return (
                np.log(self.in_bounds(x), dtype="float")
                + np.log(self.evaluate_constraints(x), dtype="float")
                - np.log(self.upper_bounds - self.lower_bounds).sum()
            )
        
    def from_unit_hypercube(self, x):
        x_out = x.copy()
        for n in self.names:
            x_out[n] = (self.bounds[n][1] - self.bounds[n][0]) * x[
                n
            ] + self.bounds[n][0]
        return x_out
    
    def convert_to_model_parameters(self, x: dict, noise: bool = False) -> dict:
        if self.rescale:
            x["a_2"] = 1 - x["a_1"]
        elif self.use_ratio:
            x["a_2"] = x["a_ratio"].item() * x["a_1"]

        if "f2" not in x:
            x["f2"] = x["f1"] - (x["domega"] / (2 * np.pi))
        if "phi_2" not in x:
            x["phi_2"] = np.mod(x["phi_1"] + x["dphi"], 2 * np.pi)
        parameters = self.model_parameters
        if noise:
            parameters += {"noise"}
        y = {k: x[k] for k in parameters if k in x}
        return y
    
    def signal_model(self, x):
        """Return the signal for a given point"""
        x = live_points_to_dict(x)
        x.update(self.constant_parameters)
        x = self.convert_to_model_parameters(x)
        with torch.inference_mode():
            return signal_model(self.x_data, **x).cpu().numpy()

    def signal_model_with_noise(self, x, noise_scale):
        """Return the signal for a given point"""
        x = live_points_to_dict(x)
        x.update(self.constant_parameters)
        x = self.convert_to_model_parameters(x)
        x["noise_scale"] = noise_scale
        with torch.inference_mode():
            return signal_model_with_noise(self.x_data, **x).cpu().numpy()

    def log_likelihood(self, x):
        x = live_points_to_dict(x)
        x.update(self.constant_parameters)
        x_model = self.convert_to_model_parameters(x)
        with torch.inference_mode():
            logl = (
                (
                    log_likelihood(
                        self.x_data,
                        self.y_data,
                        self.n_samples,
                        sigma_noise=x["sigma_noise"],
                        **x_model
                    )
                )
                .cpu()
                .numpy()
            )
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
    Compute the signal in both halves of the photodiode with a given offset of d
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
    photodiode_left:
    photodiode_right:
    photodiode_sum:
    photodiode_diff:
    """
    photodiode_left = gaussian_cdf(edges, loc=-1 * d, scale=omega / 2) - gaussian_cdf(
        gap, loc=-1 * d, scale=omega / 2
    )
    photodiode_right = gaussian_cdf(edges, loc=1 * d, scale=omega / 2) - gaussian_cdf(
        gap, loc=1 * d, scale=omega / 2
    )
    return photodiode_right - photodiode_left


def signal_model(
    x_data: torch.Tensor,
    photodiode_gap: float,
    photodiode_size: float,
    a_1: float,
    f1: float,
    phi_1: float,
    tau_1: float,
    a_2: float,
    f2: float,
    phi_2: float,
    tau_2: float,
    x_offset: float,
    beam_radius: float,
) -> torch.Tensor:
    """get the model for the peaks"""
    y_1 = decaying_sine(x_data, a_1, f1, phi_1, tau_1)
    y_2 = decaying_sine(x_data, a_2, f2, phi_2, tau_2)
    disp = y_1 + y_2 + x_offset
    Diff = int_from_disp(disp, photodiode_gap, photodiode_size, beam_radius)
    Difft = torch.fft.rfft(Diff, dim=1)
    peaks_total = torch.max(torch.abs(Difft), dim=1)[0]
    return peaks_total


@torch.jit.script
def log_likelihood(
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    n_samples: int,
    sigma_noise: float,
    photodiode_gap: float,
    photodiode_size: float,
    a_1: float,
    f1: float,
    phi_1: float,
    tau_1: float,
    a_2: float,
    f2: float,
    phi_2: float,
    tau_2: float,
    x_offset: float,
    beam_radius: float,
) -> torch.Tensor:
    norm_const = -0.5 * n_samples * math.log(2 * math.pi * sigma_noise**2)
    y_signal = signal_model(
        x_data,
        photodiode_gap,
        photodiode_size,
        a_1,
        f1,
        phi_1,
        tau_1,
        a_2,
        f2,
        phi_2,
        tau_2,
        x_offset,
        beam_radius,
    )
    return norm_const + torch.sum(
        -0.5 * ((y_data - y_signal) ** 2 / (sigma_noise**2)),
        dtype=y_data.dtype,
    )


def int_from_disp_with_noise(
    d: torch.Tensor,
    gap: float,
    edges: float,
    omega: float,
    rin_scale: float,
) -> torch.Tensor:
    """
    Compute the signal in both halves of the photodiode with a given offset of d
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
    photodiode_left:
    photodiode_right:
    photodiode_sum:
    photodiode_diff:
    """
    photodiode_left = gaussian_cdf(edges, loc=-1 * d, scale=omega / 2) - gaussian_cdf(
        gap, loc=-1 * d, scale=omega / 2
    )
    photodiode_right = gaussian_cdf(edges, loc=1 * d, scale=omega / 2) - gaussian_cdf(
        gap, loc=1 * d, scale=omega / 2
    )
    rin = rin_scale * torch.randn_like(photodiode_right)

    photodiode_left = photodiode_left * (1 + rin)
    photodiode_right = photodiode_right * (1 + rin)


    return photodiode_right - photodiode_left

def signal_model_with_noise(
    x_data: torch.Tensor,
    photodiode_gap: float,
    photodiode_size: float,
    a_1: float,
    f1: float,
    phi_1: float,
    tau_1: float,
    a_2: float,
    f2: float,
    phi_2: float,
    tau_2: float,
    x_offset: float,
    beam_radius: float,
    noise_scale: float
) -> torch.Tensor:
    """get the model for the peaks"""
    y_1 = decaying_sine(x_data, a_1, f1, phi_1, tau_1)
    y_2 = decaying_sine(x_data, a_2, f2, phi_2, tau_2)
    disp = y_1 + y_2 + x_offset
    Diff = int_from_disp_with_noise(disp, photodiode_gap, photodiode_size, beam_radius, rin_scale=noise_scale)
    Difft = torch.fft.rfft(Diff, dim=1)
    peaks_total = torch.max(torch.abs(Difft), dim=1)[0]
    return peaks_total