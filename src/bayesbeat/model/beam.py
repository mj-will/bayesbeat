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

    cuda_likelihood = True
    _model_parameters = None

    def __init__(
        self,
        x_data,
        y_data,
        *,
        photodiode_size: float,
        pre_fft_sample_rate: float,
        samples_per_measurement: int,
        prior_bounds: Optional[dict] = None,
        decay_constraint: bool = False,
        amplitude_constraint: bool = False,
        rescale: bool = False,
        reduce_factor: float = 0.1,
        device: str = "cpu",
        **kwargs,
    ):
        self.rescale = rescale
        self.vectorised_likelihood = False

        self.sample_rate = pre_fft_sample_rate
        self.samples_per_measurement = samples_per_measurement
        self.reduce_factor = reduce_factor
        self.decay_constraint = decay_constraint
        self.amplitude_constraint = amplitude_constraint

        self.constant_parameters = dict(
            f1=2000.0,
            phi_1=0.0,
            photodiode_size=photodiode_size,
        )

        bounds = {
            "a_1": [1e-7, 1e-3],
            "a_2": [1e-7, 1e-3],
            "tau_1": [10, 8000],
            "tau_2": [10, 8000],
            "dphi": [0, 2 * np.pi],
            "domega": [0, 5],
            "beam_radius": [1e-4, 1e-2],
            "photodiode_gap": [1e-4, 1e-2],
            "x_offset": [0, 1e-3],
            "sigma_amp_noise": [0, 1],
            "sigma_constant_noise": [0, 1],
        }

        if self.rescale:
            bounds["a_1"] = [0.1, 1.0]
            bounds.pop("a_2")

        if prior_bounds is not None:
            bounds.update(prior_bounds)

        if "a_ratio" in bounds:
            bounds.pop("a_2")

        for k, v in kwargs.items():
            if k in self.model_parameters:
                bounds.pop(k)
                self.constant_parameters[k] = v

        self.names = list(bounds.keys())
        self.bounds = bounds

        self.log_prior_constant = -np.log(
            self.upper_bounds - self.lower_bounds
        ).sum()

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

    @property
    def valid_parameters(self):
        additional = {
            "dphi",
            "domega",
            "sigma_amp_noise",
            "sigma_constant_noise",
        }
        return self.model_parameters.union(additional)

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
        times_full = torch.from_numpy(times_full).to(torch.float64).to(device)
        if y is not None:
            y = torch.from_numpy(y).to(device)
        return times_full, y

    def evaluate_constraints(self, x):
        """Evaluate any prior constraints"""
        out = np.ones(x.size, dtype=bool)
        if self.decay_constraint:
            out &= x["tau_1"] > x["tau_2"]
        if self.amplitude_constraint:
            out &= x["a_1"] > x["a_2"]
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
                + self.log_prior_constant
            )

    def convert_to_model_parameters(
        self, x: dict, noise: bool = False
    ) -> dict:
        x.update(self.constant_parameters)
        if self.rescale:
            x["a_2"] = 1 - x["a_1"]
        elif "a_ratio" in x:
            x["a_2"] = x["a_ratio"] * x["a_1"]

        if "f2" not in x:
            x["f2"] = x["f1"] - (x["domega"] / (2 * np.pi))
        if "phi_2" not in x:
            x["phi_2"] = np.mod(x["phi_1"] - x["dphi"], 2 * np.pi)
        y = {k: x[k] for k in self.model_parameters if k in x}
        return y

    def signal_model(self, x):
        """Return the signal for a given point"""
        x = live_points_to_dict(x)
        x = self.convert_to_model_parameters(x)
        with torch.inference_mode():
            return signal_model(self.x_data, **x).cpu().numpy()

    def signal_model_with_noise(self, x, noise_scale):
        """Return the signal for a given point"""
        raise NotImplementedError()
        x = live_points_to_dict(x)
        x = self.convert_to_model_parameters(x)
        with torch.inference_mode():
            return signal_model_with_noise(self.x_data, **x).cpu().numpy()

    def log_likelihood(self, x):
        x = live_points_to_dict(x)
        sigma_amp_noise = x.pop(
            "sigma_amp_noise",
            self.constant_parameters.get("sigma_amp_noise", 0.0),
        )
        sigma_constant_noise = x.pop(
            "sigma_constant_noise",
            self.constant_parameters.get("sigma_constant_noise", 0.0),
        )
        x = self.convert_to_model_parameters(x)
        with torch.inference_mode():
            logl = (
                log_likelihood(
                    self.x_data,
                    self.y_data,
                    sigma_amp_noise=sigma_amp_noise,
                    sigma_constant_noise=sigma_constant_noise,
                    **x,
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
    photodiode_left = gaussian_cdf(
        edges, loc=-1 * d, scale=omega / 2
    ) - gaussian_cdf(gap, loc=-1 * d, scale=omega / 2)
    photodiode_right = gaussian_cdf(
        edges, loc=1 * d, scale=omega / 2
    ) - gaussian_cdf(gap, loc=1 * d, scale=omega / 2)
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
    sigma_amp_noise: float,
    sigma_constant_noise: float,
) -> torch.Tensor:
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
    sigma2 = (sigma_amp_noise * y_signal) ** 2 + sigma_constant_noise**2
    norm_const = torch.log(2 * np.pi * sigma2)
    res = (y_data - y_signal) ** 2 / sigma2
    logl = -0.5 * torch.sum(norm_const + res, dtype=y_data.dtype)
    return logl


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
    photodiode_left = gaussian_cdf(
        edges, loc=-1 * d, scale=omega / 2
    ) - gaussian_cdf(gap, loc=-1 * d, scale=omega / 2)
    photodiode_right = gaussian_cdf(
        edges, loc=1 * d, scale=omega / 2
    ) - gaussian_cdf(gap, loc=1 * d, scale=omega / 2)
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
    noise_scale: float,
) -> torch.Tensor:
    """get the model for the peaks"""
    y_1 = decaying_sine(x_data, a_1, f1, phi_1, tau_1)
    y_2 = decaying_sine(x_data, a_2, f2, phi_2, tau_2)
    disp = y_1 + y_2 + x_offset
    Diff = int_from_disp_with_noise(
        disp,
        photodiode_gap,
        photodiode_size,
        beam_radius,
        rin_scale=noise_scale,
    )
    Difft = torch.fft.rfft(Diff, dim=1)
    peaks_total = torch.max(torch.abs(Difft), dim=1)[0]
    return peaks_total
