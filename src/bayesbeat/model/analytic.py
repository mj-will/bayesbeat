"""Analytic model derived by Bryan Barr"""

from typing import Optional

from nessai.livepoint import live_points_to_dict
import numpy as np

from .base import BaseModel, UniformPriorMixin

try:
    from numba import jit
except ImportError:

    def jit(*args, **kwargs):
        return lambda f: f


class AnalyticGaussianBeam(UniformPriorMixin, BaseModel):
    """Analytic Gaussian Beam Model."""

    constant_parameters: dict
    """Dictionary of constant parameters"""

    def __init__(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        *,
        photodiode_gap: float,
        photodiode_size: float,
        beam_radius: Optional[float] = None,
        x_offset: Optional[float] = None,
        sigma_noise: Optional[float] = None,
        prior_bounds: Optional[dict] = None,
        rescale: bool = False,
        decay_constraint: bool = False,
    ) -> None:
        super().__init__(x_data, y_data)

        self.photodiode_gap = photodiode_gap
        self.photodiode_size = photodiode_size
        self.decay_constraint = decay_constraint

        if rescale is True:
            raise NotImplementedError

        # Use the naming convention that Bryan used
        self.constant_parameters = dict(
            x_e=photodiode_size,
            x_g=photodiode_gap,
        )

        bounds = {
            "a_1": [5e-6, 1e-4],
            "a_2": [5e-6, 1e-4],
            "tau_1": [100, 1000],
            "tau_2": [100, 1000],
            "domega": [0.01, 1.0],
            "dphi": [0, 2 * np.pi],
        }

        if beam_radius is None:
            bounds["sigma_beam"] = [1e-3, 1e-2]
        else:
            self.constant_parameters["sigma_beam"] = beam_radius / 2.0

        if x_offset is None:
            bounds["x_offset"] = [-1e3, 1e-3]
        else:
            self.constant_parameters["x_offset"] = x_offset

        if sigma_noise is None:
            bounds["sigma_noise"] = [1e-5, 1.0]
        else:
            self.constant_parameters["sigma_noise"] = sigma_noise

        if prior_bounds:
            if not all([key in bounds] for key in prior_bounds):
                raise RuntimeError(
                    "Prior bounds contains invalid keys!"
                    f"Allowed keys are: {bounds.keys()}"
                )
            bounds.update(prior_bounds)

        self.names = list(bounds.keys())
        self.bounds = bounds

    def evaluate_constraints(self, x):
        """Evaluate any prior constraints"""
        out = np.zeros(x.size)
        if self.decay_constraint:
            out += np.log(x["tau_1"] > x["tau_2"])
        return out

    def log_prior(self, x):
        """Compute the log-prior probability"""
        with np.errstate(divide="ignore"):
            return (
                np.log(self.in_bounds(x), dtype="float")
                + self.evaluate_constraints(x)
                - np.log(self.upper_bounds - self.lower_bounds).sum()
            )

    def log_likelihood(self, x) -> np.ndarray:
        """Compute the log-likelihood"""
        x = live_points_to_dict(x, self.names)
        x.update(self.constant_parameters)
        sigma_noise = x.pop("sigma_noise")
        y_signal = np.sqrt(cubed_taylor_expansion_reduced(self.x_data, **x))

        norm_const = -0.5 * self.n_samples * np.log(2 * np.pi * sigma_noise**2)
        logl = norm_const + np.sum(
            -0.5 * ((self.y_data - y_signal) ** 2 / (sigma_noise**2)),
            axis=-1,
        )
        return logl

    def signal_model(self, x: np.ndarray) -> np.ndarray:
        x = live_points_to_dict(x, self.names)
        x.update(self.constant_parameters)
        x.pop("sigma_noise")
        return np.sqrt(cubed_taylor_expansion_reduced(self.x_data, **x))


@jit(nopython=True)
def cubed_taylor_expansion_reduced(
    time_vec: np.ndarray,
    sigma_beam: float,
    x_g: float,
    x_e: float,
    a_1: float,
    a_2: float,
    tau_1: float,
    tau_2: float,
    domega: float,
    dphi: float,
    x_offset: float,
):
    """Function version where it's been Taylor np.expanded out to mu**3"""
    # This could probably be optimised.
    return (
        9
        * a_1**4
        * a_2**2
        * (
            -np.sqrt(2) * x_e**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
            + np.sqrt(2) * x_g**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
        )
        ** 2
        * np.exp(-4 * time_vec / tau_1)
        * np.exp(-2 * time_vec / tau_2)
        / 32
        + 9
        * a_1**3
        * a_2**3
        * (
            -np.sqrt(2) * x_e**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
            + np.sqrt(2) * x_g**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
        )
        ** 2
        * np.exp(-3 * time_vec / tau_1)
        * np.exp(-3 * time_vec / tau_2)
        * np.cos(3 * domega * time_vec + 3 * dphi)
        / 16
        + 9
        * a_1**2
        * a_2**4
        * (
            -np.sqrt(2) * x_e**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
            + np.sqrt(2) * x_g**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
        )
        ** 2
        * np.exp(-2 * time_vec / tau_1)
        * np.exp(-4 * time_vec / tau_2)
        / 32
        + (
            a_1
            * (
                np.sqrt(2) * x_e**2 / (2 * np.sqrt(np.pi) * sigma_beam**3)
                - np.sqrt(2) * x_g**2 / (2 * np.sqrt(np.pi) * sigma_beam**3)
                - np.sqrt(2) * x_e**4 / (8 * np.sqrt(np.pi) * sigma_beam**5)
                + np.sqrt(2) * x_g**4 / (8 * np.sqrt(np.pi) * sigma_beam**5)
            )
            * np.exp(-time_vec / tau_1)
            + (
                -np.sqrt(2) * x_e**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
                + np.sqrt(2) * x_g**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
            )
            * (
                3 * a_1**3 * np.exp(-3 * time_vec / tau_1) / 4
                + 3
                * a_1
                * a_2**2
                * np.exp(-time_vec / tau_1)
                * np.exp(-2 * time_vec / tau_2)
                / 2
                + 3 * a_1 * x_offset**2 * np.exp(-time_vec / tau_1)
                + a_1 * np.exp(-time_vec / tau_1)
            )
        )
        ** 2
        / 2
        + (
            a_2
            * (
                np.sqrt(2) * x_e**2 / (2 * np.sqrt(np.pi) * sigma_beam**3)
                - np.sqrt(2) * x_g**2 / (2 * np.sqrt(np.pi) * sigma_beam**3)
                - np.sqrt(2) * x_e**4 / (8 * np.sqrt(np.pi) * sigma_beam**5)
                + np.sqrt(2) * x_g**4 / (8 * np.sqrt(np.pi) * sigma_beam**5)
            )
            * np.exp(-time_vec / tau_2)
            + (
                -np.sqrt(2) * x_e**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
                + np.sqrt(2) * x_g**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
            )
            * (
                3
                * a_1**2
                * a_2
                * np.exp(-2 * time_vec / tau_1)
                * np.exp(-time_vec / tau_2)
                / 2
                + 3 * a_2**3 * np.exp(-3 * time_vec / tau_2) / 4
                + 3 * a_2 * x_offset**2 * np.exp(-time_vec / tau_2)
                + a_2 * np.exp(-time_vec / tau_2)
            )
        )
        ** 2
        / 2
        + (
            3
            * a_1**2
            * a_2
            * (
                a_2
                * (
                    np.sqrt(2) * x_e**2 / (2 * np.sqrt(np.pi) * sigma_beam**3)
                    - np.sqrt(2)
                    * x_g**2
                    / (2 * np.sqrt(np.pi) * sigma_beam**3)
                    - np.sqrt(2)
                    * x_e**4
                    / (8 * np.sqrt(np.pi) * sigma_beam**5)
                    + np.sqrt(2)
                    * x_g**4
                    / (8 * np.sqrt(np.pi) * sigma_beam**5)
                )
                * np.exp(-time_vec / tau_2)
                + (
                    -np.sqrt(2) * x_e**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
                    + np.sqrt(2)
                    * x_g**2
                    / (4 * np.sqrt(np.pi) * sigma_beam**5)
                )
                * (
                    3
                    * a_1**2
                    * a_2
                    * np.exp(-2 * time_vec / tau_1)
                    * np.exp(-time_vec / tau_2)
                    / 2
                    + 3 * a_2**3 * np.exp(-3 * time_vec / tau_2) / 4
                    + 3 * a_2 * x_offset**2 * np.exp(-time_vec / tau_2)
                    + a_2 * np.exp(-time_vec / tau_2)
                )
            )
            * (
                -np.sqrt(2) * x_e**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
                + np.sqrt(2) * x_g**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
            )
            * np.exp(-2 * time_vec / tau_1)
            * np.exp(-time_vec / tau_2)
            / 4
            + 3
            * a_1
            * a_2**2
            * (
                a_1
                * (
                    np.sqrt(2) * x_e**2 / (2 * np.sqrt(np.pi) * sigma_beam**3)
                    - np.sqrt(2)
                    * x_g**2
                    / (2 * np.sqrt(np.pi) * sigma_beam**3)
                    - np.sqrt(2)
                    * x_e**4
                    / (8 * np.sqrt(np.pi) * sigma_beam**5)
                    + np.sqrt(2)
                    * x_g**4
                    / (8 * np.sqrt(np.pi) * sigma_beam**5)
                )
                * np.exp(-time_vec / tau_1)
                + (
                    -np.sqrt(2) * x_e**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
                    + np.sqrt(2)
                    * x_g**2
                    / (4 * np.sqrt(np.pi) * sigma_beam**5)
                )
                * (
                    3 * a_1**3 * np.exp(-3 * time_vec / tau_1) / 4
                    + 3
                    * a_1
                    * a_2**2
                    * np.exp(-time_vec / tau_1)
                    * np.exp(-2 * time_vec / tau_2)
                    / 2
                    + 3 * a_1 * x_offset**2 * np.exp(-time_vec / tau_1)
                    + a_1 * np.exp(-time_vec / tau_1)
                )
            )
            * (
                -np.sqrt(2) * x_e**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
                + np.sqrt(2) * x_g**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
            )
            * np.exp(-time_vec / tau_1)
            * np.exp(-2 * time_vec / tau_2)
            / 4
        )
        * np.cos(2 * domega * time_vec + 2 * dphi)
        + (
            3
            * a_1**2
            * a_2
            * (
                a_1
                * (
                    np.sqrt(2) * x_e**2 / (2 * np.sqrt(np.pi) * sigma_beam**3)
                    - np.sqrt(2)
                    * x_g**2
                    / (2 * np.sqrt(np.pi) * sigma_beam**3)
                    - np.sqrt(2)
                    * x_e**4
                    / (8 * np.sqrt(np.pi) * sigma_beam**5)
                    + np.sqrt(2)
                    * x_g**4
                    / (8 * np.sqrt(np.pi) * sigma_beam**5)
                )
                * np.exp(-time_vec / tau_1)
                + (
                    -np.sqrt(2) * x_e**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
                    + np.sqrt(2)
                    * x_g**2
                    / (4 * np.sqrt(np.pi) * sigma_beam**5)
                )
                * (
                    3 * a_1**3 * np.exp(-3 * time_vec / tau_1) / 4
                    + 3
                    * a_1
                    * a_2**2
                    * np.exp(-time_vec / tau_1)
                    * np.exp(-2 * time_vec / tau_2)
                    / 2
                    + 3 * a_1 * x_offset**2 * np.exp(-time_vec / tau_1)
                    + a_1 * np.exp(-time_vec / tau_1)
                )
            )
            * (
                -np.sqrt(2) * x_e**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
                + np.sqrt(2) * x_g**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
            )
            * np.exp(-2 * time_vec / tau_1)
            * np.exp(-time_vec / tau_2)
            / 4
            + 3
            * a_1
            * a_2**2
            * (
                a_2
                * (
                    np.sqrt(2) * x_e**2 / (2 * np.sqrt(np.pi) * sigma_beam**3)
                    - np.sqrt(2)
                    * x_g**2
                    / (2 * np.sqrt(np.pi) * sigma_beam**3)
                    - np.sqrt(2)
                    * x_e**4
                    / (8 * np.sqrt(np.pi) * sigma_beam**5)
                    + np.sqrt(2)
                    * x_g**4
                    / (8 * np.sqrt(np.pi) * sigma_beam**5)
                )
                * np.exp(-time_vec / tau_2)
                + (
                    -np.sqrt(2) * x_e**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
                    + np.sqrt(2)
                    * x_g**2
                    / (4 * np.sqrt(np.pi) * sigma_beam**5)
                )
                * (
                    3
                    * a_1**2
                    * a_2
                    * np.exp(-2 * time_vec / tau_1)
                    * np.exp(-time_vec / tau_2)
                    / 2
                    + 3 * a_2**3 * np.exp(-3 * time_vec / tau_2) / 4
                    + 3 * a_2 * x_offset**2 * np.exp(-time_vec / tau_2)
                    + a_2 * np.exp(-time_vec / tau_2)
                )
            )
            * (
                -np.sqrt(2) * x_e**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
                + np.sqrt(2) * x_g**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
            )
            * np.exp(-time_vec / tau_1)
            * np.exp(-2 * time_vec / tau_2)
            / 4
            + (
                a_1
                * (
                    np.sqrt(2) * x_e**2 / (2 * np.sqrt(np.pi) * sigma_beam**3)
                    - np.sqrt(2)
                    * x_g**2
                    / (2 * np.sqrt(np.pi) * sigma_beam**3)
                    - np.sqrt(2)
                    * x_e**4
                    / (8 * np.sqrt(np.pi) * sigma_beam**5)
                    + np.sqrt(2)
                    * x_g**4
                    / (8 * np.sqrt(np.pi) * sigma_beam**5)
                )
                * np.exp(-time_vec / tau_1)
                + (
                    -np.sqrt(2) * x_e**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
                    + np.sqrt(2)
                    * x_g**2
                    / (4 * np.sqrt(np.pi) * sigma_beam**5)
                )
                * (
                    3 * a_1**3 * np.exp(-3 * time_vec / tau_1) / 4
                    + 3
                    * a_1
                    * a_2**2
                    * np.exp(-time_vec / tau_1)
                    * np.exp(-2 * time_vec / tau_2)
                    / 2
                    + 3 * a_1 * x_offset**2 * np.exp(-time_vec / tau_1)
                    + a_1 * np.exp(-time_vec / tau_1)
                )
            )
            * (
                a_2
                * (
                    np.sqrt(2) * x_e**2 / (2 * np.sqrt(np.pi) * sigma_beam**3)
                    - np.sqrt(2)
                    * x_g**2
                    / (2 * np.sqrt(np.pi) * sigma_beam**3)
                    - np.sqrt(2)
                    * x_e**4
                    / (8 * np.sqrt(np.pi) * sigma_beam**5)
                    + np.sqrt(2)
                    * x_g**4
                    / (8 * np.sqrt(np.pi) * sigma_beam**5)
                )
                * np.exp(-time_vec / tau_2)
                + (
                    -np.sqrt(2) * x_e**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
                    + np.sqrt(2)
                    * x_g**2
                    / (4 * np.sqrt(np.pi) * sigma_beam**5)
                )
                * (
                    3
                    * a_1**2
                    * a_2
                    * np.exp(-2 * time_vec / tau_1)
                    * np.exp(-time_vec / tau_2)
                    / 2
                    + 3 * a_2**3 * np.exp(-3 * time_vec / tau_2) / 4
                    + 3 * a_2 * x_offset**2 * np.exp(-time_vec / tau_2)
                    + a_2 * np.exp(-time_vec / tau_2)
                )
            )
        )
        * np.cos(domega * time_vec + dphi)
    )


def cubed_taylor_expansion(
    time_vec: np.ndarray,
    sigma_beam: float,
    x_g: float,
    x_e: float,
    a_1: float,
    omega_1: float,
    phi_1: float,
    tau_1: float,
    a_2: float,
    omega_2: float,
    phi_2: float,
    tau_2: float,
    x_offset: float,
):
    """Function version where it's been Taylor np.expanded out to mu**3"""
    return (
        9
        * a_1**4
        * a_2**2
        * (
            -np.sqrt(2) * x_e**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
            + np.sqrt(2) * x_g**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
        )
        ** 2
        * np.exp(-4 * time_vec / tau_1)
        * np.exp(-2 * time_vec / tau_2)
        / 32
        + 9
        * a_1**3
        * a_2**3
        * (
            -np.sqrt(2) * x_e**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
            + np.sqrt(2) * x_g**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
        )
        ** 2
        * np.exp(-3 * time_vec / tau_1)
        * np.exp(-3 * time_vec / tau_2)
        * np.cos(
            3 * omega_1 * time_vec
            - 3 * omega_2 * time_vec
            + 3 * phi_1
            - 3 * phi_2
        )
        / 16
        + 9
        * a_1**2
        * a_2**4
        * (
            -np.sqrt(2) * x_e**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
            + np.sqrt(2) * x_g**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
        )
        ** 2
        * np.exp(-2 * time_vec / tau_1)
        * np.exp(-4 * time_vec / tau_2)
        / 32
        + (
            a_1
            * (
                np.sqrt(2) * x_e**2 / (2 * np.sqrt(np.pi) * sigma_beam**3)
                - np.sqrt(2) * x_g**2 / (2 * np.sqrt(np.pi) * sigma_beam**3)
                - np.sqrt(2) * x_e**4 / (8 * np.sqrt(np.pi) * sigma_beam**5)
                + np.sqrt(2) * x_g**4 / (8 * np.sqrt(np.pi) * sigma_beam**5)
            )
            * np.exp(-time_vec / tau_1)
            + (
                -np.sqrt(2) * x_e**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
                + np.sqrt(2) * x_g**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
            )
            * (
                3 * a_1**3 * np.exp(-3 * time_vec / tau_1) / 4
                + 3
                * a_1
                * a_2**2
                * np.exp(-time_vec / tau_1)
                * np.exp(-2 * time_vec / tau_2)
                / 2
                + 3 * a_1 * x_offset**2 * np.exp(-time_vec / tau_1)
                + a_1 * np.exp(-time_vec / tau_1)
            )
        )
        ** 2
        / 2
        + (
            a_2
            * (
                np.sqrt(2) * x_e**2 / (2 * np.sqrt(np.pi) * sigma_beam**3)
                - np.sqrt(2) * x_g**2 / (2 * np.sqrt(np.pi) * sigma_beam**3)
                - np.sqrt(2) * x_e**4 / (8 * np.sqrt(np.pi) * sigma_beam**5)
                + np.sqrt(2) * x_g**4 / (8 * np.sqrt(np.pi) * sigma_beam**5)
            )
            * np.exp(-time_vec / tau_2)
            + (
                -np.sqrt(2) * x_e**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
                + np.sqrt(2) * x_g**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
            )
            * (
                3
                * a_1**2
                * a_2
                * np.exp(-2 * time_vec / tau_1)
                * np.exp(-time_vec / tau_2)
                / 2
                + 3 * a_2**3 * np.exp(-3 * time_vec / tau_2) / 4
                + 3 * a_2 * x_offset**2 * np.exp(-time_vec / tau_2)
                + a_2 * np.exp(-time_vec / tau_2)
            )
        )
        ** 2
        / 2
        + (
            3
            * a_1**2
            * a_2
            * (
                a_2
                * (
                    np.sqrt(2) * x_e**2 / (2 * np.sqrt(np.pi) * sigma_beam**3)
                    - np.sqrt(2)
                    * x_g**2
                    / (2 * np.sqrt(np.pi) * sigma_beam**3)
                    - np.sqrt(2)
                    * x_e**4
                    / (8 * np.sqrt(np.pi) * sigma_beam**5)
                    + np.sqrt(2)
                    * x_g**4
                    / (8 * np.sqrt(np.pi) * sigma_beam**5)
                )
                * np.exp(-time_vec / tau_2)
                + (
                    -np.sqrt(2) * x_e**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
                    + np.sqrt(2)
                    * x_g**2
                    / (4 * np.sqrt(np.pi) * sigma_beam**5)
                )
                * (
                    3
                    * a_1**2
                    * a_2
                    * np.exp(-2 * time_vec / tau_1)
                    * np.exp(-time_vec / tau_2)
                    / 2
                    + 3 * a_2**3 * np.exp(-3 * time_vec / tau_2) / 4
                    + 3 * a_2 * x_offset**2 * np.exp(-time_vec / tau_2)
                    + a_2 * np.exp(-time_vec / tau_2)
                )
            )
            * (
                -np.sqrt(2) * x_e**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
                + np.sqrt(2) * x_g**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
            )
            * np.exp(-2 * time_vec / tau_1)
            * np.exp(-time_vec / tau_2)
            / 4
            + 3
            * a_1
            * a_2**2
            * (
                a_1
                * (
                    np.sqrt(2) * x_e**2 / (2 * np.sqrt(np.pi) * sigma_beam**3)
                    - np.sqrt(2)
                    * x_g**2
                    / (2 * np.sqrt(np.pi) * sigma_beam**3)
                    - np.sqrt(2)
                    * x_e**4
                    / (8 * np.sqrt(np.pi) * sigma_beam**5)
                    + np.sqrt(2)
                    * x_g**4
                    / (8 * np.sqrt(np.pi) * sigma_beam**5)
                )
                * np.exp(-time_vec / tau_1)
                + (
                    -np.sqrt(2) * x_e**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
                    + np.sqrt(2)
                    * x_g**2
                    / (4 * np.sqrt(np.pi) * sigma_beam**5)
                )
                * (
                    3 * a_1**3 * np.exp(-3 * time_vec / tau_1) / 4
                    + 3
                    * a_1
                    * a_2**2
                    * np.exp(-time_vec / tau_1)
                    * np.exp(-2 * time_vec / tau_2)
                    / 2
                    + 3 * a_1 * x_offset**2 * np.exp(-time_vec / tau_1)
                    + a_1 * np.exp(-time_vec / tau_1)
                )
            )
            * (
                -np.sqrt(2) * x_e**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
                + np.sqrt(2) * x_g**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
            )
            * np.exp(-time_vec / tau_1)
            * np.exp(-2 * time_vec / tau_2)
            / 4
        )
        * np.cos(
            2 * omega_1 * time_vec
            - 2 * omega_2 * time_vec
            + 2 * phi_1
            - 2 * phi_2
        )
        + (
            3
            * a_1**2
            * a_2
            * (
                a_1
                * (
                    np.sqrt(2) * x_e**2 / (2 * np.sqrt(np.pi) * sigma_beam**3)
                    - np.sqrt(2)
                    * x_g**2
                    / (2 * np.sqrt(np.pi) * sigma_beam**3)
                    - np.sqrt(2)
                    * x_e**4
                    / (8 * np.sqrt(np.pi) * sigma_beam**5)
                    + np.sqrt(2)
                    * x_g**4
                    / (8 * np.sqrt(np.pi) * sigma_beam**5)
                )
                * np.exp(-time_vec / tau_1)
                + (
                    -np.sqrt(2) * x_e**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
                    + np.sqrt(2)
                    * x_g**2
                    / (4 * np.sqrt(np.pi) * sigma_beam**5)
                )
                * (
                    3 * a_1**3 * np.exp(-3 * time_vec / tau_1) / 4
                    + 3
                    * a_1
                    * a_2**2
                    * np.exp(-time_vec / tau_1)
                    * np.exp(-2 * time_vec / tau_2)
                    / 2
                    + 3 * a_1 * x_offset**2 * np.exp(-time_vec / tau_1)
                    + a_1 * np.exp(-time_vec / tau_1)
                )
            )
            * (
                -np.sqrt(2) * x_e**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
                + np.sqrt(2) * x_g**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
            )
            * np.exp(-2 * time_vec / tau_1)
            * np.exp(-time_vec / tau_2)
            / 4
            + 3
            * a_1
            * a_2**2
            * (
                a_2
                * (
                    np.sqrt(2) * x_e**2 / (2 * np.sqrt(np.pi) * sigma_beam**3)
                    - np.sqrt(2)
                    * x_g**2
                    / (2 * np.sqrt(np.pi) * sigma_beam**3)
                    - np.sqrt(2)
                    * x_e**4
                    / (8 * np.sqrt(np.pi) * sigma_beam**5)
                    + np.sqrt(2)
                    * x_g**4
                    / (8 * np.sqrt(np.pi) * sigma_beam**5)
                )
                * np.exp(-time_vec / tau_2)
                + (
                    -np.sqrt(2) * x_e**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
                    + np.sqrt(2)
                    * x_g**2
                    / (4 * np.sqrt(np.pi) * sigma_beam**5)
                )
                * (
                    3
                    * a_1**2
                    * a_2
                    * np.exp(-2 * time_vec / tau_1)
                    * np.exp(-time_vec / tau_2)
                    / 2
                    + 3 * a_2**3 * np.exp(-3 * time_vec / tau_2) / 4
                    + 3 * a_2 * x_offset**2 * np.exp(-time_vec / tau_2)
                    + a_2 * np.exp(-time_vec / tau_2)
                )
            )
            * (
                -np.sqrt(2) * x_e**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
                + np.sqrt(2) * x_g**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
            )
            * np.exp(-time_vec / tau_1)
            * np.exp(-2 * time_vec / tau_2)
            / 4
            + (
                a_1
                * (
                    np.sqrt(2) * x_e**2 / (2 * np.sqrt(np.pi) * sigma_beam**3)
                    - np.sqrt(2)
                    * x_g**2
                    / (2 * np.sqrt(np.pi) * sigma_beam**3)
                    - np.sqrt(2)
                    * x_e**4
                    / (8 * np.sqrt(np.pi) * sigma_beam**5)
                    + np.sqrt(2)
                    * x_g**4
                    / (8 * np.sqrt(np.pi) * sigma_beam**5)
                )
                * np.exp(-time_vec / tau_1)
                + (
                    -np.sqrt(2) * x_e**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
                    + np.sqrt(2)
                    * x_g**2
                    / (4 * np.sqrt(np.pi) * sigma_beam**5)
                )
                * (
                    3 * a_1**3 * np.exp(-3 * time_vec / tau_1) / 4
                    + 3
                    * a_1
                    * a_2**2
                    * np.exp(-time_vec / tau_1)
                    * np.exp(-2 * time_vec / tau_2)
                    / 2
                    + 3 * a_1 * x_offset**2 * np.exp(-time_vec / tau_1)
                    + a_1 * np.exp(-time_vec / tau_1)
                )
            )
            * (
                a_2
                * (
                    np.sqrt(2) * x_e**2 / (2 * np.sqrt(np.pi) * sigma_beam**3)
                    - np.sqrt(2)
                    * x_g**2
                    / (2 * np.sqrt(np.pi) * sigma_beam**3)
                    - np.sqrt(2)
                    * x_e**4
                    / (8 * np.sqrt(np.pi) * sigma_beam**5)
                    + np.sqrt(2)
                    * x_g**4
                    / (8 * np.sqrt(np.pi) * sigma_beam**5)
                )
                * np.exp(-time_vec / tau_2)
                + (
                    -np.sqrt(2) * x_e**2 / (4 * np.sqrt(np.pi) * sigma_beam**5)
                    + np.sqrt(2)
                    * x_g**2
                    / (4 * np.sqrt(np.pi) * sigma_beam**5)
                )
                * (
                    3
                    * a_1**2
                    * a_2
                    * np.exp(-2 * time_vec / tau_1)
                    * np.exp(-time_vec / tau_2)
                    / 2
                    + 3 * a_2**3 * np.exp(-3 * time_vec / tau_2) / 4
                    + 3 * a_2 * x_offset**2 * np.exp(-time_vec / tau_2)
                    + a_2 * np.exp(-time_vec / tau_2)
                )
            )
        )
        * np.cos(omega_1 * time_vec - omega_2 * time_vec + phi_1 - phi_2)
    )
