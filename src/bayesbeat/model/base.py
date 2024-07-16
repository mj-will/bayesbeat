from abc import abstractmethod

import inspect
import logging
from nessai.model import Model
from nessai.livepoint import live_points_to_dict
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class BaseModel(Model):
    """Base class for models"""

    cuda_likelihood: bool = False
    """Indicates if the likelihood requires CUDA and should therefore not
    use multiprocessing.
    """

    def __init__(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
        self.x_data = x_data
        self.y_data = y_data
        self.n_samples = len(x_data)

    def set_x_data(self, x_data: np.ndarray) -> None:
        self.x_data = x_data

    def set_y_data(self, y_data: np.ndarray) -> None:
        self.y_data = y_data

    @abstractmethod
    def signal_model(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def signal_model_with_noise(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class TwoNoiseSourceModel(BaseModel):
    """Model that defines a Gaussian likelihood with two noise parameters."""

    _model_parameters = None

    def __init__(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        default_bounds: dict,
        *,
        prior_bounds: Optional[dict] = None,
        decay_constraint: bool = False,
        amplitude_constraint: bool = False,
        **kwargs,
    ) -> None:

        self.amplitude_constraint = amplitude_constraint
        self.decay_constraint = decay_constraint

        bounds = {
            "sigma_amp_noise": [0, 1],
            "sigma_constant_noise": [0, 1],
        }
        bounds.update(default_bounds)

        if prior_bounds is not None:
            bounds.update(prior_bounds)

        if "a_ratio" in bounds:
            bounds.pop("a_2")

        self.constant_parameters = dict()
        for k, v in kwargs.items():
            if k in bounds:
                bounds.pop(k)
                self.constant_parameters[k] = v

        if "sigma_amp_noise" not in bounds:
            logger.warning("Running without amplitude dependent noise!")

        self.names = list(bounds.keys())
        self.bounds = bounds
        self.log_prior_constant = -np.log(
            self.upper_bounds - self.lower_bounds
        ).sum()

        super().__init__(x_data, y_data)

    @property
    def model_parameters(self) -> list[str]:
        if self._model_parameters is None:
            params = set(
                inspect.signature(self.model_function).parameters.keys()
            )
            self._model_parameters = params
        return self._model_parameters

    def evaluate_constraints(self, x):
        """Evaluate any prior constraints"""
        out = np.ones(x.size, dtype=bool)
        if self.decay_constraint:
            out &= x["tau_1"] > x["tau_2"]
        if self.amplitude_constraint:
            out &= x["a_1"] > x["a_2"]
        return out

    def log_prior(self, x):
        """Compute the log-prior probability"""
        with np.errstate(divide="ignore"):
            return (
                np.log(self.in_bounds(x), dtype="float")
                + np.log(self.evaluate_constraints(x), dtype="float")
                + self.log_prior_constant
            )

    def convert_to_model_parameters(self, x: dict) -> dict:
        x.update(self.constant_parameters)
        if "a_ratio" in x:
            x["a_2"] = x["a_ratio"] * x["a_1"]
        y = {k: x[k] for k in self.model_parameters if k in x}
        return y

    def log_likelihood(self, x: np.ndarray) -> np.ndarray:
        """Compute the log-likelihood"""
        x = live_points_to_dict(x, self.names)
        sigma_amp_noise = x.pop(
            "sigma_amp_noise",
            self.constant_parameters.get("sigma_amp_noise", 0.0),
        )
        sigma_constant_noise = x.pop(
            "sigma_constant_noise",
            self.constant_parameters.get("sigma_constant_noise", 0.0),
        )
        x = self.convert_to_model_parameters(x)

        y_signal = self.model_function(**x)
        sigma2 = (sigma_amp_noise * y_signal) ** 2 + sigma_constant_noise**2
        norm_const = np.log(2 * np.pi * sigma2)
        res = (self.y_data - y_signal) ** 2 / sigma2
        logl = -0.5 * np.sum(norm_const + res)
        return logl

    def signal_model(self, x: np.ndarray) -> np.ndarray:
        x = live_points_to_dict(x, self.names)
        x = self.convert_to_model_parameters(x)
        return self.model_function(**x)


class UniformPriorMixin:
    """Mixin class that defines a uniform prior."""

    def log_prior(self, x: np.ndarray) -> np.ndarray:
        """Log probability for a uniform prior.

        Also checks if samples are within the prior bounds.

        Parameters
        ----------
        x : numpy.ndarray
            Array of samples.

        Returns
        -------
        numpy.ndarray
            Array of log-probabilities.
        """
        with np.errstate(divide="ignore"):
            log_p = np.log(self.in_bounds(x), dtype=float)
        log_p -= np.sum(np.log(self.upper_bounds - self.lower_bounds))
        return log_p

    def to_unit_hypercube(self, x: np.ndarray) -> np.ndarray:
        """Convert the samples to the unit-hypercube.

        Parameters
        ----------
        x : numpy.ndarray
            Array of samples.

        Returns
        -------
        numpy.ndarray
            Array of rescaled samples.
        """
        x_out = x.copy()
        for n in self.names:
            x_out[n] = (x[n] - self.bounds[n][0]) / (
                self.bounds[n][1] - self.bounds[n][0]
            )
        return x_out

    def from_unit_hypercube(self, x: np.ndarray) -> np.ndarray:
        """Convert samples from the unit-hypercube to the prior space.

        Parameters
        ----------
        x : numpy.ndarray
            Array of samples in the unit-hypercube.

        Returns
        -------
        numpy.ndarray
            Array of sample in the prior space.
        """
        x_out = x.copy()
        for n in self.names:
            x_out[n] = (self.bounds[n][1] - self.bounds[n][0]) * x[
                n
            ] + self.bounds[n][0]
        return x_out
