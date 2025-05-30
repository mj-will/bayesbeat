from abc import abstractmethod

import inspect
import logging
from nessai.model import Model
from nessai.livepoint import live_points_to_dict, numpy_array_to_live_points
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class UniformPriorMixin:
    """Mixin class that defines a uniform prior."""

    log10_parameters = None
    """Parameters that are in log10 space."""

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
        if self.log10_parameters:
            raise NotImplementedError
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
        if self.log10_parameters:
            raise NotImplementedError
        x_out = x.copy()
        for n in self.names:
            x_out[n] = (self.bounds[n][1] - self.bounds[n][0]) * x[
                n
            ] + self.bounds[n][0]
        return x_out


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


class TwoNoiseSourceModel(UniformPriorMixin, BaseModel):
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
            "mean_constant_noise": [-5, 5],
        }
        bounds.update(default_bounds)

        if prior_bounds is not None:
            bounds.update(prior_bounds)

        if "log10_a_1" in bounds:
            bounds.pop("a_1")

        if "a_ratio" in bounds:
            bounds.pop("a_2")

        if "log10_sigma_amp_noise" in bounds:
            bounds.pop("sigma_amp_noise")

        self.constant_parameters = dict()
        for k, v in kwargs.items():
            if k in bounds:
                bounds.pop(k)
                self.constant_parameters[k] = v

        if "sigma_amp_noise" not in bounds:
            logger.warning("Running without amplitude dependent noise!")

        self.log10_parameters = [p for p in bounds if p.startswith("log10_")]

        self.names = list(bounds.keys())
        self.bounds = bounds
        log_prior_constant = 0.0
        for p in self.names:
            log_prior_constant -= np.log(self.bounds[p][1] - self.bounds[p][0])
            if p in self.log10_parameters:
                log_prior_constant += np.log(np.log(10))

        self.log_prior_constant = log_prior_constant
        super().__init__(x_data, y_data)

    def new_point(self, N=1):
        """Generate new points in the prior space"""
        x = np.nan * np.ones((N, len(self.names)))
        for i, n in enumerate(self.names):
            if n in self.log10_parameters:
                x[:, i] = np.log10(
                    np.random.uniform(
                        10.0 ** self.bounds[n][0], 10.0 ** self.bounds[n][1], N
                    )
                )
            else:
                x[:, i] = np.random.uniform(
                    self.bounds[n][0], self.bounds[n][1], N
                )
        x = numpy_array_to_live_points(x, self.names)
        return x

    def new_point_log_prob(self, x):
        return self._evaluate_log_prior(x)

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

    def _evaluate_log_prior(self, x):
        # for log10 parameters, we want the prior to be uniform in x
        # so the log-prior is p(x) = 1 / (b - a) * 10^x * ln(10)
        # the log is therefore ln(p(x)) = -ln(b - a) + x * ln(10) + ln(ln(10))
        # The constant terms are include in log_prior_constant
        # so we only include the term that depends on x
        # We sum over the last dimensions since there should n different
        # value of the log-prior for n samples
        return self.log_prior_constant + np.sum(
            [x[p] * np.log(10) for p in self.log10_parameters], axis=0
        )

    def log_prior(self, x):
        """Compute the log-prior probability"""
        with np.errstate(divide="ignore"):
            return (
                np.log(self.in_bounds(x), dtype="float")
                + np.log(self.evaluate_constraints(x), dtype="float")
                + self._evaluate_log_prior(x)
            )

    def convert_to_model_parameters(self, x: dict) -> dict:
        x.update(self.constant_parameters)
        if "log10_a_1" in x:
            x["a_1"] = 10 ** x["log10_a_1"]
        if "a_ratio" in x:
            x["a_2"] = x["a_ratio"] * x["a_1"]
        y = {k: x[k] for k in self.model_parameters if k in x}
        return y

    def log_likelihood(self, x: np.ndarray) -> np.ndarray:
        """Compute the log-likelihood"""
        x = live_points_to_dict(x, self.names)
        if "log10_sigma_amp_noise" in x:
            sigma_amp_noise = 10 ** x.pop("log10_sigma_amp_noise")
        else:
            sigma_amp_noise = x.pop(
                "sigma_amp_noise",
                self.constant_parameters.get("sigma_amp_noise", 0.0),
            )
        sigma_constant_noise = x.pop(
            "sigma_constant_noise",
            self.constant_parameters.get("sigma_constant_noise", 0.0),
        )
        mean_constant_noise = x.pop(
            "mean_constant_noise",
            self.constant_parameters.get("mean_constant_noise", 0.0),
        )
        x = self.convert_to_model_parameters(x)

        if x["a_1"] <= 0 or x["a_2"] <= 0:
            logger.warning("Negative amplitudes")
            logger.warning(f"Setting log-likelihood to -inf")
            return -np.inf

        y_signal = self.model_function(**x)
        sigma2 = (sigma_amp_noise * y_signal) ** 2 + sigma_constant_noise**2
        norm_const = np.log(2 * np.pi * sigma2)
        res = (self.y_data - y_signal - mean_constant_noise) ** 2 / sigma2
        logl = -0.5 * np.sum(norm_const + res)
        return logl

    def signal_model(self, x: np.ndarray) -> np.ndarray:
        x = live_points_to_dict(x, self.names)
        x = self.convert_to_model_parameters(x)
        return self.model_function(**x)
